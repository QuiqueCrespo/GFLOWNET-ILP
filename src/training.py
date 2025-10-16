"""
Training loop with Trajectory Balance objective for GFlowNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import deque
import random

from .logic_structures import (
    Theory, Variable, get_all_variables, is_terminal,
    apply_add_atom, apply_unify_vars, get_valid_variable_pairs,
    theory_to_string
)
from .logic_engine import Example
from .graph_encoder import GraphConstructor, StateEncoder
from .gflownet_models import HierarchicalGFlowNet
from .reward import RewardCalculator


class TrajectoryStep:
    """Represents a single step in a trajectory."""

    def __init__(self, state: Theory, action_type: str, action_detail: any,
                 log_pf: torch.Tensor, next_state: Theory):
        self.state = state
        self.action_type = action_type  # 'ADD_ATOM' or 'UNIFY_VARIABLES'
        self.action_detail = action_detail
        self.log_pf = log_pf
        self.next_state = next_state


class TrajectoryReplayBuffer:
    """Replay buffer for high-reward trajectories (off-policy learning)."""

    def __init__(self, capacity: int = 100):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, trajectory: List[TrajectoryStep], reward: float):
        """Add trajectory to buffer."""
        self.buffer.append((trajectory, reward))

    def sample(self, n: int = 1) -> List[Tuple[List[TrajectoryStep], float]]:
        """Sample n trajectories with probability proportional to reward."""
        if len(self.buffer) == 0:
            return []

        # Extract rewards
        rewards = np.array([r for _, r in self.buffer])

        # Avoid negative rewards causing issues
        rewards = np.maximum(rewards, 0.001)

        # Sample with probability proportional to reward
        probs = rewards / rewards.sum()
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), p=probs, replace=False)

        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class GFlowNetTrainer:
    """Trainer for hierarchical GFlowNet using Trajectory Balance."""

    def __init__(self,
                 state_encoder: StateEncoder,
                 gflownet: HierarchicalGFlowNet,
                 graph_constructor: GraphConstructor,
                 reward_calculator: RewardCalculator,
                 predicate_vocab: List[str],
                 predicate_arities: Dict[str, int],
                 learning_rate: float = 1e-4,
                 exploration_strategy=None,
                 reward_weighted_loss: bool = False,
                 reward_scale_alpha: float = 1.0,
                 use_detailed_balance: bool = False,
                 use_replay_buffer: bool = False,
                 replay_buffer_capacity: int = 50,
                 replay_probability: float = 0.3):
        """
        Args:
            state_encoder: GNN encoder for states
            gflownet: Hierarchical GFlowNet model
            graph_constructor: Converts theories to graphs
            reward_calculator: Calculates rewards
            predicate_vocab: List of available predicates
            predicate_arities: Mapping from predicate name to arity
            learning_rate: Learning rate for optimizer
            exploration_strategy: Optional ExplorationStrategy instance
            reward_weighted_loss: If True, weight TB loss by reward
            reward_scale_alpha: Exponent for reward scaling (reward^alpha)
            use_detailed_balance: If True, use detailed balance loss instead of TB
            use_replay_buffer: If True, use replay buffer for off-policy learning
            replay_buffer_capacity: Maximum number of trajectories in replay buffer
            replay_probability: Probability of training on replayed trajectory
        """
        self.state_encoder = state_encoder
        self.gflownet = gflownet
        self.graph_constructor = graph_constructor
        self.reward_calculator = reward_calculator
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities
        self.exploration_strategy = exploration_strategy
        self.reward_weighted_loss = reward_weighted_loss
        self.reward_scale_alpha = reward_scale_alpha
        self.use_detailed_balance = use_detailed_balance
        self.use_replay_buffer = use_replay_buffer
        self.replay_probability = replay_probability

        # Learnable log partition function (for TB loss)
        self.log_Z = torch.nn.Parameter(torch.tensor([0.0]))

        # Replay buffer (for off-policy learning)
        if use_replay_buffer:
            self.replay_buffer = TrajectoryReplayBuffer(capacity=replay_buffer_capacity)
        else:
            self.replay_buffer = None

        # Optimizer
        all_params = (list(state_encoder.parameters()) +
                     list(gflownet.parameters()) +
                     [self.log_Z])
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate)
    def _recompute_step_log_pf(self, state: Theory, action_type: str, action_detail: any) -> torch.Tensor:
        """
        Re-computes the forward log probability for a single state-action step
        using the current model parameters.
        """
        # Encode the state into an embedding
        graph_data = self.graph_constructor.theory_to_graph(state)
        state_embedding, node_embeddings = self.state_encoder(graph_data)
        state_embedding = state_embedding.squeeze(0)

        # --- Get strategist log probability (ADD_ATOM vs UNIFY_VARIABLES) ---
        action_logits, _ = self.gflownet.forward_strategist(state_embedding)
        action_probs = F.softmax(action_logits, dim=-1)
        action_idx = 0 if action_type == 'ADD_ATOM' else 1
        log_prob_action = torch.log(action_probs[action_idx] + 1e-10)

        log_prob_detail = torch.tensor(0.0)

        # --- Get detailed action log probability ---
        if action_type == 'ADD_ATOM':
            atom_logits = self.gflownet.forward_atom_adder(state_embedding)
            atom_probs = F.softmax(atom_logits, dim=-1)
            pred_name = action_detail
            pred_idx = self.predicate_vocab.index(pred_name)
            log_prob_detail = torch.log(atom_probs[pred_idx] + 1e-10)

        elif action_type == 'UNIFY_VARIABLES':
            valid_pairs = get_valid_variable_pairs(state)
            # Ensure the action is still possible
            if not valid_pairs or len(get_all_variables(state)) < 2:
                return torch.tensor(-1e6)  # Penalize if action is no longer valid

            var_embeddings = node_embeddings[:len(get_all_variables(state))]
            pair_logits = self.gflownet.forward_variable_unifier(state_embedding, var_embeddings)
            pair_probs = F.softmax(pair_logits, dim=-1)

            chosen_pair = action_detail
            # Normalize pair representation (e.g., sort by id) for consistent lookup
            chosen_pair_ids = tuple(sorted((chosen_pair[0].id, chosen_pair[1].id)))
            
            pair_idx = -1
            for i, (v1, v2) in enumerate(valid_pairs):
                if tuple(sorted((v1.id, v2.id))) == chosen_pair_ids:
                    pair_idx = i
                    break
            
            if pair_idx != -1:
                log_prob_detail = torch.log(pair_probs[pair_idx] + 1e-10)
            else:
                return torch.tensor(-1e6) # Penalize if specific pair is no longer valid

        return log_prob_action + log_prob_detail

    def generate_trajectory(self, initial_state: Theory,
                           positive_examples: List[Example],
                           negative_examples: List[Example],
                           max_steps: int = 10) -> Tuple[List[TrajectoryStep], float]:
        """
        Generate a single trajectory by sampling from the current policy.

        Returns:
            - trajectory: List of trajectory steps
            - reward: Final reward for the terminal state
        """
        trajectory = []
        current_state = initial_state
        max_var_id = max([v.id for v in get_all_variables(current_state)], default=-1)
        step_count = 0

        while not is_terminal(current_state) and step_count < max_steps:
            # Encode current state
            graph_data = self.graph_constructor.theory_to_graph(current_state)
            state_embedding, node_embeddings = self.state_encoder(graph_data)
            state_embedding = state_embedding.squeeze(0)  # Remove batch dim

            # Check if body length limit reached
            rule = current_state[0]
            body_length = len(rule.body)
            max_body_length = 3
            at_max_length = body_length >= max_body_length

            # Get strategist action
            action_logits, _ = self.gflownet.forward_strategist(state_embedding)

            # Apply exploration strategy to logits
            if self.exploration_strategy:
                action_logits = self.exploration_strategy.modify_logits(
                    action_logits,
                    state=current_state,
                    step_count=step_count
                )

            # Apply action mask: prevent ADD_ATOM at max body length
            if at_max_length:
                # Mask out ADD_ATOM (index 0) by setting to -inf
                action_logits = action_logits.clone()
                action_logits[0] = float('-inf')

            action_probs = F.softmax(action_logits, dim=-1)

            # Sample action: 0=ADD_ATOM, 1=UNIFY_VARIABLES
            action = torch.multinomial(action_probs, 1).item()
            log_prob_action = torch.log(action_probs[action] + 1e-10)

            next_state = None
            log_prob_detail = None
            action_failed = False

            if action == 0:  # ADD_ATOM
                # Get atom adder logits
                atom_logits = self.gflownet.forward_atom_adder(state_embedding)

                # Apply exploration strategy to logits
                if self.exploration_strategy:
                    atom_logits = self.exploration_strategy.modify_logits(
                        atom_logits,
                        state=current_state,
                        step_count=step_count
                    )

                atom_probs = F.softmax(atom_logits, dim=-1)

                # Sample predicate
                pred_idx = torch.multinomial(atom_probs, 1).item()
                pred_name = self.predicate_vocab[pred_idx]
                pred_arity = self.predicate_arities[pred_name]

                log_prob_detail = torch.log(atom_probs[pred_idx] + 1e-10)

                # Apply action
                next_state, max_var_id = apply_add_atom(
                    current_state, pred_name, pred_arity, max_var_id
                )
                action_detail = ('ADD_ATOM', pred_name)

            else:  # UNIFY_VARIABLES
                # Get valid variable pairs
                valid_pairs = get_valid_variable_pairs(current_state)

                if not valid_pairs:
                    # No valid pairs - fall back to ADD_ATOM
                    action_failed = True

                # Get variable embeddings
                if not action_failed:
                    var_to_node = self.graph_constructor.get_variable_node_ids(current_state)
                    variables = get_all_variables(current_state)

                    if len(variables) < 2:
                        # Not enough variables - fall back to ADD_ATOM
                        action_failed = True

                if not action_failed:
                    var_embeddings = node_embeddings[:len(variables)]

                    # Get unifier logits
                    pair_logits = self.gflownet.forward_variable_unifier(
                        state_embedding, var_embeddings
                    )

                    if len(pair_logits) == 0:
                        # No valid pairs - fall back to ADD_ATOM
                        action_failed = True

                if action_failed:
                    # UNIFY_VARIABLES failed
                    if not at_max_length:
                        # Not at max length, fall back to ADD_ATOM instead
                        atom_logits = self.gflownet.forward_atom_adder(state_embedding)
                        if self.exploration_strategy:
                            atom_logits = self.exploration_strategy.modify_logits(
                                atom_logits,
                                state=current_state,
                                step_count=step_count
                            )
                        atom_probs = F.softmax(atom_logits, dim=-1)
                        pred_idx = torch.multinomial(atom_probs, 1).item()
                        pred_name = self.predicate_vocab[pred_idx]
                        pred_arity = self.predicate_arities[pred_name]
                        log_prob_detail = torch.log(atom_probs[pred_idx] + 1e-10)
                        next_state, max_var_id = apply_add_atom(
                            current_state, pred_name, pred_arity, max_var_id
                        )
                        action_detail = ('ADD_ATOM', pred_name)
                    # If at max length and action failed, next_state stays None
                    # which will cause an error when recording trajectory
                    # So we need to skip this iteration without recording
                    if next_state is None:
                        continue
                else:
                    # UNIFY_VARIABLES succeeded
                    pair_probs = F.softmax(pair_logits, dim=-1)
                    pair_idx = torch.multinomial(pair_probs, 1).item()
                    log_prob_detail = torch.log(pair_probs[pair_idx] + 1e-10)
                    var1, var2 = valid_pairs[pair_idx]
                    next_state = apply_unify_vars(current_state, var1, var2)
                    action_detail = ('UNIFY_VARIABLES', (var1, var2))

            # Combined log probability
            log_pf = log_prob_action + log_prob_detail

            # Record step
            trajectory.append(TrajectoryStep(
                state=current_state,
                action_type=action_detail[0],
                action_detail=action_detail[1],
                log_pf=log_pf,
                next_state=next_state
            ))

            current_state = next_state
            step_count += 1

        # Calculate reward for final state
        reward = self.reward_calculator.calculate_reward(
            current_state, positive_examples, negative_examples
        )

        # Apply exploration strategy to reward
        if self.exploration_strategy:
            # Get rule features for curiosity bonus
            num_atoms = sum(len(rule.body) for rule in current_state)
            num_unique_predicates = len(set(
                atom.predicate_name for rule in current_state for atom in rule.body
            ))

            reward = self.exploration_strategy.modify_reward(
                reward,
                trajectory_length=len(trajectory),
                num_atoms=num_atoms,
                num_unique_predicates=num_unique_predicates
            )

        return trajectory, reward

    def compute_trajectory_balance_loss(self, trajectory: List[TrajectoryStep],
                                       reward: float) -> torch.Tensor:
        """
        Compute Trajectory Balance loss for a trajectory.

        TB Loss: (log Z + sum(log P_F) - log R - sum(log P_B))^2

        With optional reward weighting and scaling.
        """
        # Sum of forward log probabilities
        sum_log_pf = sum(step.log_pf for step in trajectory)

        # Apply reward scaling if configured
        scaled_reward = reward ** self.reward_scale_alpha if self.reward_scale_alpha != 1.0 else reward

        # Log reward
        log_reward = torch.log(torch.tensor(scaled_reward, dtype=torch.float32) + 1e-8)

        # Backward probability: use uniform backward policy
        # P_B(s|s') = 1 / num_possible_parents(s')
        # For simplicity, assume uniform backward: log P_B ≈ -log(num_actions)
        # This is a simplified approximation
        sum_log_pb = -np.log(len(self.predicate_vocab) + 10) * len(trajectory)
        sum_log_pb = torch.tensor(sum_log_pb, dtype=torch.float32)

        # Trajectory Balance loss
        loss = (self.log_Z + sum_log_pf - log_reward - sum_log_pb) ** 2

        # Apply reward weighting if configured
        if self.reward_weighted_loss:
            # Weight by reward: high-reward trajectories get more weight
            weight = reward / (reward + 0.1)  # Normalized weight
            loss = weight * loss

        return loss

    def compute_detailed_balance_loss(self, trajectory: List[TrajectoryStep],
                                     reward: float) -> torch.Tensor:
        """
        Compute Detailed Balance loss for a trajectory.

        DB enforces: F(s) * P_F(s->s') = F(s') * P_B(s'->s)
        With terminal constraint: F(s_terminal) = R(s_terminal)

        This removes the single-Z bottleneck by learning state-specific flows.
        """
        if len(trajectory) == 0:
            return torch.tensor(0.0)

        losses = []

        # For each transition, compute detailed balance constraint
        for i, step in enumerate(trajectory):
            # Encode current and next state to get flow estimates
            # For simplicity, use log_Z as initial flow and propagate
            # In full implementation, would use separate flow network

            # Forward transition: s -> s'
            # log F(s) + log P_F(s->s')
            if i == 0:
                # Initial state flow is log_Z
                log_F_s = self.log_Z
            else:
                # Flow accumulates from initial state
                # This is simplified; ideally would use state-specific flow network
                log_F_s = self.log_Z + sum(trajectory[j].log_pf for j in range(i))

            # Forward flow
            log_forward = log_F_s + step.log_pf

            # Backward flow: log F(s') + log P_B(s'->s)
            if i == len(trajectory) - 1:
                # Terminal state: F(s_terminal) = R
                scaled_reward = reward ** self.reward_scale_alpha if self.reward_scale_alpha != 1.0 else reward
                log_F_s_next = torch.log(torch.tensor(scaled_reward, dtype=torch.float32) + 1e-8)
            else:
                # Non-terminal: accumulate forward flow
                log_F_s_next = self.log_Z + sum(trajectory[j].log_pf for j in range(i + 1))

            # Backward probability (uniform approximation)
            log_pb = torch.tensor(-np.log(len(self.predicate_vocab) + 10), dtype=torch.float32)
            log_backward = log_F_s_next + log_pb

            # Detailed balance constraint: forward flow = backward flow
            db_loss = (log_forward - log_backward) ** 2
            losses.append(db_loss)

        # Average loss over all transitions
        total_loss = torch.stack(losses).mean()

        # Apply reward weighting if configured
        if self.reward_weighted_loss:
            weight = reward / (reward + 0.1)
            total_loss = weight * total_loss

        return total_loss

    def train_step(self, initial_state: Theory,
                   positive_examples: List[Example],
                   negative_examples: List[Example]) -> Dict[str, float]:
        """
        Perform one training step, correctly handling on-policy and off-policy (replay) data.
        """
        # --- On-Policy Step ---
        # Generate a new trajectory using the current policy
        on_policy_trajectory, on_policy_reward = self.generate_trajectory(
            initial_state, positive_examples, negative_examples
        )

        if not on_policy_trajectory:
            return {'loss': 0.0, 'reward': on_policy_reward, 'trajectory_length': 0}

        # Add high-reward trajectories to the replay buffer
        if self.replay_buffer is not None and on_policy_reward > 0.7:
            self.replay_buffer.add(on_policy_trajectory, on_policy_reward)

        # Calculate the loss for the new (on-policy) trajectory
        if self.use_detailed_balance:
            total_loss = self.compute_detailed_balance_loss(on_policy_trajectory, on_policy_reward)
        else:
            total_loss = self.compute_trajectory_balance_loss(on_policy_trajectory, on_policy_reward)

        metrics_info = {'replay_used': False}

        # --- Off-Policy Step (Replay Buffer) ---
        use_replay = (self.replay_buffer is not None and
                     len(self.replay_buffer) > 0 and
                     random.random() < self.replay_probability)
        
        if use_replay:
            replayed_trajectory, replay_reward = self.replay_buffer.sample(1)[0]
            
            # **FIX:** Re-compute log probabilities for the replayed trajectory
            # using the current model. This is crucial for off-policy training.
            recomputed_log_pf_sum = torch.tensor(0.0)
            for step in replayed_trajectory:
                recomputed_log_pf_sum += self._recompute_step_log_pf(
                    step.state, step.action_type, step.action_detail
                )

            # Create a temporary trajectory object with a single summed log_pf for the loss function
            # This avoids creating a whole new list of TrajectoryStep objects.
            # NOTE: This assumes loss functions only need the sum of log_pf and the reward.
            # If they need more, this approach needs to be adjusted.
            
            # To make it work with your current loss functions, we'll build a new trajectory list.
            recomputed_trajectory_for_loss = [
                TrajectoryStep(s.state, s.action_type, s.action_detail, self._recompute_step_log_pf(s.state, s.action_type, s.action_detail), s.next_state)
                for s in replayed_trajectory
            ]

            if self.use_detailed_balance:
                off_policy_loss = self.compute_detailed_balance_loss(recomputed_trajectory_for_loss, replay_reward)
            else:
                off_policy_loss = self.compute_trajectory_balance_loss(recomputed_trajectory_for_loss, replay_reward)

            # Add the off-policy loss to the total loss for a combined gradient update
            total_loss += off_policy_loss
            metrics_info = {'replay_used': True, 'replay_reward': replay_reward}

        # Apply exploration strategy modifications to the final combined loss
        if self.exploration_strategy:
            total_loss = self.exploration_strategy.modify_loss(total_loss)

        # --- Optimization ---
        # Perform a SINGLE backward pass on the combined loss
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        # Step the exploration strategy (e.g., for parameter decay)
        if self.exploration_strategy:
            self.exploration_strategy.step()

        metrics = {
            'loss': total_loss.item(),
            'reward': on_policy_reward,
            'trajectory_length': len(on_policy_trajectory),
            'log_Z': self.log_Z.item()
        }
        metrics.update(metrics_info)

        return metrics

    def train(self, initial_state: Theory,
              positive_examples: List[Example],
              negative_examples: List[Example],
              num_episodes: int = 100,
              verbose: bool = True) -> List[Dict[str, float]]:
        """
        Train the GFlowNet for multiple episodes.

        Returns:
            List of metrics for each episode
        """
        history = []

        for episode in range(num_episodes):
            metrics = self.train_step(initial_state, positive_examples, negative_examples)
            history.append(metrics)

            if verbose and (episode % 10 == 0 or episode == num_episodes - 1):
                print(f"Episode {episode:3d} | Loss: {metrics['loss']:.4f} | "
                      f"Reward: {metrics['reward']:.4f} | "
                      f"Steps: {metrics['trajectory_length']} | "
                      f"log_Z: {metrics['log_Z']:.4f}")

        return history

    def sample_best_theory(self, initial_state: Theory,
                          positive_examples: List[Example],
                          negative_examples: List[Example],
                          num_samples: int = 10) -> Tuple[Theory, float]:
        """
        Sample multiple theories and return the one with highest reward.

        For top-N hypotheses, use sample_top_theories() instead.
        """
        theories = self.sample_top_theories(
            initial_state, positive_examples, negative_examples,
            num_samples=num_samples, top_k=1
        )

        if theories:
            return theories[0]
        else:
            return initial_state, 0.0

    def sample_top_theories(self, initial_state: Theory,
                           positive_examples: List[Example],
                           negative_examples: List[Example],
                           num_samples: int = 10,
                           top_k: int = 5) -> List[Tuple[Theory, float]]:
        """
        Sample multiple theories and return the top K by reward.

        Args:
            initial_state: Starting theory state
            positive_examples: Positive training examples
            negative_examples: Negative training examples
            num_samples: Number of theories to sample
            top_k: Number of top theories to return

        Returns:
            List of (theory, reward) tuples, sorted by reward (highest first)
        """
        sampled_theories = []

        for _ in range(num_samples):
            trajectory, reward = self.generate_trajectory(
                initial_state, positive_examples, negative_examples
            )

            if trajectory:
                final_theory = trajectory[-1].next_state
                sampled_theories.append((final_theory, reward))

        # Sort by reward (descending)
        sampled_theories.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return sampled_theories[:top_k]
