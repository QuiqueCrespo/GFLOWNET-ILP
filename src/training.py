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
    Theory, Variable, get_all_variables, is_terminal, is_valid_complete_state,
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
        self.action_type = action_type  # 'ADD_ATOM', 'UNIFY_VARIABLES' or 'TERMINATE'
        self.action_detail = action_detail
        self.log_pf = log_pf
        self.next_state = next_state


class TrajectoryReplayBuffer:
    """Replay buffer for high-reward trajectories (off-policy learning)."""

    def __init__(self, capacity: int = 100):
        self.buffer = list()
        self.capacity = capacity

    def add(self, trajectory: List[TrajectoryStep], reward: float):
        """Add trajectory to buffer."""
        if len(self.buffer) >= self.capacity:
            # Remove lowest-reward trajectory if smaller than new reward
            self.buffer.sort(key=lambda x: x[1])  # Sort by reward
            if reward > self.buffer[0][1]:
                self.buffer.pop(0)
                self.buffer.append((trajectory, reward))
        else:
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
                 max_body_length: int = 4,
                 learning_rate: float = 1e-4,
                 exploration_strategy=None,
                 reward_weighted_loss: bool = False,
                 reward_scale_alpha: float = 1.0,
                 use_detailed_balance: bool = False,
                 use_replay_buffer: bool = False,
                 replay_buffer_capacity: int = 50,
                 replay_probability: float = 0.3,
                 buffer_reward_threshold: float = 0.7):
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
        self.max_body_length = max_body_length
        self.predicate_arities = predicate_arities
        self.exploration_strategy = exploration_strategy
        self.reward_weighted_loss = reward_weighted_loss
        self.reward_scale_alpha = reward_scale_alpha
        self.use_detailed_balance = use_detailed_balance
        self.use_replay_buffer = use_replay_buffer
        self.replay_probability = replay_probability
        self.buffer_reward_threshold = buffer_reward_threshold
        self.freeze_encoder = False  # Option to freeze encoder during training

        self.embedding_cache = {}  # Cache for state embeddings

        # Learnable log partition function (for TB loss)
        self.log_Z = torch.nn.Parameter(torch.tensor([0.0]))

        # Replay buffer (for off-policy learning)
        if use_replay_buffer:
            self.replay_buffer = TrajectoryReplayBuffer(capacity=replay_buffer_capacity)
        else:
            self.replay_buffer = None

        # Safety caches
        # Cache states that cover 0 positive examples (should terminate immediately)
        self.zero_positive_cache = set()
        # Cache rewards for visited states to avoid redundant computation
        self.reward_cache = {}  # Key: theory_to_string(state), Value: reward

        # Optimizer
        if self.freeze_encoder:
            all_params = (
                list(gflownet.parameters()) +
                [self.log_Z]
            )
        else:
            all_params = (
                list(state_encoder.parameters()) +
                list(gflownet.parameters()) +
                [self.log_Z]
            )
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

        # --- Get strategist log probability ---
        action_logits = self.gflownet.forward_strategist(state_embedding)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # --- FIX: Handle all three action types ---
        if action_type == 'ADD_ATOM':
            action_idx = 0
        elif action_type == 'UNIFY_VARIABLES':
            action_idx = 1
        elif action_type == 'TERMINATE':
            action_idx = 2
        else:
            raise ValueError(f"Unknown action_type in recompute: {action_type}")
            
        log_prob_action = torch.log(action_probs[action_idx] + 1e-10)
        # --- END FIX ---

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
            variables = get_all_variables(state)
            num_vars = len(variables)

            if not valid_pairs or num_vars < 2:
                return torch.tensor(-1e6) 

            var_embeddings = node_embeddings[:num_vars]
            pair_logits = self.gflownet.forward_variable_unifier(state_embedding, var_embeddings)

            # --- Apply mask to get correct probabilities ---
            var_to_idx = {var.id: i for i, var in enumerate(variables)}
            all_scored_pairs = self.gflownet.variable_unifier.get_pair_indices(num_vars)
            masked_logits = torch.full_like(pair_logits, float('-inf'))
            valid_pair_ids_set = {tuple(sorted((v1.id, v2.id))) for v1, v2 in valid_pairs}

            logit_idx_map = {} # Maps (v_id1, v_id2) -> logit_idx
            for logit_idx, (var_i_idx, var_j_idx) in enumerate(all_scored_pairs):
                var_i = variables[var_i_idx]
                var_j = variables[var_j_idx]
                pair_ids = tuple(sorted((var_i.id, var_j.id)))

                if pair_ids in valid_pair_ids_set:
                    masked_logits[logit_idx] = pair_logits[logit_idx]
                    logit_idx_map[pair_ids] = logit_idx
            # --- End Masking ---

            pair_probs = F.softmax(masked_logits, dim=-1) # Probabilities over *valid* actions

            chosen_pair = action_detail # This is (var1, var2)
            chosen_pair_ids = tuple(sorted((chosen_pair[0].id, chosen_pair[1].id)))

            if chosen_pair_ids in logit_idx_map:
                pair_idx = logit_idx_map[chosen_pair_ids]
                log_prob_detail = torch.log(pair_probs[pair_idx] + 1e-10)
            else:
                return torch.tensor(-1e6) # Action was not valid

        elif action_type == 'TERMINATE':
            # --- FIX: Explicitly handle TERMINATE ---
            log_prob_detail = torch.tensor(0.0)
            # --- END FIX ---

        return log_prob_action + log_prob_detail

    def generate_trajectory(self, initial_state: Theory,
                           positive_examples: List[Example],
                           negative_examples: List[Example],
                           max_steps: int = 10,
                           stochastic: bool = True) -> Tuple[List[TrajectoryStep], float]:
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
        terminated_by_policy = False
        # Reason for forced termination (if any)
        force_termination_reason = None 

        while step_count < max_steps and not terminated_by_policy:
            state_key = theory_to_string(current_state)

            # 1. Check for forced termination (state covers zero positives)
            force_terminate_now = state_key in self.zero_positive_cache
            if force_terminate_now:
                force_termination_reason = "zero_positive_cache"

            # 2. Get state embeddings (from cache or compute)
            state_embedding, node_embeddings = self._get_state_embeddings(
                state_key, current_state
            )

            # 3. Get state properties for masking
            at_max_length = len(current_state[0].body) >= self.max_body_length
            valid_pairs = get_valid_variable_pairs(current_state)
            num_variables = len(get_all_variables(current_state))

            # 4. Get masked action logits from the strategist
            action_logits = self._get_masked_strategist_logits(
                state_embedding, current_state, at_max_length, 
                valid_pairs, num_variables, force_terminate_now,
                step_count
            )

            # 5. Sample action
            action, log_prob_action = self._sample_action_from_logits(action_logits, stochastic)

            # 6. Handle the chosen action
            log_prob_detail = None
            if action == 0:  # ADD_ATOM
                next_state, max_var_id, action_detail, log_prob_detail = \
                    self._handle_action_add_atom(
                        state_embedding, current_state, max_var_id, step_count, stochastic
                    )
            elif action == 1:  # UNIFY_VARIABLES
                next_state, action_detail, log_prob_detail = \
                    self._handle_action_unify_vars(
                        state_embedding, node_embeddings, current_state, valid_pairs, stochastic
                    )
            elif action == 2:  # TERMINATE
                next_state, action_detail, log_prob_detail = \
                    self._handle_action_terminate(current_state)
                terminated_by_policy = True
            else:
                raise ValueError(f"Unknown action index: {action}")

            # 7. Record the step
            log_pf = log_prob_action + log_prob_detail
            trajectory.append(TrajectoryStep(
                state=current_state,
                action_type=action_detail[0],
                action_detail=action_detail[1],
                log_pf=log_pf,
                next_state=next_state
            ))

            if terminated_by_policy:
                break

            # 8. Update state for next iteration
            current_state = next_state
            step_count += 1

        # --- End of Loop ---

        # Check if terminated by max_steps limit
        if step_count >= max_steps and not terminated_by_policy:
            force_termination_reason = "max_steps"

        # Calculate final reward
        reward = self._calculate_final_reward(
            current_state,
            positive_examples,
            negative_examples,
            trajectory,
            force_termination_reason
        )

        return trajectory, reward

    # --- Helper Methods ---

    def _get_state_embeddings(self, state_key: str, current_state: Theory):
        """Gets state embeddings, using cache if available."""
        if self.freeze_encoder and state_key in self.embedding_cache:
            return self.embedding_cache[state_key]
        
        # Not in cache, compute
        graph_data = self.graph_constructor.theory_to_graph(current_state)
        state_embedding, node_embeddings = self.state_encoder(graph_data)
        state_embedding = state_embedding.squeeze(0)  # Remove batch dim
        
        # Cache for potential reuse
        if self.freeze_encoder:
            self.embedding_cache[state_key] = (state_embedding, node_embeddings)
        return state_embedding, node_embeddings

    def _get_masked_strategist_logits(self, state_embedding, current_state: Theory,
                                      at_max_length: bool, valid_pairs: list, 
                                      num_variables: int, force_terminate_now: bool,
                                      step_count: int):
        """Gets logits from the strategist and applies all necessary masks."""
        action_logits = self.gflownet.forward_strategist(state_embedding)

        if self.exploration_strategy:
            action_logits = self.exploration_strategy.modify_logits(
                action_logits,
                state=current_state,
                step_count=step_count
            )

        logits = action_logits.clone()

        # 1. Mask ADD_ATOM at max body length
        if at_max_length:
            logits[0] = float('-inf')

        # 2. Mask TERMINATE for invalid/incomplete states
        # (Unless we are being forced to terminate by the zero_positive_cache)
        if not is_valid_complete_state(current_state) and not force_terminate_now:
            logits[2] = float('-inf') 

        # 3. Mask UNIFY_VARIABLES if not possible
        if not valid_pairs or num_variables < 2:
            logits[1] = float('-inf')

        # 4. If forced to terminate (by zero_positive_cache), mask all
        #    other actions to ensure TERMINATE is chosen.
        if force_terminate_now:
            logits[0] = float('-inf')  # Mask ADD_ATOM
            logits[1] = float('-inf')  # Mask UNIFY_VARIABLES
            # Note: Mask 2 (TERMINATE) is *not* applied, allowing it

        # 5. Add mask to the UNIFY_VARIABLES when there are no body atoms
        if current_state[0].body == []:
            logits[1] = float('-inf')
            
        return logits

    def _sample_action_from_logits(self, logits, stochastic: bool = True):
        """Samples an action from logits and returns action + log_prob."""
        action_probs = F.softmax(logits, dim=-1)
        if stochastic:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = action_probs.argmax().item()
        log_prob = torch.log(action_probs[action] + 1e-10)
        return action, log_prob

    def _handle_action_add_atom(self, state_embedding, current_state: Theory,
                                max_var_id: int, step_count: int, stochastic: bool = True):
        """Handles the ADD_ATOM action logic."""
        atom_logits = self.gflownet.forward_atom_adder(state_embedding)

        if self.exploration_strategy:
            atom_logits = self.exploration_strategy.modify_logits(
                atom_logits,
                state=current_state,
                step_count=step_count
            )

        pred_idx, log_prob_detail = self._sample_action_from_logits(atom_logits, stochastic)
        
        pred_name = self.predicate_vocab[pred_idx]
        pred_arity = self.predicate_arities[pred_name]

        next_state, new_max_var_id = apply_add_atom(
            current_state, pred_name, pred_arity, max_var_id
        )
        action_detail = ('ADD_ATOM', pred_name)
        
        return next_state, new_max_var_id, action_detail, log_prob_detail

    def _handle_action_unify_vars(self, state_embedding, node_embeddings,
                                  current_state: Theory, valid_pairs: list,
                                  stochastic: bool = True):
        """Handles the UNIFY_VARIABLES action logic."""
        variables = get_all_variables(current_state)
        num_vars = len(variables)
        var_embeddings = node_embeddings[:num_vars]

        # Get logits for ALL N*(N-1)/2 pairs
        all_pair_logits = self.gflownet.forward_variable_unifier(
            state_embedding, var_embeddings
        )

        # Get the mapping from logit index to variable IDs
        # Assumes variables are sorted by ID or {var.id: i for i, var in enumerate(variables)}
        var_to_idx = {var.id: i for i, var in enumerate(variables)}
        all_scored_pairs = self.gflownet.variable_unifier.get_pair_indices(num_vars)

        # Create a mask, initializing all logits to -inf
        masked_logits = torch.full_like(all_pair_logits, float('-inf'))

        # Create a map from the logit_index to the corresponding (var1, var2) tuple
        logit_idx_to_action = {}

        # Create a set of valid pairs (by ID) for fast lookup
        valid_pair_ids_set = {tuple(sorted((v1.id, v2.id))) for v1, v2 in valid_pairs}

        for logit_idx, (var_i_idx, var_j_idx) in enumerate(all_scored_pairs):
            var_i = variables[var_i_idx]
            var_j = variables[var_j_idx]

            pair_ids = tuple(sorted((var_i.id, var_j.id)))

            if pair_ids in valid_pair_ids_set:
                # This logit corresponds to a valid action. Unmask it.
                masked_logits[logit_idx] = all_pair_logits[logit_idx]
                # Store the actual Variable objects for this action
                logit_idx_to_action[logit_idx] = (var_i, var_j)

        # Sample from the *masked* logits
        pair_idx_from_all, log_prob_detail = self._sample_action_from_logits(masked_logits, stochastic)

        # Get the chosen action (var1, var2)
        # pair_idx_from_all is the *index in the full logit tensor*
        var1, var2 = logit_idx_to_action[pair_idx_from_all] 

        next_state = apply_unify_vars(current_state, var1, var2)
        action_detail = ('UNIFY_VARIABLES', (var1, var2))

        return next_state, action_detail, log_prob_detail

    def _handle_action_terminate(self, current_state: Theory):
        """Handles the TERMINATE action logic."""
        next_state = current_state  # State remains the same
        action_detail = ('TERMINATE', None)
        log_prob_detail = torch.tensor(0.0)
        return next_state, action_detail, log_prob_detail

    def _calculate_final_reward(self, final_state: Theory,
                                positive_examples: List[Example],
                                negative_examples: List[Example],
                                trajectory: List[TrajectoryStep],
                                force_termination_reason: str = None) -> float:
        """
        Calculates the final reward, applying caches, safety checks, 
        and exploration bonuses.
        """
        # If trajectory was forcibly terminated, assign minimal reward
        if force_termination_reason is not None:
            return 1e-6  # Small positive reward to avoid log(0)

        state_key = theory_to_string(final_state)

        # Use cached reward if available
        if state_key in self.reward_cache:
            reward = self.reward_cache[state_key]
        else:
            # Compute and cache
            scores = self.reward_calculator.get_detailed_scores(
            final_state, positive_examples, negative_examples
            )

            reward = scores['reward']
            if scores['TP'] == 0:
                # This rule covers no positives - cache for future avoidance
                self.zero_positive_cache.add(state_key)
                reward = 1e-6 # Assign minimal reward
            self.reward_cache[state_key] = reward
        


        

        # Apply exploration strategy bonus (if reward is not already minimal)
        if self.exploration_strategy and reward > 1e-6:
            num_atoms = sum(len(rule.body) for rule in final_state)
            num_unique_predicates = len(set(
                atom.predicate_name for rule in final_state for atom in rule.body
            ))

            reward = self.exploration_strategy.modify_reward(
                reward,
                trajectory_length=len(trajectory),
                num_atoms=num_atoms,
                num_unique_predicates=num_unique_predicates
            )

        return reward

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
        scaled_reward = reward ** self.reward_scale_alpha 

        # Log reward
        log_reward = torch.log(torch.tensor(scaled_reward, dtype=torch.float32) + 1e-6)

        # Backward probability: compute using the backward policy model
        # P_B(s|s') = backward probability of going from next_state to state
        sum_log_pb = torch.tensor(0.0, dtype=torch.float32)

        for step in trajectory:
            if step.action_type == 'TERMINATE':
                continue

            # Encode next state to get its embedding
            graph_data_next = self.graph_constructor.theory_to_graph(step.next_state)
            next_state_embedding, next_node_embeddings = self.state_encoder(graph_data_next)
            next_state_embedding = next_state_embedding.squeeze(0)

            # For UNIFY_VARIABLES, we also need embeddings from PREVIOUS state
            # since the backward policy needs to predict which pair in prev_state was unified
            prev_var_embeddings = None
            if step.action_type == 'UNIFY_VARIABLES':
                # Encode previous state to get variable embeddings
                graph_data_prev = self.graph_constructor.theory_to_graph(step.state)
                _, prev_node_embeddings = self.state_encoder(graph_data_prev)

                # Extract variable embeddings from previous state
                prev_variables = get_all_variables(step.state)
                prev_var_embeddings = prev_node_embeddings[:len(prev_variables)] if len(prev_variables) > 0 else None

            # Get log probability of backward transition from next_state to state
            log_pb_step = self.gflownet.get_backward_log_probability(
                next_state_embedding,
                step.next_state,
                step.state,
                step.action_type,
                step.action_detail,
                prev_var_embeddings  # Now passing prev_var_embeddings for UNIFY
            )
            sum_log_pb = sum_log_pb + log_pb_step

        # Trajectory Balance loss
        loss = (self.log_Z + sum_log_pf - log_reward - sum_log_pb) ** 2

        # Apply reward weighting if configured
        if self.reward_weighted_loss:
            # Weight by reward: high-reward trajectories get more weight
            weight = reward / (reward + 0.1)
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
            

            # Current state embedding
            graph_data = self.graph_constructor.theory_to_graph(step.state)
            state_embedding, _ = self.state_encoder(graph_data)
            state_embedding = state_embedding.squeeze(0)
            # Next state embedding
            graph_data_next = self.graph_constructor.theory_to_graph(step.next_state)
            next_state_embedding, node_embeddings_next = self.state_encoder(graph_data_next)
            next_state_embedding = next_state_embedding.squeeze(0)

            # Forward transition: s -> s'
            # log F(s) + log P_F(s->s')
            log_F_s = self.gflownet.forward_flow(state_embedding)

            # Forward flow
            log_forward = log_F_s + step.log_pf

            # Backward flow: log F(s') + log P_B(s'->s)
            if i == len(trajectory) - 1:
                # Terminal state: F(s_terminal) = R
                scaled_reward = reward ** self.reward_scale_alpha
                log_F_s_next = torch.log(torch.tensor(scaled_reward, dtype=torch.float32) + 1e-6)
            else:
                # Non-terminal: accumulate forward flow
                log_F_s_next = self.gflownet.forward_flow(next_state_embedding).squeeze()

            # Backward probability: P_B(s'->s)
            # The backward policy gives the probability of going back from next_state to state
            # For UNIFY_VARIABLES, we need embeddings from PREVIOUS state
            prev_var_embeddings = None
            if step.action_type == 'UNIFY_VARIABLES':
                # Encode previous state to get variable embeddings
                graph_data_prev = self.graph_constructor.theory_to_graph(step.state)
                _, prev_node_embeddings = self.state_encoder(graph_data_prev)

                # Extract variable embeddings from previous state
                prev_variables = get_all_variables(step.state)
                prev_var_embeddings = prev_node_embeddings[:len(prev_variables)] if len(prev_variables) > 0 else None

            log_pb = self.gflownet.get_backward_log_probability(
                next_state_embedding,
                step.next_state,
                step.state,
                step.action_type,
                step.action_detail,
                prev_var_embeddings  # Now passing prev_var_embeddings for UNIFY
            )
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
        
        metrics_info = {'replay_used': False}

        # --- Off-Policy Step (Replay Buffer) ---
        use_replay = (self.replay_buffer is not None and
                     len(self.replay_buffer) > 0 and
                     random.random() < self.replay_probability)

        
        if use_replay:

            
            replayed_trajectory, replay_reward = self.replay_buffer.sample(1)[0]
            
            # Re-compute log probabilities for the replayed trajectory
            recomputed_trajectory_for_loss = [
                TrajectoryStep(s.state, s.action_type, s.action_detail, self._recompute_step_log_pf(s.state, s.action_type, s.action_detail), s.next_state)
                for s in replayed_trajectory
            ]

            if self.use_detailed_balance:
                off_policy_loss = self.compute_detailed_balance_loss(recomputed_trajectory_for_loss, replay_reward)
            else:
                off_policy_loss = self.compute_trajectory_balance_loss(recomputed_trajectory_for_loss, replay_reward)

            # Add the off-policy loss to the total loss for a combined gradient update
            total_loss = off_policy_loss
            metrics_info = {'replay_used': True, 'replay_reward': replay_reward}
            on_policy_reward = np.nan
            on_policy_trajectory = []

        else:
            # --- On-Policy Step ---
            # Generate a new trajectory using the current policy
            on_policy_trajectory, on_policy_reward = self.generate_trajectory(
                initial_state, positive_examples, negative_examples
            )

            if not on_policy_trajectory:
                return {'loss': 0.0, 'reward': on_policy_reward, 'trajectory_length': 0}

            # Add high-reward trajectories to the replay buffer
            if self.replay_buffer is not None and on_policy_reward > self.buffer_reward_threshold:
                self.replay_buffer.add(on_policy_trajectory, on_policy_reward)

            # Calculate the loss for the new (on-policy) trajectory
            if self.use_detailed_balance:
                total_loss = self.compute_detailed_balance_loss(on_policy_trajectory, on_policy_reward)
            else:
                total_loss = self.compute_trajectory_balance_loss(on_policy_trajectory, on_policy_reward)

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
            'on_policy_reward': on_policy_reward,
            'trajectory_length': len(on_policy_trajectory) if on_policy_trajectory else len(replayed_trajectory) if use_replay else 0,
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
