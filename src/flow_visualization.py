"""
Flow Visualization: Track predicted flow for states N steps from origin.

This module visualizes how the model's predicted flow values (log F(s))
evolve during training for states at a fixed distance from the initial state.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import os

from .logic_structures import (
    Theory, get_initial_state, apply_add_atom, apply_unify_vars,
    get_valid_variable_pairs, theory_to_string
)
from .logic_engine import Example


class FlowVisualizer:
    """
    Visualizes predicted flow values for states at fixed depth from origin.

    Tracks:
    - Predicted log_F(s) from forward_flow network
    - Actual rewards R(s)
    - Evolution over training episodes
    """

    def __init__(self,
                 trainer,
                 target_predicate: str,
                 arity: int,
                 predicate_vocab: List[str],
                 predicate_arities: Dict[str, int],
                 positive_examples: List[Example],
                 negative_examples: List[Example],
                 max_depth: int = 4,
                 output_dir: str = "flow_visualizations"):
        """
        Args:
            trainer: GFlowNetTrainer instance
            target_predicate: Target predicate name (e.g., 'grandparent')
            arity: Target predicate arity
            predicate_vocab: List of available predicates
            predicate_arities: Dict mapping predicates to arities
            positive_examples: Positive examples for reward calculation
            negative_examples: Negative examples for reward calculation
            max_depth: Number of steps from origin to explore
            output_dir: Directory to save visualizations
        """
        self.trainer = trainer
        self.target_predicate = target_predicate
        self.arity = arity
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.max_depth = max_depth
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Storage for tracking over time
        self.episode_snapshots = []  # List of (episode, state_flows, state_rewards)

        # Generate states at target depth
        self.initial_state = get_initial_state(target_predicate, arity)
        self.target_states = self._generate_states_at_depth(self.initial_state, max_depth)

        print(f"FlowVisualizer initialized:")
        print(f"  Target depth: {max_depth}")
        print(f"  States at depth {max_depth}: {len(self.target_states)}")
        print(f"  Output directory: {output_dir}")

    def _generate_states_at_depth(self, initial_state: Theory, depth: int) -> List[Tuple[Theory, str]]:
        """
        Generate all reachable states at exactly 'depth' steps from initial state.

        Returns:
            List of (state, action_sequence) tuples
        """
        if depth == 0:
            return [(initial_state, "")]

        # BFS to generate states
        current_level = [(initial_state, "", 0)]  # (state, action_seq, max_var_id)
        states_at_depth = []

        for current_depth in range(depth):
            next_level = []

            for state, action_seq, max_var_id in current_level:
                # Get max_var_id if not provided
                if max_var_id == 0:
                    from .logic_structures import get_all_variables
                    vars_in_state = get_all_variables(state)
                    max_var_id = max([v.id for v in vars_in_state], default=-1)

                # Try ADD_ATOM actions
                for pred_name in self.predicate_vocab:
                    pred_arity = self.predicate_arities[pred_name]
                    new_state, new_max_var_id = apply_add_atom(
                        state, pred_name, pred_arity, max_var_id
                    )
                    new_action_seq = action_seq + f"A({pred_name})→"

                    if current_depth == depth - 1:
                        states_at_depth.append((new_state, new_action_seq))
                    else:
                        next_level.append((new_state, new_action_seq, new_max_var_id))

                # Try UNIFY_VARIABLES actions (if body is not empty)
                if len(state[0].body) > 0:
                    valid_pairs = get_valid_variable_pairs(state)

                    for var1, var2 in valid_pairs[:10]:  # Limit to prevent explosion
                        new_state = apply_unify_vars(state, var1, var2)
                        new_action_seq = action_seq + f"U({var1.id},{var2.id})→"

                        if current_depth == depth - 1:
                            states_at_depth.append((new_state, new_action_seq))
                        else:
                            next_level.append((new_state, new_action_seq, max_var_id))

            current_level = next_level

            # Limit states to prevent memory explosion
            if len(current_level) > 100:
                print(f"  Warning: Limiting states at depth {current_depth+1} to 100")
                current_level = current_level[:100]

        # Remove duplicates by state string representation
        unique_states = {}
        for state, action_seq in states_at_depth:
            state_str = theory_to_string(state)
            if state_str not in unique_states:
                unique_states[state_str] = (state, action_seq)

        return list(unique_states.values())

    def record_snapshot(self, episode: int):
        """
        Record predicted flows and actual rewards for target states at current episode.

        Args:
            episode: Current training episode number
        """
        state_flows = {}
        state_rewards = {}

        for state, action_seq in self.target_states:
            state_str = theory_to_string(state)

            # Get predicted log_F(s) from forward_flow network
            with torch.no_grad():
                graph_data = self.trainer.graph_constructor.theory_to_graph(state)
                state_embedding, _ = self.trainer.state_encoder(graph_data)
                state_embedding = state_embedding.squeeze(0)

                log_F = self.trainer.gflownet.forward_flow(state_embedding)
                state_flows[state_str] = log_F.item()

            # Get actual reward
            reward = self.trainer.reward_calculator.calculate_reward(
                state, self.positive_examples, self.negative_examples
            )
            state_rewards[state_str] = reward

        self.episode_snapshots.append({
            'episode': episode,
            'flows': state_flows,
            'rewards': state_rewards
        })

        print(f"Snapshot recorded at episode {episode}: {len(state_flows)} states")

    def plot_flow_evolution(self):
        """
        Create comprehensive visualization of flow evolution.

        Generates multiple plots:
        1. Flow vs Episode for top states (by reward)
        2. Flow vs Reward correlation over time
        3. Flow distribution evolution
        4. Individual state trajectories
        """
        if len(self.episode_snapshots) < 2:
            print("Need at least 2 snapshots to plot evolution")
            return

        # Collect data
        episodes = [snap['episode'] for snap in self.episode_snapshots]

        # Get top states by final reward
        final_rewards = self.episode_snapshots[-1]['rewards']
        sorted_states = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
        top_10_states = [state_str for state_str, _ in sorted_states[:10]]
        bottom_5_states = [state_str for state_str, _ in sorted_states[-5:]]

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ============= Plot 1: Flow evolution for top states =============
        ax1 = fig.add_subplot(gs[0, :2])

        # Define unique styles for top 5 states
        COLORS_TOP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        MARKERS_TOP = ['o', 's', '^', 'v', 'D']

        for i, state_str in enumerate(top_10_states[:5]):
            flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
            reward = final_rewards[state_str]

            # Abbreviate state string for legend
            state_abbrev = state_str[:40] + "..." if len(state_str) > 40 else state_str
            ax1.plot(episodes, flows,
                    color=COLORS_TOP[i],
                    marker=MARKERS_TOP[i],
                    linestyle='-',
                    label=f"R={reward:.3f}: {state_abbrev}",
                    linewidth=2, markersize=4)

        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Predicted log F(s)', fontsize=12)
        ax1.set_title('Flow Evolution: Top 5 States by Reward', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(alpha=0.3)

        # ============= Plot 2: Flow evolution for bottom states =============
        ax2 = fig.add_subplot(gs[0, 2])

        # Define unique styles for bottom 3 states
        COLORS_BOTTOM = ['#d62728', '#8c564b', '#e377c2']
        MARKERS_BOTTOM = ['x', '+', '*']
        LINESTYLES_BOTTOM = ['--', '-.', ':']

        for i, state_str in enumerate(bottom_5_states[:3]):
            flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
            reward = final_rewards[state_str]

            state_abbrev = state_str[:30] + "..." if len(state_str) > 30 else state_str
            ax2.plot(episodes, flows,
                    color=COLORS_BOTTOM[i],
                    marker=MARKERS_BOTTOM[i],
                    linestyle=LINESTYLES_BOTTOM[i],
                    label=f"R={reward:.3f}",
                    linewidth=2, markersize=4)

        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Predicted log F(s)', fontsize=12)
        ax2.set_title('Flow Evolution: Bottom 3 States', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(alpha=0.3)

        # ============= Plot 3: Flow vs Reward correlation (early) =============
        ax3 = fig.add_subplot(gs[1, 0])

        early_snap = self.episode_snapshots[min(3, len(self.episode_snapshots)-1)]
        early_flows = list(early_snap['flows'].values())
        early_rewards = list(early_snap['rewards'].values())

        ax3.scatter(early_rewards, early_flows, alpha=0.6, s=50, c='blue')
        ax3.set_xlabel('Actual Reward R(s)', fontsize=12)
        ax3.set_ylabel('Predicted log F(s)', fontsize=12)
        ax3.set_title(f'Flow vs Reward (Episode {early_snap["episode"]})',
                     fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)

        # Add correlation coefficient
        corr_early = np.corrcoef(early_rewards, early_flows)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr_early:.3f}',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ============= Plot 4: Flow vs Reward correlation (mid) =============
        ax4 = fig.add_subplot(gs[1, 1])

        mid_snap = self.episode_snapshots[len(self.episode_snapshots)//2]
        mid_flows = list(mid_snap['flows'].values())
        mid_rewards = list(mid_snap['rewards'].values())

        ax4.scatter(mid_rewards, mid_flows, alpha=0.6, s=50, c='orange')
        ax4.set_xlabel('Actual Reward R(s)', fontsize=12)
        ax4.set_ylabel('Predicted log F(s)', fontsize=12)
        ax4.set_title(f'Flow vs Reward (Episode {mid_snap["episode"]})',
                     fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)

        corr_mid = np.corrcoef(mid_rewards, mid_flows)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr_mid:.3f}',
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ============= Plot 5: Flow vs Reward correlation (final) =============
        ax5 = fig.add_subplot(gs[1, 2])

        final_snap = self.episode_snapshots[-1]
        final_flows = list(final_snap['flows'].values())
        final_rewards_list = list(final_snap['rewards'].values())

        ax5.scatter(final_rewards_list, final_flows, alpha=0.6, s=50, c='green')
        ax5.set_xlabel('Actual Reward R(s)', fontsize=12)
        ax5.set_ylabel('Predicted log F(s)', fontsize=12)
        ax5.set_title(f'Flow vs Reward (Episode {final_snap["episode"]})',
                     fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)

        corr_final = np.corrcoef(final_rewards_list, final_flows)[0, 1]
        ax5.text(0.05, 0.95, f'Correlation: {corr_final:.3f}',
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ============= Plot 6: Correlation evolution =============
        ax6 = fig.add_subplot(gs[2, 0])

        correlations = []
        for snap in self.episode_snapshots:
            flows_list = list(snap['flows'].values())
            rewards_list = list(snap['rewards'].values())
            corr = np.corrcoef(rewards_list, flows_list)[0, 1]
            correlations.append(corr)

        ax6.plot(episodes, correlations, marker='o', linewidth=3, color='purple', markersize=6)
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Episode', fontsize=12)
        ax6.set_ylabel('Correlation(log F, R)', fontsize=12)
        ax6.set_title('Flow-Reward Correlation Over Time', fontsize=14, fontweight='bold')
        ax6.grid(alpha=0.3)
        ax6.set_ylim([-1.1, 1.1])

        # ============= Plot 7: Flow distribution evolution =============
        ax7 = fig.add_subplot(gs[2, 1])

        # Plot distribution at early, mid, final
        snapshots_to_plot = [
            (self.episode_snapshots[min(3, len(self.episode_snapshots)-1)], 'Early', 'blue'),
            (self.episode_snapshots[len(self.episode_snapshots)//2], 'Mid', 'orange'),
            (self.episode_snapshots[-1], 'Final', 'green')
        ]

        for snap, label, color in snapshots_to_plot:
            flows_list = list(snap['flows'].values())
            ax7.hist(flows_list, bins=20, alpha=0.5, label=f'{label} (Ep {snap["episode"]})',
                    color=color, edgecolor='black')

        ax7.set_xlabel('Predicted log F(s)', fontsize=12)
        ax7.set_ylabel('Count', fontsize=12)
        ax7.set_title('Flow Distribution Evolution', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(alpha=0.3)

        # ============= Plot 8: Mean flow by reward bucket =============
        ax8 = fig.add_subplot(gs[2, 2])

        # Bucket states by reward
        final_snap = self.episode_snapshots[-1]
        reward_buckets = defaultdict(list)

        for state_str in final_snap['rewards'].keys():
            reward = final_snap['rewards'][state_str]
            flow = final_snap['flows'][state_str]

            # Bucket by reward ranges
            if reward < 0.2:
                bucket = '0.0-0.2'
            elif reward < 0.4:
                bucket = '0.2-0.4'
            elif reward < 0.6:
                bucket = '0.4-0.6'
            elif reward < 0.8:
                bucket = '0.6-0.8'
            else:
                bucket = '0.8-1.0'

            reward_buckets[bucket].append(flow)

        # Plot mean flow per bucket
        bucket_order = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        mean_flows = []
        std_flows = []

        for bucket in bucket_order:
            if bucket in reward_buckets:
                mean_flows.append(np.mean(reward_buckets[bucket]))
                std_flows.append(np.std(reward_buckets[bucket]))
            else:
                mean_flows.append(0)
                std_flows.append(0)

        x_pos = np.arange(len(bucket_order))
        ax8.bar(x_pos, mean_flows, yerr=std_flows, capsize=5, alpha=0.7,
               color='steelblue', edgecolor='black', linewidth=1.5)
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(bucket_order, rotation=45, ha='right')
        ax8.set_xlabel('Reward Range', fontsize=12)
        ax8.set_ylabel('Mean Predicted log F(s)', fontsize=12)
        ax8.set_title('Mean Flow by Reward Bucket (Final)', fontsize=14, fontweight='bold')
        ax8.grid(alpha=0.3, axis='y')

        # Overall title
        fig.suptitle(f'Flow Prediction Evolution (Depth {self.max_depth}, {len(self.target_states)} states)',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        output_path = os.path.join(self.output_dir, f'flow_evolution_depth{self.max_depth}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Flow evolution plot saved to: {output_path}")

        plt.close()

    def plot_state_trajectories(self, num_states: int = 10):
        """
        Plot individual state flow trajectories.

        Args:
            num_states: Number of states to plot (top by final reward)
        """
        if len(self.episode_snapshots) < 2:
            print("Need at least 2 snapshots to plot trajectories")
            return

        episodes = [snap['episode'] for snap in self.episode_snapshots]

        # Get top states by final reward
        final_rewards = self.episode_snapshots[-1]['rewards']
        sorted_states = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
        top_states = [state_str for state_str, _ in sorted_states[:num_states]]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define unique styles for up to 10 state trajectories
        COLORS_TRAJ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        MARKERS_TRAJ = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
        LINESTYLES_TRAJ = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

        for i, state_str in enumerate(top_states):
            flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
            reward = final_rewards[state_str]

            # Get action sequence for this state
            action_seq = None
            for s, a_seq in self.target_states:
                if theory_to_string(s) == state_str:
                    action_seq = a_seq
                    break

            # Create label
            label = f"R={reward:.3f}"
            if action_seq:
                label += f" | {action_seq[:30]}"

            # Get unique style for this line
            color_idx = i % len(COLORS_TRAJ)
            marker_idx = i % len(MARKERS_TRAJ)
            linestyle_idx = i % len(LINESTYLES_TRAJ)

            ax.plot(episodes, flows,
                   color=COLORS_TRAJ[color_idx],
                   marker=MARKERS_TRAJ[marker_idx],
                   linestyle=LINESTYLES_TRAJ[linestyle_idx],
                   label=label,
                   linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Predicted log F(s)', fontsize=12)
        ax.set_title(f'Individual State Flow Trajectories (Top {num_states} by Reward)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(alpha=0.3)

        output_path = os.path.join(self.output_dir, f'state_trajectories_depth{self.max_depth}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"State trajectories plot saved to: {output_path}")

        plt.close()

    def generate_report(self):
        """Generate a text report summarizing flow learning."""
        if len(self.episode_snapshots) < 2:
            print("Need at least 2 snapshots to generate report")
            return

        report_path = os.path.join(self.output_dir, f'flow_report_depth{self.max_depth}.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FLOW PREDICTION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Target predicate: {self.target_predicate}\n")
            f.write(f"  Depth from origin: {self.max_depth}\n")
            f.write(f"  States analyzed: {len(self.target_states)}\n")
            f.write(f"  Episodes tracked: {len(self.episode_snapshots)}\n\n")

            # Correlation evolution
            f.write("-"*80 + "\n")
            f.write("Flow-Reward Correlation Evolution:\n")
            f.write("-"*80 + "\n")

            for snap in self.episode_snapshots:
                flows_list = list(snap['flows'].values())
                rewards_list = list(snap['rewards'].values())
                corr = np.corrcoef(rewards_list, flows_list)[0, 1]

                f.write(f"Episode {snap['episode']:5d}: Correlation = {corr:+.4f}\n")

            f.write("\n")

            # Top states by reward
            f.write("-"*80 + "\n")
            f.write("Top 10 States by Final Reward:\n")
            f.write("-"*80 + "\n\n")

            final_snap = self.episode_snapshots[-1]
            sorted_states = sorted(final_snap['rewards'].items(),
                                 key=lambda x: x[1], reverse=True)

            for i, (state_str, reward) in enumerate(sorted_states[:10], 1):
                flow = final_snap['flows'][state_str]

                # Get action sequence
                action_seq = ""
                for s, a_seq in self.target_states:
                    if theory_to_string(s) == state_str:
                        action_seq = a_seq
                        break

                f.write(f"{i:2d}. Reward: {reward:.4f} | Flow: {flow:+.4f}\n")
                f.write(f"    Actions: {action_seq}\n")
                f.write(f"    State: {state_str[:80]}\n")

                # Show flow trajectory
                flows_traj = [snap['flows'][state_str] for snap in self.episode_snapshots]
                f.write(f"    Flow trajectory: ")
                f.write(" → ".join([f"{flow:+.2f}" for flow in flows_traj[::max(1, len(flows_traj)//5)]]))
                f.write("\n\n")

            # Bottom states
            f.write("-"*80 + "\n")
            f.write("Bottom 5 States by Final Reward:\n")
            f.write("-"*80 + "\n\n")

            for i, (state_str, reward) in enumerate(sorted_states[-5:], 1):
                flow = final_snap['flows'][state_str]
                f.write(f"{i}. Reward: {reward:.4f} | Flow: {flow:+.4f}\n")
                f.write(f"   State: {state_str[:80]}\n\n")

            # Statistics
            f.write("-"*80 + "\n")
            f.write("Final Statistics:\n")
            f.write("-"*80 + "\n")

            flows_list = list(final_snap['flows'].values())
            rewards_list = list(final_snap['rewards'].values())

            f.write(f"Flow statistics:\n")
            f.write(f"  Mean: {np.mean(flows_list):.4f}\n")
            f.write(f"  Std:  {np.std(flows_list):.4f}\n")
            f.write(f"  Min:  {np.min(flows_list):.4f}\n")
            f.write(f"  Max:  {np.max(flows_list):.4f}\n\n")

            f.write(f"Reward statistics:\n")
            f.write(f"  Mean: {np.mean(rewards_list):.4f}\n")
            f.write(f"  Std:  {np.std(rewards_list):.4f}\n")
            f.write(f"  Min:  {np.min(rewards_list):.4f}\n")
            f.write(f"  Max:  {np.max(rewards_list):.4f}\n\n")

            f.write("="*80 + "\n")

        print(f"Flow report saved to: {report_path}")
