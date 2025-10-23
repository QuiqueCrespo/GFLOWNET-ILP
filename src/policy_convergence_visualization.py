"""
Policy Convergence Visualization

Tracks and visualizes how forward and backward policies evolve during training.
Shows if policies are consistent and learning correctly.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import os

from .logic_structures import (
    Theory, get_initial_state, apply_add_atom, apply_unify_vars,
    get_valid_variable_pairs, get_all_variables, theory_to_string
)
from .logic_engine import Example


class PolicyConvergenceVisualizer:
    """
    Visualizes forward and backward policy convergence.

    Tracks:
    - Forward policy: P_F(action | state) for strategist
    - Backward policy: P_B(action | state) for backward strategist
    - Action-specific policies (atom adder, variable unifier, etc.)
    - Policy consistency over time
    """

    def __init__(self,
                 trainer,
                 target_predicate: str,
                 arity: int,
                 predicate_vocab: List[str],
                 predicate_arities: Dict[str, int],
                 output_dir: str = "policy_visualizations"):
        """
        Args:
            trainer: GFlowNetTrainer instance
            target_predicate: Target predicate name
            arity: Target predicate arity
            predicate_vocab: List of available predicates
            predicate_arities: Dict mapping predicates to arities
            output_dir: Directory to save visualizations
        """
        self.trainer = trainer
        self.target_predicate = target_predicate
        self.arity = arity
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Create test states for tracking
        self.test_states = self._create_test_states()

        # Storage for policy snapshots
        self.snapshots = []  # List of {episode, forward_policies, backward_policies}

        print(f"PolicyConvergenceVisualizer initialized:")
        print(f"  Test states: {len(self.test_states)}")
        print(f"  Output directory: {output_dir}")

    def _create_test_states(self) -> List[Tuple[Theory, str]]:
        """Create representative test states to track policy on."""
        states = []

        # State 1: Initial state (empty body)
        initial = get_initial_state(self.target_predicate, self.arity)
        states.append((initial, "Initial (empty body)"))

        # State 2: One atom added
        state_1atom = get_initial_state(self.target_predicate, self.arity)
        state_1atom, max_var = apply_add_atom(
            state_1atom, self.predicate_vocab[0],
            self.predicate_arities[self.predicate_vocab[0]], self.arity - 1
        )
        states.append((state_1atom, "After 1 ADD_ATOM"))

        # State 3: Two atoms added
        state_2atoms, max_var = apply_add_atom(
            state_1atom, self.predicate_vocab[0],
            self.predicate_arities[self.predicate_vocab[0]], max_var
        )
        states.append((state_2atoms, "After 2 ADD_ATOMs"))

        # State 4: One atom, then unified
        if len(get_valid_variable_pairs(state_1atom)) > 0:
            vars_to_unify = get_valid_variable_pairs(state_1atom)[0]
            state_unified = apply_unify_vars(state_1atom, vars_to_unify[0], vars_to_unify[1])
            states.append((state_unified, "After 1 ADD_ATOM + UNIFY"))

        return states

    def record_snapshot(self, episode: int):
        """
        Record current forward and backward policy distributions.

        Args:
            episode: Current training episode
        """
        forward_policies = {}
        backward_policies = {}

        with torch.no_grad():
            for state, state_name in self.test_states:
                # Get state embedding
                graph_data = self.trainer.graph_constructor.theory_to_graph(state)
                state_embedding, node_embeddings = self.trainer.state_encoder(graph_data)
                state_embedding = state_embedding.squeeze(0)

                # === FORWARD POLICY ===
                # Strategist: P_F(action_type | state)
                action_logits = self.trainer.gflownet.forward_strategist(state_embedding)
                action_probs = F.softmax(action_logits, dim=-1).numpy()

                forward_policies[state_name] = {
                    'strategist': {
                        'ADD_ATOM': action_probs[0],
                        'UNIFY_VARIABLES': action_probs[1],
                        'TERMINATE': action_probs[2]
                    }
                }

                # Atom Adder: P_F(predicate | ADD_ATOM)
                atom_logits = self.trainer.gflownet.forward_atom_adder(state_embedding)
                atom_probs = F.softmax(atom_logits, dim=-1).numpy()
                forward_policies[state_name]['atom_adder'] = {
                    pred: atom_probs[i] for i, pred in enumerate(self.predicate_vocab)
                }

                # Variable Unifier: P_F(pair | UNIFY_VARIABLES)
                variables = get_all_variables(state)
                if len(variables) >= 2:
                    var_embeddings = node_embeddings[:len(variables)]
                    pair_logits = self.trainer.gflownet.forward_variable_unifier(
                        state_embedding, var_embeddings
                    )
                    if pair_logits.numel() > 0:
                        pair_probs = F.softmax(pair_logits, dim=-1).numpy()
                        forward_policies[state_name]['variable_unifier'] = {
                            'num_pairs': len(pair_probs),
                            'max_prob': float(np.max(pair_probs)),
                            'min_prob': float(np.min(pair_probs)),
                            'entropy': float(-np.sum(pair_probs * np.log(pair_probs + 1e-10)))
                        }

                # === BACKWARD POLICY ===
                # Backward Strategist: P_B(action_type | state)
                backward_action_logits = self.trainer.gflownet.forward_backward_policy(state_embedding)
                backward_action_probs = F.softmax(backward_action_logits, dim=-1).numpy()

                backward_policies[state_name] = {
                    'strategist': {
                        'ADD_ATOM': backward_action_probs[0],
                        'UNIFY_VARIABLES': backward_action_probs[1]
                    }
                }

        # Store snapshot
        self.snapshots.append({
            'episode': episode,
            'forward': forward_policies,
            'backward': backward_policies
        })

        print(f"Policy snapshot recorded at episode {episode}")

    def plot_strategist_convergence(self):
        """Plot forward strategist policy convergence over time."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to plot convergence")
            return

        episodes = [snap['episode'] for snap in self.snapshots]
        num_states = len(self.test_states)

        fig, axes = plt.subplots(num_states, 1, figsize=(14, 4 * num_states))
        if num_states == 1:
            axes = [axes]

        for idx, (state, state_name) in enumerate(self.test_states):
            ax = axes[idx]

            # Extract probabilities over time
            add_atom_probs = [snap['forward'][state_name]['strategist']['ADD_ATOM']
                             for snap in self.snapshots]
            unify_probs = [snap['forward'][state_name]['strategist']['UNIFY_VARIABLES']
                          for snap in self.snapshots]
            terminate_probs = [snap['forward'][state_name]['strategist']['TERMINATE']
                              for snap in self.snapshots]

            # Plot
            ax.plot(episodes, add_atom_probs, marker='o', linewidth=2, markersize=5,
                   label='ADD_ATOM', color='blue')
            ax.plot(episodes, unify_probs, marker='s', linewidth=2, markersize=5,
                   label='UNIFY_VARIABLES', color='orange')
            ax.plot(episodes, terminate_probs, marker='^', linewidth=2, markersize=5,
                   label='TERMINATE', color='green')

            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(f'Forward Strategist Policy: {state_name}',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'forward_strategist_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Forward strategist plot saved to: {output_path}")
        plt.close()

    def plot_backward_strategist_convergence(self):
        """Plot backward strategist policy convergence over time."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to plot convergence")
            return

        episodes = [snap['episode'] for snap in self.snapshots]
        num_states = len(self.test_states)

        fig, axes = plt.subplots(num_states, 1, figsize=(14, 4 * num_states))
        if num_states == 1:
            axes = [axes]

        for idx, (state, state_name) in enumerate(self.test_states):
            ax = axes[idx]

            # Extract probabilities over time
            add_atom_probs = [snap['backward'][state_name]['strategist']['ADD_ATOM']
                             for snap in self.snapshots]
            unify_probs = [snap['backward'][state_name]['strategist']['UNIFY_VARIABLES']
                          for snap in self.snapshots]

            # Plot
            ax.plot(episodes, add_atom_probs, marker='o', linewidth=2, markersize=5,
                   label='ADD_ATOM', color='blue')
            ax.plot(episodes, unify_probs, marker='s', linewidth=2, markersize=5,
                   label='UNIFY_VARIABLES', color='orange')

            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(f'Backward Strategist Policy: {state_name}',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'backward_strategist_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Backward strategist plot saved to: {output_path}")
        plt.close()

    def plot_policy_consistency(self):
        """Plot consistency between forward and backward policies."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to plot consistency")
            return

        episodes = [snap['episode'] for snap in self.snapshots]
        num_states = len(self.test_states)

        fig, axes = plt.subplots(num_states, 2, figsize=(16, 4 * num_states))
        if num_states == 1:
            axes = axes.reshape(1, -1)

        for idx, (state, state_name) in enumerate(self.test_states):
            # Left plot: Forward vs Backward for ADD_ATOM
            ax_left = axes[idx, 0]

            forward_add = [snap['forward'][state_name]['strategist']['ADD_ATOM']
                          for snap in self.snapshots]
            backward_add = [snap['backward'][state_name]['strategist']['ADD_ATOM']
                           for snap in self.snapshots]

            ax_left.plot(episodes, forward_add, marker='o', linewidth=2, markersize=5,
                        label='Forward P_F', color='blue')
            ax_left.plot(episodes, backward_add, marker='s', linewidth=2, markersize=5,
                        label='Backward P_B', color='red', linestyle='--')

            # Plot difference
            diff_add = [abs(f - b) for f, b in zip(forward_add, backward_add)]
            ax_right_add = ax_left.twinx()
            ax_right_add.plot(episodes, diff_add, marker='x', linewidth=1, markersize=4,
                            label='|P_F - P_B|', color='purple', alpha=0.6)
            ax_right_add.set_ylabel('Absolute Difference', fontsize=10, color='purple')
            ax_right_add.tick_params(axis='y', labelcolor='purple')

            ax_left.set_xlabel('Episode', fontsize=12)
            ax_left.set_ylabel('Probability', fontsize=12)
            ax_left.set_title(f'ADD_ATOM: {state_name}', fontsize=12, fontweight='bold')
            ax_left.legend(loc='upper left', fontsize=10)
            ax_left.grid(alpha=0.3)
            ax_left.set_ylim([-0.05, 1.05])

            # Right plot: Forward vs Backward for UNIFY_VARIABLES
            ax_right = axes[idx, 1]

            forward_unify = [snap['forward'][state_name]['strategist']['UNIFY_VARIABLES']
                            for snap in self.snapshots]
            backward_unify = [snap['backward'][state_name]['strategist']['UNIFY_VARIABLES']
                             for snap in self.snapshots]

            ax_right.plot(episodes, forward_unify, marker='o', linewidth=2, markersize=5,
                         label='Forward P_F', color='orange')
            ax_right.plot(episodes, backward_unify, marker='s', linewidth=2, markersize=5,
                         label='Backward P_B', color='red', linestyle='--')

            # Plot difference
            diff_unify = [abs(f - b) for f, b in zip(forward_unify, backward_unify)]
            ax_right_unify = ax_right.twinx()
            ax_right_unify.plot(episodes, diff_unify, marker='x', linewidth=1, markersize=4,
                               label='|P_F - P_B|', color='purple', alpha=0.6)
            ax_right_unify.set_ylabel('Absolute Difference', fontsize=10, color='purple')
            ax_right_unify.tick_params(axis='y', labelcolor='purple')

            ax_right.set_xlabel('Episode', fontsize=12)
            ax_right.set_ylabel('Probability', fontsize=12)
            ax_right.set_title(f'UNIFY_VARIABLES: {state_name}', fontsize=12, fontweight='bold')
            ax_right.legend(loc='upper left', fontsize=10)
            ax_right.grid(alpha=0.3)
            ax_right.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'policy_consistency.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Policy consistency plot saved to: {output_path}")
        plt.close()

    def plot_atom_adder_convergence(self):
        """Plot atom adder policy convergence (predicate selection)."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to plot convergence")
            return

        episodes = [snap['episode'] for snap in self.snapshots]
        num_states = min(2, len(self.test_states))  # Show first 2 states

        fig, axes = plt.subplots(num_states, 1, figsize=(14, 4 * num_states))
        if num_states == 1:
            axes = [axes]

        for idx in range(num_states):
            state, state_name = self.test_states[idx]
            ax = axes[idx]

            # Plot each predicate's probability over time
            # Define unique styles for predicates
            COLORS_PRED = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            MARKERS_PRED = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
            LINESTYLES_PRED = ['-', '--', '-.', ':']

            for pred_idx, pred in enumerate(self.predicate_vocab):
                probs = [snap['forward'][state_name]['atom_adder'][pred]
                        for snap in self.snapshots]

                # Unique style for each predicate
                color_idx = pred_idx % len(COLORS_PRED)
                marker_idx = pred_idx % len(MARKERS_PRED)
                linestyle_idx = (pred_idx // len(COLORS_PRED)) % len(LINESTYLES_PRED)

                ax.plot(episodes, probs,
                       color=COLORS_PRED[color_idx],
                       marker=MARKERS_PRED[marker_idx],
                       linestyle=LINESTYLES_PRED[linestyle_idx],
                       linewidth=2, markersize=5,
                       label=pred)

            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(f'Atom Adder Policy: {state_name}',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'atom_adder_convergence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Atom adder plot saved to: {output_path}")
        plt.close()

    def plot_variable_unifier_entropy(self):
        """Plot entropy of variable unifier distribution over time."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to plot convergence")
            return

        episodes = [snap['episode'] for snap in self.snapshots]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Define unique styles for different states
        COLORS_STATES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        MARKERS_STATES = ['o', 's', '^', 'v']
        LINESTYLES_STATES = ['-', '--', '-.', ':']

        for idx, (state, state_name) in enumerate(self.test_states):
            # Extract entropy over time (if available)
            entropies = []
            for snap in self.snapshots:
                if 'variable_unifier' in snap['forward'][state_name]:
                    entropies.append(snap['forward'][state_name]['variable_unifier']['entropy'])
                else:
                    entropies.append(None)

            # Plot if we have data
            if any(e is not None for e in entropies):
                # Filter out None values
                valid_episodes = [ep for ep, ent in zip(episodes, entropies) if ent is not None]
                valid_entropies = [ent for ent in entropies if ent is not None]

                ax.plot(valid_episodes, valid_entropies,
                       color=COLORS_STATES[idx % len(COLORS_STATES)],
                       marker=MARKERS_STATES[idx % len(MARKERS_STATES)],
                       linestyle=LINESTYLES_STATES[idx % len(LINESTYLES_STATES)],
                       linewidth=2,
                       markersize=5,
                       label=state_name)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Entropy (nats)', fontsize=12)
        ax.set_title('Variable Unifier Policy Entropy Over Time',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)

        # Add annotation
        ax.text(0.02, 0.98, 'Lower entropy = more deterministic (converged)\nHigher entropy = more uniform (exploring)',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'variable_unifier_entropy.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Variable unifier entropy plot saved to: {output_path}")
        plt.close()

    def plot_comprehensive_dashboard(self):
        """Create comprehensive policy convergence dashboard."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to create dashboard")
            return

        episodes = [snap['episode'] for snap in self.snapshots]

        # Use first test state for dashboard
        state, state_name = self.test_states[0]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # === Plot 1: Forward Strategist ===
        ax1 = fig.add_subplot(gs[0, 0])

        add_probs = [snap['forward'][state_name]['strategist']['ADD_ATOM'] for snap in self.snapshots]
        unify_probs = [snap['forward'][state_name]['strategist']['UNIFY_VARIABLES'] for snap in self.snapshots]
        term_probs = [snap['forward'][state_name]['strategist']['TERMINATE'] for snap in self.snapshots]

        ax1.plot(episodes, add_probs, 'o-', label='ADD_ATOM', linewidth=2)
        ax1.plot(episodes, unify_probs, 's-', label='UNIFY', linewidth=2)
        ax1.plot(episodes, term_probs, '^-', label='TERMINATE', linewidth=2)
        ax1.set_ylabel('Probability')
        ax1.set_title('Forward Strategist', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])

        # === Plot 2: Backward Strategist ===
        ax2 = fig.add_subplot(gs[0, 1])

        back_add = [snap['backward'][state_name]['strategist']['ADD_ATOM'] for snap in self.snapshots]
        back_unify = [snap['backward'][state_name]['strategist']['UNIFY_VARIABLES'] for snap in self.snapshots]

        ax2.plot(episodes, back_add, 'o-', label='ADD_ATOM', linewidth=2)
        ax2.plot(episodes, back_unify, 's-', label='UNIFY', linewidth=2)
        ax2.set_ylabel('Probability')
        ax2.set_title('Backward Strategist', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])

        # === Plot 3: Policy Divergence ===
        ax3 = fig.add_subplot(gs[0, 2])

        div_add = [abs(f - b) for f, b in zip(add_probs, back_add)]
        div_unify = [abs(f - b) for f, b in zip(unify_probs, back_unify)]

        ax3.plot(episodes, div_add, 'o-', label='ADD_ATOM', linewidth=2)
        ax3.plot(episodes, div_unify, 's-', label='UNIFY', linewidth=2)
        ax3.set_ylabel('|P_F - P_B|')
        ax3.set_title('Forward-Backward Divergence', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # === Plot 4: Atom Adder ===
        ax4 = fig.add_subplot(gs[1, :2])

        # Define unique styles for predicates in dashboard
        COLORS_DASH = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        MARKERS_DASH = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
        LINESTYLES_DASH = ['-', '--', '-.', ':']

        for idx, pred in enumerate(self.predicate_vocab):
            probs = [snap['forward'][state_name]['atom_adder'][pred] for snap in self.snapshots]

            # Unique style for each predicate
            color_idx = idx % len(COLORS_DASH)
            marker_idx = idx % len(MARKERS_DASH)
            linestyle_idx = (idx // len(COLORS_DASH)) % len(LINESTYLES_DASH)

            ax4.plot(episodes, probs,
                    color=COLORS_DASH[color_idx],
                    marker=MARKERS_DASH[marker_idx],
                    linestyle=LINESTYLES_DASH[linestyle_idx],
                    label=pred,
                    linewidth=2,
                    markersize=4)

        ax4.set_ylabel('Probability')
        ax4.set_title('Atom Adder (Predicate Selection)', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 1])

        # === Plot 5: Variable Unifier Entropy ===
        ax5 = fig.add_subplot(gs[1, 2])

        if 'variable_unifier' in self.snapshots[0]['forward'][state_name]:
            entropies = [snap['forward'][state_name]['variable_unifier']['entropy']
                        if 'variable_unifier' in snap['forward'][state_name] else None
                        for snap in self.snapshots]

            valid_eps = [ep for ep, ent in zip(episodes, entropies) if ent is not None]
            valid_ents = [ent for ent in entropies if ent is not None]

            if valid_ents:
                ax5.plot(valid_eps, valid_ents, 'o-', linewidth=2, color='purple')
                ax5.set_ylabel('Entropy (nats)')
                ax5.set_title('Variable Unifier Entropy', fontweight='bold')
                ax5.grid(alpha=0.3)

        # === Plot 6: Action Distribution Stacked Area (Forward) ===
        ax6 = fig.add_subplot(gs[2, :])

        ax6.fill_between(episodes, 0, add_probs, label='ADD_ATOM', alpha=0.6, color='blue')
        ax6.fill_between(episodes, add_probs,
                        [a + u for a, u in zip(add_probs, unify_probs)],
                        label='UNIFY_VARIABLES', alpha=0.6, color='orange')
        ax6.fill_between(episodes,
                        [a + u for a, u in zip(add_probs, unify_probs)],
                        [a + u + t for a, u, t in zip(add_probs, unify_probs, term_probs)],
                        label='TERMINATE', alpha=0.6, color='green')

        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Probability')
        ax6.set_title('Forward Strategist Distribution (Stacked)', fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(alpha=0.3)
        ax6.set_ylim([0, 1])

        fig.suptitle(f'Policy Convergence Dashboard: {state_name}',
                    fontsize=16, fontweight='bold', y=0.995)

        output_path = os.path.join(self.output_dir, 'policy_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Policy dashboard saved to: {output_path}")
        plt.close()

    def generate_report(self):
        """Generate text report on policy convergence."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots to generate report")
            return

        report_path = os.path.join(self.output_dir, 'policy_convergence_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POLICY CONVERGENCE REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Episodes tracked: {len(self.snapshots)}\n")
            f.write(f"Test states: {len(self.test_states)}\n\n")

            for state, state_name in self.test_states:
                f.write("-"*80 + "\n")
                f.write(f"State: {state_name}\n")
                f.write(f"  {theory_to_string(state)}\n")
                f.write("-"*80 + "\n\n")

                # Initial and final forward strategist
                initial_fwd = self.snapshots[0]['forward'][state_name]['strategist']
                final_fwd = self.snapshots[-1]['forward'][state_name]['strategist']

                f.write("Forward Strategist:\n")
                f.write(f"  Episode {self.snapshots[0]['episode']:5d}: ADD={initial_fwd['ADD_ATOM']:.4f}, "
                       f"UNIFY={initial_fwd['UNIFY_VARIABLES']:.4f}, TERM={initial_fwd['TERMINATE']:.4f}\n")
                f.write(f"  Episode {self.snapshots[-1]['episode']:5d}: ADD={final_fwd['ADD_ATOM']:.4f}, "
                       f"UNIFY={final_fwd['UNIFY_VARIABLES']:.4f}, TERM={final_fwd['TERMINATE']:.4f}\n\n")

                # Initial and final backward strategist
                initial_back = self.snapshots[0]['backward'][state_name]['strategist']
                final_back = self.snapshots[-1]['backward'][state_name]['strategist']

                f.write("Backward Strategist:\n")
                f.write(f"  Episode {self.snapshots[0]['episode']:5d}: ADD={initial_back['ADD_ATOM']:.4f}, "
                       f"UNIFY={initial_back['UNIFY_VARIABLES']:.4f}\n")
                f.write(f"  Episode {self.snapshots[-1]['episode']:5d}: ADD={final_back['ADD_ATOM']:.4f}, "
                       f"UNIFY={final_back['UNIFY_VARIABLES']:.4f}\n\n")

                # Divergence
                div_add_initial = abs(initial_fwd['ADD_ATOM'] - initial_back['ADD_ATOM'])
                div_add_final = abs(final_fwd['ADD_ATOM'] - final_back['ADD_ATOM'])
                div_unify_initial = abs(initial_fwd['UNIFY_VARIABLES'] - initial_back['UNIFY_VARIABLES'])
                div_unify_final = abs(final_fwd['UNIFY_VARIABLES'] - final_back['UNIFY_VARIABLES'])

                f.write("Forward-Backward Divergence:\n")
                f.write(f"  ADD_ATOM:        {div_add_initial:.4f} → {div_add_final:.4f}\n")
                f.write(f"  UNIFY_VARIABLES: {div_unify_initial:.4f} → {div_unify_final:.4f}\n\n")

                # Convergence assessment
                if div_add_final < 0.1 and div_unify_final < 0.1:
                    f.write("  ✓ Policies CONVERGED (divergence < 0.1)\n")
                elif div_add_final < 0.2 and div_unify_final < 0.2:
                    f.write("  ~ Policies PARTIALLY CONVERGED (divergence < 0.2)\n")
                else:
                    f.write("  ✗ Policies NOT CONVERGED (divergence >= 0.2)\n")

                f.write("\n")

            f.write("="*80 + "\n")

        print(f"Policy report saved to: {report_path}")
