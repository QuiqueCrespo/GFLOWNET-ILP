"""
Visualize the trained GFlowNet as a directed graph.

Nodes represent states (rules under construction).
Edges represent actions with their probabilities.
"""

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

from src.logic_structures import (
    theory_to_string, get_all_variables, is_valid_complete_state,
    apply_add_atom, apply_unify_vars, get_valid_variable_pairs
)


class GFlowNetGraphVisualizer:
    """Visualizes the GFlowNet policy as a state-action graph."""

    def __init__(self, trainer, predicate_vocab, predicate_arities, max_body_length):
        self.trainer = trainer
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities
        self.max_body_length = max_body_length
        self.graph = nx.DiGraph()
        self.state_to_id = {}
        self.id_to_state = {}
        self.next_id = 0

    def get_state_id(self, state):
        """Get or create a unique ID for a state."""
        state_str = theory_to_string(state)
        if state_str not in self.state_to_id:
            self.state_to_id[state_str] = self.next_id
            self.id_to_state[self.next_id] = (state, state_str)
            self.next_id += 1
        return self.state_to_id[state_str]

    def get_action_probabilities(self, state):
        """Get action probabilities for a state."""
        # Encode state
        graph_data = self.trainer.graph_constructor.theory_to_graph(state)
        state_embedding, node_embeddings = self.trainer.state_encoder(graph_data)
        state_embedding = state_embedding.squeeze(0)

        # Get action logits
        action_logits = self.trainer.gflownet.forward_strategist(state_embedding)

        # Apply masks
        action_logits = action_logits.clone()

        # Check constraints
        body_length = len(state[0].body)
        at_max_length = body_length >= self.max_body_length
        valid_pairs = get_valid_variable_pairs(state)
        num_variables = len(get_all_variables(state))
        is_valid = is_valid_complete_state(state)

        # Mask ADD_ATOM at max length
        if at_max_length:
            action_logits[0] = float('-inf')

        # Mask TERMINATE for invalid states
        if not is_valid:
            action_logits[2] = float('-inf')

        # Mask UNIFY_VARIABLES if not possible
        if not valid_pairs or num_variables < 2:
            action_logits[1] = float('-inf')

        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs.detach().numpy(), at_max_length, valid_pairs, num_variables

    def get_atom_probabilities(self, state):
        """Get atom addition probabilities."""
        graph_data = self.trainer.graph_constructor.theory_to_graph(state)
        state_embedding, _ = self.trainer.state_encoder(graph_data)
        state_embedding = state_embedding.squeeze(0)

        atom_logits = self.trainer.gflownet.forward_atom_adder(state_embedding)
        atom_probs = F.softmax(atom_logits, dim=-1)

        return atom_probs.detach().numpy()

    def explore_from_state(self, initial_state, max_depth=3, min_prob=0.05, max_branches=3):
        """
        Explore the policy graph starting from initial state.

        Args:
            initial_state: Starting state (empty theory)
            max_depth: Maximum depth to explore
            min_prob: Minimum probability to follow an edge
            max_branches: Maximum number of branches to explore per node
        """
        queue = deque([(initial_state, 0)])  # (state, depth)
        visited = set()

        while queue:
            current_state, depth = queue.popleft()

            state_id = self.get_state_id(current_state)
            state_str = theory_to_string(current_state)

            if state_id in visited or depth >= max_depth:
                continue

            visited.add(state_id)

            # Add node to graph
            body_length = len(current_state[0].body)
            is_initial = (state_id == 0)

            # Create compact node label
            node_label = f"#{state_id}"

            self.graph.add_node(
                state_id,
                label=node_label,
                state_str=state_str,
                depth=depth,
                body_length=body_length,
                is_initial=is_initial
            )

            # Get action probabilities
            action_probs, at_max_length, valid_pairs, num_variables = self.get_action_probabilities(current_state)

            # Track top actions to explore
            action_choices = []

            # ADD_ATOM action
            if action_probs[0] > min_prob:
                atom_probs = self.get_atom_probabilities(current_state)
                top_atoms = np.argsort(atom_probs)[-max_branches:][::-1]

                for pred_idx in top_atoms:
                    if atom_probs[pred_idx] < min_prob:
                        continue

                    pred_name = self.predicate_vocab[pred_idx]
                    pred_arity = self.predicate_arities[pred_name]

                    # Simulate action
                    max_var_id = max([v.id for v in get_all_variables(current_state)], default=-1)
                    try:
                        next_state, _ = apply_add_atom(current_state, pred_name, pred_arity, max_var_id)
                        next_id = self.get_state_id(next_state)

                        edge_prob = action_probs[0] * atom_probs[pred_idx]
                        # Detailed action label with arity
                        action_label = f"ADD_ATOM({pred_name}, arity={pred_arity})"

                        self.graph.add_edge(
                            state_id, next_id,
                            action=action_label,
                            probability=edge_prob,
                            action_type='ADD_ATOM'
                        )

                        action_choices.append((next_state, depth + 1, edge_prob))
                    except:
                        pass

            # UNIFY_VARIABLES action
            if action_probs[1] > min_prob and valid_pairs:
                graph_data = self.trainer.graph_constructor.theory_to_graph(current_state)
                state_embedding, node_embeddings = self.trainer.state_encoder(graph_data)
                state_embedding = state_embedding.squeeze(0)

                variables = get_all_variables(current_state)
                var_embeddings = node_embeddings[:len(variables)]

                pair_logits = self.trainer.gflownet.forward_variable_unifier(state_embedding, var_embeddings)
                pair_probs = F.softmax(pair_logits, dim=-1).detach().numpy()

                top_pairs = np.argsort(pair_probs)[-max_branches:][::-1]

                for pair_idx in top_pairs:
                    if pair_idx >= len(valid_pairs) or pair_probs[pair_idx] < min_prob:
                        continue

                    var1, var2 = valid_pairs[pair_idx]

                    try:
                        next_state = apply_unify_vars(current_state, var1, var2)
                        next_id = self.get_state_id(next_state)

                        edge_prob = action_probs[1] * pair_probs[pair_idx]
                        # Detailed unification action
                        action_label = f"UNIFY(X{var1.id}, X{var2.id})"

                        self.graph.add_edge(
                            state_id, next_id,
                            action=action_label,
                            probability=edge_prob,
                            action_type='UNIFY_VARIABLES'
                        )

                        action_choices.append((next_state, depth + 1, edge_prob))
                    except:
                        pass

            # TERMINATE action
            if action_probs[2] > min_prob:
                # Terminal node - don't add to queue
                action_label = "TERMINATE"
                edge_prob = action_probs[2]

                # Create a terminal node
                terminal_id = self.next_id
                self.next_id += 1
                terminal_label = f"T{terminal_id}"
                self.graph.add_node(
                    terminal_id,
                    label=terminal_label,
                    state_str="[TERMINAL]",
                    depth=depth+1,
                    terminal=True
                )

                self.graph.add_edge(
                    state_id, terminal_id,
                    action=action_label,
                    probability=edge_prob,
                    action_type='TERMINATE'
                )

            # Add top choices to queue (by probability)
            action_choices.sort(key=lambda x: x[2], reverse=True)
            for next_state, next_depth, _ in action_choices[:max_branches]:
                queue.append((next_state, next_depth))

    def visualize(self, output_path='gflownet_graph.png', figsize=(24, 16)):
        """Create a visualization of the policy graph with legend."""
        if len(self.graph.nodes) == 0:
            print("No nodes in graph. Run explore_from_state() first.")
            return

        # Create figure with space for legend
        fig = plt.figure(figsize=figsize)

        # Main graph takes left 65% of space
        ax_graph = fig.add_axes([0.05, 0.05, 0.60, 0.90])
        # Legend takes right 30% of space
        ax_legend = fig.add_axes([0.68, 0.05, 0.30, 0.90])

        # Use hierarchical layout based on depth
        depths = nx.get_node_attributes(self.graph, 'depth')

        # Calculate positions
        pos = {}
        depth_counts = {}
        for node in self.graph.nodes():
            d = depths.get(node, 0)
            if d not in depth_counts:
                depth_counts[d] = 0
            depth_counts[d] += 1

        depth_positions = {}
        for node in self.graph.nodes():
            d = depths.get(node, 0)
            if d not in depth_positions:
                depth_positions[d] = 0

            x = d * 4  # Horizontal spacing by depth
            y = depth_positions[d] * 3 - (depth_counts[d] - 1) * 1.5  # Vertical spacing
            pos[node] = (x, y)
            depth_positions[d] += 1

        # Identify node types
        terminal_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('terminal', False)]
        initial_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('is_initial', False)]
        regular_nodes = [n for n in self.graph.nodes() if n not in terminal_nodes and n not in initial_nodes]

        # Draw initial node (larger, different color)
        if initial_nodes:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=initial_nodes,
                node_color='lightgreen',
                node_size=3500,
                alpha=0.95,
                ax=ax_graph,
                edgecolors='darkgreen',
                linewidths=3
            )

        # Draw regular nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=regular_nodes,
            node_color='lightblue',
            node_size=2500,
            alpha=0.9,
            ax=ax_graph,
            edgecolors='black',
            linewidths=1.5
        )

        # Draw terminal nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=terminal_nodes,
            node_color='lightcoral',
            node_size=2000,
            alpha=0.9,
            ax=ax_graph,
            node_shape='s',
            edgecolors='darkred',
            linewidths=2
        )

        # Draw node labels
        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx_labels(
            self.graph, pos, labels,
            font_size=10,
            font_weight='bold',
            ax=ax_graph
        )

        # Draw edges with thickness based on probability
        edge_colors = []
        edge_widths = []
        for u, v, data in self.graph.edges(data=True):
            prob = data.get('probability', 0)
            action_type = data.get('action_type', '')

            if action_type == 'ADD_ATOM':
                edge_colors.append('blue')
            elif action_type == 'UNIFY_VARIABLES':
                edge_colors.append('green')
            elif action_type == 'TERMINATE':
                edge_colors.append('red')
            else:
                edge_colors.append('gray')

            # Scale edge width based on probability (1 to 8 pts)
            edge_widths.append(0.5 + 7.5 * prob)

        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax_graph
        )

        # Draw edge labels with action and probability
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            action = data.get('action', '')
            prob = data.get('probability', 0)
            edge_labels[(u, v)] = f"{action}\np={prob:.3f}"

        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels,
            font_size=7,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
            ax=ax_graph
        )

        # Add action type legend to graph
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=15, markeredgecolor='darkgreen', markeredgewidth=2,
                   label='Initial State (empty rule)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=12, markeredgecolor='black', markeredgewidth=1,
                   label='Intermediate State'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral',
                   markersize=12, markeredgecolor='darkred', markeredgewidth=1.5,
                   label='Terminal State'),
            Line2D([0], [0], color='blue', lw=3, label='ADD_ATOM action'),
            Line2D([0], [0], color='green', lw=3, label='UNIFY_VARIABLES action'),
            Line2D([0], [0], color='red', lw=3, label='TERMINATE action'),
            Line2D([0], [0], color='black', lw=1, label='Edge thickness = probability'),
        ]
        ax_graph.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

        ax_graph.set_title(
            'GFlowNet Policy Graph\n(All trajectories start from empty theory)',
            fontsize=18, fontweight='bold', pad=20
        )
        ax_graph.axis('off')

        # Create the node legend on the right side
        ax_legend.axis('off')
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)

        # Title for legend
        ax_legend.text(
            0.5, 0.98, 'Node Legend\n(Rule Representations)',
            ha='center', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
        )

        # Create legend entries
        non_terminal_nodes = [n for n in self.graph.nodes() if n not in terminal_nodes]
        non_terminal_nodes = sorted(non_terminal_nodes)  # Sort by ID

        y_start = 0.92
        y_step = 0.08
        current_y = y_start

        for idx, node_id in enumerate(non_terminal_nodes):
            node_data = self.graph.nodes[node_id]
            state_str = node_data.get('state_str', 'UNKNOWN')
            node_label = node_data.get('label', f'#{node_id}')
            is_initial = node_data.get('is_initial', False)

            # Truncate very long rules for legend
            if len(state_str) > 100:
                display_str = state_str[:97] + "..."
            else:
                display_str = state_str

            # Color code by whether it's initial
            if is_initial:
                color = 'lightgreen'
                prefix = "🟢 "
            else:
                color = 'lightblue'
                prefix = ""

            # Draw legend entry
            ax_legend.text(
                0.02, current_y, f"{prefix}{node_label}:",
                ha='left', va='top', fontsize=9, fontweight='bold'
            )
            ax_legend.text(
                0.02, current_y - 0.025, display_str,
                ha='left', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                wrap=True, family='monospace'
            )

            current_y -= y_step

            # If we run out of space, stop
            if current_y < 0.05:
                if idx < len(non_terminal_nodes) - 1:
                    ax_legend.text(
                        0.5, current_y, f"...and {len(non_terminal_nodes) - idx - 1} more nodes",
                        ha='center', va='top', fontsize=8, style='italic', color='gray'
                    )
                break

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Policy graph saved to: {output_path}")

        # Print statistics
        print(f"\nGraph Statistics:")
        print(f"  - Total nodes (states): {len(self.graph.nodes)}")
        print(f"  - Total edges (actions): {len(self.graph.edges)}")
        print(f"  - Terminal nodes: {len(terminal_nodes)}")
        print(f"  - Initial nodes: {len(initial_nodes)}")

        # Print edge probability statistics
        probs = [data['probability'] for u, v, data in self.graph.edges(data=True)]
        if probs:
            print(f"\nEdge Probability Statistics:")
            print(f"  - Min: {min(probs):.4f}")
            print(f"  - Max: {max(probs):.4f}")
            print(f"  - Mean: {np.mean(probs):.4f}")
            print(f"  - Median: {np.median(probs):.4f}")

        return fig

    def print_paths(self, max_paths=5):
        """Print top probability paths through the graph."""
        print("\n" + "=" * 80)
        print("TOP PROBABILITY PATHS")
        print("=" * 80)

        # Find paths from initial state to terminal states
        initial_id = 0  # Assuming first added state is initial
        terminal_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('terminal', False)]

        all_paths = []
        for terminal_id in terminal_nodes:
            try:
                paths = list(nx.all_simple_paths(self.graph, initial_id, terminal_id))
                for path in paths:
                    # Calculate path probability
                    prob = 1.0
                    actions = []
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        if edge_data:
                            prob *= edge_data.get('probability', 0)
                            actions.append(edge_data.get('action', 'UNKNOWN'))

                    all_paths.append((prob, path, actions))
            except:
                pass

        # Sort by probability
        all_paths.sort(reverse=True, key=lambda x: x[0])

        print(f"\nShowing top {min(max_paths, len(all_paths))} paths:\n")
        for i, (prob, path, actions) in enumerate(all_paths[:max_paths], 1):
            print(f"Path {i} (probability: {prob:.6f}):")
            for j, (node_id, action) in enumerate(zip(path[:-1], actions)):
                state_str = self.graph.nodes[node_id].get('state_str', 'UNKNOWN')
                print(f"  Step {j+1}: {state_str}")
                print(f"           --[{action}]-->")

            # Print final state
            final_state_str = self.graph.nodes[path[-1]].get('state_str', 'TERMINAL')
            print(f"  Final: {final_state_str}")
            print()

    def export_legend_table(self, output_path='policy_graph_legend.txt'):
        """Export a text file with the complete node legend."""
        terminal_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('terminal', False)]
        non_terminal_nodes = [n for n in self.graph.nodes() if n not in terminal_nodes]
        non_terminal_nodes = sorted(non_terminal_nodes)

        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("GFLOWNET POLICY GRAPH - NODE LEGEND\n")
            f.write("=" * 100 + "\n\n")

            for node_id in non_terminal_nodes:
                node_data = self.graph.nodes[node_id]
                state_str = node_data.get('state_str', 'UNKNOWN')
                node_label = node_data.get('label', f'#{node_id}')
                is_initial = node_data.get('is_initial', False)
                depth = node_data.get('depth', 0)

                prefix = "[INITIAL] " if is_initial else ""
                f.write(f"{prefix}Node {node_label} (depth={depth}):\n")
                f.write(f"  {state_str}\n\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("EDGE ACTIONS\n")
            f.write("=" * 100 + "\n\n")

            for u, v, data in self.graph.edges(data=True):
                action = data.get('action', 'UNKNOWN')
                prob = data.get('probability', 0)
                u_label = self.graph.nodes[u].get('label', f'#{u}')
                v_label = self.graph.nodes[v].get('label', f'#{v}')

                f.write(f"{u_label} --[{action}]--> {v_label}  (p={prob:.4f})\n")

        print(f"✓ Legend table exported to: {output_path}")


def main():
    """Example usage - requires a trained trainer object."""
    print("=" * 80)
    print("GFLOWNET POLICY GRAPH VISUALIZATION")
    print("=" * 80)
    print("\nNote: This script should be run after training a GFlowNet.")
    print("See Demo_ILP.ipynb for integration with training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
