"""
Visualize the trained GFlowNet as a directed graph.

Nodes represent states (rules under construction).
Edges represent actions with their probabilities.
"""

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
            initial_state: Starting state
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
            node_label = f"L{body_length}"
            self.graph.add_node(state_id, label=node_label, state_str=state_str, depth=depth)

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
                        action_label = f"ADD {pred_name}"

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
                        action_label = f"UNIFY X{var1.id}=X{var2.id}"

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
                terminal_label = "TERMINAL"
                self.graph.add_node(terminal_id, label=terminal_label, state_str="[TERMINAL]", depth=depth+1, terminal=True)

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

    def visualize(self, output_path='gflownet_graph.png', figsize=(20, 12)):
        """Create a visualization of the policy graph."""
        if len(self.graph.nodes) == 0:
            print("No nodes in graph. Run explore_from_state() first.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Use hierarchical layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Or try hierarchical layout based on depth
        try:
            depths = nx.get_node_attributes(self.graph, 'depth')
            # Group by depth for better layout
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

                x = d * 3  # Horizontal spacing by depth
                y = depth_positions[d] * 2 - (depth_counts[d] - 1)  # Vertical spacing
                pos[node] = (x, y)
                depth_positions[d] += 1
        except:
            pass

        # Draw nodes
        terminal_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('terminal', False)]
        regular_nodes = [n for n in self.graph.nodes() if n not in terminal_nodes]

        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=regular_nodes,
            node_color='lightblue',
            node_size=2000,
            alpha=0.9,
            ax=ax
        )

        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=terminal_nodes,
            node_color='lightcoral',
            node_size=1500,
            alpha=0.9,
            ax=ax,
            node_shape='s'
        )

        # Draw node labels
        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold', ax=ax)

        # Draw edges with colors based on action type
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

            edge_widths.append(1 + 4 * prob)  # Width proportional to probability

        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

        # Draw edge labels (action + probability)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            action = data.get('action', '')
            prob = data.get('probability', 0)
            edge_labels[(u, v)] = f"{action}\n({prob:.3f})"

        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels,
            font_size=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ax=ax
        )

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='ADD_ATOM'),
            Line2D([0], [0], color='green', lw=2, label='UNIFY_VARIABLES'),
            Line2D([0], [0], color='red', lw=2, label='TERMINATE'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=10, label='State (Rule)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral',
                   markersize=10, label='Terminal'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        ax.set_title('GFlowNet Policy Graph\n(Node = State, Edge = Action with Probability)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Policy graph saved to: {output_path}")

        # Print statistics
        print(f"\nGraph Statistics:")
        print(f"  - Total nodes (states): {len(self.graph.nodes)}")
        print(f"  - Total edges (actions): {len(self.graph.edges)}")
        print(f"  - Terminal nodes: {len(terminal_nodes)}")

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
