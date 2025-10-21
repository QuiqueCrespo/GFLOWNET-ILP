"""
Improved graph encoder with better structural awareness.

Key improvements:
1. Edge features (argument positions, directions)
2. Graph Isomorphism Network (GIN) layers - more powerful than GCN
3. Multiple pooling strategies
4. Structural fingerprints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data


class ImprovedGraphConstructor:
    """
    Enhanced graph constructor with edge features.
    """

    def __init__(self, predicate_vocab):
        self.predicate_vocab = predicate_vocab
        self.predicate_to_idx = {p: i for i, p in enumerate(predicate_vocab)}

    def theory_to_graph(self, theory):
        """
        Convert theory to graph with EDGE FEATURES.

        Edge features encode:
        - Argument position (1st arg, 2nd arg, etc.)
        - Edge direction (head->body, body->body)
        - Predicate identity
        """
        rule = theory[0]

        # Collect all variables and create mapping
        from src.logic_structures import get_all_variables
        variables = get_all_variables(theory)
        var_to_idx = {v.id: i for i, v in enumerate(variables)}

        num_nodes = len(variables)
        if num_nodes == 0:
            # Empty graph
            return Data(
                x=torch.zeros((1, 10)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 10))
            )

        # Node features (same as before, but add more structural info)
        node_features = []
        for var in variables:
            # Count how many times this variable appears
            appearances_in_head = sum(1 for v in rule.head.args if v.id == var.id)
            appearances_in_body = sum(
                1 for atom in rule.body for v in atom.args if v.id == var.id
            )

            # Position information
            first_position = -1
            for atom in [rule.head] + list(rule.body):
                for pos, v in enumerate(atom.args):
                    if v.id == var.id and first_position == -1:
                        first_position = pos

            # Feature vector
            features = [
                1.0,  # Constant
                appearances_in_head,
                appearances_in_body,
                appearances_in_head + appearances_in_body,
                1.0 if appearances_in_head > 0 else 0.0,  # In head
                1.0 if appearances_in_body > 0 else 0.0,  # In body
                first_position if first_position != -1 else 0.0,
                len(rule.body),  # Rule complexity
                num_nodes,  # Total variables
                var.id % 10 / 10.0,  # Variable ID (modulo for normalization)
            ]

            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Build edges WITH FEATURES
        edge_index = []
        edge_features = []

        def add_edges_from_atom(atom, is_head=False, atom_idx=-1):
            """Add edges with features from an atom."""
            pred_idx = self.predicate_to_idx.get(atom.predicate_name, 0)

            for i, v1 in enumerate(atom.args):
                for j, v2 in enumerate(atom.args):
                    if i != j:
                        idx1 = var_to_idx[v1.id]
                        idx2 = var_to_idx[v2.id]

                        edge_index.append([idx1, idx2])

                        # Edge features: [arg_pos_i, arg_pos_j, is_head, pred_idx, ...]
                        edge_feat = [
                            i / max(len(atom.args), 1),  # Normalized source position
                            j / max(len(atom.args), 1),  # Normalized target position
                            1.0 if is_head else 0.0,     # Is from head
                            1.0 if not is_head else 0.0, # Is from body
                            pred_idx / max(len(self.predicate_vocab), 1),  # Predicate ID
                            1.0 if i < j else 0.0,       # Direction indicator
                            len(atom.args) / 3.0,        # Arity (normalized)
                            atom_idx / max(len(rule.body), 1) if atom_idx >= 0 else 0.0,  # Atom position in body
                            1.0,  # Constant
                            0.0,  # Reserved
                        ]
                        edge_features.append(edge_feat)

        # Add edges from head
        add_edges_from_atom(rule.head, is_head=True)

        # Add edges from body atoms
        for atom_idx, atom in enumerate(rule.body):
            add_edges_from_atom(atom, is_head=False, atom_idx=atom_idx)

        # Convert to tensors
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 10), dtype=torch.float)

        return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer - more expressive than GCN."""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()

        # MLP for node updates
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.gin = GINConv(self.mlp)
        self.edge_encoder = nn.Linear(edge_dim, in_dim)

    def forward(self, x, edge_index, edge_attr):
        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)

        # Aggregate edge features to nodes
        # For each node, sum the edge features of incoming edges
        num_nodes = x.size(0)
        edge_aggregated = torch.zeros(num_nodes, edge_emb.size(1), device=x.device)

        for i in range(edge_index.size(1)):
            target_node = edge_index[1, i]
            edge_aggregated[target_node] += edge_emb[i]

        # Combine node features with aggregated edge features
        x_combined = torch.cat([x, edge_aggregated], dim=-1)

        # Apply GIN
        return self.gin(x_combined, edge_index)


class ImprovedStateEncoder(nn.Module):
    """
    Improved state encoder with:
    - GIN layers (more powerful than GCN)
    - Edge features
    - Multiple pooling strategies
    - Residual connections
    """

    def __init__(self, node_feature_dim=10, edge_feature_dim=10,
                 embedding_dim=64, num_layers=3):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.embedding_dim = embedding_dim

        # Input projection
        self.node_encoder = nn.Linear(node_feature_dim, embedding_dim)

        # GIN layers with residual connections
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.gin_layers.append(
                GINLayer(embedding_dim, embedding_dim, edge_feature_dim)
            )
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))

        # Multiple pooling strategies
        self.pool_projection = nn.Linear(embedding_dim * 3, embedding_dim)

        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, graph_data):
        """
        Args:
            graph_data: PyG Data object with x, edge_index, edge_attr

        Returns:
            state_embedding: [1, embedding_dim]
            node_embeddings: [num_nodes, embedding_dim]
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr

        # Encode node features
        x = self.node_encoder(x)

        # Apply GIN layers with residual connections
        for gin_layer, bn in zip(self.gin_layers, self.batch_norms):
            x_new = gin_layer(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x = x + F.relu(x_new)  # Residual connection

        node_embeddings = x

        # Multiple pooling strategies
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)

        # Concatenate pooling results
        pooled = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

        # Project back to embedding_dim
        state_embedding = self.pool_projection(pooled)

        # Final MLP
        state_embedding = self.output_mlp(state_embedding)

        return state_embedding, node_embeddings


def main():
    """Example usage."""
    print("Improved Graph Encoder with Edge Features and GIN")
    print("\nKey improvements:")
    print("  1. Edge features (argument positions, directions)")
    print("  2. GIN layers (more expressive than GCN)")
    print("  3. Multiple pooling strategies")
    print("  4. Residual connections")
    print("\nReplace EnhancedGraphConstructor and EnhancedStateEncoder")
    print("with ImprovedGraphConstructor and ImprovedStateEncoder")


if __name__ == "__main__":
    main()
