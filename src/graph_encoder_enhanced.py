"""
Enhanced graph construction with rich node and edge features.

Improvements over basic encoding:
1. Rich variable features (appears_in_head, is_chain_var, etc.)
2. Rich predicate features (is_head, has_self_loop, etc.)
3. Edge features (argument_position, edge_type)
4. Hierarchical attention-based pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from typing import List, Tuple

from .logic_structures import Theory, Variable, get_all_variables


class EnhancedGraphConstructor:
    """Converts a theory into an enhanced graph representation with rich features."""

    def __init__(self, predicate_vocab: List[str]):
        self.predicate_vocab = predicate_vocab
        self.pred_to_idx = {pred: idx for idx, pred in enumerate(predicate_vocab)}

    def _get_canonical_variables(self, theory: Theory) -> List[Variable]:
        """Returns variables in deterministic order."""
        ordered_vars = []
        seen_vars = set()
        for rule in theory:
            for arg in rule.head.args:
                if isinstance(arg, Variable) and arg not in seen_vars:
                    ordered_vars.append(arg)
                    seen_vars.add(arg)
            for atom in rule.body:
                for arg in atom.args:
                    if isinstance(arg, Variable) and arg not in seen_vars:
                        ordered_vars.append(arg)
                        seen_vars.add(arg)
        return ordered_vars

    def theory_to_graph(self, theory: Theory) -> Data:
        """
        Convert theory to enhanced PyTorch Geometric graph.

        Node features:
        - Variables: [appears_in_head, appears_in_body, appears_multiple,
                      is_chain_var, total_occurrences, is_variable=1]
        - Predicates: [*one_hot_predicate, is_head, is_body, body_position,
                       has_self_loop, num_unique_vars, total_vars, is_variable=0]

        Edge features:
        - [argument_position, is_head_edge, is_body_edge]
        """
        if not theory:
            # Empty theory
            var_dim = 6
            pred_dim = len(self.predicate_vocab) + 7
            return Data(
                x=torch.zeros((1, max(var_dim, pred_dim))),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 3), dtype=torch.float),
                num_nodes=1,
                is_variable=torch.tensor([False]),
                is_head=torch.tensor([False]),
                is_body=torch.tensor([False])
            )

        rule = theory[0]  # Assuming single rule per theory
        node_features = []
        edge_index = []
        edge_features = []

        # Analyze variable usage patterns
        head_vars = set(rule.head.args)
        body_vars = set()
        var_counts = {}

        for atom in rule.body:
            for var in atom.args:
                body_vars.add(var)
                var_counts[var] = var_counts.get(var, 0) + 1

        # Build variable nodes with rich features
        variables = self._get_canonical_variables(theory)
        var_to_node = {var: idx for idx, var in enumerate(variables)}

        for var in variables:
            var_features = [
                1.0 if var in head_vars else 0.0,              # appears_in_head
                1.0 if var in body_vars else 0.0,              # appears_in_body
                1.0 if var_counts.get(var, 0) > 1 else 0.0,   # appears_multiple
                1.0 if (var in body_vars and var not in head_vars) else 0.0,  # is_chain_var
                float(var_counts.get(var, 0)),                 # total_occurrences
                1.0                                             # is_variable flag
            ]
            node_features.append(var_features)

        node_id = len(variables)

        # Build predicate nodes - HEAD
        head_self_loop = len(set(rule.head.args)) < len(rule.head.args)
        head_features = [
            *self._one_hot_predicate(rule.head.predicate_name),  # predicate type
            1.0,                                      # is_head
            0.0,                                      # is_body
            0.0,                                      # body_position
            1.0 if head_self_loop else 0.0,         # has_self_loop
            float(len(set(rule.head.args))),         # num_unique_vars
            float(len(rule.head.args)),              # total_vars
            0.0                                       # is_variable flag
        ]
        node_features.append(head_features)
        head_node_id = node_id
        node_id += 1

        # Add edges from head with argument position
        for arg_pos, arg in enumerate(rule.head.args):
            if arg in var_to_node:
                var_node = var_to_node[arg]
                # Bidirectional edges with position info
                edge_index.extend([[var_node, head_node_id], [head_node_id, var_node]])
                edge_feat = [float(arg_pos), 1.0, 0.0]  # position, is_head_edge, is_body_edge
                edge_features.extend([edge_feat, edge_feat])

        # Build predicate nodes - BODY
        for body_pos, atom in enumerate(rule.body):
            atom_self_loop = len(set(atom.args)) < len(atom.args)
            atom_features = [
                *self._one_hot_predicate(atom.predicate_name),
                0.0,                                      # is_head
                1.0,                                      # is_body
                float(body_pos),                          # body_position
                1.0 if atom_self_loop else 0.0,         # has_self_loop
                float(len(set(atom.args))),              # num_unique_vars
                float(len(atom.args)),                   # total_vars
                0.0                                       # is_variable flag
            ]
            node_features.append(atom_features)
            atom_node_id = node_id
            node_id += 1

            # Add edges with argument position
            for arg_pos, arg in enumerate(atom.args):
                if arg in var_to_node:
                    var_node = var_to_node[arg]
                    edge_index.extend([[var_node, atom_node_id], [atom_node_id, var_node]])
                    edge_feat = [float(arg_pos), 0.0, 1.0]  # position, is_head_edge, is_body_edge
                    edge_features.extend([edge_feat, edge_feat])

        # Pad features to same length (max of var and pred dimensions)
        var_dim = 6
        pred_dim = len(self.predicate_vocab) + 7
        max_dim = max(var_dim, pred_dim)

        # Pad all features to max_dim
        padded_features = []
        for feat in node_features:
            if len(feat) < max_dim:
                feat = feat + [0.0] * (max_dim - len(feat))
            padded_features.append(feat)

        # Convert to tensors
        x = torch.tensor(padded_features, dtype=torch.float)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 3), dtype=torch.float)

        # Create masks for hierarchical pooling
        num_nodes = len(node_features)
        is_variable = torch.tensor([f[-1] == 1.0 for f in node_features])
        is_head = torch.zeros(num_nodes, dtype=torch.bool)
        is_head[len(variables)] = True  # Head is first predicate node
        is_body = torch.zeros(num_nodes, dtype=torch.bool)
        if num_nodes > len(variables) + 1:
            is_body[len(variables)+1:] = True

        data = Data(
            x=x,
            edge_index=edge_index_t,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            is_variable=is_variable,
            is_head=is_head,
            is_body=is_body
        )

        return data

    def _one_hot_predicate(self, pred_name: str) -> List[float]:
        """One-hot encoding for predicate."""
        vec = [0.0] * len(self.predicate_vocab)
        if pred_name in self.pred_to_idx:
            vec[self.pred_to_idx[pred_name]] = 1.0
        return vec

    def get_variable_node_ids(self, theory: Theory):
        """Get mapping from variables to their node IDs in the graph."""
        from typing import Dict
        var_to_node = {}
        variables = self._get_canonical_variables(theory)
        for node_id, var in enumerate(variables):
            var_to_node[var] = node_id
        return var_to_node


class EnhancedStateEncoder(nn.Module):
    """
    GNN-based encoder with:
    1. Separate encoders for variables and predicates
    2. Edge-aware message passing (GAT with edge features)
    3. Hierarchical attention-based pooling
    """

    def __init__(self, predicate_vocab_size: int, embedding_dim: int, num_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Variable features: 6 dimensions
        self.var_feature_dim = 6
        # Predicate features: vocab_size + 7 dimensions
        self.pred_feature_dim = predicate_vocab_size + 7
        # Padded node features: max of the two
        self.node_feature_dim = max(self.var_feature_dim, self.pred_feature_dim)
        # Edge features: 3 dimensions
        edge_feature_dim = 3

        # Separate initial encoders (take full padded feature vector)
        self.var_encoder = nn.Linear(self.node_feature_dim, embedding_dim)
        self.pred_encoder = nn.Linear(self.node_feature_dim, embedding_dim)

        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_feature_dim, embedding_dim)

        # Graph Attention Network layers with edge features
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                GATConv(embedding_dim, embedding_dim, heads=4, concat=False,
                       edge_dim=embedding_dim)
            )

        # Attention-based pooling for different node types
        self.var_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        self.pred_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Combiner for final graph embedding
        self.combiner = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_data: Enhanced graph with rich features

        Returns:
            - graph_embedding: [1, embedding_dim]
            - node_embeddings: [num_nodes, embedding_dim]
        """
        # Separate encoding by node type
        var_mask = graph_data.is_variable
        pred_mask = ~var_mask

        # Initialize node embeddings
        x = torch.zeros(graph_data.num_nodes, self.embedding_dim)

        # All features are already padded to same length, so we can encode directly
        if var_mask.any():
            x[var_mask] = self.var_encoder(graph_data.x[var_mask])

        if pred_mask.any():
            x[pred_mask] = self.pred_encoder(graph_data.x[pred_mask])

        # Encode edge features
        if graph_data.edge_attr.shape[0] > 0:
            edge_attr = self.edge_encoder(graph_data.edge_attr)
        else:
            edge_attr = None

        # Message passing with edge features
        for conv in self.conv_layers:
            if edge_attr is not None:
                x = conv(x, graph_data.edge_index, edge_attr)
            else:
                x = conv(x, graph_data.edge_index)
            x = F.relu(x)

        node_embeddings = x

        # Hierarchical attention pooling
        var_embeddings = x[var_mask]
        pred_embeddings = x[pred_mask]

        if var_embeddings.shape[0] > 0:
            var_attn_scores = self.var_attention(var_embeddings)
            var_attn_weights = F.softmax(var_attn_scores, dim=0)
            var_pool = (var_attn_weights * var_embeddings).sum(dim=0, keepdim=True)
        else:
            var_pool = torch.zeros(1, self.embedding_dim)

        if pred_embeddings.shape[0] > 0:
            pred_attn_scores = self.pred_attention(pred_embeddings)
            pred_attn_weights = F.softmax(pred_attn_scores, dim=0)
            pred_pool = (pred_attn_weights * pred_embeddings).sum(dim=0, keepdim=True)
        else:
            pred_pool = torch.zeros(1, self.embedding_dim)

        # Combine variable and predicate pools
        combined = torch.cat([var_pool, pred_pool], dim=-1)
        graph_embedding = self.combiner(combined)

        return graph_embedding, node_embeddings
