"""
Graph construction and GNN-based state encoder for theory representation.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List, Dict, Tuple

from .logic_structures import Theory, Variable, get_all_variables


class GraphConstructor:
    """Converts a theory into a graph representation."""

    def __init__(self, predicate_vocab: List[str]):
        """
        Args:
            predicate_vocab: List of all possible predicate names
        """
        self.predicate_vocab = predicate_vocab
        self.pred_to_idx = {pred: idx for idx, pred in enumerate(predicate_vocab)}
    def _get_canonical_variables(self, theory: Theory) -> List[Variable]:
        """
        Returns a list of all unique variables in the theory, ordered deterministically
        by their first appearance in a top-to-bottom, left-to-right traversal.
        """
        ordered_vars = []
        seen_vars = set()
        for rule in theory:
            # Head
            for arg in rule.head.args:
                if isinstance(arg, Variable) and arg not in seen_vars:
                    ordered_vars.append(arg)
                    seen_vars.add(arg)
            # Body
            for atom in rule.body:
                for arg in atom.args:
                    if isinstance(arg, Variable) and arg not in seen_vars:
                        ordered_vars.append(arg)
                        seen_vars.add(arg)
        return ordered_vars
    
    def theory_to_graph(self, theory: Theory) -> Data:
        """
        Convert theory to PyTorch Geometric graph.

        Graph structure:
        - One node per predicate instance (head and each body atom)
        - One node per unique variable
        - Edges connect variables to predicates they appear in
        """
        if not theory:
            # Empty theory - return minimal graph
            return Data(
                x=torch.zeros((1, len(self.predicate_vocab) + 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=1
            )

        node_features = []
        node_types = []
        node_to_id = {}
        edges = []
        node_id = 0

        # Collect all unique variables using the CANONICAL ordering
        variables = self._get_canonical_variables(theory) # <-- THE ONLY CHANGE NEEDED HERE

        var_to_node = {}
        for var in variables:
            var_to_node[var] = node_id
            node_to_id[f'var_{var.id}'] = node_id
            node_types.append('var')
            # Variable node feature: [0, 0, ..., 0, 1] (last dim indicates variable)
            var_feature = [0] * len(self.predicate_vocab) + [1]
            node_features.append(var_feature)
            node_id += 1

        # Process each rule
        for rule_idx, rule in enumerate(theory):
            # Add head predicate node
            head_node = node_id
            node_to_id[f'rule_{rule_idx}_head'] = head_node
            node_types.append('pred')

            # One-hot encoding for predicate + 0 for variable indicator
            pred_feature = [0] * len(self.predicate_vocab) + [0]
            if rule.head.predicate_name in self.pred_to_idx:
                pred_feature[self.pred_to_idx[rule.head.predicate_name]] = 1
            node_features.append(pred_feature)
            node_id += 1

            # Connect head predicate to its argument variables
            for arg in rule.head.args:
                if isinstance(arg, Variable) and arg in var_to_node:
                    var_node = var_to_node[arg]
                    edges.append([var_node, head_node])
                    edges.append([head_node, var_node])

            # Add body atom nodes
            for atom_idx, atom in enumerate(rule.body):
                atom_node = node_id
                node_to_id[f'rule_{rule_idx}_body_{atom_idx}'] = atom_node
                node_types.append('pred')

                pred_feature = [0] * len(self.predicate_vocab) + [0]
                if atom.predicate_name in self.pred_to_idx:
                    pred_feature[self.pred_to_idx[atom.predicate_name]] = 1
                node_features.append(pred_feature)
                node_id += 1

                # Connect body atom to its argument variables
                for arg in atom.args:
                    if isinstance(arg, Variable) and arg in var_to_node:
                        var_node = var_to_node[arg]
                        edges.append([var_node, atom_node])
                        edges.append([atom_node, var_node])

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, num_nodes=len(node_features))

    def get_variable_node_ids(self, theory: Theory) -> Dict[Variable, int]:
        """Get mapping from variables to their node IDs in the graph."""
        var_to_node = {}
        variables = get_all_variables(theory)
        for node_id, var in enumerate(variables):
            var_to_node[var] = node_id
        return var_to_node


class StateEncoder(nn.Module):
    """GNN-based encoder that converts theory graph to embedding vector."""

    def __init__(self, node_feature_dim: int, embedding_dim: int, num_layers: int = 3):
        """
        Args:
            node_feature_dim: Dimension of input node features
            embedding_dim: Dimension of output embeddings
            num_layers: Number of GCN layers
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feature_dim, embedding_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(embedding_dim, embedding_dim))

    def forward(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_data: PyTorch Geometric graph data

        Returns:
            - graph_embedding: Single vector representing the entire graph [embedding_dim]
            - node_embeddings: Embeddings for each node [num_nodes, embedding_dim]
        """
        x = graph_data.x
        edge_index = graph_data.edge_index

        # Apply GCN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = torch.relu(x)

        node_embeddings = x

        # Global pooling to get graph-level embedding
        # If batch attribute exists, use it; otherwise assume single graph
        if hasattr(graph_data, 'batch'):
            graph_embedding = global_mean_pool(x, graph_data.batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)

        return graph_embedding, node_embeddings


def batch_theories_to_graphs(theories: List[Theory], graph_constructor: GraphConstructor) -> Batch:
    """Convert a batch of theories to a batched graph."""
    graphs = [graph_constructor.theory_to_graph(theory) for theory in theories]
    return Batch.from_data_list(graphs)
