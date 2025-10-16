"""
Hierarchical GFlowNet models: Strategist and Tacticians.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .logic_structures import Variable


class StrategistGFlowNet(nn.Module):
    """
    High-level policy that chooses between ADD_ATOM and UNIFY_VARIABLES actions.
    Also outputs state flow F(s).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Policy head: outputs logits for 2 actions
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [ADD_ATOM, UNIFY_VARIABLES]
        )

        # Flow head: outputs log F(s)
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            - action_logits: [batch_size, 2] or [2]
            - log_flow: [batch_size, 1] or [1]
        """
        action_logits = self.policy_net(state_embedding)
        log_flow = self.flow_net(state_embedding)
        return action_logits, log_flow


class AtomAdderGFlowNet(nn.Module):
    """
    Tactician that selects which predicate to add to the theory.
    """

    def __init__(self, embedding_dim: int, num_predicates: int, hidden_dim: int = 128):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_predicates)
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            predicate_logits: [batch_size, num_predicates] or [num_predicates]
        """
        return self.policy_net(state_embedding)


class VariableUnifierGFlowNet(nn.Module):
    """
    Tactician that selects which pair of variables to unify.
    Uses attention mechanism to score variable pairs.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Query and key projections for attention
        self.query_net = nn.Linear(embedding_dim, hidden_dim)
        self.key_net = nn.Linear(embedding_dim, hidden_dim)

        # Additional transformation for state context
        self.context_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, state_embedding: torch.Tensor,
                variable_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [embedding_dim] - global state representation
            variable_embeddings: [num_vars, embedding_dim] - embeddings for each variable

        Returns:
            pair_logits: [num_pairs] - logits for each valid variable pair
        """
        num_vars = variable_embeddings.size(0)

        if num_vars < 2:
            # Not enough variables to form a pair
            return torch.tensor([], dtype=torch.float)

        # Generate queries and keys for all variables
        queries = self.query_net(variable_embeddings)  # [num_vars, hidden_dim]
        keys = self.key_net(variable_embeddings)  # [num_vars, hidden_dim]

        # Add state context
        context = self.context_net(state_embedding)  # [hidden_dim]
        queries = queries + context.unsqueeze(0)
        keys = keys + context.unsqueeze(0)

        # Compute attention scores for all pairs (i, j) where i < j
        pair_logits = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                # Score for unifying variable i and j
                score = torch.dot(queries[i], keys[j])
                pair_logits.append(score)

        return torch.stack(pair_logits) if pair_logits else torch.tensor([], dtype=torch.float)

    def get_pair_indices(self, num_vars: int) -> List[Tuple[int, int]]:
        """Get list of (i, j) pairs where i < j."""
        pairs = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                pairs.append((i, j))
        return pairs


class HierarchicalGFlowNet(nn.Module):
    """
    Complete hierarchical GFlowNet combining strategist and tacticians.
    """

    def __init__(self, embedding_dim: int, num_predicates: int, hidden_dim: int = 128):
        super().__init__()

        self.strategist = StrategistGFlowNet(embedding_dim, hidden_dim)
        self.atom_adder = AtomAdderGFlowNet(embedding_dim, num_predicates, hidden_dim)
        self.variable_unifier = VariableUnifierGFlowNet(embedding_dim, hidden_dim)

    def forward_strategist(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get strategist action logits and flow."""
        return self.strategist(state_embedding)

    def forward_atom_adder(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get atom adder predicate logits."""
        return self.atom_adder(state_embedding)

    def forward_variable_unifier(self, state_embedding: torch.Tensor,
                                 variable_embeddings: torch.Tensor) -> torch.Tensor:
        """Get variable unifier pair logits."""
        return self.variable_unifier(state_embedding, variable_embeddings)

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from all submodels."""
        params = []
        params.extend(self.strategist.parameters())
        params.extend(self.atom_adder.parameters())
        params.extend(self.variable_unifier.parameters())
        return params
