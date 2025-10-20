"""
Hierarchical GFlowNet models: Strategist and Tacticians.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .logic_structures import Variable


class StrategistGFlowNet(nn.Module):
    """
    High-level policy that chooses between ADD_ATOM, UNIFY_VARIABLES or TERMINATE actions.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Policy head: outputs logits for 3 actions
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [ADD_ATOM, UNIFY_VARIABLES, TERMINATE]
        )


    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            - action_logits: [batch_size, 2] or [2]
        """
        action_logits = self.policy_net(state_embedding)
        return action_logits


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

class ForwardFlow(nn.Module):
    """
    Simple MLP to predict forward flow log F(s') from state embedding.
    Used for computing backward probabilities.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            log_flow: [batch_size, 1] or [1]
        """
        return self.flow_net(state_embedding)
    
class BackwardStrategist(nn.Module):
    """
    Backward strategist that predicts which action type was taken to reach current state.
    Given state s', predicts whether ADD_ATOM or UNIFY_VARIABLES was used to get there.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Policy head: outputs logits for 2 action types (ADD_ATOM, UNIFY_VARIABLES)
        # Note: TERMINATE is not a backward action since we can't go back from termination
        self.policy_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [ADD_ATOM, UNIFY_VARIABLES]
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            action_logits: [batch_size, 2] or [2]
        """
        return self.policy_net(state_embedding)


class BackwardAtomRemover(nn.Module):
    """
    Backward atom remover that predicts which predicate was added to reach current state.
    Given state s', predicts which atom in the body was the last one added.
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


class BackwardVariableSplitter(nn.Module):
    """
    Backward variable splitter that predicts which variables were unified to reach current state.
    Given state s', predicts which pair of variables existed before unification.
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
            pair_logits: [num_vars] - logits for each variable being the unified one
        """
        num_vars = variable_embeddings.size(0)

        if num_vars < 1:
            return torch.tensor([], dtype=torch.float)

        # Generate queries for all variables
        queries = self.query_net(variable_embeddings)  # [num_vars, hidden_dim]

        # Add state context
        context = self.context_net(state_embedding)  # [hidden_dim]
        queries = queries + context.unsqueeze(0)

        # Compute scores for each variable being the unified variable
        # Higher score = more likely this variable was created by unification
        scores = torch.sum(queries * context.unsqueeze(0), dim=1)  # [num_vars]

        return scores


class SophisticatedBackwardPolicy(nn.Module):
    """
    Sophisticated hierarchical backward policy that predicts the specific action
    that was performed to reach the current state.

    This mirrors the forward policy structure but operates in reverse:
    - BackwardStrategist: predicts action type (ADD_ATOM vs UNIFY_VARIABLES)
    - BackwardAtomRemover: predicts which predicate was added
    - BackwardVariableSplitter: predicts which variables were unified
    """

    def __init__(self, embedding_dim: int, num_predicates: int, predicate_vocab: List[str], hidden_dim: int = 128):
        super().__init__()

        self.strategist = BackwardStrategist(embedding_dim, hidden_dim)
        self.atom_remover = BackwardAtomRemover(embedding_dim, num_predicates, hidden_dim)
        self.variable_splitter = BackwardVariableSplitter(embedding_dim, hidden_dim)
        self.predicate_vocab = predicate_vocab  # Store predicate vocabulary for index mapping

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get backward strategist action logits."""
        return self.strategist(state_embedding)

    def get_log_probability(self, next_state_embedding: torch.Tensor,
                          next_state, previous_state,
                          action_type: str, action_detail: any,
                          variable_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the log probability of the specific backward transition P_B(s'->s).

        Args:
            next_state_embedding: Embedding of the next state (s')
            next_state: The next state (Theory) - s'
            previous_state: The previous state (Theory) - s
            action_type: The forward action that was taken ('ADD_ATOM' or 'UNIFY_VARIABLES')
            action_detail: Details of the action (predicate name or variable pair)
            variable_embeddings: Embeddings of variables in next_state (needed for UNIFY_VARIABLES)

        Returns:
            log_prob: Log probability of the backward transition P_B(s'->s)
        """
        # Step 1: Get backward strategist probability
        action_logits = self.strategist(next_state_embedding)
        action_log_probs = F.log_softmax(action_logits, dim=-1)

        # Map action type to index
        action_idx = 0 if action_type == 'ADD_ATOM' else 1
        log_prob_action_type = action_log_probs[action_idx]

        # Step 2: Get action-specific probability
        if action_type == 'ADD_ATOM':
            # Predict which predicate was added
            # action_detail is the predicate name that was added

            # Get atom remover logits
            atom_logits = self.atom_remover(next_state_embedding)
            atom_log_probs = F.log_softmax(atom_logits, dim=-1)

            # Map the predicate name to its index in the vocabulary
            if action_detail is not None and action_detail in self.predicate_vocab:
                predicate_idx = self.predicate_vocab.index(action_detail)
                log_prob_detail = atom_log_probs[predicate_idx]
            else:
                # Fallback: if predicate not found, use small penalty
                # This shouldn't happen in normal operation
                log_prob_detail = torch.tensor(-10.0)  # Large negative log prob

        elif action_type == 'UNIFY_VARIABLES':
            # Predict which variables were unified in the forward direction
            # action_detail is a tuple (var1, var2) from the PREVIOUS state that were unified
            # The challenge: we're looking at the NEXT state, which has fewer variables

            # For variable unification, we need embeddings from the PREVIOUS state
            # since that's where the two variables existed before being unified
            # However, we only have next_state embeddings here

            # Practical solution: The backward variable splitter predicts which variable
            # in the current (next) state was the result of unification
            # We use a heuristic: uniform probability over valid pairs in previous state

            if action_detail is not None:
                # We know which pair was unified: var1, var2
                # For backward probability, we can use a simpler estimate:
                # P_B(unify var1,var2 | next_state) ≈ 1 / num_possible_pairs_in_prev_state

                # Get number of variables in previous state to estimate num possible pairs
                from .logic_structures import get_all_variables, get_valid_variable_pairs

                prev_vars = get_all_variables(previous_state)
                num_prev_vars = len(prev_vars)

                if num_prev_vars >= 2:
                    # Approximate: uniform distribution over all valid pairs
                    # Number of pairs = n*(n-1)/2
                    num_pairs = num_prev_vars * (num_prev_vars - 1) // 2
                    import math
                    log_prob_detail = torch.tensor(-math.log(max(num_pairs, 1)), dtype=torch.float32)
                else:
                    log_prob_detail = torch.tensor(-10.0)
            else:
                log_prob_detail = torch.tensor(-10.0)
        else:
            log_prob_detail = torch.tensor(0.0)

        return log_prob_action_type + log_prob_detail
    
class UniformBackwardPolicy(nn.Module):
    """
    Uniform backward policy that assigns uniform probability over possible parent states.
    Used when no learned backward policy is desired.
    """

    def __init__(self, num_predicates: int):
        super().__init__()
        self.num_predicates = num_predicates

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            action_logits: [batch_size, 2] or [2] for action types (ADD_ATOM, UNIFY_VARIABLES)
            Note: Returns uniform logits (zeros) which correspond to uniform probabilities after softmax
        """
        if state_embedding.dim() == 1:
            # Single state, return uniform logits (zeros for uniform distribution)
            return torch.zeros(2)  # 2 action types: ADD_ATOM, UNIFY_VARIABLES
        else:
            # Batch of states
            batch_size = state_embedding.size(0)
            return torch.zeros(batch_size, 2)

    def get_log_probability(self, next_state_embedding: torch.Tensor, previous_state) -> torch.Tensor:
        """
        Compute the log probability of transitioning backward from next_state to previous_state.
        For uniform backward policy, this is uniform over possible parent states.

        Args:
            next_state_embedding: Embedding of the next state (s')
            previous_state: The previous state (s) - used to determine num parents

        Returns:
            log_prob: Log probability of the backward transition P_B(s'->s)
        """
        # For uniform backward policy: P_B(s'->s) = 1 / num_possible_parents(s')
        # We need to estimate how many different states could lead to next_state

        from .logic_structures import get_all_variables
        import math

        # Estimate number of possible parent states that could lead to next_state:
        # 1. ADD_ATOM parents: One parent for each predicate that could have been added
        #    Number = num_predicates (each predicate gives a different parent state)
        # 2. UNIFY_VARIABLES parents: One parent for each pair of variables that could have been unified
        #    Number = n*(n-1)/2 where n is number of variables in previous state

        # Get number of variables in previous state to estimate unification parents
        if previous_state and len(previous_state) > 0:
            prev_vars = get_all_variables(previous_state)
            num_prev_vars = len(prev_vars)

            # Count possible parent states:
            # - From ADD_ATOM: num_predicates different parent states
            num_add_parents = self.num_predicates

            # - From UNIFY_VARIABLES: n*(n-1)/2 different parent states (one per variable pair)
            num_unify_parents = (num_prev_vars * (num_prev_vars - 1) // 2) if num_prev_vars >= 2 else 0

            # Total possible parents = sum of parents from both action types
            num_possible_parents = num_add_parents + num_unify_parents
        else:
            # Fallback: only ADD_ATOM possible
            num_possible_parents = self.num_predicates

        # Uniform probability: P_B(s'->s) = 1 / num_possible_parents
        return torch.tensor(-math.log(max(1, num_possible_parents)), dtype=torch.float32)

class HierarchicalGFlowNet(nn.Module):
    """
    Complete hierarchical GFlowNet combining strategist and tacticians.
    """

    def __init__(self, embedding_dim: int, num_predicates: int, num_backward_actions: int = 2,
                 hidden_dim: int = 128, use_sophisticated_backward: bool = True, predicate_vocab: List[str] = None):
        super().__init__()

        self.strategist = StrategistGFlowNet(embedding_dim, hidden_dim)
        self.atom_adder = AtomAdderGFlowNet(embedding_dim, num_predicates, hidden_dim)
        self.variable_unifier = VariableUnifierGFlowNet(embedding_dim, hidden_dim)
        self.forward_flow_net = ForwardFlow(embedding_dim, hidden_dim)

        # Choose between sophisticated learned backward policy or simple uniform policy
        if use_sophisticated_backward:
            if predicate_vocab is None:
                raise ValueError("predicate_vocab is required when use_sophisticated_backward=True")
            self.backward_policy = SophisticatedBackwardPolicy(embedding_dim, num_predicates, predicate_vocab, hidden_dim)
        else:
            # Uniform policy needs num_predicates to estimate parent states
            self.backward_policy = UniformBackwardPolicy(num_predicates)

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

    def forward_flow(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get forward flow log F(s') from state embedding."""
        return self.forward_flow_net(state_embedding)
    
    def forward_backward_policy(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get backward action logits from state embedding."""
        return self.backward_policy(state_embedding)

    def get_backward_log_probability(self, next_state_embedding: torch.Tensor,
                                    next_state, previous_state,
                                    action_type: str, action_detail: any,
                                    variable_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Compute backward log probability P_B(s'->s) for a transition.
        Handles both sophisticated and uniform backward policies.

        Args:
            next_state_embedding: Embedding of the next state (s')
            next_state: The next state (Theory) - s'
            previous_state: The previous state (Theory) - s
            action_type: The forward action that was taken ('ADD_ATOM' or 'UNIFY_VARIABLES')
            action_detail: Details of the action (predicate name or variable pair)
            variable_embeddings: Embeddings of variables in next_state (optional, for sophisticated policy)

        Returns:
            log_prob: Log probability of the backward transition P_B(s'->s)
        """
        if isinstance(self.backward_policy, SophisticatedBackwardPolicy):
            # Sophisticated policy needs all parameters
            return self.backward_policy.get_log_probability(
                next_state_embedding,
                next_state,
                previous_state,
                action_type,
                action_detail,
                variable_embeddings
            )
        else:
            # Simple uniform backward policy only needs embeddings and previous state
            return self.backward_policy.get_log_probability(
                next_state_embedding,
                previous_state
            )

    def get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from all submodels."""
        params = []
        params.extend(self.strategist.parameters())
        params.extend(self.atom_adder.parameters())
        params.extend(self.variable_unifier.parameters())
        params.extend(self.forward_flow_net.parameters())
        params.extend(self.backward_policy.parameters())
        return params
