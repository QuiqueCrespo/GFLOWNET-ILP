"""
Hierarchical GFlowNet models: Strategist and Tacticians.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .logic_structures import get_all_variables, get_valid_variable_pairs

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


    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [batch_size, embedding_dim] or [embedding_dim]

        Returns:
            - action_logits: [batch_size, 3] or [3]
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
            pair_log_probs: [num_pairs] - log probabilities for each valid variable pair
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
        pair_scores = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                # Score for unifying variable i and j
                score = torch.dot(queries[i], keys[j])
                pair_scores.append(score)
        
        if not pair_scores:
            return torch.tensor([], dtype=torch.float)

        # Convert scores to log probabilities
        pair_scores_tensor = torch.stack(pair_scores)
        return pair_scores_tensor
    

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
    Given next_state s' and previous_state s, predicts which pair of variables in s were unified.

    Uses pairwise attention over variables in the previous state to score all possible pairs.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Query and key projections for pairwise attention
        self.query_net = nn.Linear(embedding_dim, hidden_dim)
        self.key_net = nn.Linear(embedding_dim, hidden_dim)

        # State context conditioning
        self.context_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # MLP to score variable pairs based on their interaction
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, next_state_embedding: torch.Tensor,
                prev_variable_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all possible variable pairs in the previous state.

        Args:
            next_state_embedding: [embedding_dim] - embedding of next state (after unification)
            prev_variable_embeddings: [num_prev_vars, embedding_dim] - embeddings of variables
                                      in the PREVIOUS state (before unification)

        Returns:
            pair_logits: [num_pairs] - logits for each possible pair (i, j) where i < j
                         Returns empty tensor if num_prev_vars < 2
        """
        num_vars = prev_variable_embeddings.size(0)

        if num_vars < 2:
            # Not enough variables to form a pair
            return torch.tensor([], dtype=torch.float)

        # Get state context
        context = self.context_net(next_state_embedding)  # [hidden_dim]

        # Generate queries and keys for all variables
        queries = self.query_net(prev_variable_embeddings)  # [num_vars, hidden_dim]
        keys = self.key_net(prev_variable_embeddings)  # [num_vars, hidden_dim]

        # Add context to queries and keys
        queries = queries + context.unsqueeze(0)
        keys = keys + context.unsqueeze(0)

        # Score all pairs (i, j) where i < j
        pair_scores = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                # Concatenate query and key for this pair
                pair_repr = torch.cat([queries[i], keys[j]], dim=0)  # [hidden_dim * 2]

                # Score this pair
                score = self.pair_scorer(pair_repr)  # [1]
                pair_scores.append(score.squeeze())

        if not pair_scores:
            return torch.tensor([], dtype=torch.float)

        # Return raw logits (not log_softmax)
        pair_logits = torch.stack(pair_scores)  # [num_pairs]
        return pair_logits

    def get_pair_indices(self, num_vars: int) -> List[Tuple[int, int]]:
        """
        Get list of (i, j) pairs in the order they are scored.
        (Mirrors the forward unifier's helper).
        """
        pairs = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                pairs.append((i, j))
        return pairs


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
            variable_embeddings: Embeddings of variables in PREVIOUS state (needed for UNIFY_VARIABLES)

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
            # This part was already correct
            atom_logits = self.atom_remover(next_state_embedding)
            atom_log_probs = F.log_softmax(atom_logits, dim=-1)

            if action_detail is not None and action_detail in self.predicate_vocab:
                predicate_idx = self.predicate_vocab.index(action_detail)
                log_prob_detail = atom_log_probs[predicate_idx]
            else:
                log_prob_detail = torch.tensor(-10.0)

        elif action_type == 'UNIFY_VARIABLES':
            
            # --- START OF FIX ---
            
            # We need to import this function
            

            if action_detail is None or variable_embeddings is None:
                return torch.tensor(-10.0)

            valid_pairs = get_valid_variable_pairs(previous_state)
            all_prev_variables = get_all_variables(previous_state)
            num_prev_vars = len(all_prev_variables)

            if not valid_pairs or num_prev_vars < 2:
                return torch.tensor(-1e6)

            # Get logits for ALL possible pairs from the previous state
            pair_logits = self.variable_splitter(next_state_embedding, variable_embeddings)

            # Check if model returned any logits
            if pair_logits.numel() == 0:
                 return torch.tensor(-1e6)

            # --- Apply mask to get correct probabilities ---
            
            # Use the new helper method to get the pairs in the correct order
            all_scored_pairs = self.variable_splitter.get_pair_indices(num_prev_vars)
            
            # Ensure the number of logits matches our expectation
            if len(all_scored_pairs) != pair_logits.shape[0]:
                # This would be a major internal error
                return torch.tensor(-1e9) 

            masked_logits = torch.full_like(pair_logits, float('-inf'))
            valid_pair_ids_set = {tuple(sorted((v1.id, v2.id))) for v1, v2 in valid_pairs}
            
            pair_id_to_logit_idx = {} # Maps (v_id1, v_id2) -> logit_idx

            for logit_idx, (var_i_idx, var_j_idx) in enumerate(all_scored_pairs):
                var_i = all_prev_variables[var_i_idx]
                var_j = all_prev_variables[var_j_idx]
                pair_ids = tuple(sorted((var_i.id, var_j.id)))

                if pair_ids in valid_pair_ids_set:
                    masked_logits[logit_idx] = pair_logits[logit_idx]
                    pair_id_to_logit_idx[pair_ids] = logit_idx
            
            # --- End Masking ---
            
            pair_log_probs = F.log_softmax(masked_logits, dim=-1) # Probabilities over *valid* actions

            # Get the ID of the action that was actually taken
            chosen_pair = action_detail # This is (var1, var2)
            chosen_pair_ids = tuple(sorted((chosen_pair[0].id, chosen_pair[1].id)))

            if chosen_pair_ids in pair_id_to_logit_idx:
                correct_logit_idx = pair_id_to_logit_idx[chosen_pair_ids]
                log_prob_detail = pair_log_probs[correct_logit_idx]
            else:
                # This action wasn't valid, or was masked.
                log_prob_detail = torch.tensor(-1e6)
            
            # --- END OF FIX ---

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
        Returns uniform logits (zeros) for the two backward action types.
        """
        if state_embedding.dim() == 1:
            return torch.zeros(2)  # [ADD_ATOM, UNIFY_VARIABLES]
        else:
            batch_size = state_embedding.size(0)
            return torch.zeros(batch_size, 2)

    def get_log_probability(self, next_state_embedding: torch.Tensor,
                            next_state) -> torch.Tensor:
        """
        Compute the log probability of transitioning backward from next_state (s')
        to any parent state (s).
        
        P_B(s'->s) = 1 / N(s'), where N(s') is the number of possible parents of s'.

        Args:
            next_state_embedding: Embedding of the next state (s')
            next_state: The next state (Theory) - s'

        Returns:
            log_prob: Log probability P_B(s'->s)
        """
        from .logic_structures import get_all_variables
        import math

        # A correct uniform policy's probability depends on the state s' (next_state),
        # not the state s (previous_state). We must estimate the number of
        # valid parent states for s'.

        num_possible_parents = 0

        # Case 1: Parents from 'ADD_ATOM'
        # Any atom in the body could have been the last one added.
        if next_state and len(next_state) > 0:
            num_possible_parents += len(next_state[0].body)
        
        # Case 2: Parents from 'UNIFY_VARIABLES'
        # Any variable could be the result of a unification.
        # This is a very rough estimate, but better than the old logic.
        # A more complex (but correct) way would be to count "splittable" variables.
        # For simplicity, we'll estimate based on total variables.
        if next_state and len(next_state) > 0:
            variables = get_all_variables(next_state)
            num_vars = len(variables)
            # A state with N vars could have come from a state with N+1 vars.
            # The number of pairs in that N+1 state is (N+1)*N/2.
            # This is a heuristic estimate.
            num_unify_parents = (num_vars * (num_vars + 1) // 2) if num_vars >= 1 else 0
            num_possible_parents += num_unify_parents

        # If there are no atoms and no variables, it must have come from
        # the initial state via an ADD_ATOM action (of which there are num_predicates)
        if num_possible_parents == 0:
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

    def forward_strategist(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get strategist action logits."""
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
            variable_embeddings: Embeddings of variables in PREVIOUS state (optional, for sophisticated policy)

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
            # Simple uniform backward policy
            return self.backward_policy.get_log_probability(
                next_state_embedding,
                next_state  # Pass next_state
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