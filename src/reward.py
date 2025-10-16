"""
Reward function for evaluating generated theories.
"""

from typing import List
from .logic_structures import Theory
from .logic_engine import LogicEngine, Example


class RewardCalculator:
    """Calculates reward for a theory based on coverage and simplicity."""

    def __init__(self, logic_engine: LogicEngine,
                 weight_pos: float = 0.8,
                 weight_neg: float = 0.8,
                 weight_simplicity: float = 0.01,
                 disconnected_var_penalty: float = 0.2,
                 self_loop_penalty: float = 0.3,
                 free_var_penalty: float = 1.0):
        """
        Args:
            logic_engine: Logic engine for evaluating entailment
            weight_pos: Weight for positive example coverage (default: 0.8, heavily penalize false negatives)
            weight_neg: Weight for negative example consistency - not covering them (default: 0.2, penalize false positives)
            weight_simplicity: Weight for simplicity (default: 0.01, minimal - fewer atoms is nice but not critical)
            disconnected_var_penalty: Penalty per disconnected variable (default: 0.2)
            self_loop_penalty: Penalty per self-loop (default: 0.3)
            free_var_penalty: Penalty per free variable in head (default: 1.0, CRITICAL - makes rule invalid)
        """
        self.logic_engine = logic_engine
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.weight_simplicity = weight_simplicity
        self.disconnected_var_penalty = disconnected_var_penalty
        self.self_loop_penalty = self_loop_penalty
        self.free_var_penalty = free_var_penalty

    def _count_disconnected_variables(self, theory: Theory) -> int:
        """
        Count truly disconnected variables - those that appear in body atoms
        that share NO variables with the connected component containing the head.

        Example:
        - grandparent(X0,X1) :- parent(X0,X2), parent(X2,X1)
          X2 is a chain variable but CONNECTED (shares vars with head atoms)
        - grandparent(X0,X5) :- parent(X0,X3), parent(X4,X5), parent(X6,X7)
          X6,X7 are DISCONNECTED (atom parent(X6,X7) shares no vars with others)
        """
        if not theory:
            return 0

        rule = theory[0]
        head_vars = set(rule.head.args)

        # Build connected component starting from head variables
        connected_vars = set(head_vars)
        changed = True

        while changed:
            changed = False
            for atom in rule.body:
                atom_vars = set(arg for arg in atom.args
                              if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable')
                # If atom shares any variable with connected component, add all its variables
                if atom_vars & connected_vars:
                    new_vars = atom_vars - connected_vars
                    if new_vars:
                        connected_vars.update(new_vars)
                        changed = True

        # Count variables that are not in the connected component
        all_body_vars = set()
        for atom in rule.body:
            for arg in atom.args:
                if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable':
                    all_body_vars.add(arg)

        disconnected = all_body_vars - connected_vars
        return len(disconnected)

    def _count_self_loops(self, theory: Theory) -> int:
        """Count atoms where the same variable appears multiple times."""
        if not theory:
            return 0

        count = 0
        for rule in theory:
            # Check head
            if len(set(rule.head.args)) < len(rule.head.args):
                count += 1

            # Check body atoms
            for atom in rule.body:
                if len(set(atom.args)) < len(atom.args):
                    count += 1

        return count

    def _count_free_variables(self, theory: Theory) -> int:
        """
        Count free variables - variables that appear in the head but NOT in the body.

        Free variables in the head make rules overly general and ungrounded.

        Example:
        - grandparent(X0, X1) :- parent(X2, X3)
          X0 and X1 are FREE (appear in head but not body) - INVALID!
        - grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)
          No free variables - all head vars appear in body - VALID!
        """
        if not theory:
            return 0

        rule = theory[0]
        head_vars = set(arg for arg in rule.head.args
                       if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable')

        body_vars = set()
        for atom in rule.body:
            for arg in atom.args:
                if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable':
                    body_vars.add(arg)

        free_vars = head_vars - body_vars
        return len(free_vars)

    def calculate_reward(self, theory: Theory,
                        positive_examples: List[Example],
                        negative_examples: List[Example]) -> float:
        """
        Calculate reward for a theory.

        Reward components:
        1. Positive coverage: fraction of positive examples entailed by theory
        2. Consistency: 1 - (fraction of negative examples entailed by theory)
        3. Simplicity: penalty for complexity (number of atoms in body)
        4. Structural penalties: disconnected variables, self-loops

        Returns:
            Reward value (guaranteed to be > 0 for numerical stability)
        """
        # 1. Positive example coverage
        if positive_examples:
            pos_covered = sum(1 for ex in positive_examples
                            if self.logic_engine.entails(theory, ex))
            score_pos = pos_covered / len(positive_examples)
        else:
            score_pos = 0.0

        # 2. Negative example consistency (should NOT cover negatives)
        if negative_examples:
            neg_covered = sum(1 for ex in negative_examples
                            if self.logic_engine.entails(theory, ex))
            score_neg = 1.0 - (neg_covered / len(negative_examples))
        else:
            score_neg = 1.0

        # 3. Simplicity (Occam's razor) - prefer simpler theories
        total_atoms = sum(len(rule.body) for rule in theory)
        simplicity = 1.0 / (1.0 + total_atoms)

        # Penalty for rules that cover EVERYTHING (both all positives AND all negatives)
        # These are uninformative - they don't discriminate
        # Rules with variable unification in the head (like target(X, X)) are fine
        if score_pos == 1.0 and score_neg == 0.0:
            # Covers all positives AND all negatives = uninformative
            uninformative_penalty = 0.9
        else:
            uninformative_penalty = 0.0

        # 4. Structural penalties
        num_disconnected = self._count_disconnected_variables(theory)
        num_self_loops = self._count_self_loops(theory)
        num_free_vars = self._count_free_variables(theory)

        disconnected_penalty_value = self.disconnected_var_penalty * num_disconnected
        self_loop_penalty_value = self.self_loop_penalty * num_self_loops
        free_var_penalty_value = self.free_var_penalty * num_free_vars

        if score_pos == 0.0:
            reward = 1e-6 # Return minimum reward
        else:
            accuracy = score_pos * score_neg

            reward = (0.9 * accuracy +
                0.1 * simplicity -
                uninformative_penalty -
                disconnected_penalty_value -
                self_loop_penalty_value -
                free_var_penalty_value)

        # Ensure minimum reward for numerical stability (avoid log(0))
        return max(reward, 1e-6)

    def get_detailed_scores(self, theory: Theory,
                           positive_examples: List[Example],
                           negative_examples: List[Example]) -> dict:
        """
        Get detailed breakdown of reward components.
        """
        # Positive coverage
        if positive_examples:
            pos_covered = sum(1 for ex in positive_examples
                            if self.logic_engine.entails(theory, ex))
            score_pos = pos_covered / len(positive_examples)
        else:
            pos_covered = 0
            score_pos = 0.0

        # Negative consistency
        if negative_examples:
            neg_covered = sum(1 for ex in negative_examples
                            if self.logic_engine.entails(theory, ex))
            score_neg = 1.0 - (neg_covered / len(negative_examples))
        else:
            neg_covered = 0
            score_neg = 1.0

        # Simplicity
        total_atoms = sum(len(rule.body) for rule in theory)
        simplicity = 1.0 / (1.0 + total_atoms)

        # Uninformative rule penalty
        if score_pos == 1.0 and score_neg == 0.0:
            uninformative_penalty = 0.9
        else:
            uninformative_penalty = 0.0

        # Structural penalties
        num_disconnected = self._count_disconnected_variables(theory)
        num_self_loops = self._count_self_loops(theory)
        num_free_vars = self._count_free_variables(theory)

        disconnected_penalty_value = self.disconnected_var_penalty * num_disconnected
        self_loop_penalty_value = self.self_loop_penalty * num_self_loops
        free_var_penalty_value = self.free_var_penalty * num_free_vars

        if score_pos == 0.0:
            reward = 1e-6 # Return minimum reward
            accuracy = 0.0
        else:
            accuracy = score_pos * score_neg
            reward = (0.9 * accuracy +
                0.1 * simplicity -
                uninformative_penalty -
                disconnected_penalty_value -
                self_loop_penalty_value -
                free_var_penalty_value)

        return {
            'reward': max(reward, 1e-6),
            'pos_covered': pos_covered,
            'pos_total': len(positive_examples),
            'pos_score': score_pos,
            'neg_covered': neg_covered,
            'neg_total': len(negative_examples),
            'neg_score': score_neg,
            'total_atoms': total_atoms,
            'simplicity': simplicity,
            'uninformative_penalty': uninformative_penalty,
            'accuracy': accuracy,
            'num_disconnected_vars': num_disconnected,
            'disconnected_penalty': disconnected_penalty_value,
            'num_self_loops': num_self_loops,
            'self_loop_penalty': self_loop_penalty_value,
            'num_free_vars': num_free_vars,
            'free_var_penalty': free_var_penalty_value
        }
