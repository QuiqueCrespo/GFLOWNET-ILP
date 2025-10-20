"""
Reward function for evaluating generated theories.
"""

from typing import List
from .logic_structures import Theory
from .logic_engine import LogicEngine, Example


class RewardCalculator:
    """Calculates reward for a theory based on coverage and simplicity."""

    def __init__(self, logic_engine: LogicEngine,
                 weight_precision: float = 0.5,
                 weight_recall: float = 0.5,
                 weight_simplicity: float = 0.01,
                 disconnected_var_penalty: float = 0.2,
                 self_loop_penalty: float = 0.3,
                 free_var_penalty: float = 1.0,
                 use_f1: bool = False):
        """
        Initialize reward calculator using confusion matrix metrics.

        Confusion Matrix:
            Theory entails example:     YES         NO
            Positive example:           TP          FN
            Negative example:           FP          TN

        Metrics:
            - Precision = TP / (TP + FP)  [fraction of entailed examples that are positive]
            - Recall = TP / (TP + FN)     [fraction of positive examples that are entailed]
            - F1-score = 2 * (Precision * Recall) / (Precision + Recall)
            - Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Reward Formula (default):
            R = w_precision * precision + w_recall * recall + w_simplicity * simplicity - penalties

        Reward Formula (if use_f1=True):
            R = F1-score + w_simplicity * simplicity - penalties

        Args:
            logic_engine: Logic engine for evaluating entailment
            weight_precision: Weight for precision (default: 0.5)
                             Precision penalizes false positives (covering negatives)
            weight_recall: Weight for recall (default: 0.5)
                          Recall penalizes false negatives (missing positives)
            weight_simplicity: Weight for simplicity bonus (default: 0.01)
            disconnected_var_penalty: Penalty per disconnected variable (default: 0.2)
            self_loop_penalty: Penalty per self-loop (default: 0.3)
            free_var_penalty: Penalty per free variable in head (default: 1.0)
            use_f1: If True, use F1-score instead of weighted precision+recall (default: False)

        Theoretical Notes:
            - Confusion matrix provides complete picture of classification performance
            - Precision and recall are standard metrics in machine learning
            - F1-score is harmonic mean, giving balanced importance to both
            - Minimum reward of 1e-6 ensures GFlowNet numerical stability
        """
        self.logic_engine = logic_engine
        self.weight_precision = weight_precision
        self.weight_recall = weight_recall
        self.weight_simplicity = weight_simplicity
        self.disconnected_var_penalty = disconnected_var_penalty
        self.self_loop_penalty = self_loop_penalty
        self.free_var_penalty = free_var_penalty
        self.use_f1 = use_f1

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
        Calculate reward using confusion matrix metrics.

        Confusion Matrix:
            - TP (True Positives): Positive examples that theory entails
            - FN (False Negatives): Positive examples that theory does NOT entail
            - FP (False Positives): Negative examples that theory entails (BAD!)
            - TN (True Negatives): Negative examples that theory does NOT entail

        Metrics:
            - Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            - Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            - F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0

        Returns:
            Reward value (guaranteed to be > 0 for numerical stability)
        """
        # Compute confusion matrix
        TP = sum(1 for ex in positive_examples if self.logic_engine.entails(theory, ex))
        FN = len(positive_examples) - TP
        FP = sum(1 for ex in negative_examples if self.logic_engine.entails(theory, ex))
        TN = len(negative_examples) - FP

        # Compute precision and recall
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            # Theory entails nothing - precision is undefined, use 0
            precision = 0.0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            # No positive examples - recall is undefined, use 0
            recall = 0.0

        # Compute F1-score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Simplicity (Occam's razor) - prefer simpler theories
        total_atoms = sum(len(rule.body) for rule in theory)
        simplicity = 1.0 / (1.0 + total_atoms)

        # Structural penalties
        num_disconnected = self._count_disconnected_variables(theory)
        num_self_loops = self._count_self_loops(theory)
        num_free_vars = self._count_free_variables(theory)

        disconnected_penalty_value = self.disconnected_var_penalty * num_disconnected
        self_loop_penalty_value = self.self_loop_penalty * num_self_loops
        free_var_penalty_value = self.free_var_penalty * num_free_vars

        # Special case: uninformative rules (cover everything - all positives AND all negatives)
        # These don't discriminate: TP=all positives, FP=all negatives, FN=0, TN=0
        if TP > 0 and FN == 0 and TN == 0 and FP == len(negative_examples) and FP > 0:
            uninformative_penalty = 0.9
        else:
            uninformative_penalty = 0.0

        # Calculate reward based on mode
        if self.use_f1:
            # Use F1-score as primary metric (harmonic mean)
            reward = (f1_score +
                     self.weight_simplicity * simplicity -
                     uninformative_penalty -
                     disconnected_penalty_value -
                     self_loop_penalty_value -
                     free_var_penalty_value)
        else:
            # Use weighted precision + recall (more flexible)
            reward = (self.weight_precision * precision +
                     self.weight_recall * recall +
                     self.weight_simplicity * simplicity -
                     uninformative_penalty -
                     disconnected_penalty_value -
                     self_loop_penalty_value -
                     free_var_penalty_value)

        # Ensure minimum reward for numerical stability (avoid log(0) in GFlowNet)
        return max(reward, 1e-6)

    def get_detailed_scores(self, theory: Theory,
                           positive_examples: List[Example],
                           negative_examples: List[Example]) -> dict:
        """
        Get detailed breakdown of reward components with full confusion matrix.
        Uses the SAME formula as calculate_reward() for consistency.
        """
        # Compute confusion matrix
        TP = sum(1 for ex in positive_examples if self.logic_engine.entails(theory, ex))
        FN = len(positive_examples) - TP
        FP = sum(1 for ex in negative_examples if self.logic_engine.entails(theory, ex))
        TN = len(negative_examples) - FP

        # Compute precision and recall
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0

        # Compute F1-score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Compute accuracy
        total = TP + TN + FP + FN
        if total > 0:
            accuracy = (TP + TN) / total
        else:
            accuracy = 0.0

        # Simplicity
        total_atoms = sum(len(rule.body) for rule in theory)
        simplicity = 1.0 / (1.0 + total_atoms)

        # Structural penalties
        num_disconnected = self._count_disconnected_variables(theory)
        num_self_loops = self._count_self_loops(theory)
        num_free_vars = self._count_free_variables(theory)

        disconnected_penalty_value = self.disconnected_var_penalty * num_disconnected
        self_loop_penalty_value = self.self_loop_penalty * num_self_loops
        free_var_penalty_value = self.free_var_penalty * num_free_vars

        # Special case: uninformative rules
        if TP > 0 and FN == 0 and TN == 0 and FP == len(negative_examples) and FP > 0:
            uninformative_penalty = 0.9
        else:
            uninformative_penalty = 0.0

        # Calculate reward (SAME formula as calculate_reward)
        if self.use_f1:
            reward = (f1_score +
                     self.weight_simplicity * simplicity -
                     uninformative_penalty -
                     disconnected_penalty_value -
                     self_loop_penalty_value -
                     free_var_penalty_value)
        else:
            reward = (self.weight_precision * precision +
                     self.weight_recall * recall +
                     self.weight_simplicity * simplicity -
                     uninformative_penalty -
                     disconnected_penalty_value -
                     self_loop_penalty_value -
                     free_var_penalty_value)

        return {
            'reward': max(reward, 1e-6),
            # Confusion matrix
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            # Derived metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            # Weighted components
            'precision_weighted': self.weight_precision * precision,
            'recall_weighted': self.weight_recall * recall,
            # Simplicity
            'total_atoms': total_atoms,
            'simplicity': simplicity,
            'simplicity_weighted': self.weight_simplicity * simplicity,
            # Penalties
            'uninformative_penalty': uninformative_penalty,
            'num_disconnected_vars': num_disconnected,
            'disconnected_penalty': disconnected_penalty_value,
            'num_self_loops': num_self_loops,
            'self_loop_penalty': self_loop_penalty_value,
            'num_free_vars': num_free_vars,
            'free_var_penalty': free_var_penalty_value
        }
