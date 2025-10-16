"""
Simple logic engine for evaluating theories against examples.
This is a basic forward-chaining evaluator for Datalog-style rules.
"""

from typing import List, Set, Tuple, Dict
from .logic_structures import Theory, Atom, Variable


class Example:
    """Represents a ground fact (example)."""

    def __init__(self, predicate_name: str, args: Tuple[str, ...]):
        self.predicate_name = predicate_name
        self.args = args

    def __eq__(self, other):
        return (isinstance(other, Example) and
                self.predicate_name == other.predicate_name and
                self.args == other.args)

    def __hash__(self):
        return hash((self.predicate_name, self.args))

    def __repr__(self):
        args_str = ', '.join(self.args)
        return f"{self.predicate_name}({args_str})"


class LogicEngine:
    """Simple forward-chaining logic engine."""

    def __init__(self, max_depth: int = 10, background_facts: List[Example] = None):
        """
        Args:
            max_depth: Maximum recursion depth for proof search
            background_facts: Known facts (ground atoms) that are always true
        """
        self.max_depth = max_depth
        self.background_facts = set(background_facts) if background_facts else set()

    def entails(self, theory: Theory, example: Example) -> bool:
        """
        Check if the theory entails the given example (ground fact).
        Uses SLD resolution (backward chaining).
        """
        # Check if already in background facts
        if example in self.background_facts:
            return True

        if not theory:
            return False

        # Try to prove the goal using SLD resolution
        return self._prove_goal(theory, example, depth=0)

    def _prove_goal(self, theory: Theory, goal: Example, depth: int) -> bool:
        """
        Prove a ground goal using SLD resolution.

        Args:
            theory: List of rules
            goal: The ground fact to prove
            depth: Current recursion depth
        """
        if depth > self.max_depth:
            return False

        # Check if goal is in background facts
        if goal in self.background_facts:
            return True

        # Try each rule in the theory
        for rule in theory:
            # Try to unify the goal with the rule head
            substitution = self._unify_atom_with_example(rule.head, goal)

            if substitution is not None:
                # If body is empty, the goal is proven
                if not rule.body:
                    return True

                # Get all variables that appear in the head
                head_vars = set()
                for arg in rule.head.args:
                    is_variable = (hasattr(arg, '__class__') and
                                  arg.__class__.__name__ == 'Variable')
                    if is_variable:
                        head_vars.add(arg)

                # Get all variables that appear in the body
                body_vars = set()
                for atom in rule.body:
                    for arg in atom.args:
                        is_variable = (hasattr(arg, '__class__') and
                                      arg.__class__.__name__ == 'Variable')
                        if is_variable:
                            body_vars.add(arg)

                # Check for disconnected head variables
                # Head variables must appear in the body (safety condition)
                if not head_vars.issubset(body_vars):
                    # Disconnected variables - rule is unsafe
                    continue

                # Try to prove the body with the substitution
                if self._prove_body(theory, rule.body, substitution, depth + 1):
                    return True

        return False

    def _unify_atom_with_example(self, atom: Atom, example: Example):
        """
        Try to unify an atom (with variables) with a ground example.
        Returns substitution dict if successful, None otherwise.

        Note: This must work with any Variable class that supports equality and hashing.
        """
        # Check if predicates match
        if atom.predicate_name != example.predicate_name:
            return None

        # Check if arities match
        if len(atom.args) != len(example.args):
            return None

        substitution = {}
        for atom_arg, example_arg in zip(atom.args, example.args):
            # Check if atom_arg is a Variable by checking if it has attributes
            # that distinguish it from a string constant
            is_variable = (hasattr(atom_arg, '__class__') and
                          atom_arg.__class__.__name__ == 'Variable')

            if is_variable:
                # Variable in atom - try to bind it
                if atom_arg in substitution:
                    # Variable already bound, check consistency
                    if substitution[atom_arg] != example_arg:
                        return None
                else:
                    # Bind variable to the constant from the example
                    substitution[atom_arg] = example_arg
            else:
                # Constant in atom - must match exactly
                if atom_arg != example_arg:
                    return None

        return substitution

    def _prove_body(self, theory: Theory, body: List[Atom],
                   substitution: Dict[Variable, str], depth: int) -> bool:
        """
        Prove all atoms in the body using SLD resolution.

        This implements proper backward chaining:
        - If an atom is fully ground, prove it recursively
        - If an atom has unbound variables, find all facts that match and try each binding
        """
        if not body:
            # Empty body means success
            return True

        # Take the first atom
        first_atom = body[0]
        remaining_body = body[1:]

        # Apply current substitution to the first atom
        # Check if all variables in the atom are bound
        ground_example = self._apply_substitution(first_atom, substitution)

        if ground_example is not None:
            # Atom is fully ground - prove it recursively
            if self._prove_goal(theory, ground_example, depth):
                # If proven, continue with the rest of the body
                return self._prove_body(theory, remaining_body, substitution, depth)
            else:
                return False
        else:
            # Atom has unbound variables - try all matching facts
            for fact in self.background_facts:
                if fact.predicate_name == first_atom.predicate_name:
                    # Try to unify with this fact
                    new_subst = self._try_unify_with_fact(first_atom, fact, substitution)

                    if new_subst is not None:
                        # Unification succeeded - try to prove the rest of the body
                        if self._prove_body(theory, remaining_body, new_subst, depth):
                            return True

            # No fact matched - proof fails
            return False

    def _try_unify_with_fact(self, atom: Atom, fact: Example, existing_subst: dict):
        """
        Try to unify an atom with a fact, extending existing substitution.
        Returns new substitution if successful, None otherwise.
        """
        if atom.predicate_name != fact.predicate_name:
            return None
        if len(atom.args) != len(fact.args):
            return None

        new_subst = existing_subst.copy()
        for atom_arg, fact_arg in zip(atom.args, fact.args):
            # Check if atom_arg is a Variable
            is_variable = (hasattr(atom_arg, '__class__') and
                          atom_arg.__class__.__name__ == 'Variable')

            if is_variable:
                if atom_arg in new_subst:
                    # Variable already bound, check consistency
                    if new_subst[atom_arg] != fact_arg:
                        return None
                else:
                    # Bind variable
                    new_subst[atom_arg] = fact_arg
            else:
                # Constant, must match
                if atom_arg != fact_arg:
                    return None

        return new_subst

    def _apply_substitution(self, atom: Atom, substitution: dict):
        """
        Apply a substitution to an atom to get a ground example.
        Returns None if there are unbound variables.
        """
        ground_args = []
        for arg in atom.args:
            # Check if arg is a Variable
            is_variable = (hasattr(arg, '__class__') and
                          arg.__class__.__name__ == 'Variable')

            if is_variable:
                if arg in substitution:
                    ground_args.append(substitution[arg])
                else:
                    # Unbound variable
                    return None
            else:
                ground_args.append(arg)

        return Example(atom.predicate_name, tuple(ground_args))

    def get_coverage(self, theory: Theory, examples: List[Example]) -> float:
        """
        Get the fraction of examples covered by the theory.
        """
        if not examples:
            return 0.0

        covered = sum(1 for ex in examples if self.entails(theory, ex))
        return covered / len(examples)
