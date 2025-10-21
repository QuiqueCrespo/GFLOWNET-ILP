from typing import List, Set, Tuple, Dict, Optional, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We need Atom, Variable, and Theory, but also the Rule class
# I am assuming:
# class Rule:
#     def __init__(self, head: Atom, body: List[Atom]):
#         self.head = head
#         self.body = body
from .logic_structures import Theory, Atom, Variable, Rule


class Example:
    """
    Represents a ground fact (example) - a predicate with constant arguments.
    (This class is unchanged)
    """

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
    """
    Prolog-style logic engine using SLD resolution (top-down evaluation).

    IMPORTANT LIMITATIONS:
    - Uses top-down evaluation (like Prolog), not bottom-up (like Datalog)
    - Vulnerable to infinite loops with left-recursive rules
    - Use right-recursive rules for transitive closures

    Performance optimizations:
    - Theory is validated once and stored (not re-checked per query)
    - Facts are indexed by predicate for O(1) lookup
    """

    def __init__(self, max_depth: int = 10, background_facts: List[Example] = None,
                 theory: Theory = None, warn_left_recursion: bool = True):
        """
        Initialize logic engine.

        Args:
            max_depth: Maximum proof search depth (prevents infinite loops)
            background_facts: Known facts (ground atoms)
            theory: Optional theory to set immediately
            warn_left_recursion: If True, warn about potentially left-recursive rules
        """
        self.max_depth = max_depth
        self.background_facts = set(background_facts) if background_facts else set()
        self.warn_left_recursion = warn_left_recursion

        # Create index for fast fact lookup (O(1) instead of O(n))
        self._fact_index: Dict[str, Set[Example]] = {}
        for fact in self.background_facts:
            if fact.predicate_name not in self._fact_index:
                self._fact_index[fact.predicate_name] = set()
            self._fact_index[fact.predicate_name].add(fact)

        # Stored theory (validated once)
        self.theory: Theory = None
        self._theory_safe: bool = False

        if theory is not None:
            self.set_theory(theory)

    def set_theory(self, theory: Theory) -> None:
        """
        Set and validate the theory.

        This method:
        1. Validates rule safety (no free variables in head)
        2. Detects potentially left-recursive rules
        3. Caches the validated theory for efficient querying

        Args:
            theory: List of rules to validate and store

        Raises:
            ValueError: If any rule is unsafe (has free variables)
        """
        if not theory:
            self.theory = []
            self._theory_safe = True
            return

        # Check safety for all rules
        unsafe_rules = []
        for i, rule in enumerate(theory):
            if not self._is_safe_rule(rule):
                unsafe_rules.append((i, rule))

        if unsafe_rules:
            error_msg = "Unsafe rules detected (free variables in head):\n"
            for i, rule in unsafe_rules:
                error_msg += f"  Rule {i}: {rule}\n"
            raise ValueError(error_msg)

        # Check for left-recursion
        if self.warn_left_recursion:
            left_recursive = self._detect_left_recursion(theory)
            if left_recursive:
                import warnings
                warning_msg = "WARNING: Potentially left-recursive rules detected.\n"
                warning_msg += "Top-down evaluation may loop infinitely on these rules:\n"
                for pred, rule_idx in left_recursive:
                    warning_msg += f"  Predicate '{pred}' in rule {rule_idx}\n"
                warning_msg += "Consider rewriting to right-recursive form.\n"
                warnings.warn(warning_msg, UserWarning)

        # Store validated theory
        self.theory = theory
        self._theory_safe = True

    def _detect_left_recursion(self, theory: Theory) -> List[Tuple[str, int]]:
        """
        Detect potentially left-recursive rules.

        A rule is left-recursive if:
        - The head predicate appears as the first body atom
        - Example: path(X,Y) :- path(X,Z), edge(Z,Y)  [BAD]

        Returns:
            List of (predicate_name, rule_index) tuples for left-recursive rules
        """
        left_recursive = []

        for i, rule in enumerate(theory):
            if not rule.body:
                continue  # Empty body, not recursive

            head_predicate = rule.head.predicate_name
            first_body_predicate = rule.body[0].predicate_name

            if head_predicate == first_body_predicate:
                left_recursive.append((head_predicate, i))

        return left_recursive

    def entails(self, theory: Theory = None, example: Example = None) -> bool:
        """
        Check if the theory entails the given example (ground fact).

        Supports two calling patterns:
        1. entails(theory, example) - old API, validates theory each time (backward compatible)
        2. entails(example=example) - new API, uses stored theory (efficient)

        Args:
            theory: Theory to use (optional if theory set via set_theory())
            example: Ground fact to prove

        Returns:
            True if theory ⊢ example, False otherwise
        """
        # Handle backward compatibility: entails(theory, example)
        if theory is not None and example is None:
            # Old API: first arg might be example
            if isinstance(theory, Example):
                example = theory
                theory = self.theory
            # Otherwise theory is actually a theory, example is second arg (already None)

        # Determine which theory to use
        if theory is None:
            if self.theory is None:
                raise ValueError("No theory provided. Use set_theory() first or pass theory parameter.")
            theory = self.theory
            use_stored = True
        else:
            use_stored = False

        # Quick check: already in background facts?
        if example in self.background_facts:
            return True

        if not theory:
            return False

        # Only validate if using ad-hoc theory (not stored)
        if not use_stored:
            for rule in theory:
                if not self._is_safe_rule(rule):
                    # logging.warning(f"Warning: Unsafe rule {rule}")
                    return False

        # Convert the ground example into the initial goal
        initial_goal = Atom(example.predicate_name, example.args)

        try:
            # _solve is a generator. We just need to know if it
            # finds at least *one* solution (proof).
            first_solution = next(self._solve(theory, [initial_goal], {}, 0), None)
            return first_solution is not None
        except RecursionError:
            return False
            
    def _is_safe_rule(self, rule: Rule) -> bool:
        """
        Check if rule is safe (all head variables appear in body).

        Special case: Empty body rules are always safe (they are facts/axioms).
        For example: human(X) :- (means "X is human for all X")
        """

        head_vars = {arg for arg in rule.head.args if isinstance(arg, Variable)}
        body_vars = set()
        for atom in rule.body:
            body_vars.update(arg for arg in atom.args if isinstance(arg, Variable))

        return head_vars.issubset(body_vars)

    # ------------------------------------------------------------------
    # NEW AND REWRITTEN HELPER METHODS
    # ------------------------------------------------------------------

    def _unify(self, term1, term2, subst: Dict) -> Optional[Dict]:
        """
        Unify two terms (constants or Variables) under a substitution.
        This is a core helper for unification.
        """
        
        # 'walk' finds the ultimate binding of a variable
        def walk(item, s):
            if isinstance(item, Variable) and item in s:
                return walk(s[item], s) # Follow the chain
            return item

        term1 = walk(term1, subst)
        term2 = walk(term2, subst)

        if term1 == term2:
            return subst  # Already unified
        if isinstance(term1, Variable):
            # No occurs check needed for Datalog
            return {**subst, term1: term2}
        if isinstance(term2, Variable):
            # No occurs check needed for Datalog
            return {**subst, term2: term1}
        
        return None  # Mismatch, e.g., 'alice' vs 'bob'

    def _unify_atoms(self, atom1: Atom, atom2: Atom, 
                     subst: Dict) -> Optional[Dict]:
        """
        Unify two atoms, returning the new substitution or None.
        """
        # Predicates and arity must match
        if atom1.predicate_name != atom2.predicate_name:
            return None
        if len(atom1.args) != len(atom2.args):
            return None

        current_subst = subst.copy()
        for arg1, arg2 in zip(atom1.args, atom2.args):
            new_subst = self._unify(arg1, arg2, current_subst)
            if new_subst is None:
                return None  # Unification failed
            current_subst = new_subst
            
        return current_subst

    def _standardize_apart(self, rule: Rule, depth: int) -> Rule:
        """
        Rename all variables in a rule to be unique for this depth.
        e.g., Variable(0) at depth 1 -> Variable(1000)
        This is CRITICAL for recursion to avoid variable conflicts.
        """
        var_map = {}

        def rename(item):
            if isinstance(item, Variable):
                if item not in var_map:
                    # Create a new Variable with a unique ID
                    # Use depth * 1000 + original_id to ensure uniqueness
                    new_id = depth * 1000 + item.id
                    var_map[item] = Variable(new_id)
                return var_map[item]
            return item # It's a constant

        new_head = Atom(rule.head.predicate_name,
                        tuple(rename(arg) for arg in rule.head.args))
        new_body = [Atom(atom.predicate_name,
                         tuple(rename(arg) for arg in atom.args))
                    for atom in rule.body]

        # Assuming Rule has a simple constructor: Rule(head, body)
        return Rule(new_head, new_body)

    def _substitute_in_atom(self, atom: Atom, subst: Dict) -> Atom:
        """
        Apply a substitution to an atom, returning a new atom.
        """
        def walk(item, s):
            if isinstance(item, Variable) and item in s:
                return walk(s[item], s) # Follow the chain
            return item

        new_args = []
        for arg in atom.args:
            new_args.append(walk(arg, subst))
        
        return Atom(atom.predicate_name, tuple(new_args))

    # ------------------------------------------------------------------
    # THE NEW CORE LOGIC
    # ------------------------------------------------------------------

    def _solve(self, theory: Theory, goals: List[Atom], 
               substitution: Dict, depth: int) -> Iterator[Dict]:
        """
        The core SLD resolution engine.
        This is a generator that yields all successful substitutions (proofs).
        """
        
        # 1. Depth limit check
        if depth > self.max_depth:
            return

        # 2. Base Case: No goals left means this branch is a success!
        if not goals:
            yield substitution
            return

        # 3. Recursive Step: Try to solve the first goal
        
        # Apply current substitution to the first goal
        first_goal = self._substitute_in_atom(goals[0], substitution)
        remaining_goals = goals[1:]

        # A. Try to match against background facts
        if first_goal.predicate_name in self._fact_index:
            for fact in self._fact_index[first_goal.predicate_name]:
                # Convert fact (Example) to an Atom for unification
                fact_atom = Atom(fact.predicate_name, fact.args)
                
                new_subst = self._unify_atoms(first_goal, fact_atom, substitution)
                
                if new_subst is not None:
                    # Success! This fact matched.
                    # Now, try to solve the *rest* of the goals
                    # with the new substitution.
                    yield from self._solve(theory, remaining_goals, new_subst, depth)

        # B. Try to match against rule heads (Rule Chaining)
        for rule in theory:
            # CRITICAL: Rename variables in the rule
            fresh_rule = self._standardize_apart(rule, depth)
            
            # Try to unify the goal with the rule's head
            new_subst = self._unify_atoms(first_goal, fresh_rule.head, substitution)
            
            if new_subst is not None:
                # Success! The rule head matched.
                # The new goals are the rule's body + the remaining goals.
                new_goals = fresh_rule.body + remaining_goals
                
                # Recursively solve the new, expanded goal list
                yield from self._solve(theory, new_goals, new_subst, depth + 1)

    # ------------------------------------------------------------------
    # (Original get_coverage method is fine and requires no changes)
    # ------------------------------------------------------------------
    
    def get_coverage(self, theory: Theory, examples: List[Example]) -> float:
        """
        Get the fraction of examples covered by the theory.
        (This method is unchanged)
        """
        if not examples:
            return 0.0

        covered = sum(1 for ex in examples if self.entails(theory, ex))
        return covered / len(examples)