import unittest
import sys
import os

# --- Key Change: Adjust Python path to find the 'src' folder ---
# This adds the parent directory (project_root) to the system path.
# This allows us to import from 'src' as if it were a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------------

# Now, import from the 'src' folder
from src.logic_engine import LogicEngine, Example
from typing import List, Set, Tuple, Dict, Union    

# ---------------------------------------------------------------------------
# Helper classes required by the LogicEngine, normally in 'logic_structures.py'
# ---------------------------------------------------------------------------

class Variable:
    """Represents a logic variable, e.g., 'X', 'Y'."""
    def __init__(self, name: str):
        self.name = name.upper()
    def __repr__(self) -> str:
        return self.name
    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name
    def __hash__(self) -> int:
        return hash(self.name)

# Define a type hint for arguments in an Atom
AtomArg = Union[Variable, str]

class Atom:
    """Represents a predicate with arguments, e.g., parent(X, 'b')."""
    def __init__(self, predicate_name: str, args: Tuple[AtomArg, ...]):
        self.predicate_name = predicate_name
        self.args = args
    def __repr__(self) -> str:
        args_str = ', '.join(map(str, self.args))
        return f"{self.predicate_name}({args_str})"

class Rule:
    """Represents a logic rule, e.g., Head :- Body."""
    def __init__(self, head: Atom, body: List[Atom]):
        self.head = head
        self.body = body
    def __repr__(self) -> str:
        if not self.body:
            return f"{self.head}."
        body_str = ', '.join(map(str, self.body))
        return f"{self.head} :- {body_str}."

# A Theory is just a list of Rules
Theory = List[Rule]


# ---------------------------------------------------------------------------
# Main Test Suite for LogicEngine
# ---------------------------------------------------------------------------

class TestLogicEngine(unittest.TestCase):
    """Test suite for the LogicEngine class."""

    def setUp(self):
        """Set up common variables, facts, and rules for tests."""
        # Define common logic variables for use in tests
        self.X = Variable("X")
        self.Y = Variable("Y")
        self.Z = Variable("Z")

        # Define a common set of background facts about a family tree
        self.family_facts = [
            Example("parent", ("a", "b")),  # a is parent of b
            Example("parent", ("b", "c")),  # b is parent of c
            Example("parent", ("a", "d")),  # a is parent of d
        ]

    ##
    # --- Basic Entailment Tests ---
    ##

    def test_entails_from_background_facts(self):
        """Tests if the engine can find facts directly in the background set."""
        engine = LogicEngine(background_facts=self.family_facts)
        # This fact is in the background knowledge
        self.assertTrue(engine.entails([], Example("parent", ("a", "b"))))
        # This fact is not
        self.assertFalse(engine.entails([], Example("parent", ("c", "a"))))

    def test_entails_with_simple_rule(self):
        """Tests a single, non-recursive rule with two atoms in the body."""
        # Theory: grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        grandparent_rule = Rule(
            head=Atom("grandparent", (self.X, self.Z)),
            body=[
                Atom("parent", (self.X, self.Y)),
                Atom("parent", (self.Y, self.Z))
            ]
        )
        theory = [grandparent_rule]
        engine = LogicEngine(background_facts=self.family_facts)

        # 'a' is a grandparent of 'c' via 'b'
        self.assertTrue(engine.entails(theory, Example("grandparent", ("a", "c"))))
        # 'a' is not a grandparent of 'b'
        self.assertFalse(engine.entails(theory, Example("grandparent", ("a", "b"))))

    def test_entails_with_axiom_rule(self):
        """Tests a rule with an empty body (a fact defined in the theory)."""
        # Theory: is_mortal('socrates').
        axiom_rule = Rule(head=Atom("is_mortal", ("socrates",)), body=[])
        theory = [axiom_rule]
        engine = LogicEngine()

        self.assertTrue(engine.entails(theory, Example("is_mortal", ("socrates",))))
        self.assertFalse(engine.entails(theory, Example("is_mortal", ("plato",))))

    ##
    # --- Recursive and Depth Tests ---
    ##

    def test_entails_with_recursive_rule(self):
        """Tests recursive rules for finding ancestors."""
        # Theory:
        # 1. ancestor(X, Y) :- parent(X, Y).
        # 2. ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        ancestor_base = Rule(
            head=Atom("ancestor", (self.X, self.Y)),
            body=[Atom("parent", (self.X, self.Y))]
        )
        ancestor_recursive = Rule(
            head=Atom("ancestor", (self.X, self.Z)),
            body=[
                Atom("parent", (self.X, self.Y)),
                Atom("ancestor", (self.Y, self.Z))
            ]
        )
        theory = [ancestor_base, ancestor_recursive]
        engine = LogicEngine(background_facts=self.family_facts)

        # Entailed by the base case rule
        self.assertTrue(engine.entails(theory, Example("ancestor", ("a", "b"))))
        # Entailed by the recursive rule
        self.assertTrue(engine.entails(theory, Example("ancestor", ("a", "c"))))
        # Not entailed
        self.assertFalse(engine.entails(theory, Example("ancestor", ("c", "a"))))

    def test_max_depth_limitation(self):
        """Tests that proofs fail if they exceed the max recursion depth."""
        ancestor_base = Rule(
            head=Atom("ancestor", (self.X, self.Y)),
            body=[Atom("parent", (self.X, self.Y))]
        )
        ancestor_recursive = Rule(
            head=Atom("ancestor", (self.X, self.Z)),
            body=[
                Atom("parent", (self.X, self.Y)),
                Atom("ancestor", (self.Y, self.Z))
            ]
        )
        theory = [ancestor_base, ancestor_recursive]

        # Proving ancestor('a', 'c') requires a depth of 2.
        # With max_depth=1, the recursive step to prove ancestor('b','c') will fail.
        engine_low_depth = LogicEngine(max_depth=1, background_facts=self.family_facts)
        self.assertFalse(engine_low_depth.entails(theory, Example("ancestor", ("a", "c"))))

        # With sufficient depth, it should succeed.
        engine_high_depth = LogicEngine(max_depth=2, background_facts=self.family_facts)
        self.assertTrue(engine_high_depth.entails(theory, Example("ancestor", ("a", "c"))))

    ##
    # --- Engine Limitations and Coverage ---
    ##

    def test_limitation_on_proving_intermediate_goals(self):
        """
        Tests a key limitation: the engine cannot prove an intermediate goal
        to bind a variable; it must find a binding in the existing background facts.
        """
        # Theory:
        #   p(X) :- q(X, Y).
        #   q(a, b) :- .
        # Goal: p(a)
        # To prove p(a), it needs to prove q(a, Y). 'Y' is an unbound variable.
        # The engine will look for q(a, _) in background_facts, but it won't
        # use the rule q(a, b) to derive it. This test verifies this behavior.
        theory = [
            Rule(head=Atom("p", (self.X,)), body=[Atom("q", (self.X, self.Y))]),
            Rule(head=Atom("q", ("a", "b")), body=[])
        ]
        engine = LogicEngine()

        # The proof for p('a') should fail because q('a', Y) cannot be found in facts.
        self.assertFalse(engine.entails(theory, Example("p", ("a",))))

    def test_get_coverage(self):
        """Tests the coverage calculation over a set of examples."""
        grandparent_rule = Rule(
            head=Atom("grandparent", (self.X, self.Z)),
            body=[
                Atom("parent", (self.X, self.Y)),
                Atom("parent", (self.Y, self.Z))
            ]
        )
        theory = [grandparent_rule]
        engine = LogicEngine(background_facts=self.family_facts)

        examples = [
            Example("grandparent", ("a", "c")),  # Provable by rule
            Example("parent", ("a", "b")),       # True from background facts
            Example("grandparent", ("b", "d")),  # Not provable
            Example("parent", ("c", "a")),       # Not in background facts
        ]

        # 2 out of 4 examples are covered (provable).
        coverage = engine.get_coverage(theory, examples)
        self.assertAlmostEqual(coverage, 0.5)

    def test_get_coverage_with_no_examples(self):
        """Tests that coverage is 0 if the example list is empty."""
        engine = LogicEngine()
        self.assertEqual(engine.get_coverage([], []), 0.0)


# This allows the test suite to be run from the command line
if __name__ == '__main__':
    unittest.main(verbosity=2)