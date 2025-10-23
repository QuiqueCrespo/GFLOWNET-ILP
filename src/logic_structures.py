"""
Core data structures for First-Order Logic (FOL) representation.
"""

from typing import List, Tuple
from collections import namedtuple

# Use integers for unique variable IDs
Variable = namedtuple('Variable', ['id'])

# Predicate name and its arguments (variables)
Atom = namedtuple('Atom', ['predicate_name', 'args'])

# A rule has a head atom and a body (list of atoms)
Rule = namedtuple('Rule', ['head', 'body'])

# The state is the theory, which is a list of rules
Theory = List[Rule]


def get_all_variables(theory: Theory) -> List[Variable]:
    """Extract all unique variables from a theory."""
    variables = set()
    for rule in theory:
        # Variables from head
        for arg in rule.head.args:
            if isinstance(arg, Variable):
                variables.add(arg)
        # Variables from body
        for atom in rule.body:
            for arg in atom.args:
                if isinstance(arg, Variable):
                    variables.add(arg)
    return list(variables)


def get_all_predicates(theory: Theory) -> List[str]:
    """Extract all unique predicate names from a theory."""
    predicates = set()
    for rule in theory:
        predicates.add(rule.head.predicate_name)
        for atom in rule.body:
            predicates.add(atom.predicate_name)
    return list(predicates)


def theory_to_string(theory: Theory) -> str:
    """Convert theory to human-readable string."""
    lines = []
    for rule in theory:
        head_str = f"{rule.head.predicate_name}({', '.join(f'X{v.id}' for v in rule.head.args)})"
        if rule.body:
            body_str = ', '.join(
                f"{atom.predicate_name}({', '.join(f'X{v.id}' for v in atom.args)})"
                for atom in rule.body
            )
            lines.append(f"{head_str} :- {body_str}.")
        else:
            lines.append(f"{head_str}.")
    return '\n'.join(lines)


def get_initial_state(target_predicate: str, arity: int) -> Theory:
    """
    Create initial state with a rule that has the target predicate in the head
    and an empty body (or a 'true' placeholder).
    """
    head_vars = [Variable(id=i) for i in range(arity)]
    head = Atom(predicate_name=target_predicate, args=tuple(head_vars))
    return [Rule(head=head, body=[])]


def is_valid_complete_state(theory: Theory) -> bool:
    """
    Check if the theory is valid and complete (potentially terminal).

    A state is valid and complete if:
    - All head variables appear in the body (no free variables)
    - Has at least one body atom

    This means the policy CAN choose to terminate at this state,
    but is not forced to - it can continue adding more atoms.
    """
    if not theory:
        return False

    rule = theory[0]

    # Must have at least one body atom
    if len(rule.body) == 0:
        return False

    # Check for free variables in head
    head_vars = set(arg for arg in rule.head.args if isinstance(arg, Variable))

    body_vars = set()
    for atom in rule.body:
        for arg in atom.args:
            if isinstance(arg, Variable):
                body_vars.add(arg)

    free_vars = head_vars - body_vars

    # Valid and complete if no free variables
    return len(free_vars) == 0


def is_terminal(theory: Theory) -> bool:
    """
    Check if the theory is in a terminal state (trajectory must stop).

    A theory is FORCED to be terminal if:
    1. It has reached maximum body length (checked in training loop), OR
    2. The policy explicitly chose TERMINATE action

    NOTE: This function is used in the main training loop to check
    if we should stop trajectory generation. The actual termination
    is controlled by:
    - Max body length constraint (action masking)
    - Policy choosing TERMINATE action

    For compatibility, returns True if state is valid and complete,
    but the policy should use is_valid_complete_state() to decide
    if TERMINATE is allowed.
    """
    # For backward compatibility and loop termination
    # This should rarely be called now since we rely on policy TERMINATE
    return is_valid_complete_state(theory)


def apply_add_atom(theory: Theory, predicate_name: str, arity: int,
                   max_var_id: int) -> Tuple[Theory, int]:
    """
    Add an atom with the given predicate to the body of the first rule.
    Creates new variables for the atom's arguments.

    Returns: (new_theory, updated_max_var_id)
    """
    if not theory:
        return theory, max_var_id

    rule = theory[0]

    # Create new variables for the atom
    new_vars = [Variable(id=max_var_id + i + 1) for i in range(arity)]
    new_atom = Atom(predicate_name=predicate_name, args=tuple(new_vars))

    # Add atom to body
    new_body = list(rule.body) + [new_atom]
    new_rule = Rule(head=rule.head, body=new_body)

    return [new_rule], max_var_id + arity


def apply_unify_vars(theory: Theory, var1: Variable, var2: Variable) -> Theory:
    """
    Unify two variables by replacing all occurrences of var2 with var1.
    """
    if not theory:
        return theory

    def replace_var(args):
        return tuple(var1 if arg == var2 else arg for arg in args)

    new_rules = []
    for rule in theory:
        # Replace in head
        new_head = Atom(
            predicate_name=rule.head.predicate_name,
            args=replace_var(rule.head.args)
        )

        # Replace in body
        new_body = [
            Atom(predicate_name=atom.predicate_name, args=replace_var(atom.args))
            for atom in rule.body
        ]

        new_rules.append(Rule(head=new_head, body=new_body))

    return new_rules


def get_valid_variable_pairs(theory: Theory) -> List[Tuple[Variable, Variable]]:
    """
    Get all valid pairs of variables that can be unified.
    A valid pair consists of two different variables.
    Excludes pairs where both variables are in the head (would create self-loops).
    """
    if not theory:
        return []

    rule = theory[0]
    head_vars = set(arg for arg in rule.head.args if isinstance(arg, Variable))

    variables = get_all_variables(theory)
    pairs = []

    for i, v1 in enumerate(variables):
        for v2 in variables[i+1:]:
            if v1 != v2:
                # Exclude pairs where BOTH are in head (would create self-loops like grandparent(X0, X0))
                if v1 in head_vars and v2 in head_vars:
                    continue  # Skip this pair
                pairs.append((v1, v2))

    return pairs
