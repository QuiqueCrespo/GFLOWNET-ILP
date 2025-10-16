"""Test free variable detection logic."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Theory, Atom, Variable, Rule, is_terminal, theory_to_string

print("="*80)
print("FREE VARIABLE DETECTION TEST")
print("="*80)

# Test cases with different variable patterns
test_cases = [
    {
        "name": "No free variables (chain)",
        "rule": Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(0), Variable(2))),
                Atom('parent', (Variable(2), Variable(1)))
            ]
        ),
        "expected_free": set()
    },
    {
        "name": "One free variable X1",
        "rule": Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(0), Variable(2)))
            ]
        ),
        "expected_free": {Variable(1)}
    },
    {
        "name": "Both variables free",
        "rule": Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(2), Variable(3)))
            ]
        ),
        "expected_free": {Variable(0), Variable(1)}
    },
    {
        "name": "X0 appears in body, X1 free",
        "rule": Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(0), Variable(0))),
                Atom('parent', (Variable(2), Variable(3)))
            ]
        ),
        "expected_free": {Variable(1)}
    },
    {
        "name": "Self-loop, X0 free (X1 in body)",
        "rule": Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(1), Variable(1)))
            ]
        ),
        "expected_free": {Variable(0)}
    },
    {
        "name": "Example from user: X6,X1 head, only X1,X5 in body",
        "rule": Rule(
            head=Atom('grandparent', (Variable(6), Variable(1))),
            body=[
                Atom('parent', (Variable(1), Variable(5))),
                Atom('parent', (Variable(5), Variable(5))),
                Atom('parent', (Variable(6), Variable(6)))
            ]
        ),
        "expected_free": set()  # Both X6 and X1 appear in body
    }
]

all_passed = True

for i, test in enumerate(test_cases, 1):
    theory = [test["rule"]]
    rule = test["rule"]

    # Calculate free variables
    head_vars = set(rule.head.args)
    body_vars = set()
    for atom in rule.body:
        body_vars.update(atom.args)
    free_vars = head_vars - body_vars

    expected = test["expected_free"]
    passed = free_vars == expected

    print(f"\nTest {i}: {test['name']}")
    print(f"  Rule: {theory_to_string(theory)}")
    print(f"  Head vars: {{{', '.join(f'X{v.id}' for v in sorted(head_vars, key=lambda x: x.id))}}}")
    print(f"  Body vars: {{{', '.join(f'X{v.id}' for v in sorted(body_vars, key=lambda x: x.id))}}}")
    print(f"  Free vars (detected): {{{', '.join(f'X{v.id}' for v in sorted(free_vars, key=lambda x: x.id))}}}")
    print(f"  Free vars (expected): {{{', '.join(f'X{v.id}' for v in sorted(expected, key=lambda x: x.id))}}}")
    print(f"  is_terminal(): {is_terminal(theory)}")
    print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")

    if not passed:
        all_passed = False
        print(f"  ERROR: Expected {expected} but got {free_vars}")

print("\n" + "="*80)
print("OVERALL RESULT")
print("="*80)

if all_passed:
    print("\n✓ ALL TESTS PASSED")
else:
    print("\n✗ SOME TESTS FAILED")
    print("\nPossible issues:")
    print("1. Variable comparison not working correctly (Variable equality)")
    print("2. Set operations not working as expected")
    print("3. Body variable collection missing some atoms")

print("\n" + "="*80)
print("DEBUGGING: Variable Equality")
print("="*80)

# Test Variable equality
v0a = Variable(0)
v0b = Variable(0)
v1 = Variable(1)

print(f"\nVariable(0) == Variable(0): {v0a == v0b}")
print(f"Variable(0) is Variable(0): {v0a is v0b}")
print(f"Variable(0) == Variable(1): {v0a == v1}")
print(f"hash(Variable(0)) == hash(Variable(0)): {hash(v0a) == hash(v0b)}")

# Test set operations
set_a = {Variable(0), Variable(1)}
set_b = {Variable(0), Variable(2)}
print(f"\n{{X0, X1}} - {{X0, X2}} = {{{', '.join(f'X{v.id}' for v in (set_a - set_b))}}}")
