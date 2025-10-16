"""Diagnose logic engine - check if SLD resolution is working correctly."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Theory, Atom, Variable, Rule, theory_to_string
from src.logic_engine import LogicEngine, Example

print("="*80)
print("LOGIC ENGINE DIAGNOSIS")
print("="*80)

# Simple test case
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie'))
]

# Test disconnected rule
disconnected_rule = [Rule(
    head=Atom('grandparent', (Variable(0), Variable(1))),
    body=[
        Atom('parent', (Variable(2), Variable(3))),
        Atom('parent', (Variable(4), Variable(5)))
    ]
)]

# Test perfect rule
perfect_rule = [Rule(
    head=Atom('grandparent', (Variable(0), Variable(1))),
    body=[
        Atom('parent', (Variable(0), Variable(2))),
        Atom('parent', (Variable(2), Variable(1)))
    ]
)]

# Test examples
positive_example = Example('grandparent', ('alice', 'charlie'))
negative_example = Example('grandparent', ('alice', 'alice'))

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)

print("\n" + "="*80)
print("TEST 1: Disconnected Rule")
print("="*80)
print(f"Rule: {theory_to_string(disconnected_rule)}")
print(f"\nThis rule has X0,X1 in head but X2,X3,X4,X5 in body (disconnected)")
print(f"It should NOT be able to prove any grandparent examples.\n")

# Test positive example
result_pos = logic_engine.entails(disconnected_rule, positive_example)
print(f"Can prove {positive_example.predicate_name}('alice', 'charlie')? {result_pos}")

if result_pos:
    print("  ⚠️  BUG: Disconnected rule should NOT prove this example!")
else:
    print("  ✓ Correct: Disconnected rule cannot prove this example")

# Test negative example
result_neg = logic_engine.entails(disconnected_rule, negative_example)
print(f"Can prove {negative_example.predicate_name}('alice', 'alice')? {result_neg}")

if result_neg:
    print("  ⚠️  BUG: Disconnected rule should NOT prove this example!")
else:
    print("  ✓ Correct: Disconnected rule cannot prove this example")

print("\n" + "="*80)
print("TEST 2: Perfect Rule")
print("="*80)
print(f"Rule: {theory_to_string(perfect_rule)}")
print(f"\nThis rule connects head and body variables correctly.")
print(f"It should prove positive but not negative examples.\n")

# Test positive example
result_pos = logic_engine.entails(perfect_rule, positive_example)
print(f"Can prove {positive_example.predicate_name}('alice', 'charlie')? {result_pos}")

if result_pos:
    print("  ✓ Correct: Perfect rule proves positive example")
else:
    print("  ⚠️  BUG: Perfect rule should prove positive example!")

# Test negative example
result_neg = logic_engine.entails(perfect_rule, negative_example)
print(f"Can prove {negative_example.predicate_name}('alice', 'alice')? {result_neg}")

if result_neg:
    print("  ⚠️  Issue: Rule proves negative example (might be ok if 'alice' is parent of 'alice' via transitivity)")
else:
    print("  ✓ Correct: Perfect rule does not prove negative example")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("""
If the disconnected rule can prove examples, this indicates a bug in SLD resolution:
- Variables in head should be bound during proving
- If X0,X1 in head are not bound by body atoms, the query should fail
- The current implementation might be succeeding with unbound variables

This would explain why rules with 0 actual coverage get reported as having 100% coverage.

Possible causes:
1. SLD resolution doesn't check if all head variables are bound
2. Unification is too permissive
3. The proving mechanism doesn't track variable bindings correctly
""")
