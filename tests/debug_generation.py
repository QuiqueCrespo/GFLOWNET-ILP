"""Debug why generation stops with free variables."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import get_initial_state, is_terminal, theory_to_string

initial_state = get_initial_state('grandparent', 2)

print("="*80)
print("DEBUG: Initial State")
print("="*80)
rule_str = theory_to_string(initial_state)
print(f"State: {rule_str}")
print(f"Is terminal: {is_terminal(initial_state)}")

rule = initial_state[0]
print(f"Body length: {len(rule.body)}")
print(f"Head vars: {', '.join(f'X{v.id}' for v in rule.head.args)}")

head_vars = set(rule.head.args)
body_vars = set()
for atom in rule.body:
    body_vars.update(atom.args)
free_vars = head_vars - body_vars

print(f"Body vars: {', '.join(f'X{v.id}' for v in body_vars) if body_vars else 'none'}")
print(f"Free vars: {', '.join(f'X{v.id}' for v in free_vars)}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if free_vars and len(rule.body) == 0:
    print("Initial state has empty body and free variables")
    print("Expected: is_terminal = False")
    print(f"Actual: is_terminal = {is_terminal(initial_state)}")

    if not is_terminal(initial_state):
        print("✓ Correct! Loop should continue...")
    else:
        print("✗ BUG: Loop will exit immediately!")
