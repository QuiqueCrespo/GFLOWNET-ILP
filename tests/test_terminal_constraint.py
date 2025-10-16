"""Test that rules with free variables cannot be terminal states."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Theory, Atom, Variable, Rule, is_terminal

print("="*80)
print("TERMINAL STATE CONSTRAINT TEST")
print("="*80)

# Test cases
test_rules = {
    "Empty body (2 free vars)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[]
    )],

    "1 body atom, 1 free var": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[Atom('parent', (Variable(0), Variable(2)))]
    )],

    "2 body atoms, no free vars": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(1)))
        ]
    )],

    "1 body atom, no free vars (both in body)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[Atom('parent', (Variable(0), Variable(1)))]
    )],

    "3 body atoms, 1 free var (max length)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(3))),
            Atom('parent', (Variable(3), Variable(4)))
        ]
    )],

    "3 body atoms, no free vars": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(1))),
            Atom('parent', (Variable(1), Variable(3)))
        ]
    )]
}

print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

for name, theory in test_rules.items():
    rule = theory[0]

    # Calculate free variables
    head_vars = set(rule.head.args)
    body_vars = set()
    for atom in rule.body:
        body_vars.update(atom.args)
    free_vars = head_vars - body_vars

    # Check if terminal
    terminal = is_terminal(theory)

    # Format output
    print(f"\n{name}:")
    print(f"  Body length: {len(rule.body)}")
    print(f"  Free variables: {len(free_vars)} ({', '.join(f'X{v.id}' for v in free_vars) if free_vars else 'none'})")
    print(f"  Terminal: {terminal}")

    # Validate logic
    if len(rule.body) < 3:
        if free_vars:
            expected = False  # Should NOT be terminal (has free vars, not at max length)
            status = "✓" if terminal == expected else "✗ ERROR"
            print(f"  Expected: {expected} (free vars present) {status}")
        else:
            expected = True  # CAN be terminal (no free vars)
            status = "✓" if terminal == expected else "✗ ERROR"
            print(f"  Expected: {expected} (no free vars) {status}")
    else:
        # At max length, always terminal
        expected = True
        status = "✓" if terminal == expected else "✗ ERROR"
        print(f"  Expected: {expected} (max length reached) {status}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
Terminal state logic:

1. **Has free variables AND body length < 3:**
   → NOT terminal (must continue adding atoms to ground head variables)

2. **No free variables (all head vars in body):**
   → IS terminal (valid rule, can stop generation)

3. **Body length >= 3 (regardless of free vars):**
   → IS terminal (force stop to prevent infinite generation)

This ensures:
- Rules with free variables CANNOT terminate early
- Only valid, grounded rules can be terminal states (before max length)
- System prevents generation of semantically invalid rules

Expected behavior:
- Empty body → NOT terminal (free vars)
- 1 atom, 1 free var → NOT terminal (must continue)
- 1 atom, no free vars → IS terminal (valid, can stop)
- 2 atoms, no free vars → IS terminal (valid, can stop)
- 3 atoms, any free vars → IS terminal (max length, force stop)

This hard constraint is MUCH better than just penalizing free variables:
- Penalty: rule can still be generated, just gets low reward
- Constraint: rule CANNOT be generated as terminal state
- Constraint is stricter and more principled
""")
