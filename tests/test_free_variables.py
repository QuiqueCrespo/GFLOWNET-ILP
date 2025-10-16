"""Test free variable detection and penalty."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Theory, Atom, Variable, Rule
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

print("="*80)
print("FREE VARIABLE PENALTY TEST")
print("="*80)

background = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie'))
]

positive_examples = [Example('grandparent', ('alice', 'charlie'))]
negative_examples = [Example('grandparent', ('alice', 'alice'))]

logic_engine = LogicEngine(background_facts=background)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3,
    free_var_penalty=1.0  # CRITICAL penalty
)

# Test cases
test_rules = {
    "Correct (no free vars)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(1)))
        ]
    )],

    "Free variable X1": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(3)))
        ]
    )],

    "Both free (X0 and X1)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(2), Variable(3)))
        ]
    )],

    "Empty body (all free)": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[]
    )]
}

print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

for name, theory in test_rules.items():
    print(f"\n{name}:")
    rule = theory[0]

    # Format rule
    head_str = f"{rule.head.predicate_name}({', '.join(f'X{v.id}' for v in rule.head.args)})"
    if rule.body:
        body_str = ', '.join(f"{a.predicate_name}({', '.join(f'X{v.id}' for v in a.args)})" for a in rule.body)
        rule_str = f"{head_str} :- {body_str}"
    else:
        rule_str = f"{head_str}."

    print(f"  Rule: {rule_str}")

    scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

    print(f"  Free variables: {scores['num_free_vars']}")
    print(f"  Free var penalty: -{scores['free_var_penalty']:.2f}")
    print(f"  Base reward: {scores['reward']:.4f}")
    print(f"  Coverage: {scores['pos_covered']}/{scores['pos_total']} pos, "
          f"{scores['neg_covered']}/{scores['neg_total']} neg")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
Free variables in the head create overly general, ungrounded rules.

Example: grandparent(X0, X1) :- parent(X2, X3)
- X0 and X1 can match ANY constants
- Rule says "X0 is grandparent of X1 if SOMEONE (X2) is parent of SOMEONE (X3)"
- This is logically unsound!

Correct rule: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)
- All head variables (X0, X1) appear in body
- X0 and X1 are properly grounded through the body atoms
- This is logically sound

The free variable penalty (default: 1.0) should be CRITICAL because:
1. Free variables make rules semantically invalid
2. They should be completely prevented, not just discouraged
3. Even with exploration bonuses, they should get negative total rewards

Expected impact:
- Correct rule:   base reward ~ 0.9 (no penalty)
- 1 free var:     base reward ~ -0.1 (huge penalty)
- 2 free vars:    base reward ~ -1.0 (massive penalty)
- Empty body:     base reward ~ -2.0 (completely invalid)
""")
