"""Diagnose why rules with 0 pos, 0 neg coverage get high rewards."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Theory, Atom, Variable, Rule, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

print("="*80)
print("REWARD CALCULATION DIAGNOSIS")
print("="*80)

# Simple test case
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie'))
]

positive_examples = [
    Example('grandparent', ('alice', 'charlie'))
]

negative_examples = [
    Example('grandparent', ('alice', 'alice')),
    Example('grandparent', ('bob', 'bob'))
]

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3,
    free_var_penalty=1.0
)

# Test different rules
test_rules = [
    {
        "name": "Rule with 0 coverage (disconnected vars)",
        "theory": [Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(2), Variable(3))),
                Atom('parent', (Variable(4), Variable(5)))
            ]
        )]
    },
    {
        "name": "Rule with 0 coverage (self-loops)",
        "theory": [Rule(
            head=Atom('grandparent', (Variable(0), Variable(0))),
            body=[
                Atom('parent', (Variable(0), Variable(0)))
            ]
        )]
    },
    {
        "name": "Perfect rule (100% pos, 0% neg)",
        "theory": [Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[
                Atom('parent', (Variable(0), Variable(2))),
                Atom('parent', (Variable(2), Variable(1)))
            ]
        )]
    },
    {
        "name": "Uninformative rule (100% pos, 100% neg)",
        "theory": [Rule(
            head=Atom('grandparent', (Variable(0), Variable(1))),
            body=[Atom('parent', (Variable(0), Variable(1)))]
        )]
    }
]

print("\n" + "="*80)
print("TEST RULES")
print("="*80)

for test in test_rules:
    theory = test["theory"]
    name = test["name"]

    print(f"\n{name}:")
    print(f"  Rule: {theory_to_string(theory)}")

    # Get detailed scores
    scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

    # Calculate reward manually
    reward = reward_calc.calculate_reward(theory, positive_examples, negative_examples)

    print(f"  Positive coverage: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
    print(f"  Negative coverage: {scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")
    print(f"  Accuracy: {scores['accuracy']:.4f}")
    print(f"  Base reward (before penalties): {scores['reward']:.4f}")
    print(f"  Disconnected vars: {scores['num_disconnected_vars']} (penalty: -{scores['disconnected_penalty']:.2f})")
    print(f"  Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
    print(f"  Free vars: {scores.get('num_free_vars', 0)} (penalty: -{scores.get('free_var_penalty', 0):.2f})")
    print(f"  Final reward: {reward:.4f}")

    # Check for the bug
    if scores['pos_covered'] == 0 and scores['neg_covered'] == 0:
        if reward > 0.5:
            print(f"  ⚠️  BUG: Rule covers 0 examples but has high reward {reward:.4f}!")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
Expected behavior:
1. Rule with 0 coverage should have reward close to 0
2. Perfect rule (100% pos, 0% neg) should have reward close to 1
3. Uninformative rule (covers everything) should have low reward

If rules with 0 coverage get high rewards, this indicates:
- Reward floor bug might still be present
- Accuracy calculation issue when no examples are covered
- Exploration bonus inflating rewards

Let's check the reward formula:
reward = 0.9 * accuracy + 0.1 * simplicity - penalties

For 0 coverage:
- pos_score = 0 (0/1 positives)
- neg_score = 1 (0/2 negatives not covered = good)
- accuracy = 0.5 * (0 + 1) = 0.5
- This gives base reward of 0.9 * 0.5 = 0.45

Then penalties are applied. If there are large penalties (e.g., -0.4 for disconnected),
the final reward should be ~0.05, not high.

If we see high rewards, the issue is likely in:
1. Accuracy calculation for 0 coverage
2. Negative score being inflated
3. Penalties not being applied correctly
""")
