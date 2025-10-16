"""Diagnose why 0 pos coverage gives non-zero rewards."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Atom, Variable, Rule
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

print("="*80)
print("ZERO POSITIVE COVERAGE REWARD DIAGNOSIS")
print("="*80)

# Simulate the benchmark cases
test_cases = [
    {
        "name": "1d_flip example",
        "rule": Rule(
            head=Atom('out', (Variable(6), Variable(2), Variable(2))),
            body=[
                Atom('in', (Variable(2), Variable(6), Variable(2))),
                Atom('max_position', (Variable(6),)),
                Atom('x22', (Variable(7),))
            ]
        ),
        "pos_examples": [Example('out', ('a', 'b', 'c'))] * 31,  # 31 positive examples
        "neg_examples": [Example('out', ('x', 'y', 'z'))] * 915,  # 915 negative examples
        "background": []
    },
    {
        "name": "andersen example (self-loops)",
        "rule": Rule(
            head=Atom('pt', (Variable(6), Variable(6))),
            body=[
                Atom('store', (Variable(6), Variable(6))),
                Atom('store', (Variable(6), Variable(6))),
                Atom('store', (Variable(6), Variable(6)))
            ]
        ),
        "pos_examples": [Example('pt', ('a', 'b'))] * 7,  # 7 positive examples
        "neg_examples": [],  # 0 negative examples
        "background": []
    },
    {
        "name": "iggp-buttons-goal (self-loops, equal pos/neg)",
        "rule": Rule(
            head=Atom('goal', (Variable(0), Variable(0), Variable(0))),
            body=[
                Atom('int_20', (Variable(0),)),
                Atom('int_73', (Variable(0),)),
                Atom('int_91', (Variable(0),))
            ]
        ),
        "pos_examples": [Example('goal', ('a', 'b', 'c'))] * 22,  # 22 positive examples
        "neg_examples": [Example('goal', ('x', 'y', 'z'))] * 22,  # 22 negative examples
        "background": []
    }
]

for test in test_cases:
    print(f"\n{'='*80}")
    print(f"TEST: {test['name']}")
    print(f"{'='*80}")

    theory = [test['rule']]

    logic_engine = LogicEngine(max_depth=5, background_facts=test['background'])
    reward_calc = RewardCalculator(
        logic_engine,
        disconnected_var_penalty=0.2,
        self_loop_penalty=0.3,
        free_var_penalty=1.0
    )

    # Get detailed scores
    scores = reward_calc.get_detailed_scores(
        theory,
        test['pos_examples'],
        test['neg_examples']
    )

    reward = reward_calc.calculate_reward(
        theory,
        test['pos_examples'],
        test['neg_examples']
    )

    print(f"\nRule structure:")
    print(f"  Head: {test['rule'].head.predicate_name}({', '.join(str(arg) for arg in test['rule'].head.args)})")
    print(f"  Body: {len(test['rule'].body)} atoms")

    print(f"\nCoverage:")
    print(f"  Positive: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
    print(f"  Negative: {scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")

    print(f"\nReward components:")
    print(f"  pos_score: {scores['pos_score']:.4f}")
    print(f"  neg_score: {scores['neg_score']:.4f}")
    print(f"  accuracy: {scores['accuracy']:.4f}")
    print(f"  simplicity: {scores['simplicity']:.4f}")
    print(f"  uninformative_penalty: {scores['uninformative_penalty']:.4f}")

    print(f"\nStructural issues:")
    print(f"  Disconnected vars: {scores['num_disconnected_vars']} (penalty: -{scores['disconnected_penalty']:.2f})")
    print(f"  Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
    print(f"  Free vars: {scores.get('num_free_vars', 0)} (penalty: -{scores.get('free_var_penalty', 0):.2f})")

    print(f"\nFinal reward: {reward:.4f}")

    # Manually calculate what reward should be
    if scores['pos_score'] == 0.0:
        print(f"\n⚠️  pos_score = 0.0, so reward should be 1e-6")
        print(f"   But actual reward is {reward:.4f}")

        if reward > 0.01:
            print(f"   ❌ BUG: Reward is too high!")

            # Check the calculation path
            print(f"\n   Checking reward calculation:")
            print(f"   1. pos_score = {scores['pos_score']} (should trigger 1e-6 floor)")
            print(f"   2. neg_score = {scores['neg_score']}")
            print(f"   3. accuracy = {scores['accuracy']} (should be 0 since pos_score = 0)")
            print(f"   4. Base reward = 0.9 * {scores['accuracy']} + 0.1 * {scores['simplicity']} = {0.9*scores['accuracy'] + 0.1*scores['simplicity']:.4f}")
            print(f"   5. Total penalties = {scores['disconnected_penalty'] + scores['self_loop_penalty'] + scores.get('free_var_penalty', 0):.2f}")
            print(f"   6. Before floor: {0.9*scores['accuracy'] + 0.1*scores['simplicity'] - scores['disconnected_penalty'] - scores['self_loop_penalty'] - scores.get('free_var_penalty', 0):.4f}")
        else:
            print(f"   ✓ Correct: Reward is near-zero")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("""
If we see high rewards (>0.5) for rules with 0 positive coverage, the issue is:

1. The reward calculation has a floor of 1e-6 when pos_score = 0
2. BUT the trainer might be using a DIFFERENT reward calculation
3. OR the best_reward is from DURING training when the rule had different coverage
4. OR there's a bug in how get_detailed_scores vs calculate_reward handle 0 coverage

Need to check if the training loop is using a different reward calculation.
""")
