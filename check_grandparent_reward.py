"""Check reward calculation for grandparent test."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import Atom, Variable, Rule
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

# The rule found
rule = Rule(
    head=Atom('grandparent', (Variable(0), Variable(0))),
    body=[
        Atom('parent', (Variable(0), Variable(0))),
        Atom('parent', (Variable(0), Variable(0))),
        Atom('parent', (Variable(0), Variable(0)))
    ]
)

theory = [rule]

# Background knowledge
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
    Example('parent', ('charlie', 'diana')),
    Example('parent', ('eve', 'frank'))
]

# Examples
positive_examples = [
    Example('grandparent', ('alice', 'charlie')),
    Example('grandparent', ('bob', 'diana'))
]

negative_examples = [
    Example('grandparent', ('alice', 'alice')),
    Example('grandparent', ('bob', 'bob')),
    Example('grandparent', ('alice', 'bob')),
    Example('grandparent', ('eve', 'charlie'))
]

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3,
    free_var_penalty=1.0
)

# Calculate reward
reward = reward_calc.calculate_reward(theory, positive_examples, negative_examples)
scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

print("Rule: grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0), parent(X0, X0).")
print(f"\nReward: {reward:.6f}")
print(f"Accuracy: {scores['accuracy']:.4f}")
print(f"Simplicity: {scores['simplicity']:.4f}")
print(f"Pos coverage: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
print(f"Neg coverage: {scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")
print(f"Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
print(f"\nExpected: 1e-6 since pos_score = 0.0")
print(f"Actual: {reward:.6f}")

if reward > 0.01:
    print("\n⚠️  BUG: Reward should be ~0.000001 for 0 positive coverage!")
else:
    print("\n✓ Correct: Reward is near-zero for 0 positive coverage")
