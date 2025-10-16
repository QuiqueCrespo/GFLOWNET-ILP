"""
Check that reward increases monotonically as we perform the correct sequence of actions.

This verifies that the reward landscape encourages the model to take steps toward
the correct rule: grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
"""

import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import *
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

# Define background knowledge
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
    Example('parent', ('eve', 'frank')),
    Example('parent', ('frank', 'grace')),
]

# Define examples
positive_examples = [
    Example('grandparent', ('alice', 'charlie')),
    Example('grandparent', ('eve', 'grace')),
]

negative_examples = [
    Example('grandparent', ('alice', 'alice')),
    Example('grandparent', ('bob', 'bob')),
    Example('grandparent', ('alice', 'eve')),
    Example('grandparent', ('bob', 'frank')),
]

# Initialize
logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(logic_engine)

print("=" * 80)
print("REWARD MONOTONICITY CHECK")
print("=" * 80)
print(f"\nBackground facts: {len(background_facts)}")
print(f"Positive examples: {len(positive_examples)}")
print(f"Negative examples: {len(negative_examples)}")
print("\nCorrect sequence to goal:")
print("  grandparent(X0, X1).")
print("  → ADD_ATOM('parent')")
print("  → UNIFY(X0, X2)")
print("  → ADD_ATOM('parent')")
print("  → UNIFY(X3, X4)")
print("  → UNIFY(X1, X5)")
print("  = grandparent(X0, X1) :- parent(X0, X3), parent(X3, X1).")
print("\n" + "=" * 80)

# Track rewards
rewards = []

# Step 0: Initial state
state = get_initial_state('grandparent', 2)
max_var_id = 1  # X0 and X1 are 0 and 1
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 0: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

# Step 1: ADD_ATOM('parent')
state, max_var_id = apply_add_atom(state, 'parent', 2, max_var_id)
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 1: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

# Step 2: UNIFY(X0, X2) - connects head var to first parent
vars_list = get_all_variables(state)
var_X0 = Variable(id=0)
var_X2 = Variable(id=2)
state = apply_unify_vars(state, var_X0, var_X2)
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 2: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

# Step 3: ADD_ATOM('parent') - second parent
state, max_var_id = apply_add_atom(state, 'parent', 2, max_var_id)
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 3: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

# Step 4: UNIFY(X3, X4) - chain the two parents
vars_list = get_all_variables(state)
var_X3 = Variable(id=3)
var_X4 = Variable(id=4)
state = apply_unify_vars(state, var_X3, var_X4)
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 4: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

# Step 5: UNIFY(X1, X5) - connect to output
vars_list = get_all_variables(state)
var_X1 = Variable(id=1)
var_X5 = Variable(id=5)
state = apply_unify_vars(state, var_X1, var_X5)
reward = reward_calc.calculate_reward(state, positive_examples, negative_examples)
rewards.append(reward)
print(f"\nStep 5: {theory_to_string(state)}")
print(f"  Reward: {reward:.6f}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check for alternative bad paths
print("\nComparison with degenerate 1-step path:")
bad_state = get_initial_state('grandparent', 2)
var_X0 = Variable(id=0)
var_X1 = Variable(id=1)
bad_state = apply_unify_vars(bad_state, var_X0, var_X1)
bad_reward = reward_calc.calculate_reward(bad_state, positive_examples, negative_examples)
print(f"  grandparent(X0, X0).")
print(f"  Reward: {bad_reward:.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Check monotonicity
monotonic = all(rewards[i] >= rewards[i-1] for i in range(1, len(rewards)))
print(f"\nReward trajectory: {' → '.join(f'{r:.6f}' for r in rewards)}")
print(f"\nMonotonic increasing: {'YES' if monotonic else 'NO'}")

if not monotonic:
    print("\n⚠️  PROBLEM: Rewards do NOT increase monotonically!")
    print("This means intermediate steps have lower rewards than earlier steps,")
    print("which creates a local minimum that gradient descent will get stuck in.")

    # Find decreases
    for i in range(1, len(rewards)):
        if rewards[i] < rewards[i-1]:
            print(f"  • Step {i-1} → {i}: {rewards[i-1]:.6f} → {rewards[i]:.6f} (DECREASE)")
else:
    print("\n✓ Rewards increase monotonically along the correct path.")
    print("The model should be able to learn this path through gradient descent.")

print(f"\nFinal reward (correct rule): {rewards[-1]:.6f}")
print(f"Degenerate reward (1-step): {bad_reward:.6f}")
if rewards[-1] > bad_reward:
    print(f"✓ Correct rule has higher reward ({rewards[-1]:.6f} > {bad_reward:.6f})")
else:
    print(f"⚠️  Degenerate has equal/higher reward ({bad_reward:.6f} >= {rewards[-1]:.6f})")
