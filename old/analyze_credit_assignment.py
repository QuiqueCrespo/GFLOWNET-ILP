"""
Analyze why the model doesn't learn to avoid UNIFY_VARIABLES at the first step.

The issue: At the initial state grandparent(X0, X1) :-, unifying X0 and X1
creates grandparent(X0, X0) :- which can never cover positive examples like
grandparent(alice, charlie). This should lead to reward ≈ 0, but the model
learns 50-50 probabilities for ADD_ATOM vs UNIFY_VARIABLES.
"""

import sys
sys.path.insert(0, '/Users/jq23948/Documents/GFLOWNET-ILP')

import torch
from src.logic_structures import get_initial_state, apply_unify_vars, Variable
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator

# Problem setup
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
]

positive_examples = [
    Example('grandparent', ('alice', 'charlie')),
]

negative_examples = [
    Example('grandparent', ('alice', 'alice')),
]

# Create initial state
initial_state = get_initial_state('grandparent', 2)
print("Initial state:", initial_state)
print(f"  Rule: {initial_state[0].head.predicate_name}({', '.join(str(v.id) for v in initial_state[0].head.args)}) :- {initial_state[0].body}")

# Apply UNIFY_VARIABLES to X0 and X1
var0 = Variable(0)
var1 = Variable(1)
unified_state = apply_unify_vars(initial_state, var0, var1)
print("\nAfter unifying X0 and X1:")
print(f"  Rule: {unified_state[0].head.predicate_name}({', '.join(str(v.id) for v in unified_state[0].head.args)}) :- {unified_state[0].body}")

# Calculate reward for unified state (even if we add atoms later)
logic_engine = LogicEngine(max_depth=10, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    weight_precision=0.5,
    weight_recall=0.5,
    weight_simplicity=0.05,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3,
    free_var_penalty=1.0,
    use_f1=True
)

# The unified state has an empty body, so we can't calculate reward yet
# But we can check the self-loop penalty
scores = reward_calc.get_detailed_scores(unified_state, positive_examples, negative_examples)
print("\nScores for unified state (empty body):")
for key, value in scores.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value}")

print("\n" + "="*80)
print("THEORETICAL ANALYSIS")
print("="*80)

print("""
The problem: After unifying X0 and X1, we get grandparent(X0, X0) :- ...

This creates a self-loop in the head, which means:
1. Self-loop penalty will be applied (0.3 per self-loop)
2. The rule can only match examples where both arguments are the same
3. Positive examples like grandparent(alice, charlie) will NEVER be covered
4. True Positives (TP) will always be 0
5. Final reward will be ≈ 1e-6 (minimum reward)

WHY THE MODEL MIGHT NOT LEARN:

1. **Replay Buffer Bias**:
   - Replay buffer only stores trajectories with reward > 0.7
   - Bad trajectories (starting with UNIFY) are not stored
   - If replay_probability = 0.5, then 50% of training uses replayed good trajectories
   - This dilutes the gradient signal from bad on-policy trajectories

2. **Log_Z Compensation**:
   - The learnable log_Z parameter might absorb the difference
   - Instead of reducing P_F(UNIFY | initial_state), the model increases log_Z
   - This minimizes loss without fixing the policy

3. **Gradient Variance**:
   - Stochastic sampling means UNIFY is only chosen ~33% of the time
   - Fewer samples = higher variance in gradient estimates
   - Slow convergence

4. **Backward Probability Issues**:
   - For UNIFY_VARIABLES, backward probability uses a heuristic: P_B = 1 / num_pairs
   - This might not accurately reflect the actual backward dynamics
   - See gflownet_models.py:362-366

RECOMMENDED FIXES:

1. **Reduce replay_probability** (or disable replay buffer during early training)
   - Ensures more on-policy learning from bad trajectories

2. **Add trajectory diversity loss**
   - Penalize the model for always taking the same action

3. **Use importance sampling for replay buffer**
   - Weight replayed trajectories by their probability ratio

4. **Check state embeddings**
   - Verify that initial_state and unified_state have different embeddings
   - If embeddings are too similar, the policy can't distinguish states

5. **Reduce reward_scale_alpha**
   - Current value of 10.0 creates huge differences in log_reward
   - This might cause numerical instability
   - Try 1.0 or 2.0 instead
""")
