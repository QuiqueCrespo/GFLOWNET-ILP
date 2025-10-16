# True Reward Selection Fix

## Issue

The benchmark pipeline was selecting the "best" rule based on **training reward** (which includes exploration bonuses), not the **true reward** (actual performance without bonuses).

### Problem Example

```
Rule with 0 positive coverage:
  Training reward: 0.7756 (with trajectory length bonus ~0.8)
  True reward: 0.000001 (actual performance)

This rule was being selected as "best" due to high training reward!
```

## Root Cause

In `run_benchmark_pipeline.py` line 188 (before fix):

```python
if reward > best_reward:  # reward includes exploration bonus!
    best_reward = reward
    best_rule = theory_to_string(theory)
```

The `reward` returned from `generate_trajectory()` includes the exploration bonus from:
- Trajectory length bonus: +0.1 per step
- Entropy bonus
- Temperature scaling

## The Fix

Modified the selection logic to use TRUE reward (lines 180-203):

```python
# Sample best rule every 10 episodes based on TRUE reward (no exploration bonus)
if episode % 10 == 0:
    trajectory, training_reward = trainer.generate_trajectory(...)

    if trajectory:
        theory = trajectory[-1].next_state

        # Calculate TRUE reward (without exploration bonus)
        true_reward = reward_calc.calculate_reward(
            theory,
            benchmark_data['positive_examples'],
            benchmark_data['negative_examples']
        )

        # Select best rule based on TRUE reward, not training reward
        if true_reward > best_true_reward:
            best_true_reward = true_reward
            best_reward = training_reward  # Keep for comparison
            best_rule_theory = theory  # Store the theory object
            best_rule = theory_to_string(theory)
```

Key changes:
1. Calculate true reward explicitly using `reward_calc.calculate_reward()`
2. Compare `true_reward > best_true_reward` for selection
3. Store both `best_true_reward` and `best_reward` for reporting
4. Store `best_rule_theory` (the actual theory object) to avoid stochastic regeneration

## Verification

### Before Fix
```
grandparent benchmark (500 episodes):
  Training reward: 0.5994
  True reward: 0.000001
  Coverage: 0/2 pos (0.0%), 0/4 neg (0.0%)
  Rule: grandparent(X2, X4) :- parent(X2, X2), parent(X4, X5), parent(X6, X7).

❌ Selected a bad rule with 0 coverage!
```

### After Fix
```
grandparent benchmark (1000 episodes):
  Training reward: 1.4277
  True reward: 0.925000
  Coverage: 2/2 pos (100.0%), 0/4 neg (0.0%)
  Rule: grandparent(X6, X5) :- parent(X6, X4), parent(X4, X5), parent(X6, X7).

✅ Selected a perfect rule with 100% pos, 0% neg coverage!
```

## Why Both Rewards Matter

### Training Reward (with bonuses)
- **Purpose**: Guides learning during training
- **Includes**: Exploration bonuses to encourage diverse trajectories
- **Use**: Drive the GFlowNet policy learning

### True Reward (no bonuses)
- **Purpose**: Evaluate actual rule quality
- **Excludes**: All exploration bonuses
- **Use**: Select the best rule, report final performance

## Impact

### Before
- Rules selected based on exploration bonuses
- High-reward rules often had 0 actual coverage
- Misleading "best" rules reported

### After
- Rules selected based on true performance
- Best rules actually have good coverage
- Accurate evaluation and reporting

## Reporting

The pipeline now shows both rewards for transparency:

```
BENCHMARK: grandparent
================================================================================
  Training reward (with bonus): 1.4277
  True reward (no bonus): 0.925000
  Coverage: 2/2 pos (100.0%), 0/4 neg (0.0%)
```

This makes it clear:
- **Training reward**: What the model sees during learning (incentivizes exploration)
- **True reward**: Actual rule quality (what we care about for evaluation)

## Files Modified

- `run_benchmark_pipeline.py`:
  - Lines 163-164: Added `best_rule_theory` and `best_true_reward` tracking
  - Lines 180-203: Modified selection logic to use true reward
  - Lines 214-240: Use stored theory object instead of regenerating
  - Line 256: Report `best_true_reward` instead of recalculated `true_reward`
  - Lines 270-271: Display both training and true rewards

## Related Documents

- `LOGIC_ENGINE_FIX.md`: Logic engine safety condition fix
- `EXPLORATION_BONUS_EXPLANATION.md`: Why exploration bonuses exist and how they work
- `PIPELINE_COVERAGE_UPDATE.md`: Coverage metrics addition
