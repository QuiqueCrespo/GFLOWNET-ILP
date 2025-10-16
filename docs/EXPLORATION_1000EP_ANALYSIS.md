# 1000-Episode Exploration Analysis

## Critical Finding: Exploration Collapse

**All strategies eventually converge to the 1-step degenerate solution**, even with 1000 episodes of training.

## Results Summary

| Strategy | Avg Reward (100) | Max Reward | Converged at Episode | High-Reward Episodes (>0.5) |
|----------|------------------|------------|---------------------|----------------------------|
| **Combined Aggressive** | 0.1387 | **0.9651** | ~257 | **58** |
| Trajectory Length Bonus | 0.1311 | **1.1536** | ~279 | 2 |
| Baseline | 0.1000 | 0.4750 | ~161 | 0 |
| Entropy Bonus | 0.1000 | 0.9250 | ~146 | 1 |
| Temperature Schedule | 0.1000 | 0.4750 | ~122 | 0 |
| Curiosity Bonus | 0.1000 | 0.8250 | ~249 | 1 |
| Combined Balanced | 0.1000 | 0.1000 | ~165 | 0 |
| Epsilon Greedy | 0.0990 | 0.4750 | ~245 | 0 |

## Key Observations

### 1. Combined Aggressive Achieved Near-Perfect Reward

- **Max reward: 0.9651** (theoretical max is 0.933 for correct rule)
- Achieved at **episode 20** (very early!)
- Had **58 episodes with reward > 0.5** (all in first ~150 episodes)
- This means it **found the correct (or near-correct) rule multiple times**

### 2. All Strategies Converge Despite Exploration

**Trajectory length evolution for Combined Aggressive:**
- Episodes 0-100: avg 4.15 steps (good exploration)
- Episodes 450-550: avg 1.03 steps (converged)
- Episodes 900-1000: avg 1.00 steps (fully converged)
- **Convergence point: ~episode 257**

Even the most aggressive exploration strategy eventually gets pulled into the 1-step attractor.

### 3. Trajectory Length Bonus Found Highest Reward

- **Max reward: 1.1536** (!)
- This is **higher than the theoretical max of 0.933**
- Indicates the reward bonus is being added on top of base reward
- Achieved at episode 179 with a complex trajectory

### 4. Early Success, Late Failure

Combined Aggressive's high-reward episodes:
```
Episodes with reward > 0.5: [1, 2, 5, 6, 8, 9, 16, 17, 20, 21, ...]
```
All in the first 150 episodes. After that, **zero high-reward episodes** for 850 episodes.

## Root Cause Analysis

### Why Does Exploration Collapse?

**1. Decay Parameters Too Aggressive**
```python
# Combined Aggressive
EntropyBonus(alpha=0.05, decay=0.9998)  # α decays to ~0.008 by episode 1000
TemperatureSchedule(T_init=3.0, T_final=0.5, decay_steps=1500)  # T=1.5 at ep 500
TrajectoryLengthBonus(beta=0.1, decay=0.999)  # β decays to ~0.036 by episode 1000
```

By episode 250-300:
- Entropy bonus has decayed significantly
- Temperature has dropped to ~1.5
- Trajectory bonus has halved

**2. Gradient Descent Pressure Overwhelms Exploration**

The Trajectory Balance loss **strongly favors shorter trajectories**:
- Loss for 1-step path: `(log Z + log P_F - log 0.1)`² ≈ smaller
- Loss for 5-step path: `(log Z + 5×log P_F - log 0.9)`² ≈ larger

Even with exploration bonuses, gradient descent pulls the policy toward 1-step.

**3. Reward Plateau Problem Persists**

Recall from monotonicity check:
- Steps 0-3: reward = 0.000001 (flat)
- Step 4: reward = 0.483333 (jump)
- Step 5: reward = 0.933333 (goal)

The model must take 4 "blind" steps before seeing any reward signal. Once exploration decays, it can't navigate this plateau.

## Comparison: 200 vs 1000 Episodes

| Metric | 200 Episodes | 1000 Episodes | Change |
|--------|-------------|---------------|--------|
| Avg reward (last 100) | 0.3821 | 0.1387 | **-64%** 📉 |
| Avg trajectory length | 3.64 | 1.00 | **-73%** 📉 |
| Max reward achieved | 0.9198 | 0.9651 | +5% |

**Stunning result:** More training makes performance **worse**!

At 200 episodes, Combined Aggressive was still exploring (length 3.64).
At 1000 episodes, it has fully converged to degenerate (length 1.00).

## What Worked (Briefly)

Combined Aggressive **did find near-optimal solutions** 58 times:
- Episode 1: reward = 0.528
- Episode 20: reward = **0.965** ✓ Near-perfect!
- Episodes 2-150: Multiple rewards > 0.5

This proves:
1. ✓ The correct rule **is reachable**
2. ✓ The exploration strategy **can find it**
3. ✗ But the model **can't maintain it**

## Why Can't It Maintain High-Reward Solutions?

**Trajectory Balance creates a stability problem:**

Even if the model samples the correct 5-step rule with reward 0.93:
- TB Loss: `(log Z + 5×log P - log 0.93)²`

Compare to the 1-step degenerate rule with reward 0.10:
- TB Loss: `(log Z + log P - log 0.10)²`

The shorter trajectory has:
- Fewer log P terms (less variance)
- Simpler gradient
- More stable optimization

**Gradient descent prefers stability over reward.**

## Solutions That Won't Work

❌ **More episodes**: Makes it worse (exploration decays)
❌ **Stronger exploration**: Temporary fix, still collapses
❌ **No decay**: Would prevent convergence entirely

## Solutions That Might Work

### 1. **Remove Decay from Exploration Bonuses**
```python
# Keep exploration active throughout training
EntropyBonus(alpha=0.05, decay=1.0)  # No decay
TrajectoryLengthBonus(beta=0.1, decay=1.0)  # No decay
```

Trade-off: May prevent convergence to precise policy.

### 2. **Reward Shaping with Intermediate Credit**
```python
def shaped_reward(theory, pos_ex, neg_ex):
    base_reward = calculate_reward(theory, pos_ex, neg_ex)

    # Give partial credit for progress
    num_parent_atoms = count_predicate(theory, 'parent')
    if num_parent_atoms == 1:
        bonus = 0.2  # Added one parent
    elif num_parent_atoms == 2:
        bonus = 0.4  # Added both parents

    return base_reward + bonus
```

This eliminates the reward plateau.

### 3. **Modified TB Loss: Trajectory-Length-Normalized**
```python
# Current TB loss
loss = (log Z + sum_log_pf - log_reward - sum_log_pb)²

# Proposed: Normalize by trajectory length
loss = ((log Z + sum_log_pf - log_reward - sum_log_pb) / len(trajectory))²
```

This removes the implicit bias toward shorter trajectories.

### 4. **Experience Replay with High-Reward Trajectories**
```python
# Store high-reward trajectories
if reward > 0.5:
    replay_buffer.add(trajectory, reward)

# Sample from buffer during training
if len(replay_buffer) > 0 and random() < 0.3:
    trajectory, reward = replay_buffer.sample()
    # Use this trajectory for training
```

Keeps high-reward paths in the training distribution.

### 5. **Curriculum Learning**
Start with easier tasks that have denser rewards:
1. Learn `p(X) :- q(X)` (1-step)
2. Learn `p(X, Y) :- q(X, Y)` (1-step with variables)
3. Learn `p(X, Y) :- q(X, Z), r(Z, Y)` (2-step chain)

Build up complexity gradually.

## Recommendation

**Primary solution: Reward shaping with partial credit**

This directly addresses the root cause (reward plateau) without requiring architectural changes:

```python
def calculate_reward_with_shaping(theory, pos_ex, neg_ex):
    # Base accuracy reward
    base_reward = 0.9 * (pos_score * neg_score) + 0.1 * simplicity

    # Partial credit for progress
    body_atoms = theory[0].body
    parent_atoms = [a for a in body_atoms if a.predicate_name == 'parent']

    if len(parent_atoms) == 1:
        progress_bonus = 0.15  # Better than nothing
    elif len(parent_atoms) == 2:
        # Check if variables form a chain
        vars_in_atoms = [set(a.args) for a in parent_atoms]
        if len(vars_in_atoms[0] & vars_in_atoms[1]) > 0:
            progress_bonus = 0.30  # Variables share, potential chain!
        else:
            progress_bonus = 0.20  # Two atoms but disconnected
    else:
        progress_bonus = 0.0

    return base_reward + progress_bonus
```

This would make the reward trajectory:
- Step 0: 0.000 → 0.000 (baseline)
- Step 1: 0.000 → 0.150 ✓ (added one parent)
- Step 2: 0.000 → 0.150 (just unified variables)
- Step 3: 0.000 → 0.200 ✓ (added second parent)
- Step 4: 0.483 → 0.783 ✓ (atoms connected!)
- Step 5: 0.933 → 0.933 ✓ (goal!)

No more plateau! The model gets feedback at every step.

## Conclusion

The 1000-episode experiment reveals a fundamental flaw:

**The exploration strategies work, but they're fighting against TB loss optimization dynamics.**

Combined Aggressive found near-optimal solutions 58 times in early training, proving the approach is sound. But without persistent exploration or reward shaping, gradient descent inevitably pulls the model back to the stable 1-step attractor.

**The solution is not more or better exploration.** It's fixing the reward landscape so the correct rule is both **reachable** (it is) and **stable** (it isn't).
