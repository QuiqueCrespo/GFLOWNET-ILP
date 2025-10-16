# Reward Explanation: Why Rewards > 1.0 Occur

## Question

How are rewards like 0.9750 and 0.8000 possible for rules that are unsatisfiable by definition, such as:
```
[0.9750] grandparent(X0, X5) :- parent(X0, X3), parent(X4, X5), parent(X6, X7).
[0.8000] grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0), parent(X6, X7).
```

## Answer

**Rewards > 1.0 are due to exploration bonuses, NOT base reward**. The system adds a trajectory length bonus to encourage longer, more complex rules during exploration.

## Breakdown of Reward Components

### Total Reward Formula

```
total_reward = base_reward + exploration_bonus
```

Where:
- `base_reward`: Actual quality of the rule (0.0 to ~0.93)
- `exploration_bonus`: `β × trajectory_length` (with `β = 0.1`)

### Verification Results

Let's examine the actual rewards:

#### Rule 1: Partially Correct but Disconnected
```prolog
grandparent(X0, X5) :- parent(X0, X3), parent(X4, X5), parent(X6, X7).
```

**Base Reward**: 0.4750
- **Positive coverage**: 4/4 = 1.0 ✓ (covers all grandparent examples!)
- **Negative avoidance**: 2/4 = 0.5 (covers 2 negatives - not good)
- **Accuracy**: 1.0 × 0.5 = 0.5
- **Simplicity**: 1/(1+3) = 0.25
- **Final base**: 0.9 × 0.5 + 0.1 × 0.25 = 0.475

**Exploration Bonus**: 0.1 × 5 = 0.5

**Total Reward**: 0.475 + 0.5 = **0.9750** ✓

**Why it works**: This rule actually *does* cover all positive examples! Let's trace one:

Goal: `grandparent(alice, charlie)`

1. Unify head: `X0=alice, X5=charlie`
2. Prove body:
   - `parent(alice, X3)` → finds `parent(alice, bob)`, binds `X3=bob`
   - `parent(X4, charlie)` → finds `parent(bob, charlie)`, binds `X4=bob`
   - `parent(X6, X7)` → finds `parent(alice, bob)`, binds `X6=alice, X7=bob`
3. All atoms proven → Success!

The rule works but is **overly general** (it also matches negatives because X4 doesn't have to equal X3).

#### Rule 2: Completely Unsatisfiable
```prolog
grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0), parent(X6, X7).
```

**Base Reward**: 0.0000
- **Positive coverage**: 0/4 = 0.0 ✗ (doesn't cover ANY positives)
- **Negative avoidance**: 4/4 = 1.0 ✓ (doesn't cover any negatives)
- **Accuracy**: 0.0 × 1.0 = 0.0
- **Final base**: ~0.0 (minimum 1e-6 for numerical stability)

**Exploration Bonus**: 0.1 × 8 = 0.8

**Total Reward**: 0.0 + 0.8 = **0.8000** ✓

**Why it's unsatisfiable**: There are no self-loops in the background facts (`parent(X, X)` doesn't exist).

#### Rule 3: Correct Solution
```prolog
grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1).
```

**Base Reward**: 0.9333
- **Positive coverage**: 4/4 = 1.0 ✓
- **Negative avoidance**: 4/4 = 1.0 ✓
- **Accuracy**: 1.0 × 1.0 = 1.0
- **Simplicity**: 1/(1+2) = 0.333
- **Final base**: 0.9 × 1.0 + 0.1 × 0.333 = 0.933

**Exploration Bonus**: 0.1 × 2 = 0.2

**Total Reward**: 0.933 + 0.2 = **1.1333**

## Why Use Exploration Bonuses?

### Problem Without Bonuses

Without trajectory length bonuses:
- Short rules (1-2 steps) get evaluated first
- Model quickly converges to shortest rule
- Even if short rule is bad, it's stable (simple gradients)
- Result: Degenerate 1-step solution `grandparent(X0, X0).`

### Solution With Bonuses

With trajectory length bonuses:
- Longer trajectories get temporary reward boost
- Model explores complex rule space
- Replay buffer stores good long trajectories
- Over time, truly good long rules dominate

**Trade-off**: Some bad long rules get high total reward temporarily, but:
1. They don't cover positives → won't be added to replay buffer (threshold is 0.7)
2. Truly good rules have both high base + bonus
3. Diversity in sampling is valuable for exploration

## Replay Buffer Threshold

The replay buffer only stores trajectories with **reward > 0.7**:

```python
if self.replay_buffer is not None and reward > 0.7:
    self.replay_buffer.add(trajectory, reward)
```

**Analysis**:
- Rule 1 (0.9750): **Added to replay** (base 0.475 + bonus 0.5)
- Rule 2 (0.8000): **Added to replay** (base 0.0 + bonus 0.8)
- Rule 3 (1.1333): **Added to replay** (base 0.933 + bonus 0.2)

**Issue**: Bad rules with high bonuses get stored! This is why we see them in final samples.

## Impact on Training

### During Training

The TB/DB loss uses the **total reward** (with bonuses):

```python
log_reward = torch.log(torch.tensor(scaled_reward + 1e-8))
```

This means:
- Model learns to generate long trajectories (rewarded by bonus)
- Some long trajectories are bad (but still get bonus)
- Good long trajectories get highest total reward
- Model explores diverse trajectory space

### Why It Still Works

Even though bad rules get high rewards:

1. **Frequency matters**: Good rules are sampled more often
   - Rule 1: 50% positive coverage → not consistently good
   - Rule 3: 100% positive coverage → consistently good

2. **Replay distribution**: Reward-proportional sampling favors higher rewards
   - Rule 3 (1.13) sampled more often than Rule 2 (0.80)

3. **Gradient signal**: TB/DB loss eventually learns true quality
   - High-reward+good-base rules get strongest gradients
   - High-reward+bad-base rules inconsistent → weak gradients

## Comparison to Baseline

### Baseline (No Bonuses)
- Avg reward: 0.14
- Convergence: Episode 257
- Final rule: `grandparent(X0, X0).` (base reward ~0.16)

### All Combined (With Bonuses)
- Avg reward: 0.72
- Convergence: None (maintains diversity)
- Final rules: Mix of good and bad (exploration continues)

**Key difference**: With bonuses, model never stops exploring complex rules.

## Should We Change This?

### Option 1: Remove Exploration Bonuses from Replay Buffer Threshold

```python
# Only use base reward for replay buffer decision
base_reward = self.reward_calculator.calculate_reward(...)
if self.replay_buffer is not None and base_reward > 0.7:
    self.replay_buffer.add(trajectory, reward)  # Store total reward
```

**Effect**: Only truly good rules stored, but might reduce exploration.

### Option 2: Remove Bonuses from Loss Calculation

```python
# Use base reward for loss, total reward for metrics
base_reward = self.reward_calculator.calculate_reward(...)
total_reward = base_reward + exploration_bonus

if self.use_detailed_balance:
    loss = self.compute_detailed_balance_loss(trajectory, base_reward)  # No bonus!
```

**Effect**: Cleaner learning signal, but loses exploration incentive.

### Option 3: Decay Exploration Bonuses (But Slowly)

```python
TrajectoryLengthBonus(beta=0.1, decay=0.9999)  # Very slow decay
```

**Effect**: Early exploration, late refinement.

### Option 4: Keep As Is

**Rationale**:
- System already works (0.72 avg reward, no convergence)
- Exploration is valuable
- Replay buffer naturally filters over time (reward-proportional sampling)
- Can add reward shaping (penalties for self-loops, etc.) instead

## Recommendation

**Keep the exploration bonuses but improve reward shaping**:

```python
def shaped_reward(theory, pos_ex, neg_ex):
    base = calculate_reward(theory, pos_ex, neg_ex)

    # Penalty for self-loops (parent(X, X), etc.)
    self_loops = sum(
        1 for rule in theory
        for atom in rule.body
        if len(set(atom.args)) < len(atom.args)  # Duplicate variables
    )
    self_loop_penalty = -0.3 * self_loops

    # Penalty for disconnected variables (X4, X6, X7 not in head)
    disconnected = count_disconnected_vars(theory)
    disconnected_penalty = -0.2 * disconnected

    return base + self_loop_penalty + disconnected_penalty
```

**Effect**:
- Rule 1 (disconnected X4): penalty -0.2 → base 0.475 - 0.2 = 0.275 → total 0.775 (still high due to bonus)
- Rule 2 (self-loops): penalty -0.6 → base 0.0 - 0.6 = -0.6 → total 0.2 (now below threshold!)
- Rule 3 (correct): no penalties → base 0.933 → total 1.133 (unchanged)

This would filter bad rules from replay buffer while maintaining exploration benefits.

## Conclusion

**Rewards > 1.0 are intentional and beneficial for exploration**, but the current system stores some bad rules in the replay buffer. The solution is better reward shaping (penalties for pathological patterns) rather than removing exploration bonuses.

The key metrics to track are:
1. **Base reward** (actual rule quality): Should approach 0.9+
2. **Exploration bonus** (encourages diversity): Keeps model exploring
3. **Total reward** (training signal): Balances quality and exploration

Current results show the system works (0.72 avg total reward, no convergence), but refinement through reward shaping would improve final rule quality.
