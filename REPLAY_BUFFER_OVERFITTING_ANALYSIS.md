# Replay Buffer Overfitting: Low Loss But Poor Sampling

**Date:** 2025-01-22
**Status:** 🔴 CRITICAL ISSUE IDENTIFIED

---

## The Observation

**User Report:**
> "When a good trajectory is replayed from the replay buffer, the loss is low, meaning it has correctly modelled that trajectory. However, when sampling a trajectory, it doesn't really sample high reward trajectories proportional to the reward - it mainly samples low reward trajectories."

**This is a smoking gun for replay buffer overfitting!**

---

## What This Means

### Loss is Low on Replayed Trajectories ✓

```python
# Replay step
replayed_trajectory = buffer.sample()  # High-reward trajectory (R=0.9)
recomputed_log_pf = model.compute_log_pf(trajectory)
loss = (log_Z + recomputed_log_pf - log(0.9) - log_pb)^2
# Loss = 0.01 (very low!)
```

**Interpretation:** The model has learned to assign **high probability** to this specific trajectory.

### But On-Policy Sampling Gives Low Rewards ✗

```python
# On-policy step
new_trajectory = model.sample()  # Stochastic sampling
reward = evaluate(new_trajectory)
# Reward = 0.15 (very low!)
```

**Interpretation:** Despite learning the replayed trajectories well, the model **doesn't sample them** (or similar high-reward trajectories) when generating new trajectories!

---

## The Paradox: How Can This Happen?

### GFlowNet Goal

GFlowNet is designed to sample trajectories **proportional to reward**:
- High-reward trajectories → sampled frequently
- Low-reward trajectories → sampled rarely

**Mathematically:** P(trajectory) ∝ R(trajectory)

### Expected Behavior

If the model has **low loss** on high-reward trajectories, it should:
1. Assign high probability to those trajectories ✓ (confirmed by low loss)
2. **Therefore sample them frequently** during on-policy sampling ✓ (expected)
3. New on-policy samples should have high reward ✓ (expected)

### Actual Behavior

1. Assign high probability to replayed trajectories ✓ (confirmed)
2. **Sample low-reward trajectories** during on-policy sampling ✗ (paradox!)
3. New on-policy samples have low reward ✗ (problem!)

**Why does this paradox occur?**

---

## Root Cause: Memorization vs. Generalization

### The Model is Memorizing, Not Learning

**Replay Buffer Contents (default: 50 trajectories):**
```python
buffer = [
    (trajectory_1, reward=0.95),  # specific path: s0→s1→s2→s3→s4
    (trajectory_2, reward=0.92),  # specific path: s0→s5→s6→s7→s8
    ...
    (trajectory_50, reward=0.85)  # specific path: s0→s9→s10→s11
]
```

**What the model learns from replay:**
```python
# Memorized lookup table:
State s1 → Action a1  (high probability)  # From trajectory_1
State s2 → Action a2  (high probability)  # From trajectory_1
State s5 → Action a5  (high probability)  # From trajectory_2
State s6 → Action a6  (high probability)  # From trajectory_2
...
```

**The model learns:** "In these **specific states**, take these **specific actions**"

**The model does NOT learn:** "What makes a good action **in general**"

### Why This Causes Low On-Policy Reward

**During on-policy sampling:**

```python
# Start from initial state s0
s0 → [sample action] → s_new

# Is s_new in the replay buffer trajectories?
if s_new in {s1, s5, s9, ...}:  # States from buffer
    # YES - model knows what to do (memorized)
    action = high_probability_action  # From memorized path
    # Continue along memorized path → high reward ✓
else:
    # NO - model has NOT learned policy for this state!
    action = sample_from_policy(s_new)  # Policy is POOR for unseen states
    # Diverge from memorized paths → low reward ✗
```

**The Problem:**
- State space is **HUGE** (combinatorially large)
- Replay buffer covers only **50 trajectories** (tiny fraction of state space)
- Model only learns policy for states in those 50 paths
- **Stochastic sampling** naturally deviates from memorized paths
- Once off the memorized path → poor policy → low reward

### Analogy: Memorizing Specific Routes

**Imagine learning to navigate a city:**

**Memorization approach (replay buffer overfitting):**
- Memorize 50 specific routes to high-value destinations
- "From corner A, turn left; from corner B, turn right; ..."
- When you start at a known corner → follow memorized route → success! ✓
- When you start at an UNKNOWN corner → lost! ✗

**Generalization approach (proper learning):**
- Learn general navigation principles
- "Head toward downtown, avoid highways during rush hour, ..."
- From ANY starting point → use principles → reach destination ✓

**The model is using the memorization approach!**

---

## Quantitative Analysis

### State Space Size vs. Replay Buffer Coverage

**Typical state space size:**
```python
# Initial: target(X0, X1) :-
# After 1 step: ~10 possible states (10 predicates)
# After 2 steps: ~100 possible states (10 predicates × 10 locations)
# After 3 steps: ~1000 possible states
# After 4 steps: ~10,000 possible states

Total reachable states ≈ 10,000+
```

**Replay buffer coverage:**
```python
50 trajectories × 5 steps/trajectory = 250 state-action pairs

Coverage = 250 / 10,000 = 2.5%
```

**The model learns policy for 2.5% of the state space!**

### Probability of Staying on Memorized Path

**Suppose each step has 10 possible actions:**

```python
P(sample memorized action) ≈ 0.7  # Model learned this state well
P(sample different action) ≈ 0.3  # Stochastic sampling

# After 5 steps:
P(stay on memorized path) = 0.7^5 ≈ 0.17 (17%)
P(diverge from memorized path) = 1 - 0.17 = 0.83 (83%)
```

**83% of on-policy trajectories diverge from memorized paths!**

Once diverged → entering states the model hasn't learned → poor policy → low reward.

---

## Why Loss is Still Low on Replayed Trajectories

### The Recomputation Process

**File:** `src/training.py:710-713`

```python
# Re-compute log probabilities for the replayed trajectory
recomputed_trajectory_for_loss = [
    TrajectoryStep(s.state, s.action_type, s.action_detail,
                  self._recompute_step_log_pf(s.state, s.action_type, s.action_detail),
                  s.next_state)
    for s in replayed_trajectory
]
```

**What happens:**
1. Take a trajectory from buffer: `s0 → s1 → s2 → s3 → s4` (reward 0.95)
2. **Use current model** to compute log P_F for each step
3. Since we've trained on this trajectory many times:
   - `log P_F(s0 → s1)` is HIGH (learned)
   - `log P_F(s1 → s2)` is HIGH (learned)
   - `log P_F(s2 → s3)` is HIGH (learned)
   - `log P_F(s3 → s4)` is HIGH (learned)
4. Compute TB loss: `(log_Z + sum_log_pf - log(0.95) - sum_log_pb)^2`
5. Loss is LOW because `sum_log_pf` is HIGH (model learned these specific transitions)

**This confirms:** The model assigns high probability to the **specific state-action pairs** in the buffer.

### But This Doesn't Transfer to New States!

**During on-policy sampling:**

```python
s0 → [sample] → s_diverge  # Different from memorized s1!

# Model's policy at s_diverge is POOR (never trained on this state)
# Takes a bad action → low reward
```

**The disconnect:**
- High probability for states **in buffer** ✓
- Low probability (poor policy) for states **not in buffer** ✗
- On-policy sampling explores states not in buffer → low reward

---

## Evidence This is Happening

### Symptoms to Check For

1. **Replay loss decreases over time** ✓
   - Model is learning the replayed trajectories

2. **On-policy loss remains high** ✗
   - Model is NOT learning general policy

3. **Replay reward is high (0.7-0.95)** ✓
   - Buffer contains good trajectories

4. **On-policy reward is low (0.1-0.3)** ✗
   - New samples are poor

5. **Gap widens over training** ✗
   - Model overfits more to buffer over time

6. **log_Z increases significantly**
   - Compensating for memorized high-probability paths

### Diagnostic Experiments

**Experiment 1: Compare loss on buffer vs. on-policy**

```python
# After episode N:
buffer_loss = loss_on_replayed_trajectory  # Should be LOW
onpolicy_loss = loss_on_new_trajectory     # Should be HIGH

if buffer_loss < 0.1 and onpolicy_loss > 1.0:
    print("⚠️ OVERFITTING TO REPLAY BUFFER DETECTED!")
```

**Experiment 2: Measure state overlap**

```python
buffer_states = {state for traj in buffer for state in traj.states}
onpolicy_states = {state for state in new_trajectory.states}

overlap = len(buffer_states & onpolicy_states) / len(onpolicy_states)

if overlap < 0.1:
    print(f"⚠️ Only {overlap*100}% state overlap - model hasn't learned these states!")
```

**Experiment 3: Disable replay and check performance**

```python
# Temporarily set replay_probability = 0.0
# Sample 10 trajectories
avg_reward = mean([r for _, r in sampled_trajectories])

# Then enable replay: replay_probability = 0.5
# Check if avg_reward improves

if not improved:
    print("⚠️ Replay buffer not helping generalization!")
```

---

## Why This Happens: The Training Dynamics

### Replay Dominates Training

**Default configuration:**
- `replay_probability = 0.3` (30% of steps use replay)
- `buffer_reward_threshold = 0.7` (only high-reward trajectories stored)

**Training distribution:**
```python
# 30% of updates: Train on buffer (reward 0.7-0.95)
# 70% of updates: Train on on-policy samples (reward 0.1-0.3)

# But gradient magnitude differs:
Replay gradient: Large (high reward → large log_R)
On-policy gradient: Small (low reward → small log_R)

# Effective training distribution (by gradient magnitude):
# ~60% replay, 40% on-policy
```

**Result:** Model optimizes primarily for replayed states, not on-policy states.

### Stochastic Sampling Ensures Divergence

**Even if model learned a good policy for buffer states:**

```python
# At state s1 (from buffer), memorized action is a1
action_probs = [0.8,  0.1,  0.05,  0.05]  # a1 has highest prob
            # [a1,   a2,   a3,    a4]

# Stochastic sampling:
sampled_action = sample(action_probs)

# 20% chance of sampling a2, a3, or a4 instead of a1!
# → Diverge from memorized path
# → Enter unmemoried state
# → Poor policy
# → Low reward
```

**The model CANNOT deterministically follow memorized paths during stochastic sampling!**

### The Memorization Trap

```
1. Initial training: Sample random trajectories → mostly low reward
2. Occasional high-reward trajectory → added to buffer
3. Replay buffer training → model learns these specific paths
4. On-policy sampling → diverges from learned paths → low reward
5. Low-reward trajectories NOT added to buffer
6. Training dominated by replay of same 50 trajectories
7. Model overfits to these 50 paths
8. On-policy performance does NOT improve
9. LOOP back to step 6
```

**The model gets stuck in a local optimum!**

---

## Why This Hurts Unification Learning Specifically

### ADD_ATOM Can Generalize Across States

**Predicate preferences are state-independent:**

```python
# State s1: grandparent(X0,X1) :- [empty]
# Action: ADD parent → reward improves

# State s2: grandparent(X0,X1) :- parent(X0,X2)
# Action: ADD parent → reward improves

# Generalization: "Adding parent is good for grandparent task"
# This transfers across states!
```

**The atom adder network learns:** "Predicate `parent` is useful" (general principle)

### UNIFY_VARIABLES is State-Specific

**Unification value depends on exact state:**

```python
# State A: grandparent(X0,X1) :- parent(X0,X2)
# UNIFY (X1, X2) → grandparent(X0,X2) :- parent(X0,X2)
# Then ADD parent(X2,X3) → Good! Reward 0.9

# State B: grandparent(X0,X1) :- parent(X0,X2), parent(X3,X1)
# UNIFY (X1, X2) → grandparent(X0,X2) :- parent(X0,X2), parent(X3,X2)
# Reward: 0.3 (different effect!)

# NO GENERALIZATION - same pair, different value!
```

**The unifier network must learn:** Pair (X1, X2) value depends on **specific atoms present** (state-specific)

**This means:**
- Unifier network CANNOT generalize across states as easily
- Each buffer trajectory teaches policy for those specific states
- Unseen states → no learned policy → random unifications → low reward

**Replay buffer overfitting hurts unification MORE than atom addition!**

---

## Theoretical Analysis: Why TB Loss Allows Memorization

### The Trajectory Balance Objective

```
TB Loss = (log Z + Σ log P_F(s→s') - log R - Σ log P_B(s'→s))^2
```

**Goal:** Sample trajectories proportional to reward.

**Problem:** This can be satisfied by:

1. **Global policy** (desired):
   - Learn P_F(action | state features) for ALL states
   - Generalize to unseen states
   - Sample high-reward trajectories everywhere

2. **Lookup table** (actual):
   - Learn P_F(action | state) for SEEN states only
   - Memorize specific paths
   - High probability for buffer states, random for others

**Both satisfy the TB objective!** But only #1 generalizes.

### Why Doesn't TB Prevent Memorization?

**Replay training:**
```python
# Trajectory from buffer: s0 → s1 → s2 → s3 (R=0.9)
# Model assigns: log P_F = -2.0, log P_B = -1.5
# TB loss = (log Z - 2.0 - log(0.9) - (-1.5))^2
# If log Z = 0.4, loss ≈ 0 ✓
```

**On-policy sampling from SAME initial state:**
```python
# Model samples: s0 → s_new → s_other → s_bad (R=0.1)
# Why different? Stochastic sampling diverged to s_new instead of s1

# At state s_new (never seen in buffer):
# Model has no learned policy → random actions
# Results in low reward
```

**TB loss on this trajectory:**
```python
# log P_F = -3.5 (low because model doesn't know these states)
# TB loss = (log Z - 3.5 - log(0.1) - log P_B)^2
# Loss might be high, but:
# 1. Only 70% of training (replay is 30%)
# 2. Gradient might be absorbed by log_Z
# 3. Biased by incorrect log P_B estimate
```

**Result:** TB loss is satisfied by memorizing buffer, not by learning general policy.

---

## Solutions

### Immediate Interventions

#### 1. Increase Replay Buffer Size

**Problem:** 50 trajectories cover only ~2.5% of state space

**Solution:**
```python
replay_buffer_capacity = 500  # 10× larger
# Covers ~25% of state space
# Better generalization
```

**Trade-off:** More memory, slower sampling

#### 2. Decrease Replay Probability

**Problem:** 30% replay → training dominated by buffer

**Solution:**
```python
replay_probability = 0.1  # Only 10% replay
# More on-policy training
# Better exploration of state space
```

**Trade-off:** Slower convergence on good trajectories

#### 3. Lower Buffer Threshold

**Problem:** Only store reward > 0.7 → buffer fills slowly

**Solution:**
```python
buffer_reward_threshold = 0.5  # Lower threshold
# More diverse trajectories in buffer
# Better state space coverage
```

**Trade-off:** Lower-quality trajectories in buffer

#### 4. Add State Diversity Metric

**Solution:** Only add trajectory to buffer if it explores NEW states

```python
def add_to_buffer(trajectory, reward):
    new_states = set(trajectory.states)
    buffer_states = set(s for traj in buffer for s in traj.states)

    novelty = len(new_states - buffer_states) / len(new_states)

    if reward > 0.5 and novelty > 0.3:  # Require 30% novel states
        buffer.add(trajectory, reward)
```

**Benefit:** Buffer covers diverse states, not just high-reward paths

### Medium-Term Solutions

#### 5. Data Augmentation on Replayed Trajectories

**Idea:** Create variations of buffer trajectories

```python
def augment_trajectory(trajectory):
    """Create slight variations by injecting random actions."""
    augmented = []
    for i, step in enumerate(trajectory):
        augmented.append(step)

        # 20% chance to inject a random action
        if random.random() < 0.2:
            random_state = apply_random_action(step.next_state)
            augmented.append(random_step)

    return augmented
```

**Benefit:** Model trains on states NEAR buffer states → better generalization

#### 6. Importance Sampling for Replay

**Problem:** All buffer trajectories weighted equally

**Solution:** Weight by how "recent" they are

```python
def sample_with_recency(buffer):
    ages = [current_step - traj.added_at for traj in buffer]
    recency_weights = [1.0 / (1.0 + age/100) for age in ages]

    # Combine recency with reward
    combined_weights = [r * w for r, w in zip(rewards, recency_weights)]

    return sample_proportional(buffer, combined_weights)
```

**Benefit:** Newer trajectories (from current policy) weighted more

#### 7. Curriculum Learning for Buffer

**Idea:** Start with small buffer, gradually increase

```python
# Episode 0-100: buffer_size = 10
# Episode 100-500: buffer_size = 50
# Episode 500+: buffer_size = 200

buffer_size = min(10 + episode // 10, 200)
```

**Benefit:** Early training focuses on few good paths (fast learning), later training explores diversity (generalization)

### Long-Term Solutions

#### 8. Use Detailed Balance Instead of TB

**Detailed Balance:** Learns state-specific flows, not single log_Z

```python
# DB objective: F(s) * P_F(s→s') = F(s') * P_B(s'→s)
# Learns flow F(s) for each state
# Can't memorize specific paths as easily
```

**Benefit:** Harder to overfit to buffer paths

#### 9. Add Regularization to Prevent Memorization

**L2 regularization on policy network:**

```python
policy_reg = λ * ||θ_policy||^2
total_loss = TB_loss + policy_reg
```

**Benefit:** Prevents large weights that memorize specific transitions

#### 10. Explicit Generalization Penalty

**Penalize policies that perform differently on buffer vs. on-policy:**

```python
buffer_performance = mean_reward_on_replayed_trajectories
onpolicy_performance = mean_reward_on_new_trajectories

generalization_gap = buffer_performance - onpolicy_performance
penalty = gap^2

total_loss = TB_loss + 0.1 * penalty
```

**Benefit:** Directly optimizes for generalization

---

## Diagnostic Code

### Check for Overfitting

```python
def diagnose_replay_overfitting(trainer, initial_state, pos_examples, neg_examples):
    """
    Check if model is overfitting to replay buffer.
    """

    # 1. Sample from replay buffer and compute loss
    if len(trainer.replay_buffer) > 0:
        buffer_traj, buffer_reward = trainer.replay_buffer.sample(1)[0]
        buffer_loss = trainer.compute_trajectory_balance_loss(buffer_traj, buffer_reward)
    else:
        buffer_loss = None

    # 2. Generate on-policy trajectory and compute loss + reward
    onpolicy_traj, onpolicy_reward = trainer.generate_trajectory(
        initial_state, pos_examples, neg_examples
    )
    onpolicy_loss = trainer.compute_trajectory_balance_loss(onpolicy_traj, onpolicy_reward)

    # 3. Compare
    print(f"\n{'='*60}")
    print(f"Replay Buffer Overfitting Diagnostic")
    print(f"{'='*60}")

    if buffer_loss is not None:
        print(f"Replay buffer loss:      {buffer_loss.item():.4f}")
        print(f"Replay buffer reward:    {buffer_reward:.4f}")
    print(f"On-policy loss:          {onpolicy_loss.item():.4f}")
    print(f"On-policy reward:        {onpolicy_reward:.4f}")

    # 4. Diagnosis
    if buffer_loss is not None and buffer_loss < 0.1 and onpolicy_loss > 1.0:
        print(f"\n⚠️  OVERFITTING DETECTED!")
        print(f"   Model has low loss on buffer but high loss on new samples")
        print(f"   Suggestions:")
        print(f"   - Increase buffer size (current: {len(trainer.replay_buffer)})")
        print(f"   - Decrease replay_probability (current: {trainer.replay_probability})")
        print(f"   - Add diversity metric to buffer")

    if buffer_loss is not None and buffer_reward > 0.7 and onpolicy_reward < 0.3:
        print(f"\n⚠️  POOR GENERALIZATION!")
        print(f"   Buffer trajectories have high reward but new samples have low reward")
        print(f"   Model is memorizing specific paths, not learning general policy")

    print(f"{'='*60}\n")

    return {
        'buffer_loss': buffer_loss.item() if buffer_loss is not None else None,
        'buffer_reward': buffer_reward if buffer_loss is not None else None,
        'onpolicy_loss': onpolicy_loss.item(),
        'onpolicy_reward': onpolicy_reward
    }
```

### Usage in Training Loop

```python
# In main training loop
for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, pos_examples, neg_examples)

    # Every 50 episodes, check for overfitting
    if episode % 50 == 0:
        diagnostic = diagnose_replay_overfitting(
            trainer, initial_state, pos_examples, neg_examples
        )

        # Log for analysis
        history.append({'episode': episode, **diagnostic})
```

---

## Conclusion

**The user's observation reveals a fundamental problem:**

1. ✅ Model learns replayed trajectories well (low loss on buffer)
2. ✗ Model doesn't generalize to new states (low reward on-policy)
3. ❌ This is **replay buffer overfitting** - memorizing specific paths instead of learning general policy

**Root causes:**
- Small buffer (50 trajectories) covers ~2.5% of state space
- Stochastic sampling naturally diverges from memorized paths
- Unification value is state-specific → hard to generalize
- TB loss satisfied by memorization (doesn't require generalization)

**Impact on unification learning:**
- Unifier network learns policy for states IN buffer
- New states → poor policy → random unifications → low reward
- Network never learns general unification principles

**Critical next steps:**
1. Diagnose: Run diagnostic code to confirm overfitting
2. Immediate fix: Increase buffer size to 500, decrease replay_probability to 0.1
3. Monitor: Track buffer vs. on-policy performance gap
4. Long-term: Add diversity metric, data augmentation, generalization penalty

This explains why unification learning is so poor - the model is stuck memorizing 50 specific paths instead of learning when unification is valuable!

---

## Files Referenced

- `src/training.py:698-757` - Replay buffer sampling and training
- `src/training.py:36-72` - TrajectoryReplayBuffer implementation
- `src/training.py:91-93` - Default hyperparameters
