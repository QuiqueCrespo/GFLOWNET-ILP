# Policy Convergence Visualization Guide

## What This Shows

The policy visualizer tracks how **forward** and **backward** policies evolve and converge during training.

### Forward Policy (P_F)
- **Strategist**: P_F(action_type | state) → {ADD_ATOM, UNIFY_VARIABLES, TERMINATE}
- **Atom Adder**: P_F(predicate | ADD_ATOM) → which predicate to add
- **Variable Unifier**: P_F(pair | UNIFY_VARIABLES) → which variable pair to unify

### Backward Policy (P_B)
- **Backward Strategist**: P_B(action_type | state) → {ADD_ATOM, UNIFY_VARIABLES}
- Used in computing Trajectory Balance loss

---

## Quick Integration

### 1. Initialize
```python
from src.policy_convergence_visualization import PolicyConvergenceVisualizer

policy_viz = PolicyConvergenceVisualizer(
    trainer=trainer,
    target_predicate='grandparent',
    arity=2,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    output_dir=f"{visualizer.run_dir}/policy_viz"
)
```

### 2. Record During Training
```python
# Before training
policy_viz.record_snapshot(episode=0)

# During training (every 100 episodes)
if episode % 100 == 0:
    policy_viz.record_snapshot(episode=episode)

# After training
policy_viz.record_snapshot(episode=num_episodes)
```

### 3. Visualize
```python
policy_viz.plot_strategist_convergence()
policy_viz.plot_backward_strategist_convergence()
policy_viz.plot_policy_consistency()
policy_viz.plot_atom_adder_convergence()
policy_viz.plot_variable_unifier_entropy()
policy_viz.plot_comprehensive_dashboard()
policy_viz.generate_report()
```

---

## Interpreting the Visualizations

### 1. Forward Strategist Convergence

**What it shows**: How P_F(action_type | state) evolves over training.

**What to look for**:

✅ **Good signs**:
- Policy becomes **more deterministic** (one action dominates)
- For states with body atoms: P(TERMINATE) increases if state is good
- For initial state: P(ADD_ATOM) high, P(UNIFY) low (after masking fix)
- Smooth convergence (not oscillating)

❌ **Bad signs**:
- Policy stays **uniform** (0.33, 0.33, 0.33) throughout training
- Oscillating wildly (no convergence)
- P(UNIFY) high when only bad pairs available

**Example (Good)**:
```
Episode 0:    P(ADD)=0.33, P(UNIFY)=0.33, P(TERM)=0.33  (random)
Episode 5000: P(ADD)=0.70, P(UNIFY)=0.15, P(TERM)=0.15  (learning)
Episode 10000: P(ADD)=0.85, P(UNIFY)=0.05, P(TERM)=0.10 (converged)
```

### 2. Backward Strategist Convergence

**What it shows**: How P_B(action_type | state) evolves.

**What to look for**:

✅ **Good signs**:
- Similar pattern to forward policy
- Converges to similar distribution as forward (consistency)

❌ **Bad signs**:
- Very different from forward policy (divergence)
- Not learning at all (stays uniform)

### 3. Policy Consistency (Forward vs Backward)

**What it shows**: Side-by-side comparison of P_F and P_B with divergence.

**Critical metric**: `|P_F - P_B|` (absolute difference)

✅ **Good signs**:
- Divergence **decreases** over training
- Final divergence < 0.1 (policies agree)
- Purple line (divergence) trends downward

❌ **Bad signs**:
- Divergence **increases** or stays high
- Final divergence > 0.2 (policies disagree)
- Divergence oscillates

**Why this matters**:
- Trajectory Balance assumes consistent forward/backward policies
- High divergence → biased gradients → poor learning

**Example (Good)**:
```
Episode 0:    |P_F(ADD) - P_B(ADD)| = 0.15  (random initialization)
Episode 5000: |P_F(ADD) - P_B(ADD)| = 0.08  (converging)
Episode 10000: |P_F(ADD) - P_B(ADD)| = 0.03 (consistent!)
```

**Example (Bad)**:
```
Episode 0:    |P_F(ADD) - P_B(ADD)| = 0.12
Episode 5000: |P_F(ADD) - P_B(ADD)| = 0.18  (diverging!)
Episode 10000: |P_F(ADD) - P_B(ADD)| = 0.25 (very inconsistent)
```

### 4. Atom Adder Convergence

**What it shows**: P_F(predicate | ADD_ATOM) for each predicate.

**What to look for**:

✅ **Good signs**:
- If one predicate is better, its probability increases
- Converges to deterministic choice (one predicate ~0.9+)

❌ **Bad signs**:
- Stays uniform (no learning)
- Wrong predicate gets high probability

**Example**:
If `parent` is the only useful predicate:
```
Episode 0:    P(parent)=0.50 (assuming 2 predicates)
Episode 5000: P(parent)=0.75 (learning)
Episode 10000: P(parent)=0.95 (converged to correct choice)
```

### 5. Variable Unifier Entropy

**What it shows**: Entropy of P_F(pair | UNIFY_VARIABLES) distribution.

**Entropy interpretation**:
- **High entropy** (e.g., 2.0+): Uniform distribution, exploring many pairs
- **Low entropy** (e.g., 0.5): Deterministic, prefers specific pairs

✅ **Good signs**:
- Entropy **decreases** over training (becoming more deterministic)
- Final entropy < 1.0 (strong preferences)

❌ **Bad signs**:
- Entropy stays high (not learning preferences)
- Entropy increases (becoming more random)

**Calculation**:
```
H = -Σ p_i * log(p_i)

Uniform over 6 pairs: H ≈ 1.79
Strong preference (0.7, 0.1, ...): H ≈ 0.92
Very deterministic (0.95, 0.01, ...): H ≈ 0.20
```

### 6. Comprehensive Dashboard

**What it shows**: All-in-one view with 6 plots.

**Quick diagnostic**:
1. Top-left: Forward strategist → should converge
2. Top-middle: Backward strategist → should match forward
3. Top-right: Divergence → should decrease to <0.1
4. Middle: Atom adder → should pick best predicate
5. Middle-right: Unifier entropy → should decrease
6. Bottom: Stacked area → visual confirmation of convergence

---

## Common Patterns

### Pattern 1: Healthy Convergence

```
Forward Strategist:
  - Becomes more deterministic
  - Settles on good actions

Backward Strategist:
  - Mirrors forward policy
  - Divergence < 0.1

Atom Adder:
  - Learns predicate preferences
  - Converges to best choice

Variable Unifier:
  - Entropy decreases
  - Learns to avoid bad pairs
```

**Diagnosis**: ✅ Model is learning correctly!

### Pattern 2: No Learning

```
Forward Strategist:
  - Stays uniform or random
  - No convergence

Backward Strategist:
  - Also random

Policy Consistency:
  - High divergence (>0.2)

Entropy:
  - Stays high
```

**Diagnosis**: ❌ Model is NOT learning!

**Possible causes**:
1. log_Z is compensating (check if log_Z > 10)
2. Gradients not flowing (check grad norms)
3. Learning rate too low
4. Reward scaling issues

### Pattern 3: Forward Learns, Backward Doesn't

```
Forward Strategist:
  - Converges nicely

Backward Strategist:
  - Stays random or changes slowly

Policy Consistency:
  - High divergence
```

**Diagnosis**: ⚠️ Backward policy not learning properly

**Possible causes**:
1. Backward policy has separate learning rate (should match forward)
2. Backward policy architecture issues
3. Not enough gradient signal

**Fix**: Check optimizer includes backward policy parameters

### Pattern 4: Oscillation

```
All policies:
  - Keep changing back and forth
  - No stable convergence
```

**Diagnosis**: ⚠️ Training instability

**Possible causes**:
1. Learning rate too high
2. Replay buffer causing issues
3. Reward scaling creating extreme gradients

**Fix**: Reduce learning rate, reduce reward_scale_alpha

---

## Integration with Other Diagnostics

### Combine with Flow Visualization

```python
# Check both flow and policy convergence
flow_viz.plot_flow_evolution()
policy_viz.plot_comprehensive_dashboard()
```

**Look for correlation**:
- If flow learning is good (correlation > 0.5) → policy should also converge
- If flow is bad but policy converges → log_Z is compensating
- If both are bad → fundamental learning issue

### Combine with Reward Tracking

```python
# Plot rewards alongside policy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(episodes, rewards)
plt.title('Rewards')

plt.subplot(1, 2, 2)
# Policy convergence plot
```

**Look for correlation**:
- Rewards increasing + Policy converging = ✅ Good!
- Rewards flat + Policy converging = ⚠️ Converging to suboptimal
- Rewards increasing + Policy not converging = ⚠️ Luck, not learning

---

## Debugging Checklist

If policies are not converging:

- [ ] Check log_Z value (should be -5 to +5)
- [ ] Check gradient norms for forward/backward policies
- [ ] Check if optimizer includes all policy parameters
- [ ] Check learning rate (try 1e-4)
- [ ] Check reward_scale_alpha (try 1.0 instead of 10.0)
- [ ] Check replay buffer (is it diluting gradients?)
- [ ] Check if masking is correct (UNIFY blocked when appropriate)
- [ ] Run flow visualization to see if flow learning works

---

## Quick Reference

| Plot | Good Sign | Bad Sign |
|------|-----------|----------|
| Forward Strategist | Converges to deterministic | Stays uniform |
| Backward Strategist | Mirrors forward | Very different |
| Consistency | Divergence < 0.1 | Divergence > 0.2 |
| Atom Adder | Picks best predicate | Stays uniform |
| Unifier Entropy | Decreases to < 1.0 | Stays high > 1.5 |

---

## Files Generated

- `forward_strategist_convergence.png` - Forward policy evolution
- `backward_strategist_convergence.png` - Backward policy evolution
- `policy_consistency.png` - Forward vs Backward comparison
- `atom_adder_convergence.png` - Predicate selection evolution
- `variable_unifier_entropy.png` - Pair selection entropy
- `policy_dashboard.png` - All-in-one overview
- `policy_convergence_report.txt` - Text summary

---

## Summary

Policy convergence visualization helps you understand:

1. **Is the model learning?** (policies converging vs staying random)
2. **Are policies consistent?** (forward/backward divergence)
3. **What actions is it learning?** (which actions dominate)
4. **Is learning stable?** (smooth convergence vs oscillation)

This is essential for debugging GFlowNet training and ensuring the model learns correctly!
