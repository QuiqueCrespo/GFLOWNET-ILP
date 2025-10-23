# Flow Visualization Guide

This guide shows how to use the `FlowVisualizer` to track and visualize how the model's predicted flow values evolve during training.

---

## What Does This Visualize?

The flow visualizer tracks **predicted flow values** `log F(s)` for states at a fixed distance (e.g., 4 steps) from the initial state, and shows:

1. **How flow predictions change over training** - Does the model learn to assign higher flow to better states?
2. **Correlation between flow and reward** - Does `log F(s)` correlate with actual reward `R(s)`?
3. **Flow distribution evolution** - How does the distribution of predicted flows change?
4. **Individual state trajectories** - How does the flow for specific states evolve?

### Why This Matters

In GFlowNet theory:
- **Terminal states**: `F(s_terminal) = R(s_terminal)` (flow equals reward)
- **Non-terminal states**: `F(s) = sum over children F(s')` (flow is sum of child flows)

If the model is learning correctly:
- States leading to high rewards should have high flow
- Flow predictions should improve over training
- Correlation between `log F(s)` and `R(s)` should increase

---

## Usage in Notebook

### Step 1: Import and Initialize

Add this after creating the trainer:

```python
from src.flow_visualization import FlowVisualizer

# Initialize visualizer
flow_viz = FlowVisualizer(
    trainer=trainer,
    target_predicate='grandparent',
    arity=2,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    positive_examples=positive_examples,
    negative_examples=negative_examples,
    max_depth=4,  # Number of steps from origin
    output_dir=f"{visualizer.run_dir}/flow_viz"
)

print(f"\\nFlow Visualizer: Tracking {len(flow_viz.target_states)} states at depth {flow_viz.max_depth}")
```

### Step 2: Record Snapshots During Training

Modify your training loop to record snapshots periodically:

```python
# Training loop
num_episodes = config['num_episodes']
initial_state = get_initial_state('grandparent', 2)

# Record initial flow predictions (before training)
flow_viz.record_snapshot(episode=0)

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

    # ... existing code for recording metrics ...

    # Record flow snapshot every 100 episodes
    if episode % 100 == 0 and episode > 0:
        flow_viz.record_snapshot(episode=episode)
        print(f"  [Flow snapshot recorded at episode {episode}]")

# Record final snapshot
flow_viz.record_snapshot(episode=num_episodes)
```

### Step 3: Generate Visualizations After Training

Add this after training completes:

```python
print("\\n" + "="*80)
print("GENERATING FLOW VISUALIZATIONS")
print("="*80)

# Generate main flow evolution plot
flow_viz.plot_flow_evolution()

# Generate individual state trajectories
flow_viz.plot_state_trajectories(num_states=15)

# Generate text report
flow_viz.generate_report()

print("\\nFlow visualizations complete!")
```

---

## Understanding the Visualizations

### Plot 1: Flow Evolution for Top States

**What it shows**: How `log F(s)` changes over training for the 5 best states (by final reward).

**What to look for**:
- ✅ **Good**: Flow increases over training for high-reward states
- ❌ **Bad**: Flow is flat or decreases for high-reward states
- ❌ **Bad**: All states have similar flow regardless of reward

**Interpretation**:
- If flows are increasing → Model is learning to assign higher flow to good states
- If flows are flat → Model is not learning flow correctly (possible issues with loss or gradients)

### Plot 2: Flow Evolution for Bottom States

**What it shows**: How `log F(s)` changes for low-reward states.

**What to look for**:
- ✅ **Good**: Flow decreases or stays low for bad states
- ❌ **Bad**: Flow increases for bad states

### Plots 3-5: Flow vs Reward Correlation

**What it shows**: Scatter plots of `log F(s)` vs `R(s)` at early, mid, and final training.

**What to look for**:
- ✅ **Good**: Positive correlation that increases over time
- ✅ **Good**: Tighter clustering around a positive trend line at final
- ❌ **Bad**: No correlation or negative correlation
- ❌ **Bad**: Correlation decreases over time

**Interpretation**:
- Correlation > 0.5 → Model has learned reasonable flow prediction
- Correlation < 0.3 → Model struggles to predict flow correctly
- Increasing correlation → Learning is working

### Plot 6: Correlation Evolution

**What it shows**: How the correlation coefficient changes over training episodes.

**What to look for**:
- ✅ **Good**: Steady increase from ~0 to >0.5
- ✅ **Good**: Stabilization at high correlation (>0.6)
- ❌ **Bad**: Oscillating or decreasing correlation
- ❌ **Bad**: Stuck at low correlation (<0.3)

**Critical Diagnostic**: This plot tells you if flow learning is working at all!

### Plot 7: Flow Distribution Evolution

**What it shows**: Histogram of `log F(s)` values at early, mid, and final training.

**What to look for**:
- ✅ **Good**: Distribution spreads out over training (model differentiates states)
- ✅ **Good**: Final distribution is multimodal (different flow levels for different state qualities)
- ❌ **Bad**: Distribution stays narrow (model assigns similar flow to all states)
- ❌ **Bad**: Distribution doesn't change much

### Plot 8: Mean Flow by Reward Bucket

**What it shows**: Average `log F(s)` for states grouped by reward ranges.

**What to look for**:
- ✅ **Good**: Monotonically increasing bars (higher reward → higher flow)
- ✅ **Good**: Clear separation between reward buckets
- ❌ **Bad**: Flat bars (no differentiation)
- ❌ **Bad**: Non-monotonic (e.g., 0.6-0.8 has lower flow than 0.4-0.6)

**Ideal pattern**: Each bar should be noticeably higher than the previous one.

---

## Example: Interpreting Results

### Scenario 1: Good Learning

```
Correlation Evolution:
  Episode 0:    0.05  (random initialization)
  Episode 1000: 0.32  (starting to learn)
  Episode 5000: 0.58  (good progress)
  Episode 10000: 0.72 (strong correlation)

Flow vs Reward (Final): Clear positive trend, tight clustering

Mean Flow by Reward Bucket:
  0.0-0.2: -8.5
  0.2-0.4: -6.2
  0.4-0.6: -4.1
  0.6-0.8: -2.3
  0.8-1.0: -0.5  ← Monotonically increasing ✓
```

**Diagnosis**: Flow learning is working well! The model correctly assigns higher flow to better states.

### Scenario 2: Poor Learning

```
Correlation Evolution:
  Episode 0:    0.02
  Episode 1000: 0.08
  Episode 5000: 0.12
  Episode 10000: 0.15  ← Barely improving

Flow vs Reward (Final): Random scatter, no visible trend

Mean Flow by Reward Bucket:
  0.0-0.2: -5.1
  0.2-0.4: -4.9
  0.4-0.6: -5.2
  0.6-0.8: -5.0
  0.8-1.0: -4.8  ← All similar, no pattern
```

**Diagnosis**: Flow learning is NOT working. Possible causes:
1. log_Z is compensating instead of flow network learning (see PIPELINE_ANALYSIS.md Bug #5)
2. Gradients not flowing to forward_flow network
3. Learning rate too low for forward_flow network
4. Reward scaling creating numerical issues

**Debug steps**:
```python
# Check if forward_flow network has gradients
for name, param in trainer.gflownet.forward_flow_net.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT")

# Check log_Z value
print(f"log_Z: {trainer.log_Z.item():.4f}")
# If log_Z > 10.0, it's probably compensating
```

### Scenario 3: Learning Then Collapsing

```
Correlation Evolution:
  Episode 0:    0.03
  Episode 2000: 0.55  ← Good progress
  Episode 5000: 0.42  ← Starting to decline
  Episode 10000: 0.28 ← Collapsed!
```

**Diagnosis**: Model was learning but then collapsed. Possible causes:
1. Replay buffer overfitting (only replaying same good trajectories)
2. Catastrophic forgetting
3. Learning rate too high, causing instability

**Fixes**:
- Reduce replay_probability
- Add gradient clipping
- Reduce learning rate

---

## Advanced Usage

### Custom Depth

To explore different depths:

```python
# Initialize multiple visualizers
flow_viz_depth2 = FlowVisualizer(..., max_depth=2, output_dir="flow_viz_depth2")
flow_viz_depth4 = FlowVisualizer(..., max_depth=4, output_dir="flow_viz_depth4")
flow_viz_depth6 = FlowVisualizer(..., max_depth=6, output_dir="flow_viz_depth6")

# Record all during training
for episode in range(num_episodes):
    # ... training step ...

    if episode % 100 == 0:
        flow_viz_depth2.record_snapshot(episode)
        flow_viz_depth4.record_snapshot(episode)
        flow_viz_depth6.record_snapshot(episode)
```

**Why multiple depths?**:
- **Depth 2**: Close to origin, fewer states, easier to learn
- **Depth 4**: Mid-range, good balance
- **Depth 6**: Far from origin, many states, harder to learn

You can compare correlation evolution across depths to see if the model learns flow better for nearby vs distant states.

### Fine-Grained Snapshots

For detailed analysis:

```python
# Record every 10 episodes for first 1000 episodes
if episode <= 1000 and episode % 10 == 0:
    flow_viz.record_snapshot(episode)

# Then every 100 for the rest
elif episode % 100 == 0:
    flow_viz.record_snapshot(episode)
```

---

## Computational Cost

### Memory
- Stores flow/reward for all states at depth N
- For depth 4 with 2 predicates: ~50-200 states
- Per snapshot: ~2KB
- 100 snapshots: ~200KB (negligible)

### Time
- Per snapshot: ~0.5-2 seconds (depends on number of states)
- 100 snapshots during 10k episode training: ~1-3 minutes total
- Plotting: ~5-10 seconds

**Recommendation**: Record every 100 episodes (not too frequent).

---

## Troubleshooting

### Issue: "Need at least 2 snapshots to plot"

**Cause**: You only called `record_snapshot()` once or not at all.

**Fix**: Call `record_snapshot()` at multiple points during training:
```python
flow_viz.record_snapshot(episode=0)  # Before training
# ... training loop with periodic snapshots ...
flow_viz.record_snapshot(episode=num_episodes)  # After training
```

### Issue: "Warning: Limiting states at depth N to 100"

**Cause**: Too many states at the specified depth (combinatorial explosion).

**Fix**: Either:
- Use a smaller depth (e.g., 3 instead of 4)
- Or accept that we're sampling 100 representative states

### Issue: Visualizations show no learning

**Possible causes**:
1. Flow network not getting gradients
2. log_Z compensating
3. Learning rate issues
4. Reward scaling issues

**Debug**:
```python
# Check log_Z value
print(f"log_Z: {trainer.log_Z.item():.4f}")

# Check forward_flow gradients
for param in trainer.gflownet.forward_flow_net.parameters():
    if param.grad is not None:
        print(f"Flow network grad norm: {param.grad.norm():.6f}")
```

---

## Summary

The Flow Visualizer provides critical insights into whether your GFlowNet is learning correctly:

✅ **Use it to diagnose**:
- Is flow learning working?
- Are flows correlating with rewards?
- Is the model differentiating between good and bad states?

✅ **Look for**:
- Increasing correlation over time (target: >0.5)
- Higher flow for higher-reward states
- Monotonic relationship in reward buckets

❌ **Red flags**:
- Flat or decreasing correlation
- No differentiation between states
- Random scatter in flow vs reward plots

This visualization is essential for understanding if the core GFlowNet learning is working!
