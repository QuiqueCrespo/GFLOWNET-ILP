# Flow Visualization - Quick Start

## What Is This?

A comprehensive tool to visualize how your GFlowNet learns to predict **flow values** for states during training.

**Flow** in GFlowNet: `F(s)` represents the "mass" flowing through state `s`. For terminal states, `F(s_terminal) = R(s_terminal)` (flow equals reward). The model must learn to correctly assign flow to all states.

---

## Quick Integration (3 Steps)

### 1. Initialize (before training)

```python
from src.flow_visualization import FlowVisualizer

flow_viz = FlowVisualizer(
    trainer=trainer,
    target_predicate='grandparent',
    arity=2,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    positive_examples=positive_examples,
    negative_examples=negative_examples,
    max_depth=4,
    output_dir=f"{visualizer.run_dir}/flow_viz"
)
```

### 2. Record during training

```python
# Before training starts
flow_viz.record_snapshot(episode=0)

# During training loop
for episode in range(num_episodes):
    metrics = trainer.train_step(...)

    # Every 100 episodes
    if episode % 100 == 0 and episode > 0:
        flow_viz.record_snapshot(episode=episode)

# After training ends
flow_viz.record_snapshot(episode=num_episodes)
```

### 3. Visualize (after training)

```python
flow_viz.plot_flow_evolution()
flow_viz.plot_state_trajectories(num_states=15)
flow_viz.generate_report()
```

---

## What You Get

### Main Visualization: `flow_evolution_depth4.png`

8 subplots showing:
1. **Top states flow evolution** - Do high-reward states get high flow?
2. **Bottom states flow evolution** - Do low-reward states stay low?
3. **Early correlation** - Flow vs Reward at start of training
4. **Mid correlation** - Flow vs Reward at middle
5. **Final correlation** - Flow vs Reward at end (should be strong!)
6. **Correlation over time** - Is it increasing? (target: >0.5)
7. **Flow distribution** - Is model differentiating states?
8. **Flow by reward bucket** - Monotonic increase? (good sign!)

### State Trajectories: `state_trajectories_depth4.png`

Individual flow trajectories for top 15 states showing exactly how each state's predicted flow changes over training.

### Text Report: `flow_report_depth4.txt`

Detailed analysis:
- Correlation evolution table
- Top 10 states with their flow trajectories
- Bottom 5 states for comparison
- Summary statistics

---

## Interpreting Results

### ✅ Good Signs

- **Correlation increasing**: 0.05 → 0.35 → 0.58 → 0.72
- **High-reward states have high flow**: Clear separation in Plot 8
- **Correlation > 0.5 by end of training**
- **Flow distribution spreads out** over training (Plot 7)

### ❌ Warning Signs

- **Flat correlation**: Stays around 0.1-0.2 throughout training
- **No differentiation**: All states have similar flow
- **Correlation decreasing**: Was 0.5, dropped to 0.3
- **Random scatter**: No visible trend in Flow vs Reward plots

### 🔧 What To Do If Flow Learning Is Broken

**Check log_Z**:
```python
print(trainer.log_Z.item())  # Should be -5 to +5
# If > 10 or < -10: log_Z is compensating instead of learning
```

**Check gradients**:
```python
for name, param in trainer.gflownet.forward_flow_net.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.6f}")
# Should see non-zero gradients (>1e-5)
```

**Possible fixes**:
1. Use Detailed Balance instead of Trajectory Balance
2. Reduce `reward_scale_alpha` from 10.0 to 1.0
3. Reduce `log_Z` learning rate
4. Fix replay buffer (see PIPELINE_ANALYSIS.md Bug #1)

---

## Files Included

- **src/flow_visualization.py** - Main implementation
- **FLOW_VISUALIZATION_GUIDE.md** - Detailed documentation
- **notebook_flow_visualization.py** - Copy-paste cells for notebook

---

## Example Output

```
Flow Visualizer: Tracking 87 states at depth 4

Episode 0: Correlation = -0.03 (random)
Episode 1000: Correlation = +0.28 (learning)
Episode 5000: Correlation = +0.54 (good)
Episode 10000: Correlation = +0.71 (excellent!)

Top state: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)
  Reward: 0.9156
  Flow trajectory: -8.2 → -6.5 → -3.8 → -1.2 → -0.4
  ✓ Flow increased as expected!
```

---

## Performance Impact

- **Memory**: ~200KB for 100 snapshots (negligible)
- **Time**: ~1-2 seconds per snapshot
- **Total**: ~2-3 minutes for 100 snapshots over 10k episode training

---

## Why This Matters

**GFlowNet Theory**: The model must learn both:
1. **Policy** P_F(a|s) - which actions to take
2. **Flow** F(s) - how much "mass" flows through each state

If flow learning is broken:
- Loss can still decrease (via log_Z compensation)
- But policy won't learn correctly
- Model won't converge to optimal behavior

**This visualization tells you if flow learning is actually working!**

---

## Quick Reference

| Plot | What to Check | Good Sign | Bad Sign |
|------|---------------|-----------|----------|
| Correlation Evolution | Is it increasing? | >0.5 by end | <0.3 throughout |
| Flow by Reward Bucket | Monotonic? | Clear increase | Flat bars |
| Top States Flow | Increasing? | Steady rise | Flat or decreasing |
| Flow Distribution | Spreading out? | Multimodal final | Stays narrow |

---

## Next Steps

1. **Add to notebook** using `notebook_flow_visualization.py`
2. **Run training** with flow tracking
3. **Check correlation** - is it >0.5?
4. **If broken** - follow diagnostics in FLOW_VISUALIZATION_GUIDE.md

The flow visualization is essential for debugging GFlowNet training!
