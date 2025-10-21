# Embedding Trajectory Visualization Guide

## Overview

The embedding trajectory visualizer shows how the encoder's representation of rules **evolves step-by-step** as they are constructed during GFlowNet sampling. This is crucial for understanding whether the encoder is learning meaningful representations.

---

## What Gets Visualized

The system samples multiple trajectories and tracks the embedding at each construction step:

```
Step 0: grandparent(X0, X1) :-
        Embedding: [0.12, -0.45, 0.78, ...]

Step 1: grandparent(X0, X1) :- parent(X0, X2)
        Embedding: [0.15, -0.42, 0.81, ...]  ← Changed!

Step 2: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)
        Embedding: [0.23, -0.38, 0.92, ...]  ← Changed again!
```

---

## The 6 Visualizations

### 1. **PCA 2D Trajectory Paths** (`embedding_trajectory_pca_2d.png`)

**What it shows:**
- Each colored line = one trajectory
- Points = individual steps
- Square = start, Star = end
- Numbers on points = step index

**How to interpret:**

✅ **GOOD SIGNS:**
- Trajectories fan out and explore different regions
- High-reward trajectories (shown in legend) separate from low-reward
- Clear progression from start to end points

❌ **BAD SIGNS:**
- All trajectories cluster together in one small region
- No visible separation between high and low reward paths
- Embeddings barely move from initial state

**What this tells you:**
- Whether the encoder creates diverse representations for different rules
- Whether construction steps lead to meaningful changes in embedding space

---

### 2. **PCA 3D Trajectory Paths** (`embedding_trajectory_pca_3d.png`)

**What it shows:**
- Same as 2D but with an additional dimension
- Rotate view to see different angles

**Why it's useful:**
- Sometimes trajectories overlap in 2D but separate in 3D
- Gives fuller picture of embedding space structure

**How to interpret:**
- Same principles as 2D visualization
- Look for vertical separation if horizontal looks cluttered

---

### 3. **t-SNE Trajectory Paths** (`embedding_trajectory_tsne.png`)

**What it shows:**
- Non-linear dimensionality reduction (better at preserving local structure)
- Similar layout to PCA plots

**Key difference from PCA:**
- t-SNE emphasizes **local neighborhoods**
- Better at showing clusters
- Distances between far-apart points are less meaningful

**How to interpret:**

✅ **GOOD:**
- Clear clusters of similar states
- Trajectories form coherent paths (not scattered randomly)

❌ **BAD:**
- Everything scattered uniformly (no structure)
- Trajectories jump around randomly

---

### 4. **Similarity Heatmap** (`embedding_trajectory_similarity.png`)

**What it shows:**
- Every step of every trajectory compared to every other step
- Blue lines separate different trajectories
- Color: Green = similar, Red = different

**How to interpret:**

✅ **GOOD STRUCTURE:**
```
Trajectory 1:  [green green green |              ]
               [green green green |              ]
               [green green green |              ]
               [-------------------|--------------|]
Trajectory 2:  [                  | red  red  red]
               [                  | red  red  red]

→ Block diagonal structure
→ Steps within a trajectory are similar
→ Different trajectories are different
```

❌ **BAD STRUCTURE:**
```
All entries are uniformly green (>0.95 similarity)
→ All states look the same to the encoder!
```

**What this tells you:**
- Whether the encoder distinguishes between different rules
- Whether sequential steps in a trajectory are related
- Overall embedding diversity

---

### 5. **Distance Evolution** (`embedding_trajectory_distance_evolution.png`)

**What it shows:**
Two plots:

**Plot A: Distance from Initial State**
- How far the embedding has moved from the starting point
- Each line = one trajectory

**Plot B: Distance from Previous Step**
- How much each action changes the embedding
- Spikes = actions that significantly change representation

**How to interpret:**

✅ **GOOD PATTERNS:**

Plot A (Distance from initial):
- Lines steadily increase (moving away from initial state)
- High-reward trajectories reach farther distances
- Different trajectories reach different distances

Plot B (Step distances):
- Non-zero values (actions are changing embeddings)
- Variable distances (different actions have different effects)
- Larger steps early, smaller refinements later

❌ **BAD PATTERNS:**

Plot A:
- All lines stay near zero (embeddings not changing)
- All trajectories reach same distance regardless of reward

Plot B:
- All values near zero (actions don't change embeddings)
- Uniform distances (all actions have identical effect)

**Example interpretation:**
```
If Plot B shows:
Step 1: Distance = 0.5   ← Added first atom (big change)
Step 2: Distance = 0.4   ← Added second atom (big change)
Step 3: Distance = 0.1   ← Unified variables (small change)

→ This is GOOD! Different actions have different impacts.
```

---

### 6. **Action-Colored Visualization** (`embedding_trajectory_by_action.png`)

**What it shows:**
- All steps from all trajectories in PCA space
- Color = action type that led to this state
  - Blue = ADD_ATOM
  - Green = UNIFY_VARIABLES
  - Red = TERMINATE
  - Gold = FINAL state

**How to interpret:**

✅ **GOOD CLUSTERING:**
```
      Blue points        Green points
      (ADD_ATOM)         (UNIFY)
         ●●●               ▲▲▲
         ●●●               ▲▲▲
         ●●●               ▲▲▲

     Red points
     (TERMINATE)
        ■■■

→ Different action types occupy different regions
→ Encoder recognizes structural differences
```

❌ **BAD OVERLAP:**
```
All colors completely overlapped in same region
→ Encoder can't distinguish what action was taken
→ All states look the same regardless of structure
```

**What this tells you:**
- Whether the encoder learns action-specific patterns
- Whether different construction strategies are distinguishable

---

## Comprehensive Diagnosis

### Scenario 1: Healthy Embedding Learning ✅

**What you see:**
- PCA/t-SNE: Trajectories fan out, clear paths
- Similarity: Block diagonal structure
- Distance evolution: Steady increase from initial, variable step sizes
- Action clustering: Clear separation between action types

**Interpretation:**
- ✅ Encoder is learning meaningful representations
- ✅ Different rules occupy different embedding regions
- ✅ Construction steps lead to meaningful changes
- ✅ GFlowNet has useful state information to make decisions

---

### Scenario 2: Embedding Collapse ❌

**What you see:**
- PCA/t-SNE: All trajectories cluster in tiny region
- Similarity: Uniform high similarity (>0.95 everywhere)
- Distance evolution: All distances near zero
- Action clustering: All colors completely overlapped

**Interpretation:**
- ❌ Encoder producing nearly identical embeddings for everything
- ❌ GFlowNet receives no useful state information
- ❌ Policy cannot differentiate between good and bad partial rules

**Solution:**
- Apply contrastive pre-training (see `contrastive_pretraining.py`)
- Use improved architecture with edge features (see `improved_graph_encoder.py`)
- Add auxiliary contrastive loss during training

---

### Scenario 3: Partial Success 🟡

**What you see:**
- PCA/t-SNE: Some separation but trajectories still cluster
- Similarity: Some block structure but high baseline similarity (>0.80)
- Distance evolution: Small but non-zero changes
- Action clustering: Slight separation but much overlap

**Interpretation:**
- 🟡 Encoder learning something but not enough
- 🟡 May be converging to suboptimal representations
- 🟡 Needs more training or better architecture

**Solution:**
- Continue pre-training for more epochs
- Increase number of base rules for pre-training
- Consider temperature tuning in contrastive loss

---

## Connecting to GFlowNet Performance

### High Embedding Diversity → Good GFlowNet Performance

If embeddings are diverse and meaningful:
1. Policy can distinguish good partial rules from bad ones
2. High-reward trajectories navigate to different regions than low-reward
3. Action selection is informed by actual state quality
4. Training converges to good solutions

### Low Embedding Diversity → Poor GFlowNet Performance

If embeddings are collapsed:
1. Policy receives same input for all states
2. Cannot learn which actions lead to rewards
3. Exploration is essentially random
4. Training fails to find good rules (zero reward problem)

---

## Practical Usage

### In Jupyter Notebook:

```python
from visualize_embedding_trajectories import EmbeddingTrajectoryVisualizer

# Create visualizer
viz = EmbeddingTrajectoryVisualizer(trainer, graph_constructor, state_encoder)

# Sample trajectories and collect embeddings
trajectories_data = viz.collect_trajectory_embeddings(
    initial_state,
    positive_examples,
    negative_examples,
    num_trajectories=10,
    max_steps=5
)

# Generate all visualizations
viz.visualize_all(trajectories_data, output_dir='results')
```

### Interpreting Results:

1. **Start with PCA 2D** - Quick overview of trajectory diversity
2. **Check similarity heatmap** - Confirm embeddings are actually different
3. **Examine distance evolution** - See if actions change representations
4. **Review action clustering** - Verify encoder learns action patterns

---

## When to Run This Analysis

### Timing:

1. **After contrastive pre-training** - Verify pre-training worked
2. **During GFlowNet training** - Monitor if embeddings stay diverse
3. **After full training** - Understand what was learned

### Red Flags to Watch For:

⚠️ **During training:**
- Embedding diversity decreases over time (collapse)
- High and low reward trajectories become indistinguishable
- Step distances approach zero

⚠️ **After training:**
- Similarity > 0.95 across all steps (pre-training didn't work)
- No separation between action types (encoder learned nothing)
- Reward doesn't correlate with trajectory path

---

## Advanced Analysis

### Comparing Pre-training vs No Pre-training:

Run the visualization twice:
1. With untrained encoder (before pre-training)
2. With pre-trained encoder (after pre-training)

**Expected improvement:**
- Similarity drops from ~0.99 to <0.85
- Trajectories spread from clustered → fanned out
- Action clusters become visible

### Monitoring During Training:

Sample trajectories every N episodes:
```python
if episode % 500 == 0:
    viz.visualize_all(...)
```

**Watch for:**
- Embedding collapse (diversity decreases)
- Trajectory convergence (all paths become similar)
- Loss of action differentiation

---

## FAQ

**Q: Why do I need 6 different visualizations?**

A: Each reveals different aspects:
- PCA/t-SNE: Overall structure and diversity
- Similarity heatmap: Quantitative confirmation
- Distance evolution: Temporal dynamics
- Action clustering: Pattern learning verification

**Q: How many trajectories should I sample?**

A:
- 10 trajectories: Quick check
- 20-50 trajectories: Thorough analysis
- 100+ trajectories: Statistical confidence

**Q: What if t-SNE looks different from PCA?**

A: Normal! They emphasize different properties:
- PCA: Global structure, linear relationships
- t-SNE: Local structure, clusters

Both should show diversity if encoder is working.

**Q: All my trajectories are low reward. Is that okay?**

A: For this visualization, diversity matters more than reward:
- Low reward but diverse paths: Encoder is working, policy needs training
- Low reward and identical paths: Both encoder and policy need work

---

## Summary Checklist

After running these visualizations, check:

- [ ] Trajectories occupy > 50% of the visible space (not clustered in corner)
- [ ] Similarity heatmap shows block diagonal structure
- [ ] Distance from initial state increases over steps
- [ ] Different action types cluster in different regions
- [ ] High and low reward trajectories are visually distinguishable

If all boxes checked: ✅ Encoder is learning well!
If <3 boxes checked: ⚠️ Consider pre-training or architecture improvements
If 0 boxes checked: ❌ Encoder has collapsed, apply fixes immediately

---

## References

See also:
- `EMBEDDING_IMPROVEMENT_GUIDE.md` - Fixes for embedding problems
- `contrastive_pretraining.py` - Pre-training implementation
- `improved_graph_encoder.py` - Better architecture options
