# Visualization Line Style Fix Plan

## Problem
Multiple lines in plots have identical styles (color, linestyle, marker), making them impossible to distinguish.

## Files to Fix
1. `src/flow_visualization.py`
2. `src/policy_convergence_visualization.py`

---

## Unique Style Generator

Create a style generator that ensures no two lines have the same combination:

```python
# Define style components
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Tab10 colors
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+', '<', '>']

def get_unique_style(index):
    """
    Returns a unique combination of color, linestyle, and marker.
    Supports up to 480 unique combinations (10 colors × 4 linestyles × 12 markers).
    """
    color_idx = index % len(COLORS)
    style_idx = (index // len(COLORS)) % len(LINESTYLES)
    marker_idx = (index // (len(COLORS) * len(LINESTYLES))) % len(MARKERS)

    return {
        'color': COLORS[color_idx],
        'linestyle': LINESTYLES[style_idx],
        'marker': MARKERS[marker_idx],
        'linewidth': 2,
        'markersize': 5
    }
```

---

## Changes Needed

### src/flow_visualization.py

#### Fix 1: Lines 211-218 (Top 5 states evolution)
**Current:**
```python
for i, state_str in enumerate(top_10_states[:5]):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    state_abbrev = state_str[:40] + "..." if len(state_str) > 40 else state_str
    ax1.plot(episodes, flows, marker='o', label=f"R={reward:.3f}: {state_abbrev}",
            linewidth=2, markersize=4)
```

**Fixed:**
```python
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', '^', 'v', 'D']

for i, state_str in enumerate(top_10_states[:5]):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    state_abbrev = state_str[:40] + "..." if len(state_str) > 40 else state_str
    ax1.plot(episodes, flows,
            color=COLORS[i],
            marker=MARKERS[i],
            linestyle='-',
            label=f"R={reward:.3f}: {state_abbrev}",
            linewidth=2, markersize=4)
```

#### Fix 2: Lines 229-235 (Bottom 3 states)
**Current:**
```python
for i, state_str in enumerate(bottom_5_states[:3]):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    state_abbrev = state_str[:30] + "..." if len(state_str) > 30 else state_str
    ax2.plot(episodes, flows, marker='x', label=f"R={reward:.3f}",
            linewidth=2, markersize=4, linestyle='--')
```

**Fixed:**
```python
COLORS_BOTTOM = ['#d62728', '#8c564b', '#e377c2']
MARKERS_BOTTOM = ['x', '+', '*']
LINESTYLES_BOTTOM = ['--', '-.', ':']

for i, state_str in enumerate(bottom_5_states[:3]):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    state_abbrev = state_str[:30] + "..." if len(state_str) > 30 else state_str
    ax2.plot(episodes, flows,
            color=COLORS_BOTTOM[i],
            marker=MARKERS_BOTTOM[i],
            linestyle=LINESTYLES_BOTTOM[i],
            label=f"R={reward:.3f}",
            linewidth=2, markersize=4)
```

#### Fix 3: Lines 435-436 (State trajectories)
**Current:**
```python
for i, state_str in enumerate(top_states):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    # ... label creation ...
    ax.plot(episodes, flows, marker='o', label=label,
           linewidth=2, markersize=5, alpha=0.8)
```

**Fixed:**
```python
# Define color palette and markers for up to 10 states
COLORS_TRAJ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS_TRAJ = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
LINESTYLES_TRAJ = ['-', '--', '-.', ':', '-']  # Repeat solid after 4

for i, state_str in enumerate(top_states):
    flows = [snap['flows'][state_str] for snap in self.episode_snapshots]
    reward = final_rewards[state_str]
    # ... label creation ...

    # Get unique style for this line
    style_idx = i % len(COLORS_TRAJ)
    linestyle_idx = i % len(LINESTYLES_TRAJ)

    ax.plot(episodes, flows,
           color=COLORS_TRAJ[style_idx],
           marker=MARKERS_TRAJ[style_idx],
           linestyle=LINESTYLES_TRAJ[linestyle_idx],
           label=label,
           linewidth=2, markersize=5, alpha=0.8)
```

---

### src/policy_convergence_visualization.py

#### Fix 4: Lines 355-359 (Atom adder - multiple predicates)
**Current:**
```python
for pred in self.predicate_vocab:
    probs = [snap['forward'][state_name]['atom_adder'][pred]
            for snap in self.snapshots]
    ax.plot(episodes, probs, marker='o', linewidth=2, markersize=5,
           label=pred)
```

**Fixed:**
```python
COLORS_PRED = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS_PRED = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
LINESTYLES_PRED = ['-', '--', '-.', ':']

for idx, pred in enumerate(self.predicate_vocab):
    probs = [snap['forward'][state_name]['atom_adder'][pred]
            for snap in self.snapshots]

    # Unique style for each predicate
    color_idx = idx % len(COLORS_PRED)
    marker_idx = idx % len(MARKERS_PRED)
    linestyle_idx = (idx // len(COLORS_PRED)) % len(LINESTYLES_PRED)

    ax.plot(episodes, probs,
           color=COLORS_PRED[color_idx],
           marker=MARKERS_PRED[marker_idx],
           linestyle=LINESTYLES_PRED[linestyle_idx],
           linewidth=2, markersize=5,
           label=pred)
```

#### Fix 5: Lines 400-401 (Variable unifier entropy - multiple states)
**Current:**
```python
for state, state_name in self.test_states:
    # ... entropy extraction ...
    if any(e is not None for e in entropies):
        # ...
        ax.plot(valid_episodes, valid_entropies, marker='o', linewidth=2,
               markersize=5, label=state_name)
```

**Fixed:**
```python
COLORS_STATES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS_STATES = ['o', 's', '^', 'v']
LINESTYLES_STATES = ['-', '--', '-.', ':']

for idx, (state, state_name) in enumerate(self.test_states):
    # ... entropy extraction ...
    if any(e is not None for e in entropies):
        # ...
        ax.plot(valid_episodes, valid_entropies,
               color=COLORS_STATES[idx % len(COLORS_STATES)],
               marker=MARKERS_STATES[idx % len(MARKERS_STATES)],
               linestyle=LINESTYLES_STATES[idx % len(LINESTYLES_STATES)],
               linewidth=2,
               markersize=5,
               label=state_name)
```

#### Fix 6: Lines 482-483 (Dashboard atom adder)
**Current:**
```python
for pred in self.predicate_vocab:
    probs = [snap['forward'][state_name]['atom_adder'][pred] for snap in self.snapshots]
    ax4.plot(episodes, probs, 'o-', label=pred, linewidth=2, markersize=4)
```

**Fixed:**
```python
COLORS_DASH = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS_DASH = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

for idx, pred in enumerate(self.predicate_vocab):
    probs = [snap['forward'][state_name]['atom_adder'][pred] for snap in self.snapshots]

    ax4.plot(episodes, probs,
            color=COLORS_DASH[idx % len(COLORS_DASH)],
            marker=MARKERS_DASH[idx % len(MARKERS_DASH)],
            linestyle='-',
            label=pred,
            linewidth=2,
            markersize=4)
```

---

## Implementation Strategy

1. Define color/marker/linestyle palettes at the top of each function
2. Use modulo indexing to cycle through styles
3. Ensure each line gets a unique combination

## Testing

After fixes, verify:
1. All lines in each plot are visually distinguishable
2. Legend matches line styles correctly
3. Colors are colorblind-friendly (using standard matplotlib Tab10)
