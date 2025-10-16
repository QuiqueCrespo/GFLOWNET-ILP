# Reward Shaping Implementation Summary

## Overview

This document summarizes the implementation of reward shaping penalties to address pathological rule patterns in the GFlowNet-ILP system.

## Problem

The previous session identified that bad rules were getting stored in the replay buffer because exploration bonuses inflated their total rewards:

```
[0.9750] grandparent(X0, X5) :- parent(X0, X3), parent(X4, X5), parent(X6, X7).
[0.8000] grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0), parent(X6, X7).
```

**Root causes:**
1. **Disconnected variables**: Variables appear in body atoms that share NO connection with the head
   - Example: `parent(X6, X7)` in above rule - X6, X7 disconnected from rest
2. **Self-loops**: Same variable appears multiple times in one atom
   - Example: `parent(X0, X0)` when no self-parent relationships exist
3. **Exploration bonuses**: `total_reward = base_reward + (0.1 × trajectory_length)` inflates bad rules

## Solution: Structural Penalties

### Implementation

Modified `src/reward.py` to add configurable penalties:

```python
class RewardCalculator:
    def __init__(self, logic_engine: LogicEngine,
                 ...
                 disconnected_var_penalty: float = 0.2,
                 self_loop_penalty: float = 0.3):
```

### Disconnected Variable Detection

**Key improvement**: Detect *truly disconnected* variables using connected component analysis:

```python
def _count_disconnected_variables(self, theory):
    """
    Count variables in body atoms that share NO variables
    with the connected component containing the head.
    """
    # Build connected component starting from head variables
    connected_vars = set(rule.head.args)
    changed = True

    while changed:
        changed = False
        for atom in rule.body:
            atom_vars = set(atom.args)
            # If atom shares any variable with connected component,
            # add all its variables
            if atom_vars & connected_vars:
                new_vars = atom_vars - connected_vars
                if new_vars:
                    connected_vars.update(new_vars)
                    changed = True

    # Count variables NOT in connected component
    all_body_vars = set()
    for atom in rule.body:
        all_body_vars.update(atom.args)

    disconnected = all_body_vars - connected_vars
    return len(disconnected)
```

**Why this is better**: Chain variables (like X2 in `grandparent(X0,X1) :- parent(X0,X2), parent(X2,X1)`) are correctly identified as CONNECTED, not disconnected.

### Self-Loop Detection

```python
def _count_self_loops(self, theory):
    """Count atoms where the same variable appears multiple times."""
    count = 0
    for rule in theory:
        # Check head
        if len(set(rule.head.args)) < len(rule.head.args):
            count += 1

        # Check body atoms
        for atom in rule.body:
            if len(set(atom.args)) < len(atom.args):
                count += 1

    return count
```

### Penalty Application

```python
def calculate_reward(self, theory, positive_examples, negative_examples):
    # ... calculate accuracy, simplicity ...

    # Structural penalties
    num_disconnected = self._count_disconnected_variables(theory)
    num_self_loops = self._count_self_loops(theory)

    disconnected_penalty_value = self.disconnected_var_penalty * num_disconnected
    self_loop_penalty_value = self.self_loop_penalty * num_self_loops

    reward = (0.9 * accuracy +
        0.1 * simplicity -
        uninformative_penalty -
        disconnected_penalty_value -
        self_loop_penalty_value)

    return max(reward, 1e-6)
```

## Test Results

### Reward Shaping Verification (`examples/test_reward_shaping.py`)

**Correct Rule**:
- `grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)`
- Disconnected variables: 0 ✓
- Self-loops: 0 ✓
- **Base reward: 0.9333** (excellent)

**Disconnected Variables**:
- `grandparent(X0, X5) :- parent(X0, X3), parent(X4, X5), parent(X6, X7)`
- Disconnected variables: 2 (X6, X7 in isolated atom) ✓
- Penalty: -0.40
- **Base reward: 0.0750** (poor)

**Self-Loop**:
- `grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0)`
- Self-loops: 3 (head + 2 body atoms) ✓
- Penalty: -0.90
- **Base reward: 0.0000** (minimum)

**With Exploration Bonuses**:
- Correct: 0.9333 + 0.20 = **1.1333** (highest)
- Disconnected: 0.0750 + 0.30 = 0.3750
- Self-loop: 0.0000 + 0.20 = 0.2000

**Conclusion**: Even with exploration bonuses, correct rules now have significantly higher total rewards!

## Combined Improvements Test

### Configuration

Test script: `examples/test_combined_improvements.py`

**Baseline**:
- Original graph encoding
- NO reward shaping penalties
- Paper improvements (detailed balance + replay buffer + reward weighting)

**Combined**:
- Enhanced graph encoding (rich features + attention pooling)
- Reward shaping penalties (disconnected: 0.2, self-loop: 0.3)
- Paper improvements (detailed balance + replay buffer + reward weighting)

**Settings**:
- 2000 episodes each
- 4 positive examples, 4 negative examples
- 7 background facts

### Results

*(Test currently running - results will be added when complete)*

## Expected Impact

1. **Replay Buffer Quality**: Fewer pathological rules stored
   - Disconnected/self-loop rules get base reward < 0.5
   - Correct rules get base reward > 0.9
   - Buffer should converge to high-quality rules

2. **Learning Signal**: Model receives clearer guidance
   - Pathological patterns explicitly penalized
   - Correct patterns rewarded
   - Combined with enhanced encoding, should learn faster

3. **Rule Quality**: Final generated rules should be:
   - Structurally valid (connected variables)
   - Semantically meaningful (no self-loops where inappropriate)
   - Higher accuracy on test examples

## Integration with Enhanced Encoding

The reward shaping works synergistically with the enhanced encoding:

1. **Enhanced encoding** provides:
   - Rich structural features (has_self_loop, is_chain_var)
   - Attention-based pooling (learns what matters)
   - Better state representation

2. **Reward shaping** provides:
   - Explicit penalties for bad patterns
   - Stronger gradient signal
   - Faster convergence

Together they should provide:
- **Faster learning**: Model sees patterns explicitly flagged in encoding
- **Higher quality**: Penalties guide away from pathological solutions
- **Better generalization**: Structural understanding transfers to new problems

## Next Steps

1. ✓ Verify reward shaping implementation (test_reward_shaping.py)
2. ✓ Integrate enhanced encoding (added get_variable_node_ids method)
3. ⏳ Run combined improvements test (currently running)
4. ⏳ Analyze results and compare with baseline
5. Future: Test on harder problems (ancestors, siblings, list operations)

## References

- Previous session: `docs/SESSION_SUMMARY.md`
- Reward explanation: `docs/REWARD_EXPLANATION.md`
- Enhanced encoding: `docs/IMPROVED_ENCODING_PROPOSAL.md`
- Paper improvements: `docs/PAPER_IMPROVEMENTS_SUMMARY.md`

## Code Files Modified

1. `src/reward.py`
   - Added `_count_disconnected_variables` (lines 35-76)
   - Added `_count_self_loops` (lines 78-85)
   - Modified `calculate_reward` to apply penalties (lines 115-134)
   - Updated `get_detailed_scores` to include penalty info (lines 170-204)

2. `src/graph_encoder_enhanced.py`
   - Added `get_variable_node_ids` method (lines 198-205)
   - Ensures compatibility with training code

3. `examples/test_reward_shaping.py` (new)
   - Comprehensive test of penalty calculations
   - Verifies correct disconnected variable detection
   - Verifies self-loop counting
   - Shows reward ordering with exploration bonuses

4. `examples/test_combined_improvements.py` (new)
   - Compares baseline vs combined improvements
   - 2000 episodes per configuration
   - Saves results to `analysis/combined_improvements_results.json`
