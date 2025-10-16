# Exploration Bonus Explanation

## Issue Observed

Benchmark results showed rules with 0 positive coverage getting high rewards (0.66 - 0.98):

```
1d_flip:
  Best reward: 0.7992
  Coverage: 0/31 pos (0.0%), 3/915 neg (0.3%)

andersen:
  Best reward: 0.9792
  Coverage: 0/7 pos (0.0%), 0/0 neg (0.0%)
```

## Root Cause

The benchmark pipeline uses **"aggressive" exploration strategy** which includes a **trajectory length bonus**:

```python
# From src/exploration.py lines 302-307
elif config == "aggressive":
    return CombinedExploration([
        EntropyBonus(alpha=0.05, decay=0.9998),
        TemperatureSchedule(T_init=3.0, T_final=0.5, decay_steps=1500),
        TrajectoryLengthBonus(beta=0.1, decay=0.999),  # ← This adds bonus!
    ])
```

### Trajectory Length Bonus

```python
# From src/exploration.py lines 138-141
def modify_reward(self, reward: float, trajectory_length: int = 0, **kwargs) -> float:
    """Add bonus proportional to trajectory length."""
    return reward + self.beta * trajectory_length
```

For a rule with trajectory length 8:
- True reward: 1e-6 (for 0 pos coverage)
- Trajectory bonus: 0.1 × 8 = 0.8
- **Training reward: ~0.8**

## Why This Is Intentional

The exploration bonus serves important purposes during training:

1. **Prevents getting stuck**: Encourages the model to explore longer trajectories
2. **Improves learning**: Helps the model learn from diverse rule structures
3. **Avoids premature convergence**: Prevents settling on trivial rules too quickly

## The Fix

The benchmark pipeline now reports BOTH rewards:

```
1d_flip:
  Training reward (with bonus): 0.7756
  True reward (no bonus): 0.000001
  Coverage: 0/31 pos (0.0%), 0/915 neg (0.0%)
```

This makes it clear:
- **Training reward**: What the GFlowNet sees during learning (with exploration incentives)
- **True reward**: Actual performance without bonuses (what we care about for evaluation)

## Verification

Test showing the difference:

```python
# Rule with 0 positive coverage
out(X0, X0, X0) :- x3(X0), not_end(X0), c4(X0).

# Reward calculation
True reward: 0.000001  # From reward_calculator (no bonus)
Training reward: 0.7756  # With trajectory length bonus
```

## Impact on Training

The exploration bonus does NOT affect:
- ✅ Final rule quality evaluation (uses true reward)
- ✅ Coverage metrics (based on actual proofs)
- ✅ Logical correctness (enforced by safety condition)

The exploration bonus DOES affect:
- 🔄 Which trajectories get sampled during training
- 🔄 Gradient signals for policy learning
- 🔄 Exploration vs exploitation balance

## Relationship to Logic Engine Fix

The logic engine safety condition fix (from `LOGIC_ENGINE_FIX.md`) is INDEPENDENT:

1. **Logic engine fix**: Prevents incorrect evaluation of disconnected rules
   - Before: Disconnected rules could prove examples they shouldn't
   - After: Safety condition enforced, true reward correctly near-zero

2. **Exploration bonus**: Intentional training signal for exploration
   - Before: Same exploration bonuses applied
   - After: Same exploration bonuses applied (unchanged)

Both are correct:
- Logic engine now correctly evaluates rules (true reward = 1e-6 for 0 coverage) ✓
- Exploration bonus encourages diverse trajectories during training ✓

## Recommendations

### For Evaluation
Use **true reward** (without bonuses) to assess rule quality.

### For Training
Keep exploration bonuses to:
- Help find good rules in complex search spaces
- Prevent premature convergence to trivial solutions
- Encourage learning from diverse rule structures

### For Reporting
Always show BOTH:
- Training reward (for understanding learning dynamics)
- True reward (for evaluating actual performance)

## Files Modified

- `run_benchmark_pipeline.py` (lines 200, 227-232, 251, 265-266, 347): Added true_reward calculation and display

## Related Documents

- `LOGIC_ENGINE_FIX.md`: Explains safety condition fix for disconnected variables
- `PIPELINE_COVERAGE_UPDATE.md`: Explains coverage metrics addition
