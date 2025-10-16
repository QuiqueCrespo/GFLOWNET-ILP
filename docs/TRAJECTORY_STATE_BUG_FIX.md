# Trajectory State Bug Fix

## Problem

When checking final states in trajectories, some scripts were using `trajectory[-1].state` instead of `trajectory[-1].next_state`, causing them to check the **pre-action state** instead of the **terminal state**.

## Impact

This made it appear that rules had free variables when they actually didn't, because:
- `trajectory[-1].state` = state **before** the last action
- `trajectory[-1].next_state` = state **after** the last action (the actual terminal state)

## Example

```python
# WRONG - checks state BEFORE last action
theory = trajectory[-1].state  # May have free variables

# CORRECT - checks state AFTER last action (terminal)
theory = trajectory[-1].next_state  # No free variables
```

### Concrete Example:

```
Step 3:
  state: grandparent(X0, X1) :- parent(X2, X3).  # Has free vars X0, X1
  action: UNIFY_VARIABLES (X0, X2)
  next_state: grandparent(X0, X1) :- parent(X0, X3).  # Free var X1 remains

trajectory[-1].state → Has free vars X0, X1 (WRONG!)
trajectory[-1].next_state → Has free var X1 (CORRECT)
```

## Files Fixed

1. **examples/demo_enhanced_method.py**
   - Line 134: `trajectory[-1].state` → `trajectory[-1].next_state`
   - Line 199: `trajectory[-1].state` → `trajectory[-1].next_state`

2. **examples/verify_no_free_vars.py** (previously fixed)
   - Line 79: `trajectory[-1].state` → `trajectory[-1].next_state`
   - Line 126: `trajectory[-1].state` → `trajectory[-1].next_state`

3. **examples/test_break_hypothesis.py**
   - Line 75: `trajectory[-1].state` → `trajectory[-1].next_state`

## Verification

After fix:
- `python examples/verify_no_free_vars.py` → **0% free variables** ✓
- `python examples/quick_demo_test.py` → **0 rules with free variables** ✓
- `python examples/inspect_generated_rules.py` → **0 free variables in all 20 rules** ✓

## Root Cause

The `TrajectoryStep` class stores:
```python
class TrajectoryStep:
    state: Theory         # State BEFORE action
    action_type: str
    action_detail: any
    log_pf: torch.Tensor
    next_state: Theory    # State AFTER action (result)
```

When analyzing final rules, we must use `next_state` to get the actual terminal state that resulted from the trajectory.

## Prevention

When writing new code that accesses trajectories:
- ✓ Use `trajectory[-1].next_state` for terminal state
- ✗ Don't use `trajectory[-1].state` for terminal state

The only valid use of `.state` is when iterating through trajectory steps to analyze the sequence of actions:
```python
for step in trajectory:
    print(f"From state: {step.state}")
    print(f"Action: {step.action_type}")
    print(f"To state: {step.next_state}")
```
