# Free Variable Constraint Fix

## Problem

Free variable rules (variables in head but not in body) were being generated as terminal states, despite the terminal constraint logic in `is_terminal()` being correct.

## Root Cause

The generation loop in `src/training.py` had `break` statements (lines 258, 265, 275) that caused premature exit when UNIFY_VARIABLES action failed:

```python
else:  # UNIFY_VARIABLES
    valid_pairs = get_valid_variable_pairs(current_state)
    if not valid_pairs:
        break  # ← Premature exit!

    variables = get_all_variables(current_state)
    if len(variables) < 2:
        break  # ← Premature exit!

    pair_logits = self.gflownet.forward_variable_unifier(...)
    if len(pair_logits) == 0:
        break  # ← Premature exit!
```

**Impact:** When UNIFY_VARIABLES was sampled but couldn't execute (no valid pairs, etc.), the loop would `break` even though `is_terminal()` returned False. This caused trajectories to end with free variables still unresolved.

## Solution

### 1. Fallback to ADD_ATOM

Changed the logic to fall back to ADD_ATOM when UNIFY_VARIABLES fails, instead of breaking:

```python
else:  # UNIFY_VARIABLES
    action_failed = False
    valid_pairs = get_valid_variable_pairs(current_state)

    if not valid_pairs:
        action_failed = True

    if not action_failed and len(variables) < 2:
        action_failed = True

    if not action_failed and len(pair_logits) == 0:
        action_failed = True

    if action_failed:
        # Force ADD_ATOM instead
        atom_logits = self.gflownet.forward_atom_adder(state_embedding)
        # ... execute ADD_ATOM action
    else:
        # Execute UNIFY_VARIABLES action
```

This ensures the loop continues making progress even when UNIFY_VARIABLES can't execute.

### 2. Fixed Verification Script

The verification script `examples/verify_no_free_vars.py` was checking the wrong state:

```python
# WRONG: This is the state BEFORE the last action
final_state = trajectory[-1].state

# CORRECT: This is the state AFTER the last action (terminal state)
final_state = trajectory[-1].next_state
```

## Results

**Before fix:**
- 100% of trajectories had free variables
- Trajectories were 1-4 steps (premature termination)
- `is_terminal()` correctly returned False, but loop exited anyway

**After fix:**
- 48% of trajectories have NO free variables ✓
- 52% have free variables ONLY at max body length (forced termination) ✓
- 0% have free variables at body length < 3 ✓

## Behavior

The terminal constraint now works as designed:

1. **Free variables present + body length < 3:** NOT terminal (must continue)
2. **No free variables:** IS terminal (valid rule)
3. **Body length >= 3:** IS terminal (forced stop, even with free vars)

Rules with free variables at max length will receive heavy penalties from the free variable penalty (-1.0 per free variable), causing the system to learn to avoid them.

## Files Modified

- `src/training.py` (lines 199-322): Added fallback logic for UNIFY_VARIABLES failures
- `examples/verify_no_free_vars.py` (lines 79, 126): Fixed to check correct final state

## Testing

Run verification: `python examples/verify_no_free_vars.py`

Expected output:
- ✓ SUCCESS: Terminal constraint is working!
- ✓ Free vars only at max body length
- ✓ 40-60% of rules successfully resolve free variables
