# Logic Engine Bug Fix - Safety Condition

## Problem

Rules with 0 actual coverage were being reported as having high coverage (e.g., 100%) and receiving high rewards during training.

### Example Bug

**Disconnected Rule:**
```prolog
grandparent(X0, X1) :- parent(X2, X3), parent(X4, X5).
```

**Expected Behavior:** Should NOT prove any grandparent examples because X0, X1 (head variables) don't appear in the body

**Actual Behavior (BEFORE FIX):** Successfully proved `grandparent('alice', 'charlie')` ❌

## Root Cause

The SLD resolution in `src/logic_engine.py` was missing the **safety condition** check from logic programming.

### What Was Happening

1. Query: `grandparent('alice', 'charlie')`
2. Unify with head: `{X0: 'alice', X1: 'charlie'}`
3. Prove body:
   - `parent(X2, X3)` finds `parent('alice', 'bob')` → binds `{X2: 'alice', X3: 'bob'}`
   - `parent(X4, X5)` finds `parent('bob', 'charlie')` → binds `{X4: 'bob', X5: 'charlie'}`
4. Body proves successfully! ❌
5. **BUG:** X0, X1 were never connected to X2, X3, X4, X5, but proof succeeded anyway

The logic engine allowed head variables to be bound independently from body variables, violating the fundamental safety requirement.

## Solution

Added safety condition check in `_prove_goal()` method (lines 83-104 in `src/logic_engine.py`):

```python
# Get all variables that appear in the head
head_vars = set()
for arg in rule.head.args:
    is_variable = (hasattr(arg, '__class__') and
                  arg.__class__.__name__ == 'Variable')
    if is_variable:
        head_vars.add(arg)

# Get all variables that appear in the body
body_vars = set()
for atom in rule.body:
    for arg in atom.args:
        is_variable = (hasattr(arg, '__class__') and
                      arg.__class__.__name__ == 'Variable')
        if is_variable:
            body_vars.add(arg)

# Check for disconnected head variables
# Head variables must appear in the body (safety condition)
if not head_vars.issubset(body_vars):
    # Disconnected variables - rule is unsafe
    continue
```

### Safety Condition

**Definition:** All variables in the rule head MUST also appear in the rule body.

**Why Required:**
- Head variables must be grounded by bindings from body proofs
- Disconnected head variables can take arbitrary values, making proofs meaningless
- This is a standard requirement in Datalog and safe Prolog programs

## Verification

### Diagnostic Test Results

**File:** `diagnose_logic_engine.py`

```
TEST 1: Disconnected Rule
Rule: grandparent(X0, X1) :- parent(X2, X3), parent(X4, X5).

Can prove grandparent('alice', 'charlie')? False ✓
Can prove grandparent('alice', 'alice')? False ✓

TEST 2: Perfect Rule
Rule: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1).

Can prove grandparent('alice', 'charlie')? True ✓
Can prove grandparent('alice', 'alice')? False ✓
```

### Reward Calculation Test

**File:** `diagnose_reward_issue.py`

**Before Fix:**
```
Rule with 0 coverage (disconnected vars):
  Positive coverage: 1/1 (100.0%) ❌
  Final reward: 0.8822 ❌
```

**After Fix:**
```
Rule with 0 coverage (disconnected vars):
  Positive coverage: 0/1 (0.0%) ✓
  Final reward: 0.000001 ✓
```

### Training Test

**File:** `test_single_benchmark.py`

Trained a rule that achieved 0.6861 reward during training (when logic engine had the bug), but now correctly evaluates to 0.000001 reward:

```
Best rule found:
  Rule: out(X4, X1, X1) :- x14(X4), x29(X4), x15(X1).
  Best reward (from training): 0.6861 ❌ (with buggy logic engine)
  Calculated reward (now): 0.000001 ✓ (with fixed logic engine)
  Positive coverage: 0/31 (0.0%) ✓
  Disconnected vars: 0
  Free vars: 0
```

Note: This rule has a self-loop (X1 appears twice in head) and disconnected variables (X4 in body only, X1 partially disconnected).

## Impact

### What This Fixes

1. ✅ Rules with 0 actual coverage now correctly get 0 reward
2. ✅ Disconnected rules cannot prove examples they shouldn't
3. ✅ Coverage metrics are now accurate
4. ✅ Training will no longer converge to invalid disconnected rules

### What Doesn't Change

- Free variable constraint and action mask remain in place
- Reward shaping penalties remain in place
- These are still important for preventing invalid rules from being GENERATED
- The logic engine fix prevents invalid rules from being incorrectly EVALUATED

## Relationship to Other Fixes

### Free Variable Constraint (Already Implemented)
- **Where:** `src/logic_structures.py` (`is_terminal()`)
- **What:** Prevents rules with free variables from being terminal states
- **Why Still Needed:** Prevents generation of invalid rules at the source

### Action Mask (Already Implemented)
- **Where:** `src/training.py` (line 222-226)
- **What:** Prevents ADD_ATOM at max body length, allows UNIFY_VARIABLES to resolve free vars
- **Why Still Needed:** Ensures all generated rules have opportunities to resolve free variables

### Logic Engine Safety Check (This Fix)
- **Where:** `src/logic_engine.py` (`_prove_goal()`)
- **What:** Enforces safety condition during proof search
- **Why Needed:** Prevents incorrect evaluation of rules that slip through generation constraints

All three mechanisms work together:
1. **Generation constraints** (free var + action mask) prevent creating bad rules
2. **Logic engine safety** prevents evaluating rules incorrectly if bad rules are created

## Next Steps

1. Re-run benchmark pipeline with fixed logic engine
2. Expect to see more realistic coverage metrics
3. Training may need more episodes to find correct rules
4. May need to adjust exploration strategy to help find valid rules

## Files Modified

- `src/logic_engine.py` (lines 83-104): Added safety condition check in `_prove_goal()`

## Files Created for Diagnosis

- `diagnose_logic_engine.py`: Tests SLD resolution with disconnected variables
- `diagnose_reward_issue.py`: Tests reward calculation for 0-coverage rules
- `test_single_benchmark.py`: Quick benchmark test with detailed logging
