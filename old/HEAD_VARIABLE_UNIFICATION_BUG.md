# Critical Bug: Head Variable Unification

## The Problem

**Unifying variables that appear in the head always leads to zero reward**, but the model is allowed to do it and doesn't learn to avoid it.

---

## Concrete Example

### Scenario 1: Empty Body (ALREADY MASKED)

```
State: grandparent(X0, X1) :-
Action: UNIFY X0 and X1
Result: grandparent(X0, X0) :-
```

✅ **This is already blocked** by the mask at line 371-372 in training.py:
```python
if current_state[0].body == []:
    logits[1] = float('-inf')
```

### Scenario 2: After Adding Atoms (NOT MASKED!)

```
State: grandparent(X0, X1) :- parent(X0, X2)
Valid pairs: [(X0, X1), (X0, X2), (X1, X2)]  ← includes (X0, X1)!
Action: UNIFY X0 and X1
Result: grandparent(X0, X0) :- parent(X0, X2)
```

❌ **This is NOT blocked!** The model can unify head variables at any step (after step 0).

**Why this is bad**:
- Creates self-loop in head: `grandparent(X0, X0)`
- Can only match examples where both args are same: `grandparent(alice, alice)`
- These are typically negative examples
- Self-loop penalty applies: -0.3
- Reward ≈ 1e-6 (minimum)

---

## Root Cause: `get_valid_variable_pairs` Includes Head Pairs

**File**: `src/logic_structures.py:184-195`

```python
def get_valid_variable_pairs(theory: Theory) -> List[Tuple[Variable, Variable]]:
    """
    Get all valid pairs of variables that can be unified.
    A valid pair consists of two different variables.
    """
    variables = get_all_variables(theory)  # ALL variables (head + body)
    pairs = []
    for i, v1 in enumerate(variables):
        for v2 in variables[i+1:]:
            if v1 != v2:
                pairs.append((v1, v2))  # Includes (X0, X1) from head!
    return pairs
```

**Problem**: Returns ALL pairs of variables, including pairs where **both variables are in the head**.

---

## Why Doesn't the Model Learn to Avoid This?

Given that unifying head variables always leads to ~0 reward, why does the model keep doing it?

### Reason 1: **Replay Buffer Dilution** (Bug #1)
- Bad trajectories (with head unification) have reward ≈ 1e-6
- Replay buffer only stores trajectories with reward > 0.7
- Bad trajectories are never replayed
- With `replay_probability = 0.5`, 50% of training uses good trajectories
- Gradient signal to avoid head unification is diluted

### Reason 2: **No Explicit Mask**
- Unlike the empty body case, there's no mask preventing head unification
- Model must learn from experience that this is bad
- But experience is diluted by replay buffer

### Reason 3: **Inaccurate Backward Probability** (Bug #2)
- P_B for UNIFY is learned (or uniform heuristic)
- Biased gradients make learning slower

### Reason 4: **log_Z Compensation** (Issue #5)
- log_Z can absorb differences instead of policy learning
- Loss decreases without fixing the policy

---

## The Data: Why This Shows Up

Let's trace through the valid pairs:

```python
# Initial state
grandparent(X0, X1) :-

# After ADD_ATOM: parent
grandparent(X0, X1) :- parent(X2, X3)
Variables: [X0, X1, X2, X3]
Valid pairs: [(X0,X1), (X0,X2), (X0,X3), (X1,X2), (X1,X3), (X2,X3)]
              ^^^^^^^
              HEAD PAIR - should NOT be allowed!

# After ADD_ATOM: parent again
grandparent(X0, X1) :- parent(X2, X3), parent(X4, X5)
Variables: [X0, X1, X2, X3, X4, X5]
Valid pairs: 15 total
  - (X0, X1) ← HEAD PAIR
  - (X0, X2), (X0, X3), ..., (X0, X5) ← OK (head with body)
  - (X1, X2), (X1, X3), ..., (X1, X5) ← OK (head with body)
  - (X2, X3), (X2, X4), ... ← OK (body with body)
```

**At each step**, the head pair (X0, X1) is included in valid_pairs!

**Probability of sampling head pair**:
- If 15 valid pairs, P(head pair) = 1/15 ≈ 6.7%
- If 6 valid pairs, P(head pair) = 1/6 ≈ 16.7%

This is **abnormally high** for an action that ALWAYS leads to zero reward!

---

## Solution: Mask Head Variable Pairs

### Option 1: Filter in `get_valid_variable_pairs`

Modify the function to exclude pairs where both variables are in the head:

```python
def get_valid_variable_pairs(theory: Theory) -> List[Tuple[Variable, Variable]]:
    """
    Get all valid pairs of variables that can be unified.
    Excludes pairs where both variables are in the head (creates self-loop).
    """
    if not theory:
        return []

    rule = theory[0]
    head_vars = set(arg for arg in rule.head.args if isinstance(arg, Variable))

    variables = get_all_variables(theory)
    pairs = []

    for i, v1 in enumerate(variables):
        for v2 in variables[i+1:]:
            if v1 != v2:
                # Exclude pairs where BOTH are in head
                if v1 in head_vars and v2 in head_vars:
                    continue  # Skip this pair
                pairs.append((v1, v2))

    return pairs
```

### Option 2: Add Mask in Training Loop

Alternative: Keep `get_valid_variable_pairs` as-is, but add filtering when computing masked logits:

```python
def _get_masked_strategist_logits(...):
    # ... existing code ...

    # NEW: Mask UNIFY_VARIABLES if it would unify two head variables
    if valid_pairs:
        rule = current_state[0]
        head_vars = set(arg for arg in rule.head.args if isinstance(arg, Variable))

        # Check if ALL valid pairs would unify head variables
        all_pairs_are_head_pairs = all(
            v1 in head_vars and v2 in head_vars
            for v1, v2 in valid_pairs
        )

        if all_pairs_are_head_pairs:
            logits[1] = float('-inf')  # Mask UNIFY_VARIABLES
```

**Recommended**: Option 1 (filter in `get_valid_variable_pairs`) because it's cleaner and affects all uses of the function.

---

## Impact of the Fix

### Before Fix:
```
State: grandparent(X0, X1) :- parent(X0, X2)
Valid pairs: [(X0, X1), (X0, X2), (X1, X2)]
P(UNIFY head pair) = 1/3 = 33% if UNIFY is chosen
```

### After Fix:
```
State: grandparent(X0, X1) :- parent(X0, X2)
Valid pairs: [(X0, X2), (X1, X2)]  ← (X0, X1) removed!
P(UNIFY head pair) = 0%  ← IMPOSSIBLE!
```

**Expected Results**:
1. ✅ Model can no longer create self-loops by unifying head variables
2. ✅ All remaining UNIFY actions are potentially useful
3. ✅ Average reward should increase (fewer bad trajectories)
4. ✅ Learning should be faster (less exploration of bad action space)

---

## Why This Bug Matters

This is **one of the root causes** of the 50-50 probability issue:

1. Model samples UNIFY_VARIABLES
2. Valid pairs include head pair with probability ~10-30%
3. If head pair is chosen → reward ≈ 1e-6
4. Gradient signal to reduce P(UNIFY) is weak (replay buffer, etc.)
5. Model doesn't learn to avoid UNIFY strongly

**After fixing this**: The model will never waste time exploring head unifications, and all UNIFY actions will be potentially useful!

---

## Additional Context: ILP Theory

In Inductive Logic Programming, **self-loops in the head are almost always wrong**:

### Valid Rule:
```
grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
```
Can match: grandparent(alice, charlie) ✓

### Invalid Rule (self-loop):
```
grandparent(X, X) :- parent(X, Z), parent(Z, X)
```
Can only match: grandparent(alice, alice) ✗

**Exception**: Some predicates like `equal(X, X)` or `same(X, X)` are intentionally self-referential, but these are rare and domain-specific.

For general ILP, **we should never unify head variables**.

---

## Verification After Fix

To verify the fix is working:

```python
# Test get_valid_variable_pairs
from src.logic_structures import get_initial_state, apply_add_atom, get_valid_variable_pairs, Variable

state = get_initial_state('grandparent', 2)
state, _ = apply_add_atom(state, 'parent', 2, 1)

# State: grandparent(X0, X1) :- parent(X2, X3)
pairs = get_valid_variable_pairs(state)

# Check that (X0, X1) is NOT in pairs
head_vars = {Variable(0), Variable(1)}
head_pairs = [(v1, v2) for v1, v2 in pairs
              if v1 in head_vars and v2 in head_vars]

print(f"Valid pairs: {len(pairs)}")
print(f"Head pairs: {len(head_pairs)}")  # Should be 0!

if len(head_pairs) > 0:
    print("❌ BUG: Head pairs are still allowed!")
else:
    print("✅ FIX VERIFIED: No head pairs in valid_pairs")
```

---

## Summary

**Bug**: `get_valid_variable_pairs` includes pairs where both variables are in the head, allowing the model to create self-loops.

**Impact**: Model explores bad action space, dilutes learning signal, contributes to 50-50 probability issue.

**Fix**: Filter out head pairs in `get_valid_variable_pairs`:
```python
if v1 in head_vars and v2 in head_vars:
    continue
```

**Priority**: 🔴 **CRITICAL** - This is likely a major contributor to poor learning!
