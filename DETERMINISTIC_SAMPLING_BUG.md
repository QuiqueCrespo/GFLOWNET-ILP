# Deterministic Sampling Bug: Why `stochastic=False` Still Produces Random Trajectories

**Date:** 2025-01-22
**Status:** 🔴 CRITICAL BUG
**Priority:** HIGH

---

## The Problem

**User Report:**
> "Why when I sample deterministically in the notebook I get different trajectories? How is it possible?"

**Expected Behavior:**
```python
# With stochastic=False, should get SAME trajectory every time
traj1, _ = trainer.generate_trajectory(..., stochastic=False)
traj2, _ = trainer.generate_trajectory(..., stochastic=False)
# traj1 should equal traj2
```

**Actual Behavior:**
```python
# But trajectories are DIFFERENT!
traj1, _ = trainer.generate_trajectory(..., stochastic=False)
traj2, _ = trainer.generate_trajectory(..., stochastic=False)
# traj1 ≠ traj2  ← BUG!
```

---

## Root Cause: `stochastic` Parameter Not Passed to Sub-Actions

### The Code Flow

**File:** `src/training.py:240-340`

```python
def generate_trajectory(self, ..., stochastic: bool = True):
    """Generate a trajectory."""

    while step_count < max_steps:
        # ... get state embeddings ...

        # Step 1: Sample ACTION TYPE (strategist)
        action, log_prob = self._sample_action_from_logits(
            action_logits, stochastic  # ✅ stochastic IS passed here
        )

        # Step 2: Handle specific action
        if action == 0:  # ADD_ATOM
            next_state, ..., log_prob_detail = self._handle_action_add_atom(...)
            # ❌ stochastic NOT passed to _handle_action_add_atom!

        elif action == 1:  # UNIFY_VARIABLES
            next_state, ..., log_prob_detail = self._handle_action_unify_vars(...)
            # ❌ stochastic NOT passed to _handle_action_unify_vars!
```

### The Bug in Detail

**Line 286:** Strategist sampling (DOES use stochastic)
```python
action, log_prob_action = self._sample_action_from_logits(action_logits, stochastic)
#                                                                         ^^^^^^^^^^
#                                                                         ✅ Passed!
```

**Line 422:** Predicate sampling (DOES NOT use stochastic)
```python
# In _handle_action_add_atom:
pred_idx, log_prob_detail = self._sample_action_from_logits(atom_logits)
#                                                                       ^
#                                                       ❌ stochastic NOT passed!
#                                                       Defaults to True (random)
```

**Line 474:** Variable pair sampling (DOES NOT use stochastic)
```python
# In _handle_action_unify_vars:
pair_idx, log_prob_detail = self._sample_action_from_logits(masked_logits)
#                                                                         ^
#                                                       ❌ stochastic NOT passed!
#                                                       Defaults to True (random)
```

---

## What This Means

### When You Call `stochastic=False`:

**What IS deterministic:**
- ✅ Choice between ADD_ATOM, UNIFY_VARIABLES, TERMINATE

**What is STILL RANDOM:**
- ❌ Which predicate to add (if ADD_ATOM chosen)
- ❌ Which variable pair to unify (if UNIFY_VARIABLES chosen)

### Example Trajectory

```python
# Call with stochastic=False
trajectory = generate_trajectory(..., stochastic=False)

# Run 1:
Step 1: ADD_ATOM (deterministic)
        predicate: parent (RANDOM!) ← Could be child, friend, etc.
Step 2: UNIFY_VARIABLES (deterministic)
        pair: (X0, X2) (RANDOM!) ← Could be (X1, X2), etc.
Step 3: TERMINATE (deterministic)

# Run 2: DIFFERENT trajectory!
Step 1: ADD_ATOM (deterministic - same)
        predicate: child (RANDOM - different!) ← Bug!
Step 2: UNIFY_VARIABLES (deterministic - same)
        pair: (X1, X2) (RANDOM - different!) ← Bug!
Step 3: TERMINATE (deterministic - same)
```

**Only the high-level action types are deterministic - the details are random!**

---

## Why This is a Problem

### 1. Can't Reproduce Results

**Expected:**
```python
# Set seed, sample deterministically
torch.manual_seed(42)
traj1 = generate_trajectory(..., stochastic=False)

torch.manual_seed(42)
traj2 = generate_trajectory(..., stochastic=False)

assert traj1 == traj2  # Should pass!
```

**Actual:**
```python
# Even with same seed + stochastic=False
traj1 ≠ traj2  # Still different! ✗
```

### 2. Can't Debug Policy

**When debugging, you want:**
- "Show me exactly what the model would do deterministically"
- "I want to see the highest-probability path"

**But you get:**
- Mixed deterministic + random behavior
- Can't see true greedy policy
- Can't verify model has learned correctly

### 3. Evaluation is Inconsistent

**During evaluation:**
```python
# Try to get best theory from model
best_theory = sample_best_theory(..., stochastic=False)

# Run 1: Gets lucky with random predicate choices → reward 0.9
# Run 2: Gets unlucky with random predicate choices → reward 0.3

# Which is the true "best" the model can produce?
```

### 4. Breaks Replay Buffer Logic

**When replaying:**
```python
# We recompute log probabilities for actions that were taken
# But if deterministic sampling doesn't reproduce same actions...
# The recomputed log_pf might be for DIFFERENT actions!
```

---

## Additional Sources of Non-Determinism

Even after fixing the `stochastic` parameter bug, there may be other sources:

### 1. Model Evaluation Mode

**Check if model is in eval mode:**
```python
# Before sampling, should call:
model.eval()  # Disables dropout, batch norm

# But if in training mode:
model.train()  # Dropout is active → random!
```

**File check needed:** Does `generate_trajectory` set model to eval mode?

### 2. Argmax Ties

**When multiple actions have IDENTICAL probabilities:**
```python
action_probs = [0.33, 0.33, 0.34]
#                ^^^^  ^^^^
# These two are very close - argmax may vary due to floating point!

action = action_probs.argmax()  # Which one? Undefined!
```

**PyTorch argmax behavior on ties:**
- Returns the FIRST occurrence
- But due to floating point errors, "ties" may not be exact
- Order could vary between runs

### 3. Floating Point Non-Determinism

**GPU operations are not always deterministic:**
```python
# Matrix multiplication on GPU
result = A @ B  # May vary slightly between runs

# Softmax with very small differences
logits = [2.0000001, 2.0000002, 1.9999999]
probs = softmax(logits)  # Which is max? Depends on floating point errors!
```

**PyTorch determinism:**
```python
# To make PyTorch fully deterministic:
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 4. Random Number Generator State

**If random seeds not set:**
```python
# Without seed:
random.random()  # Different each run
np.random.rand()  # Different each run
torch.rand(...)  # Different each run

# Need to set ALL of these:
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### 5. Graph Construction Order

**If node/edge ordering varies:**
```python
# If graph nodes are built from a set:
head_vars = set(rule.head.args)  # Set order is undefined!

# Then graph construction may vary:
for var in head_vars:  # Order could be different each time!
    add_node(var)
```

**Sets in Python 3.7+ maintain insertion order in CPython, but this is implementation detail**

### 6. Dictionary Iteration Order

**Similar to sets:**
```python
var_to_idx = {var.id: i for i, var in enumerate(variables)}

# If variables come from a set, order may vary
for var_id, idx in var_to_idx.items():  # Order?
    ...
```

---

## The Fix

### Primary Fix: Thread `stochastic` Through All Sampling

**File:** `src/training.py`

#### Fix 1: Update `_handle_action_add_atom` signature

**Current (line 410-432):**
```python
def _handle_action_add_atom(self, state_embedding, current_state: Theory,
                            max_var_id: int, step_count: int):
    """Handles the ADD_ATOM action logic."""
    atom_logits = self.gflownet.forward_atom_adder(state_embedding)

    # ... exploration strategy ...

    pred_idx, log_prob_detail = self._sample_action_from_logits(atom_logits)
    #                                                           ❌ No stochastic!
```

**Fixed:**
```python
def _handle_action_add_atom(self, state_embedding, current_state: Theory,
                            max_var_id: int, step_count: int, stochastic: bool = True):
    #                                                         ^^^^^^^^^^^^^^^^^^
    """Handles the ADD_ATOM action logic."""
    atom_logits = self.gflownet.forward_atom_adder(state_embedding)

    # ... exploration strategy ...

    pred_idx, log_prob_detail = self._sample_action_from_logits(atom_logits, stochastic)
    #                                                                        ^^^^^^^^^^^
```

#### Fix 2: Update `_handle_action_unify_vars` signature

**Current (line 434-483):**
```python
def _handle_action_unify_vars(self, state_embedding, node_embeddings,
                              current_state: Theory, valid_pairs: list):
    """Handles the UNIFY_VARIABLES action logic."""
    # ... masking logic ...

    pair_idx, log_prob_detail = self._sample_action_from_logits(masked_logits)
    #                                                           ❌ No stochastic!
```

**Fixed:**
```python
def _handle_action_unify_vars(self, state_embedding, node_embeddings,
                              current_state: Theory, valid_pairs: list,
                              stochastic: bool = True):
    #                                                  ^^^^^^^^^^^^^^^^^^
    """Handles the UNIFY_VARIABLES action logic."""
    # ... masking logic ...

    pair_idx, log_prob_detail = self._sample_action_from_logits(masked_logits, stochastic)
    #                                                                           ^^^^^^^^^^^
```

#### Fix 3: Update calls in `generate_trajectory`

**Current (line 290-299):**
```python
if action == 0:  # ADD_ATOM
    next_state, max_var_id, action_detail, log_prob_detail = \
        self._handle_action_add_atom(
            state_embedding, current_state, max_var_id, step_count
        )
elif action == 1:  # UNIFY_VARIABLES
    next_state, action_detail, log_prob_detail = \
        self._handle_action_unify_vars(
            state_embedding, node_embeddings, current_state, valid_pairs
        )
```

**Fixed:**
```python
if action == 0:  # ADD_ATOM
    next_state, max_var_id, action_detail, log_prob_detail = \
        self._handle_action_add_atom(
            state_embedding, current_state, max_var_id, step_count, stochastic
        )                                                           ^^^^^^^^^^
elif action == 1:  # UNIFY_VARIABLES
    next_state, action_detail, log_prob_detail = \
        self._handle_action_unify_vars(
            state_embedding, node_embeddings, current_state, valid_pairs, stochastic
        )                                                                 ^^^^^^^^^^
```

---

## Secondary Fixes: Ensure Full Determinism

### 1. Set Model to Eval Mode

```python
def generate_trajectory(self, ..., stochastic: bool = True):
    """Generate a trajectory."""

    # At the start:
    if not stochastic:
        self.gflownet.eval()  # Disable dropout
        self.state_encoder.eval()

    # ... generate trajectory ...

    # At the end (optional - restore training mode):
    # self.gflownet.train()
    # self.state_encoder.train()
```

### 2. Handle Argmax Ties Explicitly

```python
def _sample_action_from_logits(self, logits, stochastic: bool = True):
    """Samples an action from logits and returns action + log_prob."""
    action_probs = F.softmax(logits, dim=-1)
    if stochastic:
        action = torch.multinomial(action_probs, 1).item()
    else:
        # Use argmax - but be aware of ties
        action = action_probs.argmax().item()

        # Optional: Add tie-breaking with small random noise
        # logits_with_noise = logits + torch.randn_like(logits) * 1e-6
        # action = logits_with_noise.argmax().item()

    log_prob = torch.log(action_probs[action] + 1e-10)
    return action, log_prob
```

### 3. Set All Random Seeds (User's Responsibility)

**In notebook/script before sampling:**
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For full reproducibility (may slow down training):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use before sampling:
set_seed(42)
traj1 = generate_trajectory(..., stochastic=False)

set_seed(42)
traj2 = generate_trajectory(..., stochastic=False)

# Now traj1 == traj2 (should be!)
```

---

## Testing the Fix

### Test 1: Basic Determinism

```python
def test_deterministic_sampling():
    """Test that stochastic=False produces identical trajectories."""

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate first trajectory
    traj1, reward1 = trainer.generate_trajectory(
        initial_state, pos_examples, neg_examples, stochastic=False
    )

    # Reset seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate second trajectory
    traj2, reward2 = trainer.generate_trajectory(
        initial_state, pos_examples, neg_examples, stochastic=False
    )

    # Check they're identical
    assert len(traj1) == len(traj2), "Trajectory lengths differ!"

    for i, (step1, step2) in enumerate(zip(traj1, traj2)):
        assert step1.action_type == step2.action_type, f"Step {i}: action type differs"
        assert step1.action_detail == step2.action_detail, f"Step {i}: action detail differs"

    print("✅ Deterministic sampling test PASSED!")
```

### Test 2: Stochastic Sampling Produces Variation

```python
def test_stochastic_sampling_varies():
    """Test that stochastic=True produces DIFFERENT trajectories."""

    torch.manual_seed(42)
    traj1, _ = trainer.generate_trajectory(..., stochastic=True)

    torch.manual_seed(43)  # Different seed
    traj2, _ = trainer.generate_trajectory(..., stochastic=True)

    # Should be different (with high probability)
    differs = False
    for step1, step2 in zip(traj1, traj2):
        if step1.action_detail != step2.action_detail:
            differs = True
            break

    assert differs, "Stochastic sampling produced identical trajectories (unlikely!)"
    print("✅ Stochastic sampling variation test PASSED!")
```

### Test 3: Deterministic Sampling Chooses Highest Probability

```python
def test_deterministic_chooses_argmax():
    """Test that stochastic=False chooses highest-probability actions."""

    # Generate deterministic trajectory
    traj, _ = trainer.generate_trajectory(..., stochastic=False)

    # For each step, verify it chose the highest-probability action
    for step in traj:
        # Re-encode state
        graph_data = trainer.graph_constructor.theory_to_graph(step.state)
        state_embedding, node_embeddings = trainer.state_encoder(graph_data)
        state_embedding = state_embedding.squeeze(0)

        # Get action logits
        if step.action_type == 'ADD_ATOM':
            atom_logits = trainer.gflownet.forward_atom_adder(state_embedding)
            chosen_pred = step.action_detail
            chosen_idx = trainer.predicate_vocab.index(chosen_pred)
            assert chosen_idx == atom_logits.argmax().item(), \
                "Deterministic sampling didn't choose argmax predicate!"

        elif step.action_type == 'UNIFY_VARIABLES':
            # ... similar check for variable pairs ...
            pass

    print("✅ Argmax selection test PASSED!")
```

---

## Impact Analysis

### Why This Bug Matters

**For Users:**
1. ❌ Can't reproduce results
2. ❌ Can't evaluate true model capability (deterministic best case)
3. ❌ Debugging is harder (can't see greedy policy)
4. ❌ Evaluation is inconsistent (random variation in "deterministic" mode)

**For Training:**
1. ❌ Replay buffer may recompute incorrect log probabilities
2. ❌ Can't verify if model has converged (deterministic policy still random)
3. ❌ Hard to diagnose overfitting (can't compare deterministic vs stochastic gap)

**For Research:**
1. ❌ Can't compare models fairly (deterministic sampling varies)
2. ❌ Can't ablate components (need determinism for controlled experiments)
3. ❌ Results not reproducible across runs

---

## Verification After Fix

**After applying the fix, run this verification:**

```python
# 1. Set seed
set_seed(42)

# 2. Generate 5 deterministic trajectories
trajectories = []
for _ in range(5):
    set_seed(42)  # Reset to SAME seed each time
    traj, _ = trainer.generate_trajectory(..., stochastic=False)
    trajectories.append(traj)

# 3. Verify all are IDENTICAL
for i in range(1, 5):
    assert trajectories[i] == trajectories[0], f"Trajectory {i} differs from trajectory 0!"

print("✅ DETERMINISM VERIFIED!")
print(f"   All 5 trajectories with same seed are identical")
```

---

## Summary

**The Bug:**
- `stochastic` parameter only used for high-level action type selection
- Predicate selection and variable pair selection ALWAYS random
- Even with `stochastic=False`, get different trajectories

**The Fix:**
- Thread `stochastic` parameter through ALL sampling decisions
- Update `_handle_action_add_atom` and `_handle_action_unify_vars` signatures
- Pass `stochastic` when calling these methods

**Additional Considerations:**
- Set model to eval mode when `stochastic=False`
- Set random seeds for full reproducibility
- Be aware of argmax ties (rare but possible)

**Priority:** 🔴 HIGH - This breaks reproducibility and evaluation

---

## Files to Modify

1. **`src/training.py`**
   - Line 410: Add `stochastic` parameter to `_handle_action_add_atom`
   - Line 422: Pass `stochastic` to `_sample_action_from_logits`
   - Line 434: Add `stochastic` parameter to `_handle_action_unify_vars`
   - Line 474: Pass `stochastic` to `_sample_action_from_logits`
   - Lines 291, 296: Pass `stochastic` when calling handler methods

**Estimated fix time:** 5 minutes

**Testing time:** 10 minutes

**Total:** ~15 minutes to fix and verify
