# Quick Fix: Deterministic Sampling Bug

**Problem:** `stochastic=False` still produces different trajectories

**Root Cause:** `stochastic` parameter not passed to sub-actions

---

## The Fix (3 Changes)

### 1. Update `_handle_action_add_atom` (Line ~410)

**Add `stochastic` parameter:**
```python
def _handle_action_add_atom(self, state_embedding, current_state: Theory,
                            max_var_id: int, step_count: int,
                            stochastic: bool = True):  # ← ADD THIS
```

**Pass it to sampling (Line ~422):**
```python
pred_idx, log_prob_detail = self._sample_action_from_logits(atom_logits, stochastic)
#                                                                        ^^^^^^^^^^
#                                                                        ADD THIS
```

### 2. Update `_handle_action_unify_vars` (Line ~434)

**Add `stochastic` parameter:**
```python
def _handle_action_unify_vars(self, state_embedding, node_embeddings,
                              current_state: Theory, valid_pairs: list,
                              stochastic: bool = True):  # ← ADD THIS
```

**Pass it to sampling (Line ~474):**
```python
pair_idx_from_all, log_prob_detail = self._sample_action_from_logits(masked_logits, stochastic)
#                                                                                   ^^^^^^^^^^
#                                                                                   ADD THIS
```

### 3. Update calls in `generate_trajectory` (Lines ~290-299)

**Pass `stochastic` to ADD_ATOM handler:**
```python
if action == 0:  # ADD_ATOM
    next_state, max_var_id, action_detail, log_prob_detail = \
        self._handle_action_add_atom(
            state_embedding, current_state, max_var_id, step_count,
            stochastic  # ← ADD THIS
        )
```

**Pass `stochastic` to UNIFY_VARIABLES handler:**
```python
elif action == 1:  # UNIFY_VARIABLES
    next_state, action_detail, log_prob_detail = \
        self._handle_action_unify_vars(
            state_embedding, node_embeddings, current_state, valid_pairs,
            stochastic  # ← ADD THIS
        )
```

---

## Test the Fix

```python
from test_deterministic_bug import test_deterministic_sampling_bug

# Should FAIL before fix
passed = test_deterministic_sampling_bug(trainer, initial_state, pos_ex, neg_ex)

# Should PASS after fix
```

---

## Full Reproducibility (For Notebooks)

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
    # Optional: full determinism (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Before sampling:
set_seed(42)
traj1 = trainer.generate_trajectory(..., stochastic=False)

set_seed(42)
traj2 = trainer.generate_trajectory(..., stochastic=False)

# After fix: traj1 == traj2 ✓
```

---

**File:** `src/training.py`

**Estimated time:** 5 minutes

**Lines to modify:** 410, 422, 434, 474, 291, 296
