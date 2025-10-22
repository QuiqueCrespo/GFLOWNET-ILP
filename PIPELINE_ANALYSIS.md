# GFlowNet-ILP Pipeline Analysis & Bug Report

## Executive Summary

The pipeline is **theoretically sound** in most aspects, but there are **several critical bugs and theoretical issues** that prevent proper learning. The most critical issue is the **50-50 probability bug** where the model fails to learn that UNIFY_VARIABLES at the first step always leads to zero reward.

---

## ✅ Components That Are Correct

### 1. Logic Engine (src/logic_engine.py)
- **SLD Resolution**: Correctly implemented with proper unification and standardization apart
- **Variable Renaming**: Uses depth-based renaming (depth * 1000 + var_id) to avoid conflicts
- **Fact Indexing**: O(1) lookup using predicate-based indexing
- **Safety Checking**: Properly detects free variables in rule heads

### 2. Reward Calculation (src/reward.py)
- **Confusion Matrix**: Correctly computes TP, FP, TN, FN
- **Precision/Recall**: Properly calculated with edge case handling
- **F1-Score**: Harmonic mean correctly computed
- **Disconnected Variables**: BFS-based detection from head variables - correct
- **Self-Loops**: Correctly detects duplicate variables in atoms
- **Free Variables**: Correctly identifies head variables not in body

### 3. Graph Encoder (src/graph_encoder_enhanced.py)
- **Rich Features**: Variables and predicates have comprehensive feature vectors
- **Bidirectional Edges**: Correctly creates edges with position information
- **Hierarchical Pooling**: Attention-based pooling for variables and predicates
- **Feature Padding**: Handles different feature dimensions correctly

### 4. Trajectory Balance Loss (src/training.py:480-538)
- **Formula**: `(log_Z + sum_log_pf - log_reward - sum_log_pb)^2` - theoretically correct
- **TERMINATE Handling**: Correctly skips TERMINATE steps in P_B sum (line 503-505)
- **Off-Policy Replay**: Correctly recomputes log probabilities for replayed trajectories (line 654-668)

---

## 🐛 Critical Bugs & Issues

### **BUG #1: Replay Buffer Dilutes Gradient for Bad Actions** ⚠️ CRITICAL

**Location**: `src/training.py:644-677`

**Problem**:
```python
# Line 676: Off-policy loss is ADDED to on-policy loss
total_loss += off_policy_loss
```

**Why this is wrong**:
1. Replay buffer only stores trajectories with `reward > buffer_reward_threshold` (typically 0.7)
2. Bad trajectories (e.g., starting with UNIFY_VARIABLES at first step) have reward ≈ 1e-6
3. These bad trajectories are NEVER stored in the replay buffer
4. With `replay_probability = 0.5`, half of the training uses replayed good trajectories
5. The gradient signal from bad on-policy trajectories is diluted by 50%

**Example**:
- Initial state: `grandparent(X0, X1) :-`
- Action: UNIFY_VARIABLES → `grandparent(X0, X0) :-`
- This creates a self-loop and can never match `grandparent(alice, charlie)`
- Reward ≈ 1e-6, NOT stored in replay buffer
- Gradient to reduce P(UNIFY|initial_state) is weak

**Impact**: Model fails to learn to avoid bad actions strongly

**Fix**:
```python
# Option 1: Don't add off-policy loss, replace it
if use_replay:
    total_loss = off_policy_loss  # Don't add, replace
else:
    total_loss = on_policy_loss

# Option 2: Weight the losses appropriately
total_loss = on_policy_loss + replay_weight * off_policy_loss

# Option 3: Reduce replay_probability during early training
# Let model learn bad actions first, then use replay buffer
```

---

### **BUG #2: Backward Probability for UNIFY_VARIABLES Uses Heuristic** ⚠️ MODERATE

**Location**: `src/gflownet_models.py:362-366`

**Problem**:
```python
# Approximate: uniform distribution over all valid pairs
num_pairs = num_prev_vars * (num_prev_vars - 1) // 2
log_prob_detail = torch.tensor(-math.log(max(num_pairs, 1)), dtype=torch.float32)
```

**Why this is wrong**:
- The backward policy should learn which variable pair was unified
- Instead, it uses a uniform heuristic: P_B = 1 / num_pairs
- This is a crude approximation that doesn't reflect the actual backward dynamics
- In Trajectory Balance, inaccurate P_B leads to biased gradient estimates

**Impact**:
- Biased TB loss gradients
- Slower convergence
- May prevent learning optimal policy

**Fix**:
Implement a proper learned backward variable splitter that predicts which variables were unified:
```python
# Use the BackwardVariableSplitter network (already defined but not used)
# Predict which variable in next_state was result of unification
# Then infer which pair was unified based on the variable mapping
```

---

### **BUG #3: VariableUnifierGFlowNet Returns log_softmax Instead of Logits** ⚠️ MINOR

**Location**: `src/gflownet_models.py:127`

**Problem**:
```python
# Line 127
return F.log_softmax(pair_scores_tensor, dim=-1)
```

**Why this might be wrong**:
- Other tacticians (AtomAdderGFlowNet) return **logits**, not log_probs
- The training code applies softmax again: `F.softmax(pair_logits, dim=-1)` (training.py:195)
- This applies softmax twice: `softmax(log_softmax(scores)) != softmax(scores)`

**Impact**:
- Incorrect probability distribution for variable pairs
- May lead to suboptimal variable unification choices

**Fix**:
```python
# Return raw scores (logits), not log_softmax
return pair_scores_tensor  # Remove log_softmax
```

---

### **BUG #4: Reward Scaling with Alpha=10 May Cause Numerical Issues** ⚠️ MODERATE

**Location**: `src/training.py:493`

**Problem**:
```python
scaled_reward = reward ** self.reward_scale_alpha  # alpha = 10.0 in notebook
```

**Why this is problematic**:
- For `reward = 0.8`: `0.8^10 ≈ 0.107` → `log(0.107) ≈ -2.23`
- For `reward = 0.5`: `0.5^10 ≈ 0.001` → `log(0.001) ≈ -6.91`
- For `reward = 1e-6`: `(1e-6)^10 = 1e-60` → `log(1e-6) ≈ -13.82`

While this creates large differences in log-space (which is good for credit assignment), it can also:
1. Create huge gradient magnitudes for small reward differences
2. Cause numerical instability in optimization
3. Make the loss landscape very steep, leading to oscillation

**Recommendation**: Try `alpha = 1.0` or `2.0` first to ensure stable training.

---

### **ISSUE #5: Log_Z Can Absorb Differences Instead of Fixing Policy** ⚠️ MODERATE

**Location**: `src/training.py:124`

**Problem**:
```python
self.log_Z = torch.nn.Parameter(torch.tensor([0.0]))
```

**Theoretical Issue**:
The Trajectory Balance loss is:
```
L = (log_Z + sum_log_pf - log_reward - sum_log_pb)^2
```

To minimize this loss, the model has two choices:
1. **Correct approach**: Adjust policy to match rewards (reduce P_F for bad actions)
2. **Shortcut**: Increase log_Z to balance the equation without fixing the policy

If log_Z is learning too fast relative to the policy network, it can absorb the difference and prevent proper credit assignment.

**Evidence**: If you observe that:
- `log_Z` increases over time
- Policy probabilities don't change much
- Loss decreases but policy doesn't improve

Then log_Z is compensating instead of the policy learning.

**Fix**:
1. Use a smaller learning rate for log_Z
2. Or use Detailed Balance loss instead (no global Z parameter)
3. Monitor log_Z during training to detect this issue

---

## 🔍 Why 50-50 Probabilities Persist

Given the analysis above, here's why the model learns 50-50 probabilities for ADD_ATOM vs UNIFY_VARIABLES at the first step:

### Root Cause: Replay Buffer + Log_Z Compensation

1. **On-policy trajectory** (33% chance UNIFY is sampled):
   - UNIFY_VARIABLES → ... → reward ≈ 1e-6
   - Loss: `(log_Z + log_P(UNIFY) + ... - log(1e-6) - sum_log_pb)^2`
   - Gradient wants to reduce `log_P(UNIFY)`

2. **But then replay buffer kicks in** (50% of the time):
   - Replayed trajectory: ADD_ATOM → ... → reward ≈ 0.8
   - Loss: `(log_Z + log_P(ADD) + ... - log(0.1) - sum_log_pb)^2`
   - This loss is ADDED to the on-policy loss
   - The combined gradient is diluted

3. **Log_Z compensates**:
   - Instead of reducing P(UNIFY), log_Z increases to balance both equations
   - This minimizes loss without fixing the policy

### Contributing Factors:

1. **Backward probability heuristic for UNIFY** (Bug #2)
2. **Reward scaling** creating extreme log values (Bug #4)
3. **Possible state embedding similarity** (if initial_state and unified_state have similar embeddings)

---

## 🎯 Recommended Fixes (Priority Order)

### 1. **FIX REPLAY BUFFER DILUTION** (Highest Priority)

**Option A**: Don't add replay loss, use it as replacement
```python
if use_replay:
    total_loss = off_policy_loss
else:
    total_loss = on_policy_loss
```

**Option B**: Reduce replay probability during early training
```python
# Anneal replay_probability from 0.0 to 0.5 over first 50% of episodes
replay_prob = min(0.5, episode / (num_episodes * 0.5) * 0.5)
```

**Option C**: Use importance weighting for replay
```python
# Weight replay loss by probability ratio
replay_weight = 0.5  # or tune this
total_loss = on_policy_loss + replay_weight * off_policy_loss
```

### 2. **MONITOR LOG_Z** (High Priority)

Add logging to track log_Z evolution:
```python
if episode % 100 == 0:
    print(f"log_Z: {self.log_Z.item():.4f}")
```

If log_Z grows large (> 5.0), consider:
- Using Detailed Balance instead of Trajectory Balance
- Reducing log_Z learning rate
- Regularizing log_Z (add `lambda * log_Z^2` to loss)

### 3. **FIX VARIABLE UNIFIER LOG_SOFTMAX** (Medium Priority)

Change line 127 in `src/gflownet_models.py`:
```python
# Before
return F.log_softmax(pair_scores_tensor, dim=-1)

# After
return pair_scores_tensor  # Return logits
```

### 4. **REDUCE REWARD_SCALE_ALPHA** (Medium Priority)

In notebook, change:
```python
'reward_scale_alpha': 1.0,  # instead of 10.0
```

### 5. **IMPLEMENT PROPER BACKWARD POLICY FOR UNIFY** (Lower Priority)

This requires more significant changes to the BackwardVariableSplitter to properly predict which variables were unified.

---

## 📊 Experimental Verification

To verify these fixes, run experiments and check:

1. **Policy probabilities**: After training, check P(ADD_ATOM | initial_state) and P(UNIFY_VARIABLES | initial_state)
   - Should see P(ADD_ATOM) → ~0.9-1.0, P(UNIFY) → ~0.0-0.1

2. **Reward distribution**: Plot rewards of sampled trajectories
   - Should see increasing proportion of high-reward trajectories

3. **Log_Z value**: Track log_Z over training
   - Should stabilize, not grow unbounded

4. **Gradient norms**: Track gradient norms for policy vs log_Z
   - Policy gradients should dominate

---

## 🎓 Theoretical Correctness Assessment

### Overall: **7/10** (Good foundation, some critical bugs)

**Strengths**:
- ✅ Trajectory Balance loss is correctly formulated
- ✅ Logic engine is sound (SLD resolution)
- ✅ Reward calculation is comprehensive and correct
- ✅ State representation is rich and well-designed

**Weaknesses**:
- ⚠️ Replay buffer implementation dilutes on-policy learning
- ⚠️ Backward probability uses heuristics instead of learned policy
- ⚠️ Potential numerical issues with reward scaling
- ⚠️ Log_Z can act as a "cheat" to minimize loss without fixing policy

**Verdict**: The theoretical framework is solid, but implementation details prevent the model from learning properly. With the fixes above, the model should converge to the correct policy.
