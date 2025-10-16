# Root Cause Analysis: Why Model Converges to `grandparent(X0, X0)`

## Executive Summary

After extensive analysis including encoding inspection, action probabilities, reward landscape evaluation, loss function examination, and 1000-episode training runs, we identified **THREE fundamental issues** and **ONE critical bug** causing convergence to degenerate rules.

---

## Issues Identified

### ❌ CRITICAL BUG: Logic Engine Couldn't Handle Existential Variables

**Problem:** The original logic engine returned `False` for any rule with unbound variables in the body, even when those variables could be bound by background facts.

**Example:**
```prolog
Rule: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
Facts: parent(alice, bob), parent(bob, charlie)
Query: grandparent(alice, charlie)?

Old engine: FALSE (Z is unbound, give up)
Fixed engine: TRUE (try Z=bob, succeeds!)
```

**Impact:** Even with correct rules and background knowledge, the engine couldn't prove anything, so all rewards were ~0.

**Fix:** ✅ Implemented proper backtracking search that tries all possible bindings for existential variables from background facts.

---

### 🔍 ISSUE 1: No Background Knowledge (Original Setup)

**Problem:** Without `parent` facts, the logic engine cannot prove ANY grandparent relationships, even with correct rules.

**Analysis:**
```
Rule: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
Query: grandparent(alice, charlie)

To prove:
  1. Find Z such that parent(alice, Z)
  2. AND parent(Z, charlie)

Without facts: IMPOSSIBLE! No Z exists in the knowledge base.
Result: pos_score = 0 for ALL rules
```

**Consequence:**
- Reward = 0.9 × (0 × neg_score) + 0.1 × simplicity
- Reward = 0.1 × simplicity ONLY
- Model learns: "Just maximize simplicity"
- Shortest trajectory wins: 1-step unification

**Fix:** ✅ Added background_facts parameter to LogicEngine

---

### 🔍 ISSUE 2: Reward Function Bias Toward Simplicity

**Problem:** When pos_score = 0 (no background knowledge), reward reduces to simplicity only.

**Math:**
```
reward = 0.9 × accuracy + 0.1 × simplicity - penalty
accuracy = pos_score × neg_score

When pos_score = 0:
  accuracy = 0 × neg_score = 0
  reward = 0 + 0.1 × simplicity
  reward = 0.1 / (1 + num_atoms)

Maximizing reward → Minimizing num_atoms → Empty rules!
```

**Observed Behavior:**
- `grandparent(X0, X0)`: 0 atoms, reward = 0.1 ✓
- `grandparent(X0, X1) :- parent(X2, X3)`: 1 atom, reward = 0.05
- Correct rule with 2 atoms: reward = 0.033

**Model learns:** "Fewer atoms = better"

---

### 🔍 ISSUE 3: Trajectory Length Bias in Training

**Problem:** Trajectory Balance loss favors shorter trajectories.

**Analysis:**
```
TB Loss = (log Z + Σ log P_F - log R - Σ log P_B)²

Shorter trajectory:
  - Fewer actions → smaller |Σ log P_F|
  - Easier to balance with log R and log Z
  - Simpler gradient landscape

Longer trajectory:
  - More actions → larger |Σ log P_F|
  - Harder to balance
  - More complex optimization
```

**Observed:**
- Episodes 0-99: Avg 3.4 steps
- Episodes 100-999: Avg 1.0 step (converged)
- 100% of sampled theories: 1-step trajectories

**Conclusion:** Model discovered that 1-step (UNIFY only) is easiest to optimize!

---

## Training Results (1000 Episodes)

### Without Background Knowledge
```
Episodes    0-  99: Avg reward=0.0575, Avg steps=3.4
Episodes  100- 199: Avg reward=0.0988, Avg steps=1.1
Episodes  200- 299: Avg reward=0.0995, Avg steps=1.0
Episodes  300- 999: Avg reward=0.1000, Avg steps=1.0

Final: 100% of theories = grandparent(X0, X0)
```

### With Background Knowledge (After Bug Fix)
```
Correct rule reward: 0.9333
Degenerate rule reward: 0.1000

Model still converges to: grandparent(X0, X0)
```

**Why?** Even though correct rule has 9× higher reward, the model doesn't explore enough to discover the multi-step trajectory!

---

## Reward Landscape Analysis

| Theory | Atoms | Pos | Neg | Accuracy | Reward |
|--------|-------|-----|-----|----------|--------|
| `grandparent(X0, X1)` | 0 | 1/1 | 1/1 | 0.00 | 0.00 (penalty) |
| `grandparent(X0, X0)` | 0 | 0/1 | 1/1 | 0.00 | 0.10 |
| `... :- parent(X2, X3)` | 1 | 1/1 | 1/1 | 0.00 | 0.00 (penalty) |
| **CORRECT RULE** | 2 | 1/1 | 0/1 | 1.00 | **0.93** ✓ |

✅ Reward function correctly assigns highest reward to correct rule!
❌ Model doesn't explore enough to find it!

---

## Root Causes Summary

1. **Critical Bug** ✅ FIXED
   - Logic engine couldn't handle existential variables
   - Now implements proper backtracking search

2. **Missing Background Knowledge** ✅ IDENTIFIED
   - Without facts, no rule can get positive coverage
   - Need to provide parent relationships

3. **Exploration Failure** ❌ OPEN ISSUE
   - Model converges to 1-step local optimum
   - Doesn't explore multi-step trajectories enough
   - TB loss may bias toward shorter trajectories

4. **Backward Policy** ❌ OPEN ISSUE
   - Current: Uniform constant (doesn't depend on state)
   - Should: Learn state-dependent backward policy
   - Impact: Biases trajectory length preference

---

## Solutions Implemented

### ✅ Fixed Logic Engine
```python
class LogicEngine:
    def __init__(self, background_facts=None):
        self.background_facts = set(background_facts or [])

    def _prove_body(self, body, substitution, facts):
        # Try all possible bindings for unbound variables
        # from background facts
        ...
```

### ✅ Added Background Knowledge Support
```python
background = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
]
engine = LogicEngine(background_facts=background)
```

---

## Solutions Needed (Future Work)

### 1. Improve Exploration
- **Option A:** Add entropy bonus to encourage action diversity
- **Option B:** Use ε-greedy exploration
- **Option C:** Temperature-based sampling
- **Option D:** Curriculum learning (start with easier examples)

### 2. Fix Backward Policy
- **Current:** `log P_B = -log(vocab_size + 10) × length`
- **Better:** Learn state-dependent backward policy
- **Impact:** Better trajectory balance, less length bias

### 3. Reward Shaping
- **Option A:** Give partial credit for rule structure
- **Option B:** Reward intermediate progress
- **Option C:** Add diversity bonus

### 4. Training Improvements
- **Option A:** Increase learning rate for more exploration early
- **Option B:** Use replay buffer to revisit good trajectories
- **Option C:** Multi-task learning with varying difficulty

---

## Key Insights

1. **The reward function is correct!**
   - Correct rule gets 0.93 vs 0.10 for degenerate rule
   - Problem is exploration, not reward signal

2. **Trajectory Balance has implicit bias**
   - Shorter trajectories are easier to optimize
   - Model rationally chooses simplest solution

3. **Background knowledge is essential**
   - Without facts, ILP is impossible
   - Logic engine needs grounding for rules

4. **The bug was critical**
   - Original engine couldn't use background knowledge
   - Even correct rules got 0 reward

---

## Recommended Next Steps

**Immediate:**
1. ✅ Use background knowledge in all experiments
2. ✅ Verify logic engine handles complex rules
3. ⏭️ Add exploration mechanisms

**Short-term:**
4. Implement learned backward policy
5. Add entropy regularization
6. Try temperature-based sampling

**Long-term:**
7. Beam search during generation
8. Hierarchical exploration strategies
9. Meta-learning for better initialization

---

## Testing Checklist

- [x] Logic engine handles existential variables
- [x] Background knowledge integration works
- [x] Correct rule gets high reward (0.93)
- [x] Degenerate rule gets low reward (0.10)
- [ ] Model explores multi-step trajectories
- [ ] Model discovers correct rules
- [ ] Training converges to high-reward solutions

**Status:** 4/7 passing. Core infrastructure works, need exploration improvements.

---

**Analysis Date:** 2025-10-15
**1000 Episode Training:** Completed
**Critical Bug:** Fixed
**Next Priority:** Exploration mechanisms
