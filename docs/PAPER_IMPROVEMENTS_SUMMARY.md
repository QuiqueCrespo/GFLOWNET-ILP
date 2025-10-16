# Paper-Based Improvements Summary

## Overview

This document summarizes the improvements implemented based on research papers found in the `papers/` directory, specifically:

1. **SHAFT Paper** (`d4dd00392f.pdf`): "Efficient Symmetry-Aware Materials Generation via Hierarchical Generative Flow Networks"
2. **DAG-GFlowNet Paper** (`deleu22a.pdf`): "Bayesian Structure Learning with Generative Flow Networks"

## Implemented Improvements

### 1. Detailed Balance Loss (from DAG-GFlowNet)

**File**: `src/training.py` (lines 311-370)

**What it does**: Replaces the single partition function `Z` with per-transition flow balancing.

**Theory**:
- **Standard TB Loss**: `(log Z + Σ log P_F - log R - Σ log P_B)²`
- **Detailed Balance**: For each transition `s → s'`, enforce `F(s) × P_F(s→s') = F(s') × P_B(s'→s)`
- **Terminal constraint**: `F(s_terminal) = R(s_terminal)`

**Key advantage**: Removes the single-Z bottleneck where one global parameter must satisfy conflicting constraints for trajectories of different lengths and rewards.

**Implementation**:
```python
def compute_detailed_balance_loss(self, trajectory, reward):
    losses = []
    for i, step in enumerate(trajectory):
        # Compute flow at current state
        if i == 0:
            log_F_s = self.log_Z
        else:
            log_F_s = self.log_Z + sum(traj[j].log_pf for j in range(i))

        # Forward flow
        log_forward = log_F_s + step.log_pf

        # Backward flow
        if i == len(trajectory) - 1:
            # Terminal: F(s_T) = R
            log_F_s_next = log(reward)
        else:
            log_F_s_next = log_Z + sum(...)

        log_backward = log_F_s_next + log_pb

        # Balance constraint
        db_loss = (log_forward - log_backward)²
        losses.append(db_loss)

    return losses.mean()
```

**Usage**:
```python
trainer = GFlowNetTrainer(
    ...,
    use_detailed_balance=True  # Enable detailed balance loss
)
```

### 2. Replay Buffer for Off-Policy Learning (from DAG-GFlowNet)

**File**: `src/training.py` (lines 36-65)

**What it does**: Maintains a buffer of high-reward trajectories and trains on them even after exploration has decayed.

**Key components**:

**TrajectoryReplayBuffer class**:
```python
class TrajectoryReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory, reward):
        """Store trajectory"""
        self.buffer.append((trajectory, reward))

    def sample(self, n=1):
        """Sample with probability ∝ reward"""
        rewards = np.array([r for _, r in self.buffer])
        probs = rewards / rewards.sum()
        return sample based on probs
```

**Off-policy correction** (lines 130-181):
```python
def _recompute_step_log_pf(self, state, action_type, action_detail):
    """
    Re-compute log probabilities for replayed trajectories
    using CURRENT model parameters (critical for off-policy).
    """
    # Encode state with current encoder
    state_embedding = self.state_encoder(state)

    # Get current policy probabilities
    action_logits = self.gflownet.forward_strategist(state_embedding)
    action_probs = F.softmax(action_logits, dim=-1)

    # Return log prob under current policy
    return log_prob_action + log_prob_detail
```

**Training integration** (lines 372-458):
```python
def train_step(self, initial_state, pos_ex, neg_ex):
    # Generate new trajectory (on-policy)
    on_policy_trajectory, reward = self.generate_trajectory(...)

    # Add to replay buffer if high reward
    if reward > 0.7:
        self.replay_buffer.add(on_policy_trajectory, reward)

    # Compute on-policy loss
    loss = self.compute_loss(on_policy_trajectory, reward)

    # Sample from replay buffer (off-policy)
    if random.random() < self.replay_probability:
        replay_trajectory, replay_reward = self.replay_buffer.sample(1)[0]

        # CRITICAL: Re-compute log probs with current policy
        recomputed_trajectory = [
            TrajectoryStep(..., self._recompute_step_log_pf(...), ...)
            for step in replay_trajectory
        ]

        # Compute off-policy loss
        off_policy_loss = self.compute_loss(recomputed_trajectory, replay_reward)

        # Combine losses
        total_loss = on_policy_loss + off_policy_loss

    # Single backward pass
    total_loss.backward()
```

**Usage**:
```python
trainer = GFlowNetTrainer(
    ...,
    use_replay_buffer=True,
    replay_buffer_capacity=50,
    replay_probability=0.3  # 30% of time train on replayed trajectory
)
```

### 3. No-Decay Exploration Strategy

**Motivation**: Previous experiments showed that exploration decay causes convergence to degenerate 1-step solutions.

**Implementation**:
```python
from src.exploration import EntropyBonus, TrajectoryLengthBonus, TemperatureSchedule, CombinedExploration

exploration = CombinedExploration([
    EntropyBonus(alpha=0.05, decay=1.0),  # No decay!
    TrajectoryLengthBonus(beta=0.1, decay=1.0),  # No decay!
    TemperatureSchedule(T_init=2.0, T_final=2.0)  # Constant temperature!
])
```

**Key insight**: Setting `decay=1.0` means parameters never decrease, maintaining exploration throughout training.

### 4. Reward-Weighted Loss (from flow assignment experiments)

**File**: `src/training.py` (lines 304-307)

**What it does**: Weights the TB/DB loss by reward to prioritize high-reward trajectories.

```python
if self.reward_weighted_loss:
    weight = reward / (reward + 0.1)  # Normalized [0, 1]
    loss = weight * loss
```

**Effect**:
- Reward 0.9: weight = 0.90 (high priority)
- Reward 0.1: weight = 0.50 (medium priority)
- Reward 0.01: weight = 0.09 (low priority)

**Result**: Gradient descent focuses more on matching flow to high-reward states.

## Fixed LogicEngine (SLD Resolution)

**File**: `src/logic_engine.py`

**Problem**: The logic engine was using forward chaining incorrectly, breaking SLD resolution behavior needed for Prolog-like backward chaining.

**Fix**: Reimplemented proper SLD resolution:

1. **Goal-driven backward chaining**: Start with goal, find rule whose head unifies, then recursively prove body
2. **Proper substitution propagation**: Variables get bound as facts are found
3. **Correct unification**: Works with any Variable class (test vs production)

**Key changes**:
```python
def _prove_goal(self, theory, goal, depth):
    """Prove a ground goal using SLD resolution."""
    # Check background facts
    if goal in self.background_facts:
        return True

    # Try each rule
    for rule in theory:
        # Unify goal with rule head
        subst = self._unify_atom_with_example(rule.head, goal)
        if subst is not None:
            # Prove body with substitution
            if self._prove_body(theory, rule.body, subst, depth + 1):
                return True
    return False

def _prove_body(self, theory, body, substitution, depth):
    """Prove all atoms in body."""
    if not body:
        return True

    first_atom = body[0]
    remaining = body[1:]

    # Apply substitution to first atom
    ground_example = self._apply_substitution(first_atom, substitution)

    if ground_example is not None:
        # Fully ground - prove recursively
        if self._prove_goal(theory, ground_example, depth):
            return self._prove_body(theory, remaining, substitution, depth)
        return False
    else:
        # Unbound variables - try all matching facts
        for fact in self.background_facts:
            if fact.predicate_name == first_atom.predicate_name:
                new_subst = self._try_unify_with_fact(first_atom, fact, substitution)
                if new_subst is not None:
                    if self._prove_body(theory, remaining, new_subst, depth):
                        return True
        return False
```

**Variable detection** (works with any Variable class):
```python
is_variable = (hasattr(arg, '__class__') and
               arg.__class__.__name__ == 'Variable')
```

**Tests**: All 8 logic engine tests now pass.

## Experimental Configurations

The test script (`examples/test_paper_improvements.py`) compares 9 configurations:

1. **Baseline**: Standard TB loss, decaying exploration
2. **Detailed Balance**: DB loss only
3. **Replay Buffer**: TB loss + replay
4. **Reward Weighted**: TB loss + reward weighting
5. **No-Decay Exploration**: TB loss + permanent exploration
6. **DB + Replay**: Detailed balance + replay buffer
7. **Replay + Reward Weight**: Replay + reward weighting
8. **Replay + No-Decay**: Replay + permanent exploration
9. **All Combined**: DB + Replay + Reward Weight + No-Decay

## Expected Results

Based on previous experiments:

### Baseline Performance (from 1000-episode experiments):
- Avg reward (last 100): 0.14
- High-reward episodes: 58 (all in first 150 episodes)
- Converged to 1-step at episode: 257
- **Problem**: Exploration decay causes forgetting of high-reward trajectories

### Expected Improvements:

**Detailed Balance**:
- **Hypothesis**: Should delay convergence by removing single-Z bottleneck
- **Expected**: Convergence episode > 300, more distributed high-reward episodes

**Replay Buffer**:
- **Hypothesis**: Should maintain high-reward episodes throughout training
- **Expected**: High-reward episodes in later phases (200-500), avg reward > 0.3

**No-Decay Exploration**:
- **Hypothesis**: Should prevent convergence entirely or delay significantly
- **Expected**: No convergence (or convergence > 800), continuous high-reward discovery

**All Combined**:
- **Hypothesis**: Should maintain diverse high-reward trajectories indefinitely
- **Expected**:
  - No convergence to 1-step (maintains trajectory diversity)
  - Avg reward (last 100) > 0.5
  - High-reward episodes distributed throughout all phases
  - Multiple unique theories sampled at end

## Key Metrics to Track

1. **Convergence Episode**: When avg trajectory length < 1.1 for 50 consecutive episodes
2. **High-Reward Episodes**: Count of episodes with reward > 0.5
3. **Phase Distribution**:
   - Episodes 0-100 (early exploration)
   - Episodes 100-200 (exploration decay)
   - Episodes 200-500 (post-convergence)
4. **Final Theory Diversity**: Number of unique theories sampled at end
5. **Replay Buffer Usage**: How often replayed trajectories are used

## Implementation Files

- `src/training.py`: Main trainer with DB loss, replay buffer, off-policy correction
- `src/exploration.py`: Exploration strategies (existing, now with no-decay option)
- `examples/test_paper_improvements.py`: Comprehensive comparison experiment
- `tests/test_logic_engine.py`: SLD resolution tests (all passing)

## Running the Experiment

```bash
# Run full comparison (10000 episodes per configuration)
python examples/test_paper_improvements.py

# Results saved to:
# - analysis/paper_improvements_results.json
# - analysis/paper_improvements_test.log
```

## Next Steps After Results

1. **Analyze convergence patterns**: Which combinations prevent/delay convergence?
2. **Check reward distribution**: Are high-reward episodes maintained?
3. **Verify off-policy learning**: Is replay buffer effective?
4. **Test on harder problems**: Scale to more complex rule learning tasks
5. **Implement hierarchical policy** (SHAFT paper): Break down construction into stages

## References

1. **DAG-GFlowNet**: Deleu et al. (2022). "Bayesian Structure Learning with Generative Flow Networks"
   - Detailed balance loss formulation
   - Off-policy learning with replay
   - Backward transition probability estimation

2. **SHAFT**: Chen et al. (2024). "Efficient Symmetry-Aware Materials Generation via Hierarchical Generative Flow Networks"
   - Hierarchical policy decomposition
   - Physics-informed reward functions
   - Symmetry exploitation

3. **Previous Experiments**:
   - `docs/EXPLORATION_1000EP_ANALYSIS.md`: Exploration decay problem
   - `docs/FLOW_ASSIGNMENT_RESULTS.md`: Reward weighting +51% improvement
   - `docs/FLOW_ASSIGNMENT_ANALYSIS.md`: Single-Z bottleneck analysis
