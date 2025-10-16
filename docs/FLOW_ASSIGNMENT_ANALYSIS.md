# Flow Assignment Analysis

## Current TB Loss Mechanism

### Standard Trajectory Balance Loss

```python
TB_loss = (log Z + sum(log P_F) - log R - sum(log P_B))²
```

Where:
- `Z`: Partition function (total flow from initial state)
- `P_F`: Forward transition probabilities
- `R`: Reward (terminal state flow)
- `P_B`: Backward transition probabilities

**Flow conservation constraint:** `Z · P_F(τ) = R(s) · P_B(τ)`

### Current Implementation

```python
# From src/training.py line 238
loss = (self.log_Z + sum_log_pf - log_reward - sum_log_pb) ** 2
```

**Problems with current implementation:**

1. **Single learnable Z**: One `log_Z` parameter for all trajectories
2. **No explicit flow assignment**: Flow is implicitly determined by minimizing squared error
3. **Equal weight to all trajectories**: Every trajectory gets same loss weight regardless of reward

## The Core Problem

### Example Scenario

Training sees two trajectories:

**Trajectory A** (degenerate):
- Length: 1 step
- Reward: 0.1
- Forward prob: P_F = 0.5
- TB constraint: `Z × 0.5 = 0.1 × P_B`
- Desired: `Z = 0.2 × P_B` (if P_B ≈ 1)

**Trajectory B** (correct):
- Length: 5 steps
- Reward: 0.9
- Forward prob: P_F = 0.001 (product of 5 small probs)
- TB constraint: `Z × 0.001 = 0.9 × P_B`
- Desired: `Z = 900 × P_B` (if P_B ≈ 1)

**Contradiction!** Single Z must satisfy both:
- `Z ≈ 0.2` (for trajectory A)
- `Z ≈ 900` (for trajectory B)

### What Actually Happens

Gradient descent finds a compromise:
1. If Z ≈ 0.2, trajectory A has low loss (good)
2. Trajectory B has huge loss: `(log 0.2 + log 0.001 - log 0.9)² ≈ 81`
3. To reduce loss, model **increases P_F for trajectory A** (easier than finding B again)
4. Result: Converges to sampling trajectory A with P_F ≈ 1.0

**The model optimizes loss, not reward!**

## Why High-Reward Trajectories Are Unstable

From our experiments:
- Episode 20: Found rule with reward 0.965 ✓
- Episodes 21-1000: Never found again ✗

**Analysis:**

When the model finds the high-reward trajectory once:
1. TB loss is initially huge (Z too small)
2. Model has two options:
   - Option A: Increase Z and P_F for high-reward trajectory
   - Option B: Decrease P_F for high-reward trajectory (make it rare)

**Option B is easier** because:
- Increasing P_F for 5-step trajectory requires coordinated changes across 5 decisions
- Decreasing P_F only requires changing one decision
- The 1-step trajectory still provides gradient signal every episode

Result: High-reward trajectory becomes rarer, then disappears.

## Proposed Solutions

### Solution 1: Reward-Weighted TB Loss

**Idea:** Weight loss by reward to prioritize high-reward trajectories.

```python
def compute_trajectory_balance_loss_weighted(self, trajectory, reward):
    # Standard TB loss
    sum_log_pf = sum(step.log_pf for step in trajectory)
    log_reward = torch.log(torch.tensor(reward + 1e-8))
    sum_log_pb = -np.log(len(self.predicate_vocab) + 10) * len(trajectory)
    sum_log_pb = torch.tensor(sum_log_pb, dtype=torch.float32)

    tb_loss = (self.log_Z + sum_log_pf - log_reward - sum_log_pb) ** 2

    # Weight by reward (normalized)
    weight = reward / (reward + 0.1)  # 0.1 baseline prevents zero weight

    return weight * tb_loss
```

**Effect:**
- Reward 0.9: weight = 0.9 / 1.0 = 0.90 (high priority)
- Reward 0.1: weight = 0.1 / 0.2 = 0.50 (medium priority)
- Reward 0.01: weight = 0.01 / 0.11 = 0.09 (low priority)

**Benefit:** Model focuses gradient descent on matching flow to high-reward states.

### Solution 2: Detailed Balance (State-Specific Z)

**Idea:** Replace single Z with flow function F(s).

Standard TB: `Z · P_F(s→s') = R(s_T) · P_B(s'→s)`

Detailed Balance: `F(s) · P_F(s→s') = F(s') · P_B(s'→s)`

Where `F(s_T) = R(s_T)` at terminals.

**Implementation:**
```python
class StateFlowEstimator(nn.Module):
    """Estimate flow F(s) for each state."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_embedding):
        return self.flow_net(state_embedding)

def compute_detailed_balance_loss(self, trajectory, reward):
    """Compute loss with state-specific flows."""
    losses = []

    for i, step in enumerate(trajectory):
        # Get flow for current and next state
        state_emb = self.encode_state(step.state)
        next_emb = self.encode_state(step.next_state)

        F_s = self.flow_estimator(state_emb)
        F_s_next = self.flow_estimator(next_emb)

        # Forward flow: F(s) * P_F(s->s')
        forward_flow = F_s + step.log_pf

        # Backward flow: F(s') * P_B(s'->s)
        log_pb = self.estimate_backward_prob(step)
        backward_flow = F_s_next + log_pb

        # Detailed balance constraint
        loss = (forward_flow - backward_flow) ** 2
        losses.append(loss)

    # Terminal constraint: F(s_T) = R(s_T)
    terminal_emb = self.encode_state(trajectory[-1].next_state)
    F_terminal = self.flow_estimator(terminal_emb)
    terminal_loss = (F_terminal - torch.log(torch.tensor(reward))) ** 2

    return torch.stack(losses).mean() + terminal_loss
```

**Benefit:** Each state can have appropriate flow, no single-Z bottleneck.

### Solution 3: SubTB - Mixed Trajectory Balance

**Idea:** Use SubTB (Sub-Trajectory Balance) which decomposes trajectories.

```python
def compute_subtb_loss(self, trajectory, reward):
    """SubTB: Balance over sub-trajectories."""
    losses = []

    # Initial state flow
    log_Z = self.log_Z

    # For each step, balance flow
    for i, step in enumerate(trajectory):
        # Flow into this state
        if i == 0:
            log_F_in = log_Z
        else:
            # Accumulated flow from previous steps
            log_F_in = log_Z + sum(trajectory[j].log_pf for j in range(i))

        # Flow out of this state
        log_F_out = log_F_in + step.log_pf

        # Expected flow (from reward)
        remaining_steps = len(trajectory) - i - 1
        # Estimate: flow should decay toward reward
        log_F_expected = torch.log(torch.tensor(reward)) + remaining_steps * np.log(0.5)

        loss = (log_F_out - log_F_expected) ** 2
        losses.append(loss)

    return torch.stack(losses).mean()
```

**Benefit:** Balances flow at each step, catches issues earlier.

### Solution 4: Log-Reward Scaling

**Idea:** Current loss uses `log R`, but log compresses reward differences.

```python
# Current: log compresses differences
# log(0.9) = -0.105
# log(0.1) = -2.303
# Difference: 2.2

# Proposed: Use R^α for α > 1 to amplify differences
def compute_tb_loss_scaled(self, trajectory, reward, alpha=2.0):
    sum_log_pf = sum(step.log_pf for step in trajectory)

    # Scale reward to amplify differences
    scaled_reward = reward ** alpha  # α=2: 0.9²=0.81, 0.1²=0.01 (81x difference!)
    log_reward = torch.log(torch.tensor(scaled_reward + 1e-8))

    sum_log_pb = -np.log(len(self.predicate_vocab) + 10) * len(trajectory)
    sum_log_pb = torch.tensor(sum_log_pb, dtype=torch.float32)

    loss = (self.log_Z + sum_log_pf - log_reward - sum_log_pb) ** 2

    return loss
```

**Effect:**
- α=1.0: log(0.9)=-0.11, log(0.1)=-2.30 → diff = 2.2
- α=2.0: log(0.81)=-0.21, log(0.01)=-4.61 → diff = 4.4 (2x larger!)
- α=3.0: log(0.73)=-0.32, log(0.001)=-6.91 → diff = 6.6 (3x larger!)

**Benefit:** Makes high-reward trajectories more attractive in flow assignment.

### Solution 5: Trajectory Replay Buffer

**Idea:** Explicitly maintain high-reward trajectories in memory.

```python
class TrajectoryReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = []
        self.capacity = capacity

    def add(self, trajectory, reward):
        self.buffer.append((trajectory, reward))
        # Keep only top-K by reward
        self.buffer.sort(key=lambda x: x[1], reverse=True)
        self.buffer = self.buffer[:self.capacity]

    def sample(self, n=1):
        # Sample with probability proportional to reward
        rewards = np.array([r for _, r in self.buffer])
        probs = rewards / rewards.sum()
        indices = np.random.choice(len(self.buffer), size=n, p=probs)
        return [self.buffer[i] for i in indices]

# In training loop
def train_step_with_replay(self, initial_state, pos_ex, neg_ex):
    # Generate new trajectory
    trajectory, reward = self.generate_trajectory(initial_state, pos_ex, neg_ex)

    # Add to replay if good
    if reward > 0.3:
        self.replay_buffer.add(trajectory, reward)

    # Train on new trajectory
    loss_new = self.compute_trajectory_balance_loss(trajectory, reward)

    # Also train on replayed high-reward trajectory (50% of time)
    if len(self.replay_buffer) > 0 and random.random() < 0.5:
        replay_traj, replay_reward = self.replay_buffer.sample(1)[0]
        loss_replay = self.compute_trajectory_balance_loss(replay_traj, replay_reward)
        loss = 0.5 * loss_new + 0.5 * loss_replay
    else:
        loss = loss_new

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

**Benefit:** High-reward trajectories remain in training distribution even after exploration decays.

## Recommended Approach

**Combine Solutions 1 and 5:**

1. **Reward-weighted TB loss** (easy to implement, theoretically sound)
2. **Trajectory replay buffer** (practical solution to maintain high-reward paths)

### Implementation Priority

**Immediate (Easy):**
```python
# Solution 1: Reward-weighted loss
weight = reward / (reward + 0.1)
loss = weight * tb_loss
```

**Short-term (Moderate):**
```python
# Solution 5: Replay buffer
# Add TrajectoryReplayBuffer class
# Modify train_step to use replay
```

**Long-term (Research):**
```python
# Solution 2: Detailed balance with flow network
# More complex, but theoretically superior
```

**Experimental (Worth trying):**
```python
# Solution 4: Log-reward scaling
# Just change one line, might have big impact
scaled_reward = reward ** 2.0  # Try α=2.0, 3.0, 4.0
```

## Expected Impact

Based on our findings:
- Current: 58 high-reward samples → all forgotten by episode 257
- With weighted loss: High-reward samples get stronger gradient → slower forgetting
- With replay buffer: High-reward samples stay in distribution → no forgetting

**Hypothesis:** Combination should maintain high-reward trajectories throughout training.

## Next Steps

1. Implement reward-weighted TB loss (10 lines of code)
2. Run 1000-episode experiment with Combined Aggressive
3. Check if avg reward at episode 1000 > 0.5 (vs current 0.14)
4. If successful, add replay buffer for further improvement
