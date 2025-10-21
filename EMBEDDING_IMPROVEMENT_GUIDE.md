# Guide: Improving Rule Embeddings

**Problem:** Graph encoder produces very similar embeddings for all rules, even with different semantics.

**Impact:** GFlowNet cannot learn which states lead to good rewards → poor performance

---

## Approaches (Ranked by Priority)

### 🥇 1. Contrastive Pre-Training (RECOMMENDED FIRST)

**Why:** Quick to implement, proven effective, doesn't require architecture changes

**Implementation:** See `contrastive_pretraining.py`

**How it works:**
- Train encoder BEFORE GFlowNet training
- Learns to distinguish:
  - ✓ Positive pairs: Same rule with renamed variables → SIMILAR embeddings
  - ✗ Negative pairs: Different variable connections → DIFFERENT embeddings

**Usage:**
```python
from contrastive_pretraining import ContrastivePreTrainer, generate_base_rules

# Generate base rules
base_rules = generate_base_rules(predicate_vocab, predicate_arities, num_rules=100)

# Create pre-trainer
pretrainer = ContrastivePreTrainer(
    state_encoder=state_encoder,
    graph_constructor=graph_constructor,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities
)

# Pre-train for 100-500 epochs
pretrainer.pretrain(base_rules, num_epochs=200)

# Now use the pre-trained encoder in GFlowNet training
trainer = GFlowNetTrainer(
    state_encoder=state_encoder,  # Already pre-trained!
    ...
)
```

**Expected results:**
- Embeddings should show clear clusters
- Similar rules (renamed variables) → high similarity (>0.95)
- Different rules (changed connections) → low similarity (<0.7)

---

### 🥈 2. Architecture Improvements

**Why:** Makes encoder more expressive, can capture finer structural details

**Implementation:** See `improved_graph_encoder.py`

**Key improvements:**

#### a) **Edge Features** (High impact)
Current encoder only uses node features. Add edge features:
- Argument position (1st arg connects to 2nd arg)
- Edge direction (which variable flows to which)
- Predicate identity

```python
from improved_graph_encoder import ImprovedGraphConstructor, ImprovedStateEncoder

# Replace existing encoder
graph_constructor = ImprovedGraphConstructor(predicate_vocab)
state_encoder = ImprovedStateEncoder(
    node_feature_dim=10,
    edge_feature_dim=10,  # NEW: edge features
    embedding_dim=64,
    num_layers=3
)
```

**Why this helps:**
- `parent(X,Z), parent(Z,Y)` has different edge patterns than `parent(X,Z), parent(Y,Z)`
- First: X→Z→Y (chain)
- Second: X→Z←Y (convergent)

#### b) **Graph Isomorphism Network (GIN)** (Medium impact)
More powerful than current GCN layers:
- Provably more expressive for non-isomorphic graphs
- Better at distinguishing structural differences

Already implemented in `ImprovedStateEncoder`

#### c) **Multiple Pooling Strategies** (Low-Medium impact)
Combine different pooling methods:
- Mean pooling: average node features
- Max pooling: most prominent features
- Sum pooling: total graph properties

Already implemented in `ImprovedStateEncoder`

---

### 🥉 3. Training Improvements

#### a) **Auxiliary Contrastive Loss** (Medium effort, high impact)

Add contrastive loss DURING GFlowNet training:

```python
# In training loop
def train_step_with_contrastive(self, initial_state, positives, negatives, alpha=0.1):
    # Regular GFlowNet loss
    trajectory, reward = self.generate_trajectory(initial_state, positives, negatives)
    gfn_loss = self.compute_trajectory_balance_loss(trajectory, reward)

    # Contrastive loss
    # Generate augmented versions
    renamed_state = augmenter.variable_renaming(trajectory[0].state)
    different_state = augmenter.variable_connection_change(trajectory[0].state)

    emb_original = self.get_embedding(trajectory[0].state)
    emb_renamed = self.get_embedding(renamed_state)
    emb_different = self.get_embedding(different_state)

    # Encourage: similar(original, renamed), dissimilar(original, different)
    contrastive_loss = (
        -F.cosine_similarity(emb_original, emb_renamed, dim=-1).mean() +
        F.cosine_similarity(emb_original, emb_different, dim=-1).mean()
    )

    # Combined loss
    total_loss = gfn_loss + alpha * contrastive_loss

    total_loss.backward()
    self.optimizer.step()
```

#### b) **Reward-Weighted Replay** (Low effort, medium impact)

Already implemented! Just enable it:
```python
trainer = GFlowNetTrainer(
    ...
    use_replay_buffer=True,
    replay_probability=0.5,  # 50% of training uses replay
    buffer_reward_threshold=0.5,  # Only store good rules
)
```

This helps by:
- Replaying high-reward trajectories more often
- Encoder sees successful rules multiple times
- Learns to distinguish good states from bad

#### c) **Curriculum Learning** (High effort, high impact)

Train in stages:

```python
# Stage 1: Learn basic patterns (1-2 atoms)
trainer.max_body_length = 2
train(num_episodes=1000)

# Stage 2: Medium complexity (2-3 atoms)
trainer.max_body_length = 3
train(num_episodes=2000)

# Stage 3: Full complexity
trainer.max_body_length = 4
train(num_episodes=5000)
```

---

### 🏅 4. Feature Engineering

#### a) **Add Graph Topology Features** (Medium effort)

Add global structural features to each node:

```python
def enhanced_node_features(var, theory):
    # Existing features
    features = [...]

    # NEW: Topology features
    # - Degree (how many connections)
    # - Shortest path to head variables
    # - Betweenness centrality
    # - Is in a cycle?

    G = build_networkx_graph(theory)
    degree = G.degree(var.id)

    head_vars = [v.id for v in theory[0].head.args]
    shortest_path = min(
        nx.shortest_path_length(G, var.id, hv)
        for hv in head_vars if nx.has_path(G, var.id, hv)
    ) if head_vars else -1

    in_cycle = nx.is_in_cycle(G, var.id)

    features.extend([degree, shortest_path, 1.0 if in_cycle else 0.0])

    return features
```

#### b) **Structural Fingerprints** (High effort)

Add discrete structural patterns:

```python
def get_structural_fingerprint(theory):
    """
    Returns a binary vector indicating presence of patterns:
    - [1,0,0,...] = chain pattern (X→Y→Z)
    - [0,1,0,...] = convergent pattern (X→Z←Y)
    - [0,0,1,...] = divergent pattern (X→Y, X→Z)
    - etc.
    """
    fingerprint = [0] * 10

    # Check for chain
    if has_chain_pattern(theory):
        fingerprint[0] = 1

    # Check for convergent
    if has_convergent_pattern(theory):
        fingerprint[1] = 1

    # ... more patterns

    return fingerprint
```

---

### 🔬 5. Supervised Pre-Training

If you have labeled data (rule → quality score):

```python
class SupervisedPreTrainer:
    def __init__(self, encoder, graph_constructor):
        self.encoder = encoder
        self.graph_constructor = graph_constructor

        # Regression head to predict rule quality
        self.quality_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Predict quality score
            nn.Sigmoid()
        )

    def train(self, rules_with_scores):
        for rule, quality_score in rules_with_scores:
            emb = self.encoder(self.graph_constructor.theory_to_graph(rule))
            predicted_quality = self.quality_predictor(emb)

            loss = F.mse_loss(predicted_quality, quality_score)
            loss.backward()
            optimizer.step()
```

Generate labeled data:
- Sample random rules
- Evaluate them on your task
- Train encoder to predict quality from structure

---

## 📋 Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Implement contrastive pre-training
2. ✅ Run pre-training for 200 epochs
3. ✅ Test embedding quality (should see improvement)
4. ✅ Train GFlowNet with pre-trained encoder

**Expected improvement:** 50-70% reduction in embedding similarity for different rules

### Phase 2: Architecture (2-3 days)
1. ✅ Add edge features to graph constructor
2. ✅ Replace GCN with GIN layers
3. ✅ Add multiple pooling strategies
4. ✅ Re-run contrastive pre-training
5. ✅ Train GFlowNet

**Expected improvement:** 70-85% reduction in embedding similarity

### Phase 3: Training Improvements (1-2 days)
1. ✅ Add auxiliary contrastive loss during GFlowNet training
2. ✅ Enable reward-weighted replay
3. ✅ Experiment with curriculum learning

**Expected improvement:** Faster convergence, higher final rewards

---

## 🧪 Testing Improvements

After each phase, run the embedding analysis:

```python
# Test semantic equivalence
rule1 = create_rule([('parent', (0,2)), ('parent', (2,1))])  # Chain
rule2 = create_rule([('parent', (0,2)), ('parent', (1,2))])  # Convergent

emb1 = get_embedding(rule1)
emb2 = get_embedding(rule2)

similarity = cosine_similarity([emb1], [emb2])[0,0]

print(f"Similarity: {similarity:.4f}")
# Before: ~0.999 (BAD)
# Target: <0.70 (GOOD)
```

---

## 📊 Expected Results Timeline

| Approach | Implementation Time | Expected Improvement | Risk |
|----------|-------------------|---------------------|------|
| Contrastive Pre-training | 4-6 hours | 🟢 High | 🟢 Low |
| Edge Features | 6-8 hours | 🟢 High | 🟡 Medium |
| GIN Layers | 4-6 hours | 🟡 Medium | 🟢 Low |
| Auxiliary Loss | 2-4 hours | 🟡 Medium | 🟢 Low |
| Supervised Pre-training | 8-12 hours | 🟡 Medium | 🟡 Medium |
| Curriculum Learning | 4-6 hours | 🟡 Medium | 🟡 Medium |
| Structural Fingerprints | 12-16 hours | 🟡 Medium | 🔴 High |

---

## 🎯 Success Criteria

You'll know it's working when:

1. **Embedding Analysis:**
   - Similar rules (renamed vars): similarity > 0.95 ✅
   - Different rules (changed structure): similarity < 0.70 ✅
   - Embedding visualization shows clear clusters

2. **GFlowNet Training:**
   - Mean reward increases during training
   - Model finds high-reward rules (>0.8) regularly
   - Replay buffer fills with diverse good rules

3. **Policy Behavior:**
   - Policy differentiates between good and bad partial rules
   - Action probabilities vary based on state quality
   - Model doesn't collapse to uniform distribution

---

## 🚨 Common Pitfalls

1. **Over-fitting to augmentations:**
   - Use diverse augmentation strategies
   - Don't make negatives too easy to distinguish

2. **Forgetting during GFlowNet training:**
   - Use lower learning rate for encoder than policy
   - Freeze encoder for first N episodes, then fine-tune

3. **Insufficient pre-training data:**
   - Generate at least 100 diverse base rules
   - Cover different lengths, structures, predicates

4. **Temperature too high/low in contrastive loss:**
   - Start with 0.5
   - If embeddings still too similar: decrease (0.1-0.3)
   - If training unstable: increase (0.7-1.0)

---

## 📚 Further Reading

- **Contrastive Learning:** SimCLR paper (Chen et al., 2020)
- **GIN:** "How Powerful are Graph Neural Networks?" (Xu et al., 2019)
- **Graph Embeddings for Logic:** "Learning Symbolic Rules for Reasoning" (Bosnjak et al., 2017)

---

## ✅ Next Steps

1. Start with **contrastive pre-training** (quickest win)
2. If that helps but not enough, add **edge features + GIN**
3. If still struggling, add **auxiliary contrastive loss** during training
4. Consider **curriculum learning** for harder tasks

Good luck! 🚀
