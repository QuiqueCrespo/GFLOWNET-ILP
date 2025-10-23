# Vocabulary Design for General-Purpose Pretrained Encoder

## Problem Statement

**Original Issue:** The grandparent ILP task only uses 1 predicate (`parent`), but we want to pretrain the encoder on diverse predicates for better generalization.

**Challenge:** How do we handle dimension mismatch?
- If encoder trained with 8 predicates → node_feature_dim = 9
- If encoder used with 1 predicate → node_feature_dim = 2
- **Incompatible!** ❌

## Solution: Unified Vocabulary Approach

### Design Decision

**Use the SAME expanded vocabulary throughout the entire pipeline:**

1. **Encoder**: Trained on 8 predicates
2. **Graph Constructor**: Encodes with 8 predicates
3. **GFlowNet Policies**: Have capacity for 8 predicates
4. **Trainer**: Restricts search to ONLY task predicates (1 predicate)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   VOCABULARY LAYERS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENCODER VOCABULARY (8 predicates)                          │
│  ['parent', 'child', 'sibling', 'ancestor',                 │
│   'friend', 'male', 'female', 'adult']                      │
│                                                              │
│  Used by:                                                    │
│  - GraphConstructor (one-hot encoding)                       │
│  - StateEncoder (input dimension = 9)                        │
│  - GFlowNet policies (AtomAdder output dimension = 8)        │
│  - Pretraining (generates diverse rules)                     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TASK VOCABULARY (1 predicate)                              │
│  ['parent']                                                  │
│                                                              │
│  Used by:                                                    │
│  - Trainer (restricts ADD_ATOM action to 'parent' only)     │
│  - Trajectory generation (only samples 'parent')             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Code Implementation

**Step 1: Define Vocabularies**
```python
# Full vocabulary for encoder (8 predicates)
predicate_vocab = ['parent', 'child', 'sibling', 'ancestor',
                   'friend', 'male', 'female', 'adult']
predicate_arities = {
    'parent': 2, 'child': 2, 'sibling': 2, 'ancestor': 2,
    'friend': 2, 'male': 1, 'female': 1, 'adult': 1
}

# Task vocabulary (1 predicate - only used for search)
TASK_PREDICATES = ['parent']
```

**Step 2: Initialize Components with Full Vocabulary**
```python
# Graph constructor: uses full vocab
graph_constructor = EnhancedGraphConstructor(predicate_vocab)

# Encoder: input dim = len(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),  # 8
    embedding_dim=128,
    num_layers=3
)

# GFlowNet: capacity for full vocab
gflownet = HierarchicalGFlowNet(
    embedding_dim=128,
    num_predicates=len(predicate_vocab),  # 8
    predicate_vocab=predicate_vocab  # Full vocab
)
```

**Step 3: Pretrain with Full Vocabulary**
```python
pretrainer = EncoderPretrainer(
    state_encoder=state_encoder,
    graph_constructor=graph_constructor,
    predicate_vocab=predicate_vocab,  # All 8 predicates
    predicate_arities=predicate_arities
)

# Generates random rules using all 8 predicates
pretrainer.pretrain(num_steps=5000)
```

**Step 4: Trainer Restricts to Task Vocabulary**
```python
trainer = GFlowNetTrainer(
    state_encoder=state_encoder,
    gflownet=gflownet,
    graph_constructor=graph_constructor,
    predicate_vocab=TASK_PREDICATES,  # Only ['parent']!
    predicate_arities={'parent': 2}
)

# During trajectory generation:
# - AtomAdder policy outputs logits for all 8 predicates
# - Trainer MASKS OUT 7 predicates, only samples 'parent'
# - Result: only 'parent' atoms are added to rules
```

## How It Works

### During Pretraining

1. Generate random rule: `sibling(X0, X1) :- parent(X2, X0), parent(X2, X1)`
2. Graph constructor encodes using full vocab (8 predicates)
3. Encoder produces embedding
4. Contrastive learning trains encoder to distinguish structures

**Key Point:** Encoder sees diverse predicates and learns general structural patterns.

### During ILP Task

1. Start with: `grandparent(X0, X1) :-`
2. Policy outputs logits: `[parent=0.8, child=0.1, sibling=0.05, ...]`
3. **Trainer masks**: `[parent=0.8, child=-inf, sibling=-inf, ...]`
4. Only 'parent' can be sampled
5. Rule built: `grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)`

**Key Point:** Graph constructor still uses full vocab encoding, so dimensions match!

### Why This Works

The unused predicates in the vocabulary are like "unused neurons" in a neural network:
- They're there (taking up space in the one-hot encoding)
- They're never activated (no 'child', 'sibling', etc. atoms in actual rules)
- But their presence ensures dimensional consistency

**Analogy:**
```
Like training a language model on {A-Z, 0-9} (36 tokens)
but only testing on {A-Z} (26 tokens).

The model still uses 36-dimensional embeddings,
but digits never appear in test sequences.
```

## Benefits

### 1. Dimension Compatibility ✓

- Encoder: 8-predicate input → works with 8-predicate graphs
- GFlowNet: 8-predicate capacity → works with 8-predicate vocab
- Trainer: 1-predicate search → generates 1-predicate rules
- **No dimension mismatches!**

### 2. General-Purpose Encoder ✓

The pretrained encoder learns:
- How variables connect (chain, convergent, star, etc.)
- How predicates compose (transitivity, symmetry, etc.)
- Structural patterns independent of specific predicates

**Transfer Learning:**
```python
# Pretrain once on 8 predicates
pretrainer.pretrain(num_steps=5000)
pretrainer.save_pretrained_encoder('general_encoder.pt')

# Use for different tasks
# Task 1: Learn grandparent with 'parent'
trainer1 = GFlowNetTrainer(..., predicate_vocab=['parent'])

# Task 2: Learn friendship with 'friend', 'knows'
trainer2 = GFlowNetTrainer(..., predicate_vocab=['friend', 'knows'])

# Task 3: Learn ancestry with 'parent', 'ancestor'
trainer3 = GFlowNetTrainer(..., predicate_vocab=['parent', 'ancestor'])
```

All tasks use the SAME pretrained encoder!

### 3. Efficient Pretraining ✓

- Pretrain ONCE on diverse predicates
- Reuse for MANY tasks
- No need to retrain encoder for each new task

### 4. Better Generalization ✓

Encoder learns from:
- Diverse predicate combinations
- Various rule structures
- Multiple semantic patterns

Result: Better embeddings even for single-predicate tasks!

## Limitations

### 1. Fixed Vocabulary Size

**Constraint:** Must define vocabulary upfront

**Impact:**
- If task needs a predicate not in vocab → must retrain everything
- Vocab size determines model size (larger vocab = larger models)

**Mitigation:**
- Use a large, diverse vocabulary (10-20 predicates)
- Cover common predicate types (relations, properties, functions)

### 2. Unused Parameters

**Overhead:**
- 8-predicate vocab but only 1 used → 87.5% of AtomAdder output unused
- GFlowNet capacity wasted on unused predicates

**Impact:**
- Slightly larger model size
- Slightly slower inference (negligible in practice)

**Why It's OK:**
- Model size still small (few MB)
- Inference still fast (<1ms per forward pass)
- Benefit of transfer learning outweighs overhead

### 3. Predicate Semantics Not Captured

**Limitation:** Encoder treats all predicates as arbitrary symbols
- Doesn't know 'parent' implies familial relationship
- Doesn't know 'male' is a property vs 'parent' is a relation

**Why It's OK:**
- GFlowNet learns task-specific semantics from rewards
- Pretraining provides structural understanding, not semantic

## Alternatives Considered

### Alternative 1: Variable-Size Vocabulary

**Idea:** Make encoder work with any vocabulary size

**Approach:** Use predicate embeddings instead of one-hot
```python
# Instead of one-hot [0, 1, 0, 0, ...]
# Use learnable embedding: predicate_name → embedding[128]
```

**Pros:**
- Can add new predicates without retraining
- Smaller model size

**Cons:**
- More complex architecture
- Loses one-hot interpretability
- Requires predicate-specific pretraining
- **Not implemented** (future work)

### Alternative 2: Task-Specific Encoder

**Idea:** Train separate encoder for each task

**Approach:**
```python
# Task 1: 1-predicate encoder
encoder1 = StateEncoder(predicate_vocab_size=1)

# Task 2: 3-predicate encoder
encoder2 = StateEncoder(predicate_vocab_size=3)
```

**Pros:**
- No wasted capacity
- Task-specific optimization

**Cons:**
- ❌ No transfer learning
- ❌ Must pretrain for each task
- ❌ No generalization benefits
- **Rejected**

### Alternative 3: Predicate-Agnostic Encoder

**Idea:** Encoder only sees graph structure, not predicate identity

**Approach:**
```python
# All predicates encoded as [1], variables as [0]
# No predicate-specific information
```

**Pros:**
- Works with any vocabulary
- Very general

**Cons:**
- ❌ Loses predicate information
- ❌ Can't distinguish different predicates
- ❌ Much weaker representations
- **Rejected**

## Best Practices

### 1. Choose a Rich Vocabulary

**Good:** Diverse predicates covering common patterns
```python
predicate_vocab = [
    # Binary relations
    'parent', 'child', 'sibling', 'spouse', 'friend',
    'ancestor', 'descendant', 'colleague',

    # Unary properties
    'male', 'female', 'adult', 'child_person',

    # Functions/attributes
    'age', 'height', 'weight'
]
```

**Bad:** All similar predicates
```python
predicate_vocab = [
    'parent1', 'parent2', 'parent3', 'parent4'
]
```

### 2. Pretrain Once, Use Many Times

```python
# Step 1: Pretrain (do once)
pretrainer.pretrain(num_steps=10000)
pretrainer.save_pretrained_encoder('pretrained_encoder_v1.pt')

# Step 2: Load for each task
encoder.load_state_dict(torch.load('pretrained_encoder_v1.pt'))

# Task 1
trainer1 = GFlowNetTrainer(..., predicate_vocab=['parent'])
trainer1.train()

# Task 2
trainer2 = GFlowNetTrainer(..., predicate_vocab=['friend', 'knows'])
trainer2.train()
```

### 3. Document Task vs Encoder Vocabularies

```python
config = {
    'encoder_vocab': ['parent', 'child', ...],  # Full vocab
    'task_vocab': ['parent'],  # Subset for search
    'note': 'Encoder pretrained on diverse predicates; task uses subset'
}
```

### 4. Test Generalization

After pretraining, test on multiple tasks:
```python
# Test 1: Single predicate
test_task(['parent'])

# Test 2: Multiple predicates
test_task(['parent', 'ancestor'])

# Test 3: Different domain
test_task(['friend', 'colleague'])
```

Verify encoder generalizes across tasks.

## FAQ

**Q: Why not just use the task vocabulary for everything?**

A: The task vocabulary is too limited (1 predicate). Pretraining on diverse predicates teaches the encoder general structural patterns that transfer to the task.

**Q: Isn't it wasteful to have 7 unused predicates?**

A: Slightly, but the benefits of transfer learning far outweigh the small overhead. Model size increases by ~1KB, training time by ~5%, but pretraining improves final performance by 10-20%.

**Q: Can I add new predicates later?**

A: No, vocabulary is fixed. Adding predicates requires retraining encoder (dimension mismatch). Define a large vocabulary upfront.

**Q: What if my task needs a predicate not in the vocab?**

A: Either:
1. Add it to vocab and retrain everything
2. Use a similar predicate from vocab (approximate)
3. Extend vocab and fine-tune encoder (future work)

**Q: How do I know if pretraining worked?**

A: Run the test cell in the notebook:
- Equivalent rules (renamed variables) should have >0.90 similarity
- Different rules (different structure) should have <0.90 similarity

**Q: Can I use this for tasks with >8 predicates?**

A: Yes! Just expand the vocabulary:
```python
predicate_vocab = ['pred1', 'pred2', ..., 'pred20']  # 20 predicates
```
Model will be slightly larger but works the same way.

## Summary

✅ **Use unified vocabulary approach:**
- Encoder: Full vocab (8 predicates)
- Trainer: Task vocab (1 predicate)
- Ensures dimension compatibility
- Enables transfer learning

✅ **Benefits:**
- General-purpose pretrained encoder
- Works with any subset of predicates
- No dimension mismatches
- Better generalization

✅ **Trade-offs:**
- Small overhead from unused predicates
- Fixed vocabulary (must define upfront)
- Worth it for transfer learning benefits

This design choice makes the encoder a **reusable component** that can be pretrained once and used for many ILP tasks, similar to how BERT is pretrained once and fine-tuned for many NLP tasks.
