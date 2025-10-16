# Graph Encoding Analysis

## Question
Is the graph encoding sensible for representing FOL rules?

## Answer
**The encoding itself is sensible (standard hypergraph representation), but it has limitations that should be addressed through reward shaping.**

## Current Encoding Method

### Node Types
1. **Variable nodes**: One node per unique variable
   - Features: `[0, 0, ..., 0, 1]` (last dimension = 1 indicates variable)

2. **Predicate nodes**: One node per predicate instance (head + each body atom)
   - Features: One-hot encoding of predicate type + `[0]` at end

### Edges
- Bidirectional edges connect each variable to the predicates it appears in
- Example: If `parent(X0, X2)` appears, edges connect:
  - X0 node ↔ parent node
  - X2 node ↔ parent node

### Example: Correct Grandparent Rule

```prolog
grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1).
```

**Graph structure**:
```
Nodes:
  0: X0 (variable)
  1: X1 (variable)
  2: X2 (variable)
  3: grandparent (head predicate)
  4: parent (body atom 0)
  5: parent (body atom 1)

Edges:
  X0 ↔ grandparent (head)
  X1 ↔ grandparent (head)
  X0 ↔ parent (body 0)
  X2 ↔ parent (body 0)
  X2 ↔ parent (body 1)
  X1 ↔ parent (body 1)
```

**Connectivity**:
- X0 connects to: grandparent(head), parent(body0)
- X1 connects to: grandparent(head), parent(body1)
- X2 connects to: parent(body0), parent(body1)

**Chain structure**: X0 → X2 → X1 is represented implicitly through shared predicates.

## What The Encoding Captures Well

### 1. Variable Sharing
The encoding correctly represents which variables appear in which predicates:
- Correct rule: X2 connects body atoms, creating chain
- Bad rule: Disconnected variables create isolated subgraphs

### 2. Predicate Types
One-hot encoding distinguishes different predicates (parent vs grandparent).

### 3. Rule Structure
Separate nodes for head vs body atoms allow GNN to learn their different roles.

### 4. Deterministic Ordering
Canonical variable ordering (top-to-bottom, left-to-right) ensures consistency.

## What The Encoding Misses

### 1. Argument Positions
The encoding loses which argument position each variable occupies.

Example: `parent(X0, X2)` vs `parent(X2, X0)` produce the **same edges** but have **different semantics**.

**Current**:
```
parent(X0, X2): edges X0 ↔ parent, X2 ↔ parent
parent(X2, X0): edges X2 ↔ parent, X0 ↔ parent (identical!)
```

**Impact**: GNN cannot distinguish argument order. But for symmetric predicates or with enough training data, this may not matter much.

**Solution**: Add edge features indicating argument position:
```python
edge_features = []
for arg_idx, arg in enumerate(atom.args):
    edge_features.append(arg_idx)  # 0 for first arg, 1 for second arg
```

### 2. Duplicate Predicates
Multiple body atoms with same predicate create multiple nodes, but they're indistinguishable.

Example:
```prolog
grandparent(X0, X0) :- parent(X0, X0), parent(X0, X0).
```

**Graph**:
```
Nodes:
  0: X0
  1: grandparent (head)
  2: parent (body 0)
  3: parent (body 1)

Edges:
  X0 ↔ grandparent (appears twice - self-loop!)
  X0 ↔ parent (body 0, appears twice)
  X0 ↔ parent (body 1, appears twice)
```

**Issue**: Both parent atoms connect to X0 twice (self-loop), but GNN sees 12 edges and may not recognize this as pathological.

**Impact**: Self-loops get encoded but aren't explicitly flagged as problematic.

### 3. Head vs Body Distinction
While head and body atoms are separate nodes, there's no explicit feature distinguishing them.

**Current**: GNN must learn from graph structure alone that node 3 is special (head).

**Solution**: Add node type feature:
```python
node_feature = one_hot_predicate + [is_variable, is_head, is_body]
```

### 4. Variable Connectivity Patterns
The encoding represents connections but doesn't explicitly capture important patterns:

- **Chain variables** (X0 → X2 → X1): X2 appears only in body, connects head variables
- **Disconnected variables** (X6, X7): Only in body, don't connect to head
- **Self-loops**: Same variable in multiple positions of one atom

**Current**: GNN must learn to recognize these patterns from raw connectivity.

**Impact**: Requires sufficient training to learn these structural properties.

## GNN Architecture Considerations

### Current: Mean Pooling
```python
graph_embedding = x.mean(dim=0, keepdim=True)
```

**Properties**:
- Treats all nodes equally
- Permutation invariant (good for graph-level prediction)
- Loses information about relative importance

**Issues**:
- Disconnected subgraphs get same weight as connected ones
- Large graphs dilute important features

### Alternative: Hierarchical Pooling
```python
# Separate pooling for head and body
head_embedding = head_nodes.mean()
body_embedding = body_nodes.mean()
var_embedding = var_nodes.mean()

# Combine with learned weights
graph_embedding = W1*head_embedding + W2*body_embedding + W3*var_embedding
```

**Benefits**: Explicitly distinguishes head from body, preserves structure.

### Alternative: Attention-Based Pooling
```python
# Learn importance weights for each node
attention_weights = softmax(W @ node_embeddings)
graph_embedding = attention_weights @ node_embeddings
```

**Benefits**: Automatically learns which nodes are important.

## Comparison to Other Representations

### 1. Sequence-Based (Text)
Represent rule as sequence: "grandparent(X0,X1):-parent(X0,X2),parent(X2,X1)"

**Pros**: Simple, captures order
**Cons**: Loses structure, not permutation invariant

### 2. Tree-Based (AST)
Parse rule into abstract syntax tree

**Pros**: Captures hierarchical structure explicitly
**Cons**: More complex, still loses variable sharing across siblings

### 3. Hypergraph (Current)
Variables and predicates as nodes, edges show membership

**Pros**: Natural representation, captures variable sharing
**Cons**: Loses argument order, requires GNN

**Verdict**: Hypergraph is most natural for FOL rules.

## Identified Issues and Solutions

### Issue 1: Disconnected Variables Not Penalized

**Example**:
```prolog
grandparent(X0, X5) :- parent(X0, X3), parent(X4, X5), parent(X6, X7).
```

Variables X3, X4, X6, X7 don't appear in head.

**Why bad**: Rule is overly general - matches any combination where parents exist.

**Encoding**: Graph correctly shows disconnected components, but GNN doesn't know this is bad.

**Solution**: Reward function penalty:
```python
def count_disconnected_variables(theory):
    head_vars = set(theory[0].head.args)
    body_vars = set()
    for atom in theory[0].body:
        body_vars.update(atom.args)
    return len(body_vars - head_vars)

penalty = -0.2 * count_disconnected_variables(theory)
```

### Issue 2: Self-Loops Not Explicitly Flagged

**Example**:
```prolog
grandparent(X0, X0) :- parent(X0, X0).
```

**Why bad**: Requires `parent(a, a)` which likely doesn't exist (no self-loops in family tree).

**Encoding**: Graph shows X0 connected to parent twice, but doesn't flag as problematic.

**Solution**: Reward function penalty:
```python
def count_self_loops(theory):
    count = 0
    for rule in theory:
        # Check head
        if len(set(rule.head.args)) < len(rule.head.args):
            count += 1
        # Check body
        for atom in rule.body:
            if len(set(atom.args)) < len(atom.args):
                count += 1
    return count

penalty = -0.3 * count_self_loops(theory)
```

### Issue 3: Argument Order Lost

**Example**:
```prolog
grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1).  # Correct
grandparent(X0, X1) :- parent(X2, X0), parent(X1, X2).  # Wrong order
```

Both produce similar graphs (same connectivity) but have different semantics.

**Current impact**: Minimal for this problem (symmetric relations learned implicitly).

**Solution if needed**: Add edge features for argument positions.

## Is The Encoding Sensible?

### Yes, for these reasons:

1. **Standard approach**: Hypergraph representation is common in program synthesis and ILP
2. **Captures key structure**: Variable sharing and predicate relationships
3. **Deterministic**: Same theory always produces same graph
4. **Learnable**: GNN can learn relevant patterns with enough data
5. **Extensible**: Can add edge features, hierarchical pooling, etc.

### But has limitations:

1. **Argument order implicit**: GNN must learn from examples
2. **Pathological patterns not explicit**: Needs reward shaping
3. **Mean pooling crude**: Could use hierarchical or attention-based pooling

## Recommendations

### Immediate (Easy):
1. **Add reward penalties** for:
   - Disconnected variables: `-0.2 × count`
   - Self-loops: `-0.3 × count`
   - Isolated predicates: `-0.1 × count`

### Short-term (Moderate):
2. **Improve GNN architecture**:
   - Use hierarchical pooling (head vs body)
   - Add node type features (head/body/variable)
   - Try attention-based pooling

### Long-term (Research):
3. **Add edge features**:
   - Argument position (0, 1, 2, ...)
   - Edge type (head-to-var, body-to-var, var-to-var)

4. **Try alternative architectures**:
   - Graph Attention Networks (GAT)
   - Message Passing Neural Networks (MPNN)
   - Hierarchical graph networks

## Conclusion

**The encoding is sensible and follows standard practices**, but the system would benefit from:
1. **Reward shaping** to penalize pathological patterns (immediate need)
2. **Architectural improvements** to better capture structure (nice to have)

The current issues (disconnected variables, self-loops) are **not encoding failures** - they're represented correctly in the graph. The problem is that the **reward function doesn't penalize them**, leading the model to generate these patterns.

**Priority**: Fix reward function first, then consider architectural improvements if needed.
