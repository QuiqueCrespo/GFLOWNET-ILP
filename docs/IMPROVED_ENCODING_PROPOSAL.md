# Improved Rule Encoding Proposal

## Current Encoding Issues

The current hypergraph encoding loses important structural information:

1. **Argument order**: `parent(X0, X2)` vs `parent(X2, X0)` are identical
2. **Role information**: Head vs body atoms not distinguished in features
3. **Connectivity patterns**: Chains, disconnected variables, self-loops implicit
4. **Pooling**: Mean pooling treats all nodes equally

## Proposed Improved Encoding

### Option 1: Enhanced Hypergraph with Rich Features (Recommended)

#### Node Representation

**Variable Nodes**:
```python
variable_features = [
    # Appearance indicators (4 bits)
    appears_in_head,           # 1 if in head, 0 otherwise
    appears_in_body,           # 1 if in body, 0 otherwise
    appears_in_multiple_atoms, # 1 if appears in >1 atom
    is_chain_variable,         # 1 if only in body (potential chain)

    # Position encoding (using learned embeddings)
    head_position_embedding,   # Which position in head (or 0)

    # Statistics
    total_occurrences,         # How many times this variable appears

    # Node type
    1  # is_variable flag
]
```

**Predicate Nodes**:
```python
predicate_features = [
    # Predicate type (one-hot)
    *predicate_type_one_hot,   # [0,1,0,...] for predicate vocab

    # Role information
    is_head,                   # 1 if this is the head atom
    is_body,                   # 1 if this is a body atom
    body_position,             # Which position in body (0 for head)

    # Structural features
    has_self_loop,             # 1 if same var appears multiple times
    num_unique_variables,      # How many unique vars in this atom
    total_variables,           # Total var occurrences (arity)

    # Node type
    0  # is_variable flag
]
```

#### Edge Representation with Features

Instead of simple bidirectional edges, use **directed edges with features**:

```python
class EnhancedEdge:
    source: int           # Variable node
    target: int           # Predicate node
    argument_position: int # 0 for first arg, 1 for second arg, etc.
    edge_type: str        # "head_arg" or "body_arg"
```

**Example**: For `parent(X0, X2)` at body position 0:
```python
edges = [
    Edge(source=X0, target=parent, arg_position=0, type="body_arg"),
    Edge(source=X2, target=parent, arg_position=1, type="body_arg")
]
```

#### Hierarchical Graph Structure

Build a **two-level graph**:

**Level 1: Atom Graph**
- Nodes: Individual atoms (head + body atoms)
- Edges: Connect atoms that share variables
- Edge weights: Number of shared variables

**Level 2: Variable-Atom Bipartite Graph**
- Left nodes: Variables
- Right nodes: Atoms
- Edges: Variable-to-atom membership with position info

```python
def theory_to_hierarchical_graph(self, theory):
    # Level 1: Atom-level graph
    atom_nodes = []
    atom_edges = []

    for rule in theory:
        # Add head
        head_node = {
            'type': 'head',
            'predicate': rule.head.predicate_name,
            'variables': rule.head.args
        }
        atom_nodes.append(head_node)

        # Add body atoms
        for atom in rule.body:
            body_node = {
                'type': 'body',
                'predicate': atom.predicate_name,
                'variables': atom.args
            }
            atom_nodes.append(body_node)

        # Connect atoms that share variables
        for i, atom1 in enumerate(atom_nodes):
            for j, atom2 in enumerate(atom_nodes[i+1:], start=i+1):
                shared_vars = set(atom1['variables']) & set(atom2['variables'])
                if shared_vars:
                    atom_edges.append((i, j, len(shared_vars)))

    # Level 2: Variable-atom bipartite graph
    var_to_atom_edges = []
    for atom_idx, atom in enumerate(atom_nodes):
        for arg_pos, var in enumerate(atom['variables']):
            var_to_atom_edges.append((var, atom_idx, arg_pos))

    return {
        'atom_graph': (atom_nodes, atom_edges),
        'bipartite': var_to_atom_edges
    }
```

#### Advanced GNN Architecture

```python
class HierarchicalRuleEncoder(nn.Module):
    def __init__(self, node_feature_dim, embedding_dim):
        super().__init__()

        # Separate encoders for variables and predicates
        self.var_encoder = nn.Linear(var_feature_dim, embedding_dim)
        self.pred_encoder = nn.Linear(pred_feature_dim, embedding_dim)

        # Message passing with edge features
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_feature_dim, embedding_dim),
            nn.ReLU()
        )
        self.gnn = GATConv(embedding_dim, embedding_dim, edge_dim=embedding_dim)

        # Hierarchical pooling
        self.head_pool = AttentionPooling(embedding_dim)
        self.body_pool = AttentionPooling(embedding_dim)
        self.var_pool = AttentionPooling(embedding_dim)

        # Combine pools with learned weights
        self.combiner = nn.Linear(3 * embedding_dim, embedding_dim)

    def forward(self, graph_data):
        # Separate encoding for different node types
        var_mask = graph_data.x[:, -1] == 1
        pred_mask = ~var_mask

        var_features = self.var_encoder(graph_data.x[var_mask])
        pred_features = self.pred_encoder(graph_data.x[pred_mask])

        # Combine into single embedding matrix
        x = torch.zeros(graph_data.num_nodes, self.embedding_dim)
        x[var_mask] = var_features
        x[pred_mask] = pred_features

        # Message passing with edge features
        edge_features = self.edge_nn(graph_data.edge_attr)
        x = self.gnn(x, graph_data.edge_index, edge_features)

        # Hierarchical pooling
        head_emb = self.head_pool(x[graph_data.is_head])
        body_emb = self.body_pool(x[graph_data.is_body])
        var_emb = self.var_pool(x[var_mask])

        # Combine
        combined = torch.cat([head_emb, body_emb, var_emb], dim=-1)
        graph_embedding = self.combiner(combined)

        return graph_embedding, x
```

### Option 2: Sequence + Structure Hybrid (Alternative)

Combine sequence representation with structural features:

```python
def theory_to_hybrid_encoding(self, theory):
    rule = theory[0]

    # Part 1: Sequence encoding (captures order)
    tokens = []
    tokens.append('<HEAD>')
    tokens.append(rule.head.predicate_name)
    for arg in rule.head.args:
        tokens.append(f'X{arg.id}')

    tokens.append('<BODY>')
    for atom in rule.body:
        tokens.append(atom.predicate_name)
        for arg in atom.args:
            tokens.append(f'X{arg.id}')

    # Embed sequence with Transformer
    sequence_embedding = self.transformer(tokens)

    # Part 2: Structural features (explicit patterns)
    structural_features = [
        count_disconnected_variables(theory),
        count_self_loops(theory),
        count_chain_variables(theory),
        len(rule.body),  # Rule complexity
        len(get_all_variables(theory)),  # Number of variables
        is_connected_graph(theory)  # Boolean: is rule fully connected
    ]

    # Combine
    combined = torch.cat([
        sequence_embedding,
        torch.tensor(structural_features)
    ])

    return combined
```

### Option 3: Abstract Syntax Tree (AST) Representation

Treat rule as tree structure:

```python
class RuleAST:
    """
    Tree structure:
           Rule
          /    \
       Head    Body
        |       |
      Atom   [Atom, Atom, ...]
       |         |
      [Args]   [Args]
    """

    def __init__(self, theory):
        rule = theory[0]
        self.root = Node(type='rule')

        # Head subtree
        head_node = Node(type='head', predicate=rule.head.predicate_name)
        for arg_pos, arg in enumerate(rule.head.args):
            arg_node = Node(type='var', var_id=arg.id, position=arg_pos)
            head_node.add_child(arg_node)
        self.root.add_child(head_node)

        # Body subtree
        body_node = Node(type='body')
        for atom_idx, atom in enumerate(rule.body):
            atom_node = Node(type='atom', predicate=atom.predicate_name, position=atom_idx)
            for arg_pos, arg in enumerate(atom.args):
                arg_node = Node(type='var', var_id=arg.id, position=arg_pos)
                atom_node.add_child(arg_node)
            body_node.add_child(atom_node)
        self.root.add_child(body_node)

def encode_ast_with_treelstm(self, ast):
    """Encode AST using Tree-LSTM or Tree-GNN."""
    # Bottom-up encoding
    def encode_node(node):
        if node.is_leaf():
            return self.leaf_encoder(node.features)
        else:
            child_encodings = [encode_node(child) for child in node.children]
            return self.tree_lstm(node.features, child_encodings)

    return encode_node(ast.root)
```

## Comparison of Approaches

| Approach | Pros | Cons | Complexity |
|----------|------|------|------------|
| **Current Hypergraph** | Simple, standard | Loses order, implicit structure | Low |
| **Enhanced Hypergraph** (Recommended) | Captures all structure, extensible | More features to learn | Medium |
| **Sequence + Structure** | Explicit patterns, simple | Loses graph structure benefits | Medium |
| **AST Tree** | Natural hierarchy, captures order | Complex, less standard | High |

## Recommended Implementation: Enhanced Hypergraph

### Changes to `GraphConstructor`

```python
class EnhancedGraphConstructor:
    def theory_to_graph(self, theory):
        if not theory:
            return self._empty_graph()

        rule = theory[0]
        node_features = []
        edge_index = []
        edge_features = []

        # Analyze variable usage patterns
        head_vars = set(rule.head.args)
        body_vars = set()
        var_counts = {}
        for atom in rule.body:
            for var in atom.args:
                body_vars.add(var)
                var_counts[var] = var_counts.get(var, 0) + 1

        # Build variable nodes with rich features
        variables = self._get_canonical_variables(theory)
        var_to_node = {}

        for node_id, var in enumerate(variables):
            var_to_node[var] = node_id

            # Compute variable features
            var_features = [
                1.0 if var in head_vars else 0.0,  # appears_in_head
                1.0 if var in body_vars else 0.0,  # appears_in_body
                1.0 if var_counts.get(var, 0) > 1 else 0.0,  # appears_multiple
                1.0 if var in body_vars and var not in head_vars else 0.0,  # is_chain_var
                float(var_counts.get(var, 0)),  # total_occurrences
                1.0  # is_variable flag
            ]
            node_features.append(var_features)

        node_id = len(variables)

        # Build predicate nodes with rich features
        # Head
        head_self_loop = len(set(rule.head.args)) < len(rule.head.args)
        head_features = [
            *self._one_hot_predicate(rule.head.predicate_name),
            1.0,  # is_head
            0.0,  # is_body
            0.0,  # body_position
            1.0 if head_self_loop else 0.0,  # has_self_loop
            float(len(set(rule.head.args))),  # num_unique_vars
            float(len(rule.head.args)),  # total_vars
            0.0   # is_variable flag
        ]
        node_features.append(head_features)
        head_node_id = node_id
        node_id += 1

        # Add edges from head with argument position
        for arg_pos, arg in enumerate(rule.head.args):
            if arg in var_to_node:
                var_node = var_to_node[arg]
                # Directed edge: var -> pred
                edge_index.append([var_node, head_node_id])
                edge_features.append([
                    float(arg_pos),  # argument_position
                    1.0,  # is_head_edge
                    0.0   # is_body_edge
                ])
                # Reverse edge: pred -> var
                edge_index.append([head_node_id, var_node])
                edge_features.append([
                    float(arg_pos),
                    1.0,
                    0.0
                ])

        # Body atoms
        for body_pos, atom in enumerate(rule.body):
            atom_self_loop = len(set(atom.args)) < len(atom.args)
            atom_features = [
                *self._one_hot_predicate(atom.predicate_name),
                0.0,  # is_head
                1.0,  # is_body
                float(body_pos),  # body_position
                1.0 if atom_self_loop else 0.0,  # has_self_loop
                float(len(set(atom.args))),  # num_unique_vars
                float(len(atom.args)),  # total_vars
                0.0   # is_variable flag
            ]
            node_features.append(atom_features)
            atom_node_id = node_id
            node_id += 1

            # Add edges with argument position
            for arg_pos, arg in enumerate(atom.args):
                if arg in var_to_node:
                    var_node = var_to_node[arg]
                    edge_index.append([var_node, atom_node_id])
                    edge_features.append([
                        float(arg_pos),
                        0.0,  # is_head_edge
                        1.0   # is_body_edge
                    ])
                    edge_index.append([atom_node_id, var_node])
                    edge_features.append([
                        float(arg_pos),
                        0.0,
                        1.0
                    ])

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Additional graph-level attributes
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(node_features)

        # Store masks for hierarchical pooling
        data.is_variable = torch.tensor([f[-1] == 1.0 for f in node_features])
        data.is_head = torch.tensor([
            i == len(variables) for i in range(len(node_features))
        ])
        data.is_body = torch.tensor([
            i > len(variables) for i in range(len(node_features))
        ]) & ~data.is_head

        return data

    def _one_hot_predicate(self, pred_name):
        vec = [0.0] * len(self.predicate_vocab)
        if pred_name in self.pred_to_idx:
            vec[self.pred_to_idx[pred_name]] = 1.0
        return vec
```

### Changes to `StateEncoder`

```python
class EnhancedStateEncoder(nn.Module):
    def __init__(self, var_feature_dim, pred_feature_dim,
                 edge_feature_dim, embedding_dim, num_layers=3):
        super().__init__()

        # Separate initial encoders
        self.var_encoder = nn.Linear(var_feature_dim, embedding_dim)
        self.pred_encoder = nn.Linear(pred_feature_dim, embedding_dim)

        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_feature_dim, embedding_dim)

        # GNN layers with edge features (use GAT or GIN)
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                GATConv(embedding_dim, embedding_dim, edge_dim=embedding_dim)
            )

        # Attention-based pooling
        self.var_attention = nn.Linear(embedding_dim, 1)
        self.pred_attention = nn.Linear(embedding_dim, 1)

    def forward(self, graph_data):
        # Separate encoding by node type
        var_mask = graph_data.is_variable
        pred_mask = ~var_mask

        # Initialize node embeddings
        x = torch.zeros(graph_data.num_nodes, self.embedding_dim)
        x[var_mask] = self.var_encoder(
            graph_data.x[var_mask, :var_feature_dim]
        )
        x[pred_mask] = self.pred_encoder(
            graph_data.x[pred_mask, var_feature_dim:]
        )

        # Encode edge features
        edge_attr = self.edge_encoder(graph_data.edge_attr)

        # Message passing with edge features
        for conv in self.conv_layers:
            x = conv(x, graph_data.edge_index, edge_attr)
            x = F.relu(x)

        # Hierarchical attention pooling
        var_embeddings = x[var_mask]
        pred_embeddings = x[pred_mask]

        if var_embeddings.shape[0] > 0:
            var_attn = F.softmax(self.var_attention(var_embeddings), dim=0)
            var_pool = (var_attn * var_embeddings).sum(dim=0)
        else:
            var_pool = torch.zeros(self.embedding_dim)

        if pred_embeddings.shape[0] > 0:
            pred_attn = F.softmax(self.pred_attention(pred_embeddings), dim=0)
            pred_pool = (pred_attn * pred_embeddings).sum(dim=0)
        else:
            pred_pool = torch.zeros(self.embedding_dim)

        # Combine with learned weights or concatenation
        graph_embedding = torch.cat([var_pool, pred_pool]).unsqueeze(0)

        return graph_embedding, x
```

## Implementation Plan

### Phase 1: Rich Node Features (Immediate)
1. Add variable usage features (in_head, in_body, is_chain)
2. Add predicate features (is_head, has_self_loop)
3. Keep current GNN architecture
4. Expected impact: Better structural awareness

### Phase 2: Edge Features (Short-term)
1. Add argument position to edges
2. Add edge type (head/body)
3. Use GATConv or similar edge-aware layer
4. Expected impact: Captures argument order

### Phase 3: Hierarchical Pooling (Medium-term)
1. Implement attention-based pooling
2. Separate pooling for vars vs predicates
3. Expected impact: Better global representation

### Phase 4: Advanced Architecture (Long-term)
1. Try Graph Attention Networks
2. Experiment with hierarchical graph structures
3. Expected impact: Further improvements

## Expected Benefits

### Enhanced Hypergraph Benefits:

1. **Argument Order**: Edge features capture which position each variable occupies
2. **Structural Patterns**: Node features explicitly flag self-loops, chain variables, etc.
3. **Role Distinction**: Head vs body clear in features, not just structure
4. **Better Pooling**: Attention mechanism learns importance weights automatically
5. **Backward Compatible**: Can gradually add features without breaking existing code

### Quantitative Expectations:

- **Reduction in pathological rules**: 80% fewer self-loops and disconnected variables
- **Better convergence**: Model learns valid structures 2-3x faster
- **Higher quality rules**: Base reward (without bonuses) increases from ~0.5 to ~0.8+
- **More interpretable**: Attention weights show which parts of rule are important

## Conclusion

**Recommended approach**: Enhanced Hypergraph with rich features and hierarchical pooling.

**Why**:
- Builds on existing working architecture
- Addresses all identified limitations
- Incremental implementation (can add features progressively)
- Standard components (GATConv, attention pooling)
- Expected to significantly reduce pathological patterns

**Implementation priority**:
1. Rich node features (1-2 days)
2. Edge features (1-2 days)
3. Hierarchical pooling (2-3 days)
4. Testing and refinement (3-5 days)

Total effort: ~1-2 weeks for complete implementation and testing.
