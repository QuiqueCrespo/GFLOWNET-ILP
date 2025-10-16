"""Compare original and enhanced graph encodings."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

import torch
from src.logic_structures import Theory, Atom, Variable, Rule
from src.graph_encoder import GraphConstructor, StateEncoder
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder

# Define test theories
theories = {
    "Correct": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(1))),
        body=[
            Atom('parent', (Variable(0), Variable(2))),
            Atom('parent', (Variable(2), Variable(1)))
        ]
    )],

    "Disconnected": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(5))),
        body=[
            Atom('parent', (Variable(0), Variable(3))),
            Atom('parent', (Variable(4), Variable(5))),
            Atom('parent', (Variable(6), Variable(7)))
        ]
    )],

    "Self-loop": [Rule(
        head=Atom('grandparent', (Variable(0), Variable(0))),
        body=[
            Atom('parent', (Variable(0), Variable(0))),
            Atom('parent', (Variable(0), Variable(0)))
        ]
    )]
}

predicate_vocab = ['parent', 'grandparent']

# Original encoding
print("="*80)
print("ORIGINAL ENCODING")
print("="*80)

original_constructor = GraphConstructor(predicate_vocab)
original_encoder = StateEncoder(
    node_feature_dim=len(predicate_vocab) + 1,
    embedding_dim=32,
    num_layers=2
)

for name, theory in theories.items():
    print(f"\n{name} Rule:")
    rule = theory[0]
    print(f"  {rule.head.predicate_name}({', '.join(f'X{v.id}' for v in rule.head.args)}) :- ", end="")
    print(', '.join(f"{a.predicate_name}({', '.join(f'X{v.id}' for v in a.args)})" for a in rule.body))

    graph = original_constructor.theory_to_graph(theory)
    print(f"  Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Has edge features: No")

    # Encode
    graph_emb, node_emb = original_encoder(graph)
    print(f"  Graph embedding shape: {graph_emb.shape}")
    print(f"  Graph embedding: {graph_emb[0, :8].detach().numpy()}")  # First 8 dims

# Enhanced encoding
print("\n" + "="*80)
print("ENHANCED ENCODING")
print("="*80)

enhanced_constructor = EnhancedGraphConstructor(predicate_vocab)
enhanced_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=32,
    num_layers=2
)

for name, theory in theories.items():
    print(f"\n{name} Rule:")
    rule = theory[0]
    print(f"  {rule.head.predicate_name}({', '.join(f'X{v.id}' for v in rule.head.args)}) :- ", end="")
    print(', '.join(f"{a.predicate_name}({', '.join(f'X{v.id}' for v in a.args)})" for a in rule.body))

    graph = enhanced_constructor.theory_to_graph(theory)
    print(f"  Nodes: {graph.num_nodes}, Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edge features shape: {graph.edge_attr.shape}")
    print(f"  Has edge features: Yes")

    # Show variable features for first variable
    if graph.is_variable.any():
        var_idx = graph.is_variable.nonzero()[0].item()
        var_features = graph.x[var_idx].numpy()
        print(f"  Variable features (node {var_idx}): {var_features}")
        print(f"    - appears_in_head: {var_features[0]}")
        print(f"    - appears_in_body: {var_features[1]}")
        print(f"    - appears_multiple: {var_features[2]}")
        print(f"    - is_chain_var: {var_features[3]}")

    # Show head predicate features
    if graph.is_head.any():
        head_idx = graph.is_head.nonzero()[0].item()
        head_features = graph.x[head_idx].numpy()
        print(f"  Head predicate features (node {head_idx}):")
        print(f"    - is_head: {head_features[-7]}")
        print(f"    - has_self_loop: {head_features[-4]}")
        print(f"    - num_unique_vars: {head_features[-3]}")

    # Encode
    graph_emb, node_emb = enhanced_encoder(graph)
    print(f"  Graph embedding shape: {graph_emb.shape}")
    print(f"  Graph embedding: {graph_emb[0, :8].detach().numpy()}")  # First 8 dims

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print("""
Original Encoding:
  - Node features: Simple one-hot (predicate) + is_variable flag
  - Edge features: None
  - Pooling: Mean pooling (all nodes equal weight)
  - Pros: Simple, standard, works
  - Cons: Loses argument order, implicit structure patterns

Enhanced Encoding:
  - Node features: Rich (appears_in_head, is_chain_var, has_self_loop, etc.)
  - Edge features: [argument_position, is_head_edge, is_body_edge]
  - Pooling: Attention-based (learns importance weights)
  - Pros: Explicit structure patterns, captures argument order, hierarchical
  - Cons: More complex, more parameters to learn

Expected Benefits:
  - Disconnected variables explicitly flagged (is_chain_var feature)
  - Self-loops explicitly flagged (has_self_loop feature)
  - Argument order preserved (edge position features)
  - Better pooling (attention focuses on important nodes)

Next Steps:
  1. Test enhanced encoding on actual training
  2. Compare convergence speed and final rule quality
  3. Analyze attention weights to understand what model learns
  4. Combine with reward shaping for maximum effect
""")
