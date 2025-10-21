"""
Analyze graph embeddings to verify semantic equivalence detection.

Tests whether the graph encoder produces:
1. Similar embeddings for semantically equivalent rules
2. Different embeddings for semantically different rules
"""

import sys
sys.path.insert(0, '/Users/jq23948/Documents/GFLOWNET-ILP')

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from src.logic_structures import Rule, Atom, Variable
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.logic_engine import Example


def create_rule_from_atoms(head_pred, head_args, body_atoms_list):
    """
    Create a rule from specification.

    Args:
        head_pred: Head predicate name
        head_args: List of variable IDs for head
        body_atoms_list: List of (pred_name, var_ids_tuple)

    Returns:
        Theory containing the rule
    """
    head_vars = [Variable(id=vid) for vid in head_args]
    head = Atom(predicate_name=head_pred, args=tuple(head_vars))

    body_atoms = []
    for pred_name, var_ids in body_atoms_list:
        vars = [Variable(id=vid) for vid in var_ids]
        body_atoms.append(Atom(predicate_name=pred_name, args=tuple(vars)))

    rule = Rule(head=head, body=tuple(body_atoms))
    return [rule]


def get_embedding(theory, graph_constructor, state_encoder):
    """Extract embedding for a theory."""
    graph_data = graph_constructor.theory_to_graph(theory)
    state_embedding, _ = state_encoder(graph_data)
    return state_embedding.squeeze(0).detach().numpy()


def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix for embeddings."""
    return cosine_similarity(embeddings)


def main():
    print("=" * 80)
    print("GRAPH EMBEDDING ANALYSIS")
    print("=" * 80)

    # Initialize encoder
    predicate_vocab = ['parent', 'sibling', 'ancestor']
    graph_constructor = EnhancedGraphConstructor(predicate_vocab)
    state_encoder = EnhancedStateEncoder(
        predicate_vocab_size=len(predicate_vocab),
        embedding_dim=32,
        num_layers=2
    )

    print(f"\nEncoder configuration:")
    print(f"  - Embedding dim: 32")
    print(f"  - Num layers: 2")
    print(f"  - Predicate vocab: {predicate_vocab}")

    # ========================================================================
    # TEST 1: Semantically Equivalent Rules (Variable Renaming)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Semantic Equivalence - Variable Renaming")
    print("=" * 80)

    # grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
    rule1 = create_rule_from_atoms(
        'grandparent', [0, 1],
        [('parent', (0, 2)), ('parent', (2, 1))]
    )

    # grandparent(A, B) :- parent(A, C), parent(C, B)
    # (Same semantics, just renamed variables)
    rule2 = create_rule_from_atoms(
        'grandparent', [10, 11],
        [('parent', (10, 12)), ('parent', (12, 11))]
    )

    emb1 = get_embedding(rule1, graph_constructor, state_encoder)
    emb2 = get_embedding(rule2, graph_constructor, state_encoder)

    similarity = cosine_similarity([emb1], [emb2])[0, 0]

    print(f"\nRule 1: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)")
    print(f"Rule 2: grandparent(X10, X11) :- parent(X10, X12), parent(X12, X11)")
    print(f"\nCosine Similarity: {similarity:.6f}")
    print(f"Expected: ~1.0 (semantically identical)")
    print(f"Result: {'✓ PASS' if similarity > 0.99 else '✗ FAIL'}")

    # ========================================================================
    # TEST 2: Semantically Equivalent Rules (Predicate Order Swap)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Semantic Equivalence - Predicate Order")
    print("=" * 80)

    # ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)
    rule3 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('parent', (0, 2)), ('ancestor', (2, 1))]
    )

    # ancestor(X, Y) :- ancestor(Z, Y), parent(X, Z)
    # (Same semantics, swapped body order)
    rule4 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('ancestor', (2, 1)), ('parent', (0, 2))]
    )

    emb3 = get_embedding(rule3, graph_constructor, state_encoder)
    emb4 = get_embedding(rule4, graph_constructor, state_encoder)

    similarity_order = cosine_similarity([emb3], [emb4])[0, 0]

    print(f"\nRule 3: ancestor(X0, X1) :- parent(X0, X2), ancestor(X2, X1)")
    print(f"Rule 4: ancestor(X0, X1) :- ancestor(X2, X1), parent(X0, X2)")
    print(f"\nCosine Similarity: {similarity_order:.6f}")
    print(f"Expected: High similarity (order shouldn't matter much)")
    print(f"Result: {'✓ PASS' if similarity_order > 0.9 else '✗ FAIL (order sensitivity detected)'}")

    # ========================================================================
    # TEST 3: Different Semantics - Similar Syntax
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Different Semantics - Similar Syntax")
    print("=" * 80)

    # grandparent(X, Y) :- parent(X, Z), parent(Z, Y)  [correct]
    rule5 = create_rule_from_atoms(
        'grandparent', [0, 1],
        [('parent', (0, 2)), ('parent', (2, 1))]
    )

    # grandparent(X, Y) :- parent(X, Z), parent(Y, Z)  [WRONG - sibling-like]
    rule6 = create_rule_from_atoms(
        'grandparent', [0, 1],
        [('parent', (0, 2)), ('parent', (1, 2))]
    )

    emb5 = get_embedding(rule5, graph_constructor, state_encoder)
    emb6 = get_embedding(rule6, graph_constructor, state_encoder)

    similarity_diff = cosine_similarity([emb5], [emb6])[0, 0]

    print(f"\nRule 5: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)")
    print(f"Rule 6: grandparent(X0, X1) :- parent(X0, X2), parent(X1, X2)")
    print(f"\nCosine Similarity: {similarity_diff:.6f}")
    print(f"Expected: Low similarity (different semantics)")
    print(f"Result: {'✓ PASS' if similarity_diff < 0.95 else '✗ FAIL (cannot distinguish different semantics)'}")

    # ========================================================================
    # TEST 4: Different Rule Lengths
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Different Rule Lengths")
    print("=" * 80)

    # Short rule: ancestor(X, Y) :- parent(X, Y)
    rule7 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('parent', (0, 1))]
    )

    # Long rule: ancestor(X, Y) :- parent(X, Z), parent(Z, W), parent(W, Y)
    rule8 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('parent', (0, 2)), ('parent', (2, 3)), ('parent', (3, 1))]
    )

    emb7 = get_embedding(rule7, graph_constructor, state_encoder)
    emb8 = get_embedding(rule8, graph_constructor, state_encoder)

    similarity_length = cosine_similarity([emb7], [emb8])[0, 0]

    print(f"\nRule 7 (short): ancestor(X0, X1) :- parent(X0, X1)")
    print(f"Rule 8 (long):  ancestor(X0, X1) :- parent(X0, X2), parent(X2, X3), parent(X3, X1)")
    print(f"\nCosine Similarity: {similarity_length:.6f}")
    print(f"Expected: Moderate similarity (same domain, different complexity)")
    print(f"Result: Informational (no strict threshold)")

    # ========================================================================
    # TEST 5: Repeated Predicates
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Repeated Predicates")
    print("=" * 80)

    # No repeat: ancestor(X, Y) :- parent(X, Z), sibling(Z, W), parent(W, Y)
    rule9 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('parent', (0, 2)), ('sibling', (2, 3)), ('parent', (3, 1))]
    )

    # With repeat: ancestor(X, Y) :- parent(X, Z), parent(Z, W), parent(W, Y)
    rule10 = create_rule_from_atoms(
        'ancestor', [0, 1],
        [('parent', (0, 2)), ('parent', (2, 3)), ('parent', (3, 1))]
    )

    emb9 = get_embedding(rule9, graph_constructor, state_encoder)
    emb10 = get_embedding(rule10, graph_constructor, state_encoder)

    similarity_repeat = cosine_similarity([emb9], [emb10])[0, 0]

    print(f"\nRule 9 (mixed):   ancestor(X0, X1) :- parent(X0, X2), sibling(X2, X3), parent(X3, X1)")
    print(f"Rule 10 (repeat): ancestor(X0, X1) :- parent(X0, X2), parent(X2, X3), parent(X3, X1)")
    print(f"\nCosine Similarity: {similarity_repeat:.6f}")
    print(f"Expected: Moderate-to-low similarity (different predicate composition)")
    print(f"Result: Informational")

    # ========================================================================
    # COMPREHENSIVE SIMILARITY MATRIX
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SIMILARITY MATRIX")
    print("=" * 80)

    all_embeddings = np.array([emb1, emb2, emb3, emb4, emb5, emb6, emb7, emb8, emb9, emb10])
    similarity_matrix = compute_similarity_matrix(all_embeddings)

    rule_labels = [
        "R1: GP(X,Y):-P(X,Z),P(Z,Y)",
        "R2: GP(A,B):-P(A,C),P(C,B) [renamed]",
        "R3: A(X,Y):-P(X,Z),A(Z,Y)",
        "R4: A(X,Y):-A(Z,Y),P(X,Z) [swapped]",
        "R5: GP(X,Y):-P(X,Z),P(Z,Y) [correct]",
        "R6: GP(X,Y):-P(X,Z),P(Y,Z) [wrong]",
        "R7: A(X,Y):-P(X,Y) [short]",
        "R8: A(X,Y):-P(X,Z),P(Z,W),P(W,Y) [long]",
        "R9: A(X,Y):-P(X,Z),S(Z,W),P(W,Y)",
        "R10: A(X,Y):-P(X,Z),P(Z,W),P(W,Y) [repeat]"
    ]

    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=rule_labels,
        yticklabels=rule_labels,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Graph Embedding Similarity Matrix\n(Darker green = more similar)', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('embedding_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nSimilarity matrix saved to: embedding_similarity_matrix.png")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    test_results = []
    test_results.append(("Variable Renaming (R1 vs R2)", similarity, similarity > 0.99))
    test_results.append(("Predicate Order (R3 vs R4)", similarity_order, similarity_order > 0.9))
    test_results.append(("Different Semantics (R5 vs R6)", similarity_diff, similarity_diff < 0.95))

    print("\nTest Results:")
    for test_name, sim_value, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}: {sim_value:.4f}")

    all_passed = all(result[2] for result in test_results)

    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL: ✓ All critical tests passed!")
        print("The graph encoder successfully captures semantic equivalence.")
    else:
        print("OVERALL: ✗ Some tests failed")
        print("The graph encoder may need improvements to better capture semantics.")
    print("=" * 80)

    # Additional insights
    print("\nKey Insights:")
    print(f"  - Same rule with different variable IDs: {similarity:.4f} similarity")
    print(f"  - Same rule with swapped predicate order: {similarity_order:.4f} similarity")
    print(f"  - Different semantics (similar syntax): {similarity_diff:.4f} similarity")
    print(f"  - Short vs long rules: {similarity_length:.4f} similarity")
    print(f"  - Mixed vs repeated predicates: {similarity_repeat:.4f} similarity")


if __name__ == "__main__":
    main()
