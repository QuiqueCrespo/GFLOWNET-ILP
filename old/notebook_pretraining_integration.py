"""
Notebook integration code for predicate-agnostic encoder pretraining.

This module provides ready-to-use code cells that can be inserted into the Demo_ILP.ipynb notebook
to add predicate-agnostic encoder pretraining functionality using contrastive learning.

Key Idea:
- Pretrain on GENERIC predicates (pred0, pred1, ..., pred9)
- Encoder learns STRUCTURAL patterns (chains, forks, stars)
- Transfer to ANY vocabulary at task time
- Predicate names are just indices - semantics don't matter

Insert this after the model initialization (after creating state_encoder) and before training.
"""

PRETRAINING_CELL_1 = """
# =============================================================================
# PREDICATE-AGNOSTIC ENCODER PRETRAINING (Optional - Recommended)
# =============================================================================
# This cell pretrains the encoder using contrastive learning on GENERIC predicates.
# The encoder learns structural patterns that transfer to ANY vocabulary!
#
# Key Insight:
#   - Pretraining uses: pred0, pred1, ..., pred9 (generic names)
#   - Encoder learns: chains, forks, stars, disconnected components
#   - Transfer works because: chain(pred0) ≈ chain(parent) ≈ chain(friend)
#   - Same structure → same encoding, regardless of predicate names!

from src.predicate_agnostic_pretraining import PredicateAgnosticPretrainer
import matplotlib.pyplot as plt

# Configuration
USE_PRETRAINING = True  # Set to False to skip pretraining
PRETRAIN_STEPS = 2000   # Number of pretraining steps (2000-5000 recommended)
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 1e-3
PRETRAIN_TEMPERATURE = 0.5
PRETRAIN_NUM_NEGATIVES = 4
NUM_GENERIC_PREDICATES = 10  # Number of generic predicates (pred0, ..., pred9)

print("=" * 80)
print("PREDICATE-AGNOSTIC ENCODER PRETRAINING")
print("=" * 80)

if not USE_PRETRAINING:
    print("\\nPretraining DISABLED - using randomly initialized encoder")
    print("=" * 80 + "\\n")
else:
    print(f"\\nPretraining Configuration:")
    print(f"  Steps: {PRETRAIN_STEPS}")
    print(f"  Batch size: {PRETRAIN_BATCH_SIZE}")
    print(f"  Learning rate: {PRETRAIN_LR}")
    print(f"  Temperature: {PRETRAIN_TEMPERATURE}")
    print(f"  Num negatives: {PRETRAIN_NUM_NEGATIVES}")
    print(f"  Generic predicates: {NUM_GENERIC_PREDICATES} (pred0, pred1, ..., pred{NUM_GENERIC_PREDICATES-1})")
    print()
    print("Key Design:")
    print("  ✓ Pretrain on GENERIC predicates (not task-specific)")
    print("  ✓ Encoder learns STRUCTURAL patterns")
    print("  ✓ Transfer to ANY vocabulary at task time")
    print("  ✓ Predicate names are just indices - structure matters!")

    # Create predicate-agnostic pretrainer
    print("\\n" + "-" * 80)
    print("RUNNING PREDICATE-AGNOSTIC PRETRAINING")
    print("-" * 80 + "\\n")

    # IMPORTANT: max_predicates should match the StateEncoder's capacity
    # This is typically set in config['pretrain_max_capacity']
    MAX_PREDICATES = config.get('pretrain_max_capacity', 20)

    pretrainer = PredicateAgnosticPretrainer(
        state_encoder=state_encoder,
        num_generic_predicates=NUM_GENERIC_PREDICATES,
        max_predicates=MAX_PREDICATES,  # Must match StateEncoder's expected input dim
        max_arity=2,
        learning_rate=PRETRAIN_LR,
        temperature=PRETRAIN_TEMPERATURE,
        num_negatives=PRETRAIN_NUM_NEGATIVES
    )

    # Run pretraining
    pretrain_history = pretrainer.pretrain(
        num_steps=PRETRAIN_STEPS,
        batch_size=PRETRAIN_BATCH_SIZE,
        verbose=True,
        log_interval=200
    )

    print("\\n" + "-" * 80)
    print("PRETRAINING COMPLETE")
    print("-" * 80)

    # Show final metrics
    final_metrics = pretrain_history[-1]
    print(f"\\nFinal Pretraining Metrics:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Positive similarity: {final_metrics['avg_pos_sim']:.3f}")
    print(f"  Negative similarity: {final_metrics['avg_neg_sim']:.3f}")
    print(f"  Similarity gap: {final_metrics['avg_pos_sim'] - final_metrics['avg_neg_sim']:.3f}")

    # Plot training curves
    print("\\nGenerating pretraining visualizations...")

    steps = list(range(len(pretrain_history)))
    losses = [h['loss'] for h in pretrain_history]
    accuracies = [h['accuracy'] for h in pretrain_history]
    pos_sims = [h['avg_pos_sim'] for h in pretrain_history]
    neg_sims = [h['avg_neg_sim'] for h in pretrain_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(steps, losses, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Step', fontsize=11)
    axes[0, 0].set_ylabel('Contrastive Loss', fontsize=11)
    axes[0, 0].set_title('Pretraining Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(steps, accuracies, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Step', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Positive vs Negative Discrimination', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    # Similarities
    axes[1, 0].plot(steps, pos_sims, label='Equivalent Rules', linewidth=2, color='blue')
    axes[1, 0].plot(steps, neg_sims, label='Different Rules', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Step', fontsize=11)
    axes[1, 0].set_ylabel('Cosine Similarity', fontsize=11)
    axes[1, 0].set_title('Embedding Similarities', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Similarity gap
    gaps = [pos - neg for pos, neg in zip(pos_sims, neg_sims)]
    axes[1, 1].plot(steps, gaps, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Step', fontsize=11)
    axes[1, 1].set_ylabel('Similarity Gap', fontsize=11)
    axes[1, 1].set_title('Positive - Negative Similarity Gap', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{visualizer.run_dir}/encoder_pretraining_curves.png', dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {visualizer.run_dir}/encoder_pretraining_curves.png")
    plt.show()

    # Optional: Save pretrained encoder weights
    # pretrainer.save_pretrained_encoder(f'{visualizer.run_dir}/pretrained_encoder.pt')

    print("\\n" + "=" * 80)
    print("ENCODER IS NOW PRETRAINED AND READY FOR GFLOWNET TRAINING")
    print("=" * 80 + "\\n")
"""

PRETRAINING_CELL_2 = """
# =============================================================================
# TEST PRETRAINED ENCODER - TRANSFER LEARNING (Optional)
# =============================================================================
# Test whether the predicate-agnostic encoder can recognize structural patterns
# across DIFFERENT vocabularies (transfer learning!)

from sklearn.metrics.pairwise import cosine_similarity
from src.logic_structures import Rule, Atom, Variable
from src.predicate_agnostic_pretraining import FlexibleGraphConstructor
import numpy as np

def create_test_rule(head_pred, head_args, body_atoms_list):
    '''Create a rule for testing.'''
    head_vars = [Variable(id=vid) for vid in head_args]
    head = Atom(predicate_name=head_pred, args=tuple(head_vars))
    body_atoms = []
    for pred_name, var_ids in body_atoms_list:
        vars = [Variable(id=vid) for vid in var_ids]
        body_atoms.append(Atom(predicate_name=pred_name, args=tuple(vars)))
    rule = Rule(head=head, body=tuple(body_atoms))
    return [rule]

def get_test_embedding(theory, predicate_vocab, state_encoder, flex_graph_constructor):
    '''Get embedding for a theory using flexible graph constructor.'''
    graph_data = flex_graph_constructor.theory_to_graph(theory, predicate_vocab=predicate_vocab)
    state_embedding, _ = state_encoder(graph_data)
    return state_embedding.squeeze(0).detach().numpy()

if USE_PRETRAINING:
    print("=" * 80)
    print("TESTING PREDICATE-AGNOSTIC ENCODER - TRANSFER LEARNING")
    print("=" * 80)

    # Create flexible graph constructor
    flex_graph_constructor = FlexibleGraphConstructor(max_predicates=20)

    # Test 1: Same structure, same predicate - should be SIMILAR
    rule1 = create_test_rule('grandparent', [0, 1], [('parent', (0, 2)), ('parent', (2, 1))])  # Chain
    rule2 = create_test_rule('grandparent', [5, 6], [('parent', (5, 7)), ('parent', (7, 6))])  # Chain (renamed)

    emb1 = get_test_embedding(rule1, ['parent'], state_encoder, flex_graph_constructor)
    emb2 = get_test_embedding(rule2, ['parent'], state_encoder, flex_graph_constructor)
    sim_same_pred = cosine_similarity([emb1], [emb2])[0, 0]

    print("\\nTest 1: Same Structure, Same Predicate")
    print("-" * 80)
    print("Rule 1: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)")
    print("Rule 2: grandparent(X5, X6) :- parent(X5, X7), parent(X7, X6)")
    print(f"Similarity: {sim_same_pred:.4f}")
    print(f"Status: {'✓ PASS' if sim_same_pred > 0.85 else '✗ FAIL'} (Expected: >0.85)")

    # Test 2: Same structure, DIFFERENT predicate - should STILL be similar! (transfer learning)
    rule3 = create_test_rule('relationship', [0, 1], [('parent', (0, 2)), ('parent', (2, 1))])   # Chain with 'parent'
    rule4 = create_test_rule('relationship', [0, 1], [('friend', (0, 2)), ('friend', (2, 1))])   # Chain with 'friend'

    emb3 = get_test_embedding(rule3, ['parent'], state_encoder, flex_graph_constructor)
    emb4 = get_test_embedding(rule4, ['friend'], state_encoder, flex_graph_constructor)
    sim_diff_pred = cosine_similarity([emb3], [emb4])[0, 0]

    print("\\nTest 2: Same Structure, DIFFERENT Predicate (Transfer Learning)")
    print("-" * 80)
    print("Rule 3 (chain): relationship(X0, X1) :- parent(X0, X2), parent(X2, X1)")
    print("Rule 4 (chain): relationship(X0, X1) :- friend(X0, X2), friend(X2, X1)")
    print(f"Similarity: {sim_diff_pred:.4f}")
    print(f"Status: {'✓ PASS' if sim_diff_pred > 0.70 else '✗ FAIL'} (Expected: >0.70)")
    print("Note: Both are CHAIN structures → encoder should recognize similarity!")

    # Test 3: Different structures - should be DIFFERENT
    rule5 = create_test_rule('rel', [0, 1], [('parent', (0, 2)), ('parent', (2, 1))])  # Chain
    rule6 = create_test_rule('rel', [0, 1], [('parent', (0, 2)), ('parent', (1, 2))])  # Convergent

    emb5 = get_test_embedding(rule5, ['parent'], state_encoder, flex_graph_constructor)
    emb6 = get_test_embedding(rule6, ['parent'], state_encoder, flex_graph_constructor)
    sim_diff_struct = cosine_similarity([emb5], [emb6])[0, 0]

    print("\\nTest 3: Different Structures")
    print("-" * 80)
    print("Rule 5 (chain):      rel(X0, X1) :- parent(X0, X2), parent(X2, X1)")
    print("Rule 6 (convergent): rel(X0, X1) :- parent(X0, X2), parent(X1, X2)")
    print(f"Similarity: {sim_diff_struct:.4f}")
    print(f"Status: {'✓ PASS' if sim_diff_struct < 0.85 else '✗ FAIL'} (Expected: <0.85)")

    # Summary
    print("\\n" + "=" * 80)
    if sim_same_pred > 0.85 and sim_diff_pred > 0.70 and sim_diff_struct < 0.85:
        print("✓ SUCCESS: Predicate-agnostic encoder works perfectly!")
        print("  ✓ Recognizes same structures with same predicates")
        print("  ✓ Transfer learning works: same structure across different predicates!")
        print("  ✓ Distinguishes different structures")
    elif sim_diff_pred > 0.70:
        print("✓ GOOD: Transfer learning is working!")
        print(f"  Same structure across different predicates: {sim_diff_pred:.4f} similarity")
        print("  (May improve with more pretraining steps)")
    else:
        print("⚠ WARNING: Transfer learning may need more pretraining")
        print(f"  Consider increasing PRETRAIN_STEPS to 5000+")
    print("=" * 80 + "\\n")
"""

# Instructions for integrating into notebook
INTEGRATION_INSTRUCTIONS = '''
# HOW TO INTEGRATE PREDICATE-AGNOSTIC PRETRAINING INTO NOTEBOOK:

1. Open Demo_ILP.ipynb

2. Find the cell that creates the state_encoder and gflownet models
   (search for "state_encoder = EnhancedStateEncoder")

3. After that cell, insert TWO new cells:

   Cell 1: Copy the code from PRETRAINING_CELL_1 above
   Cell 2: Copy the code from PRETRAINING_CELL_2 above

4. Run the notebook:
   - The first new cell will pretrain the encoder on GENERIC predicates (pred0, ..., pred9)
   - The second new cell will test transfer learning across different vocabularies
   - Then continue with normal GFlowNet training

5. Configuration options in Cell 1:
   - USE_PRETRAINING: Set to False to skip pretraining
   - PRETRAIN_STEPS: 2000-5000 (more steps = better results)
   - NUM_GENERIC_PREDICATES: Number of generic predicates (default 10)

6. Expected results:
   - Pretraining loss should decrease
   - Accuracy should reach >0.7
   - Positive similarity should increase (equivalent rules)
   - Negative similarity should stay lower (different rules)
   - Test should show same structure with DIFFERENT predicates has >0.70 similarity (TRANSFER LEARNING!)
   - Test should show different structures have <0.85 similarity

7. Key Insight:
   The encoder is truly PREDICATE-AGNOSTIC:
   - Pretraining: Uses pred0, pred1, ..., pred9 (generic names)
   - Task: Uses parent, friend, sibling, etc. (specific names)
   - Transfer works because: chain(pred0) ≈ chain(parent) ≈ chain(friend)
   - Same structure → same encoding, regardless of predicate names!
'''

def print_instructions():
    """Print integration instructions."""
    print(INTEGRATION_INSTRUCTIONS)

if __name__ == "__main__":
    print_instructions()

    print("\n" + "=" * 80)
    print("PRETRAINING CELL 1: ENCODER PRETRAINING")
    print("=" * 80)
    print(PRETRAINING_CELL_1)

    print("\n" + "=" * 80)
    print("PRETRAINING_CELL 2: TEST PRETRAINED ENCODER")
    print("=" * 80)
    print(PRETRAINING_CELL_2)
