"""
Quick-start script: Add contrastive pre-training to your GFlowNet pipeline.

This is the EASIEST and FASTEST way to improve embedding quality.
"""

import sys
sys.path.insert(0, '/Users/jq23948/Documents/GFLOWNET-ILP')

import numpy as np
import matplotlib.pyplot as plt
from src.logic_structures import get_initial_state
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from contrastive_pretraining import ContrastivePreTrainer, generate_base_rules
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
predicate_vocab = ['parent']
predicate_arities = {'parent': 2}
embedding_dim = 32
num_layers = 2

print("=" * 80)
print("CONTRASTIVE PRE-TRAINING DEMO")
print("=" * 80)

# Step 1: Create encoder
print("\nStep 1: Creating graph encoder...")
graph_constructor = EnhancedGraphConstructor(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=embedding_dim,
    num_layers=num_layers
)

# Step 2: Test BEFORE pre-training
print("\nStep 2: Testing embeddings BEFORE pre-training...")

def create_test_rule(head_pred, head_args, body_atoms_list):
    from src.logic_structures import Rule, Atom, Variable
    head_vars = [Variable(id=vid) for vid in head_args]
    head = Atom(predicate_name=head_pred, args=tuple(head_vars))
    body_atoms = []
    for pred_name, var_ids in body_atoms_list:
        vars = [Variable(id=vid) for vid in var_ids]
        body_atoms.append(Atom(predicate_name=pred_name, args=tuple(vars)))
    rule = Rule(head=head, body=tuple(body_atoms))
    return [rule]

def get_embedding(theory):
    graph_data = graph_constructor.theory_to_graph(theory)
    state_embedding, _ = state_encoder(graph_data)
    return state_embedding.squeeze(0).detach().numpy()

# Test rules
rule_chain = create_test_rule('grandparent', [0, 1], [('parent', (0, 2)), ('parent', (2, 1))])
rule_convergent = create_test_rule('grandparent', [0, 1], [('parent', (0, 2)), ('parent', (1, 2))])

emb_chain_before = get_embedding(rule_chain)
emb_conv_before = get_embedding(rule_convergent)

sim_before = cosine_similarity([emb_chain_before], [emb_conv_before])[0, 0]
print(f"  Similarity (chain vs convergent): {sim_before:.6f}")
print(f"  Status: {'❌ TOO SIMILAR' if sim_before > 0.95 else '✅ Good'}")

# Step 3: Generate base rules for pre-training
print("\nStep 3: Generating base rules for pre-training...")
base_rules = generate_base_rules(predicate_vocab, predicate_arities, num_rules=100)
print(f"  Generated {len(base_rules)} base rules")

# Step 4: Pre-train
print("\nStep 4: Running contrastive pre-training...")
pretrainer = ContrastivePreTrainer(
    state_encoder=state_encoder,
    graph_constructor=graph_constructor,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities
)

losses = pretrainer.pretrain(base_rules, num_epochs=200, verbose=True)

# Step 5: Test AFTER pre-training
print("\nStep 5: Testing embeddings AFTER pre-training...")
emb_chain_after = get_embedding(rule_chain)
emb_conv_after = get_embedding(rule_convergent)

sim_after = cosine_similarity([emb_chain_after], [emb_conv_after])[0, 0]
print(f"  Similarity (chain vs convergent): {sim_after:.6f}")
print(f"  Status: {'✅ Good' if sim_after < 0.90 else '❌ Still too similar'}")

# Step 6: Visualize improvement
print("\nStep 6: Visualizing improvement...")

improvement = sim_before - sim_after
print(f"\n  Improvement: {improvement:.6f} reduction in similarity")
print(f"  Before: {sim_before:.6f}")
print(f"  After:  {sim_after:.6f}")

# Plot training loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(losses, linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Contrastive Loss', fontsize=12)
axes[0].set_title('Pre-training Loss Curve', fontsize=14)
axes[0].grid(alpha=0.3)

# Before/After comparison
categories = ['Before', 'After']
similarities = [sim_before, sim_after]
colors = ['red' if s > 0.90 else 'green' for s in similarities]

axes[1].bar(categories, similarities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='Target (<0.90)')
axes[1].set_ylabel('Cosine Similarity', fontsize=12)
axes[1].set_title('Embedding Similarity: Chain vs Convergent', fontsize=14)
axes[1].set_ylim([0, 1.05])
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('contrastive_pretraining_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: contrastive_pretraining_results.png")
plt.show()

# Step 7: Save pre-trained encoder (optional)
print("\nStep 7: Saving pre-trained encoder...")
import torch
torch.save(state_encoder.state_dict(), 'pretrained_encoder.pt')
print("✓ Saved to: pretrained_encoder.pt")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✓ Pre-training completed successfully!")
print(f"  - Initial similarity: {sim_before:.6f} (semantically different rules)")
print(f"  - Final similarity:   {sim_after:.6f}")
print(f"  - Improvement:        {improvement:.6f}")

if sim_after < 0.90:
    print("\n🎉 SUCCESS! The encoder can now distinguish semantic differences.")
    print("   Use this pre-trained encoder in your GFlowNet training:")
    print("\n   state_encoder.load_state_dict(torch.load('pretrained_encoder.pt'))")
else:
    print("\n⚠️  Similarity still high. Try:")
    print("   1. More pre-training epochs (500-1000)")
    print("   2. More diverse base rules (200-500)")
    print("   3. Lower temperature in contrastive loss")
    print("   4. Add edge features (see improved_graph_encoder.py)")

print("=" * 80)
