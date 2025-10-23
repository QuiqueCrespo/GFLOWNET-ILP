"""
Diagnostic: Why isn't the model learning to avoid head unification?

This script tests:
1. Are state embeddings different for states with/without head unification?
2. Does the variable unifier assign different scores to head vs body pairs?
3. Are gradients flowing to the variable unifier?
"""

import sys
sys.path.insert(0, '/Users/jq23948/Documents/GFLOWNET-ILP')

import torch
from src.logic_structures import (
    get_initial_state, apply_add_atom, apply_unify_vars,
    get_valid_variable_pairs, get_all_variables, Variable, theory_to_string
)
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet

print("="*80)
print("DIAGNOSTIC: Head Unification Learning")
print("="*80)

# Setup
predicate_vocab = ['parent']
predicate_arities = {'parent': 2}

background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
]

positive_examples = [Example('grandparent', ('alice', 'charlie'))]
negative_examples = [Example('grandparent', ('alice', 'alice'))]

logic_engine = LogicEngine(max_depth=10, background_facts=background_facts)
reward_calc = RewardCalculator(logic_engine, use_f1=True)

graph_constructor = EnhancedGraphConstructor(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=32,
    num_layers=2
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=32,
    num_predicates=len(predicate_vocab),
    hidden_dim=64,
    use_sophisticated_backward=True,
    predicate_vocab=predicate_vocab
)

# ============================================================================
# TEST 1: Are state embeddings different?
# ============================================================================

print("\n" + "-"*80)
print("TEST 1: State Embedding Similarity")
print("-"*80)

# Create two states
state1 = get_initial_state('grandparent', 2)
state1, _ = apply_add_atom(state1, 'parent', 2, 1)
# state1: grandparent(X0, X1) :- parent(X2, X3)

state2 = apply_unify_vars(state1, Variable(0), Variable(1))
# state2: grandparent(X0, X0) :- parent(X2, X3)

print(f"\nState 1 (good): {theory_to_string(state1)}")
print(f"State 2 (bad):  {theory_to_string(state2)}")

# Get embeddings
with torch.no_grad():
    graph1 = graph_constructor.theory_to_graph(state1)
    emb1, node_emb1 = state_encoder(graph1)

    graph2 = graph_constructor.theory_to_graph(state2)
    emb2, node_emb2 = state_encoder(graph2)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(
    emb1.numpy().reshape(1, -1),
    emb2.numpy().reshape(1, -1)
)[0, 0]

print(f"\nCosine similarity: {similarity:.6f}")
if similarity > 0.95:
    print("  ❌ VERY SIMILAR - Model may struggle to differentiate!")
elif similarity > 0.85:
    print("  ⚠️  QUITE SIMILAR - Differentiation may be difficult")
else:
    print("  ✅ DIFFERENT - Model should be able to differentiate")

# ============================================================================
# TEST 2: Does variable unifier score head pairs differently?
# ============================================================================

print("\n" + "-"*80)
print("TEST 2: Variable Unifier Scoring")
print("-"*80)

# Use state1 (before unification)
variables = get_all_variables(state1)
print(f"\nVariables: {[f'X{v.id}' for v in variables]}")

# Get head variables
rule = state1[0]
head_vars = set(arg for arg in rule.head.args if isinstance(arg, Variable))
print(f"Head variables: {[f'X{v.id}' for v in head_vars]}")

# Get variable embeddings
with torch.no_grad():
    graph = graph_constructor.theory_to_graph(state1)
    state_emb, node_embs = state_encoder(graph)
    var_embs = node_embs[:len(variables)]

    # Get unifier scores for all pairs
    pair_logits = gflownet.forward_variable_unifier(state_emb.squeeze(0), var_embs)

# Map logits to pairs
valid_pairs = get_valid_variable_pairs(state1)
print(f"\nValid pairs: {len(valid_pairs)}")

from src.gflownet_models import VariableUnifierGFlowNet
unifier = VariableUnifierGFlowNet(32, 64)
pair_indices = unifier.get_pair_indices(len(variables))

print("\nPair scores:")
for idx, (i, j) in enumerate(pair_indices[:10]):  # Show first 10
    if idx >= len(pair_logits):
        break

    vi, vj = variables[i], variables[j]
    score = pair_logits[idx].item()

    # Check if both are in head
    both_in_head = (vi in head_vars and vj in head_vars)
    label = "HEAD PAIR ⚠️" if both_in_head else "body pair"

    print(f"  ({i}, {j}) = (X{vi.id}, X{vj.id}): score={score:+.4f}  [{label}]")

# Compute average score for head pairs vs others
head_pair_scores = []
other_pair_scores = []

for idx, (i, j) in enumerate(pair_indices):
    if idx >= len(pair_logits):
        break

    vi, vj = variables[i], variables[j]
    score = pair_logits[idx].item()

    if vi in head_vars and vj in head_vars:
        head_pair_scores.append(score)
    else:
        other_pair_scores.append(score)

if head_pair_scores:
    print(f"\nAverage score for HEAD pairs: {sum(head_pair_scores)/len(head_pair_scores):+.4f}")
if other_pair_scores:
    print(f"Average score for OTHER pairs: {sum(other_pair_scores)/len(other_pair_scores):+.4f}")

if head_pair_scores and other_pair_scores:
    head_avg = sum(head_pair_scores)/len(head_pair_scores)
    other_avg = sum(other_pair_scores)/len(other_pair_scores)

    if head_avg < other_avg - 0.5:
        print("  ✅ Model DOES distinguish - head pairs have lower scores!")
    elif abs(head_avg - other_avg) < 0.1:
        print("  ❌ Model DOES NOT distinguish - scores are similar")
    else:
        print("  ⚠️  Model partially distinguishes")

# ============================================================================
# TEST 3: Check reward difference
# ============================================================================

print("\n" + "-"*80)
print("TEST 3: Reward Verification")
print("-"*80)

reward1 = reward_calc.calculate_reward(state1, positive_examples, negative_examples)
reward2 = reward_calc.calculate_reward(state2, positive_examples, negative_examples)

scores1 = reward_calc.get_detailed_scores(state1, positive_examples, negative_examples)
scores2 = reward_calc.get_detailed_scores(state2, positive_examples, negative_examples)

print(f"\nState 1 (no head unification):")
print(f"  Reward: {reward1:.6f}")
print(f"  TP={scores1['TP']}, FP={scores1['FP']}, Self-loops={scores1['num_self_loops']}")

print(f"\nState 2 (head unified):")
print(f"  Reward: {reward2:.6f}")
print(f"  TP={scores2['TP']}, FP={scores2['FP']}, Self-loops={scores2['num_self_loops']}")

print(f"\nReward difference: {reward1 - reward2:.6f}")
if reward2 < 0.01:
    print("  ✅ Head unification leads to ~0 reward (as expected)")
else:
    print("  ⚠️  Head unification doesn't lead to 0 reward?")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\n1. State Embedding Differentiation:")
print(f"   Similarity: {similarity:.4f}")
if similarity > 0.95:
    print("   ❌ PROBLEM: States are too similar")
else:
    print("   ✅ States are distinguishable")

print("\n2. Variable Unifier Scoring:")
if head_pair_scores and other_pair_scores:
    head_avg = sum(head_pair_scores)/len(head_pair_scores)
    other_avg = sum(other_pair_scores)/len(other_pair_scores)
    print(f"   Head pair avg: {head_avg:+.4f}")
    print(f"   Other pair avg: {other_avg:+.4f}")

    if head_avg < other_avg - 0.5:
        print("   ✅ Unifier distinguishes head pairs")
    else:
        print("   ❌ PROBLEM: Unifier doesn't distinguish head pairs")
else:
    print("   ? Could not compute (no head pairs)")

print("\n3. Reward Signal:")
print(f"   Head unification reward: {reward2:.6f}")
if reward2 < 0.01:
    print("   ✅ Clear signal that head unification is bad")
else:
    print("   ❌ PROBLEM: Signal is not clear")

print("\n" + "="*80)
print("LIKELY ROOT CAUSES (if model not learning):")
print("="*80)
print("1. 🔴 Replay buffer dilution (Bug #1)")
print("      - Bad trajectories not stored in replay buffer")
print("      - 50% of training uses good replayed trajectories")
print("      - Gradient signal diluted")
print("\n2. 🟡 log_Z compensation (Issue #5)")
print("      - log_Z absorbs difference instead of policy learning")
print("\n3. 🟡 Reward scaling with alpha=10")
print("      - Creates extreme log values, numerical issues")
print("\n4. 🟢 State/variable embeddings (if tests above show problems)")
print("      - Model physically cannot distinguish states")

print("\nRECOMMENDED FIXES (in order):")
print("1. Fix replay buffer (don't add off-policy loss, replace or weight properly)")
print("2. Monitor log_Z (if >10, it's compensating)")
print("3. Reduce reward_scale_alpha to 1.0 or 2.0")
print("4. If tests show embedding issues, improve encoder")
print("="*80)
