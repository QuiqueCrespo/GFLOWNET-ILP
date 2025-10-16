"""Debug why generation loop exits early with free variables."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

import torch
import torch.nn.functional as F
from src.logic_structures import (
    get_initial_state, is_terminal, theory_to_string,
    get_all_variables, get_valid_variable_pairs,
    apply_add_atom, apply_unify_vars
)
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet

# Setup
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie'))
]
positive_examples = [Example('grandparent', ('alice', 'charlie'))]
negative_examples = [Example('grandparent', ('alice', 'alice'))]

predicate_vocab = ['parent']
predicate_arities = {'parent': 2}

graph_constructor = EnhancedGraphConstructor(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=32,
    num_layers=2
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=32,
    num_predicates=len(predicate_vocab),
    hidden_dim=64
)

print("="*80)
print("DEBUGGING GENERATION LOOP")
print("="*80)

# Run a single generation with detailed logging
initial_state = get_initial_state('grandparent', 2)
current_state = initial_state
max_var_id = max([v.id for v in get_all_variables(current_state)], default=-1)
step_count = 0
max_steps = 10

print(f"\nInitial state: {theory_to_string(initial_state)}")
print(f"Is terminal: {is_terminal(initial_state)}")

rule = initial_state[0]
head_vars = set(rule.head.args)
body_vars = set()
for atom in rule.body:
    body_vars.update(atom.args)
free_vars = head_vars - body_vars
print(f"Free variables: {len(free_vars)} ({', '.join(f'X{v.id}' for v in free_vars)})")
print()

while not is_terminal(current_state) and step_count < max_steps:
    print(f"--- Step {step_count} ---")
    print(f"Current state: {theory_to_string(current_state)}")

    # Check free variables
    rule = current_state[0]
    head_vars = set(rule.head.args)
    body_vars = set()
    for atom in rule.body:
        body_vars.update(atom.args)
    free_vars = head_vars - body_vars
    print(f"Free variables: {len(free_vars)} ({', '.join(f'X{v.id}' for v in free_vars)})")
    print(f"Is terminal: {is_terminal(current_state)}")

    # Encode state
    graph_data = graph_constructor.theory_to_graph(current_state)
    state_embedding, node_embeddings = state_encoder(graph_data)
    state_embedding = state_embedding.squeeze(0)

    # Get action
    action_logits, _ = gflownet.forward_strategist(state_embedding)
    action_probs = F.softmax(action_logits, dim=-1)
    action = torch.multinomial(action_probs, 1).item()

    print(f"Action chosen: {'ADD_ATOM' if action == 0 else 'UNIFY_VARIABLES'}")
    print(f"Action probs: ADD_ATOM={action_probs[0].item():.4f}, UNIFY_VARIABLES={action_probs[1].item():.4f}")

    if action == 0:  # ADD_ATOM
        atom_logits = gflownet.forward_atom_adder(state_embedding)
        atom_probs = F.softmax(atom_logits, dim=-1)
        pred_idx = torch.multinomial(atom_probs, 1).item()
        pred_name = predicate_vocab[pred_idx]
        pred_arity = predicate_arities[pred_name]

        print(f"Adding atom: {pred_name} (arity {pred_arity})")

        next_state, max_var_id = apply_add_atom(
            current_state, pred_name, pred_arity, max_var_id
        )
        print(f"Next state: {theory_to_string(next_state)}")

    else:  # UNIFY_VARIABLES
        valid_pairs = get_valid_variable_pairs(current_state)
        print(f"Valid variable pairs: {len(valid_pairs)}")

        if not valid_pairs:
            print("✗ No valid pairs - BREAKING!")
            break

        variables = get_all_variables(current_state)
        print(f"All variables: {len(variables)}")

        if len(variables) < 2:
            print("✗ Less than 2 variables - BREAKING!")
            break

        var_embeddings = node_embeddings[:len(variables)]
        pair_logits = gflownet.forward_variable_unifier(
            state_embedding, var_embeddings
        )

        print(f"Pair logits length: {len(pair_logits)}")

        if len(pair_logits) == 0:
            print("✗ No pair logits - BREAKING!")
            break

        pair_probs = F.softmax(pair_logits, dim=-1)
        pair_idx = torch.multinomial(pair_probs, 1).item()
        var1, var2 = valid_pairs[pair_idx]

        print(f"Unifying: X{var1.id} with X{var2.id}")

        next_state = apply_unify_vars(current_state, var1, var2)
        print(f"Next state: {theory_to_string(next_state)}")

    current_state = next_state
    step_count += 1
    print()

print("="*80)
print("FINAL STATE")
print("="*80)
print(f"\nFinal state: {theory_to_string(current_state)}")
print(f"Is terminal: {is_terminal(current_state)}")
print(f"Steps taken: {step_count}")
print(f"Max steps: {max_steps}")

rule = current_state[0]
head_vars = set(rule.head.args)
body_vars = set()
for atom in rule.body:
    body_vars.update(atom.args)
free_vars = head_vars - body_vars

print(f"\nFree variables: {len(free_vars)} ({', '.join(f'X{v.id}' for v in free_vars)})")
print(f"Body length: {len(rule.body)}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if free_vars and not is_terminal(current_state) and step_count < max_steps:
    print("\n✗ BUG CONFIRMED: Loop exited with free variables but not terminal!")
    print("  This means one of the break statements was hit.")
elif free_vars and is_terminal(current_state):
    print("\n⚠ Free variables remain but state is terminal (max length reached)")
elif not free_vars:
    print("\n✓ No free variables - rule is valid!")
else:
    print("\n? Unexpected state")
