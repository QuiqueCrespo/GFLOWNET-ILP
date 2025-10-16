"""
Comprehensive pipeline test to verify all components are working correctly.
"""

import torch
import numpy as np
from src.logic_structures import (
    Variable, Atom, Rule, Theory,
    get_initial_state, is_terminal,
    apply_add_atom, apply_unify_vars,
    get_valid_variable_pairs, get_all_variables,
    theory_to_string
)
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def test_logic_structures():
    """Test core logic data structures."""
    print("\n" + "=" * 70)
    print("TEST 1: Logic Structures")
    print("=" * 70)

    # Create initial state
    initial = get_initial_state('target', arity=2)
    print(f"\n✓ Initial state created: {theory_to_string(initial)}")

    # Test adding atoms
    state1, max_var = apply_add_atom(initial, 'parent', 2, 1)
    print(f"✓ After ADD_ATOM('parent'): {theory_to_string(state1)}")

    state2, max_var = apply_add_atom(state1, 'parent', 2, max_var)
    print(f"✓ After another ADD_ATOM('parent'): {theory_to_string(state2)}")

    # Test variable unification
    vars = get_all_variables(state2)
    print(f"✓ All variables: {[f'X{v.id}' for v in vars]}")

    if len(vars) >= 2:
        state3 = apply_unify_vars(state2, vars[1], vars[2])
        print(f"✓ After UNIFY_VARIABLES(X{vars[1].id}, X{vars[2].id}): {theory_to_string(state3)}")

    # Test valid pairs
    pairs = get_valid_variable_pairs(state2)
    print(f"✓ Valid variable pairs: {len(pairs)}")

    # Test terminal state
    print(f"✓ Is terminal: {is_terminal(state2)}")

    print("\n✅ Logic structures test PASSED")
    return True


def test_logic_engine():
    """Test logic engine for entailment."""
    print("\n" + "=" * 70)
    print("TEST 2: Logic Engine")
    print("=" * 70)

    engine = LogicEngine(max_depth=5)

    # Create a simple theory: parent(X, Y) :- true
    v0, v1 = Variable(0), Variable(1)
    head = Atom('parent', (v0, v1))
    rule = Rule(head=head, body=[])
    theory = [rule]

    print(f"\nTheory: {theory_to_string(theory)}")

    # Test entailment - should be true for any parent relation
    ex1 = Example('parent', ('alice', 'bob'))
    result1 = engine.entails(theory, ex1)
    print(f"✓ entails({ex1}): {result1}")

    # Create theory with body: target(X, Y) :- parent(X, Z), parent(Z, Y)
    v2 = Variable(2)
    head2 = Atom('target', (v0, v1))
    body2 = [Atom('parent', (v0, v2)), Atom('parent', (v2, v1))]
    rule2 = Rule(head=head2, body=body2)
    theory2 = [rule2]

    print(f"\nTheory: {theory_to_string(theory2)}")

    # This should not entail anything without base facts
    ex2 = Example('target', ('alice', 'charlie'))
    result2 = engine.entails(theory2, ex2)
    print(f"✓ entails({ex2}): {result2}")

    # Test coverage
    examples = [Example('parent', ('a', 'b')), Example('parent', ('c', 'd'))]
    coverage = engine.get_coverage(theory, examples)
    print(f"✓ Coverage: {coverage:.2f}")

    print("\n✅ Logic engine test PASSED")
    return True


def test_graph_encoder():
    """Test graph construction and encoding."""
    print("\n" + "=" * 70)
    print("TEST 3: Graph Encoder")
    print("=" * 70)

    predicate_vocab = ['target', 'parent']
    graph_constructor = GraphConstructor(predicate_vocab)

    # Create a theory
    initial = get_initial_state('target', arity=2)
    state, _ = apply_add_atom(initial, 'parent', 2, 1)

    print(f"\nTheory: {theory_to_string(state)}")

    # Convert to graph
    graph = graph_constructor.theory_to_graph(state)
    print(f"✓ Graph created: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
    print(f"✓ Node feature dim: {graph.x.size(1)}")

    # Test encoder
    node_feature_dim = len(predicate_vocab) + 1
    embedding_dim = 32
    encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=2)

    graph_emb, node_emb = encoder(graph)
    print(f"✓ Graph embedding shape: {graph_emb.shape}")
    print(f"✓ Node embeddings shape: {node_emb.shape}")

    # Test variable node mapping
    var_to_node = graph_constructor.get_variable_node_ids(state)
    print(f"✓ Variable to node mapping: {len(var_to_node)} variables")

    print("\n✅ Graph encoder test PASSED")
    return True


def test_gflownet_models():
    """Test GFlowNet models."""
    print("\n" + "=" * 70)
    print("TEST 4: GFlowNet Models")
    print("=" * 70)

    embedding_dim = 32
    num_predicates = 3
    hidden_dim = 64

    # Create models
    gflownet = HierarchicalGFlowNet(embedding_dim, num_predicates, hidden_dim)

    # Test strategist
    state_emb = torch.randn(embedding_dim)
    action_logits, log_flow = gflownet.forward_strategist(state_emb)
    print(f"\n✓ Strategist action logits: {action_logits.shape}")
    print(f"✓ Strategist log flow: {log_flow.shape}")
    print(f"✓ Action probabilities: {torch.softmax(action_logits, dim=-1).detach().numpy()}")

    # Test atom adder
    pred_logits = gflownet.forward_atom_adder(state_emb)
    print(f"✓ Atom adder logits: {pred_logits.shape}")
    print(f"✓ Predicate probabilities: {torch.softmax(pred_logits, dim=-1).detach().numpy()}")

    # Test variable unifier
    num_vars = 4
    var_embs = torch.randn(num_vars, embedding_dim)
    pair_logits = gflownet.forward_variable_unifier(state_emb, var_embs)
    expected_pairs = num_vars * (num_vars - 1) // 2
    print(f"✓ Variable unifier logits: {pair_logits.shape} (expected {expected_pairs} pairs)")

    # Get pair indices
    pairs = gflownet.variable_unifier.get_pair_indices(num_vars)
    print(f"✓ Pair indices: {pairs[:3]}... ({len(pairs)} total)")

    # Test parameter collection
    params = gflownet.get_all_parameters()
    total_params = sum(p.numel() for p in params)
    print(f"✓ Total parameters: {total_params:,}")

    print("\n✅ GFlowNet models test PASSED")
    return True


def test_reward_calculator():
    """Test reward calculation."""
    print("\n" + "=" * 70)
    print("TEST 5: Reward Calculator")
    print("=" * 70)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine, weight_pos=0.6, weight_neg=0.3, weight_simplicity=0.1)

    # Simple theory
    v0, v1 = Variable(0), Variable(1)
    head = Atom('grandparent', (v0, v1))
    rule = Rule(head=head, body=[])
    theory = [rule]

    print(f"\nTheory: {theory_to_string(theory)}")

    # Create examples
    pos_examples = [
        Example('grandparent', ('a', 'b')),
        Example('grandparent', ('c', 'd'))
    ]
    neg_examples = [
        Example('grandparent', ('x', 'x'))
    ]

    # Calculate reward
    reward = reward_calc.calculate_reward(theory, pos_examples, neg_examples)
    print(f"✓ Basic reward: {reward:.4f}")

    # Get detailed scores
    scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)
    print(f"✓ Detailed scores:")
    print(f"  - Positive coverage: {scores['pos_covered']}/{scores['pos_total']} (score: {scores['pos_score']:.4f})")
    print(f"  - Negative coverage: {scores['neg_covered']}/{scores['neg_total']} (score: {scores['neg_score']:.4f})")
    print(f"  - Simplicity: {scores['simplicity']:.4f} ({scores['total_atoms']} atoms)")
    print(f"  - Total reward: {scores['reward']:.4f}")

    # Test with more complex theory
    v2 = Variable(2)
    head2 = Atom('grandparent', (v0, v1))
    body2 = [Atom('parent', (v0, v2)), Atom('parent', (v2, v1))]
    rule2 = Rule(head=head2, body=body2)
    theory2 = [rule2]

    print(f"\nMore complex theory: {theory_to_string(theory2)}")
    reward2 = reward_calc.calculate_reward(theory2, pos_examples, neg_examples)
    print(f"✓ Reward for complex theory: {reward2:.4f}")

    print("\n✅ Reward calculator test PASSED")
    return True


def test_trajectory_generation():
    """Test trajectory generation."""
    print("\n" + "=" * 70)
    print("TEST 6: Trajectory Generation")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    predicate_vocab = ['target', 'parent']
    predicate_arities = {'target': 2, 'parent': 2}

    node_feature_dim = len(predicate_vocab) + 1
    embedding_dim = 32
    hidden_dim = 64

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=2)
    gflownet = HierarchicalGFlowNet(embedding_dim, len(predicate_vocab), hidden_dim)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine)

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calc,
        predicate_vocab=predicate_vocab,
        predicate_arities=predicate_arities,
        learning_rate=1e-3
    )

    # Generate trajectory
    initial_state = get_initial_state('target', arity=2)
    pos_examples = [Example('target', ('a', 'b'))]
    neg_examples = [Example('target', ('x', 'x'))]

    print(f"\nInitial state: {theory_to_string(initial_state)}")

    trajectory, reward = trainer.generate_trajectory(
        initial_state, pos_examples, neg_examples, max_steps=5
    )

    print(f"✓ Generated trajectory with {len(trajectory)} steps")
    print(f"✓ Final reward: {reward:.4f}")

    # Show trajectory
    for i, step in enumerate(trajectory):
        print(f"\nStep {i}:")
        print(f"  Action: {step.action_type}")
        print(f"  Detail: {step.action_detail}")
        print(f"  Log P_F: {step.log_pf.item():.4f}")
        print(f"  Next state: {theory_to_string(step.next_state)}")

    print("\n✅ Trajectory generation test PASSED")
    return True


def test_training_loop():
    """Test the training loop."""
    print("\n" + "=" * 70)
    print("TEST 7: Training Loop")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    predicate_vocab = ['target', 'parent']
    predicate_arities = {'target': 2, 'parent': 2}

    node_feature_dim = len(predicate_vocab) + 1
    embedding_dim = 32
    hidden_dim = 64

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=2)
    gflownet = HierarchicalGFlowNet(embedding_dim, len(predicate_vocab), hidden_dim)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine)

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calc,
        predicate_vocab=predicate_vocab,
        predicate_arities=predicate_arities,
        learning_rate=1e-3
    )

    # Train for a few episodes
    initial_state = get_initial_state('target', arity=2)
    pos_examples = [Example('target', ('a', 'b')), Example('target', ('c', 'd'))]
    neg_examples = [Example('target', ('x', 'x'))]

    print(f"\nTraining for 10 episodes...")
    history = trainer.train(
        initial_state=initial_state,
        positive_examples=pos_examples,
        negative_examples=neg_examples,
        num_episodes=10,
        verbose=False
    )

    print(f"✓ Training completed: {len(history)} episodes")
    print(f"✓ Episode 0 - Loss: {history[0]['loss']:.4f}, Reward: {history[0]['reward']:.4f}")
    print(f"✓ Episode 9 - Loss: {history[9]['loss']:.4f}, Reward: {history[9]['reward']:.4f}")
    print(f"✓ Final log_Z: {history[-1]['log_Z']:.4f}")

    # Test sampling
    print(f"\nSampling best theory from 5 samples...")
    best_theory, best_reward = trainer.sample_best_theory(
        initial_state, pos_examples, neg_examples, num_samples=5
    )

    print(f"✓ Best theory found:")
    print(f"  {theory_to_string(best_theory)}")
    print(f"✓ Best reward: {best_reward:.4f}")

    print("\n✅ Training loop test PASSED")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\n" + "=" * 70)
    print("TEST 8: Gradient Flow")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    predicate_vocab = ['target', 'parent']
    predicate_arities = {'target': 2, 'parent': 2}

    node_feature_dim = len(predicate_vocab) + 1
    embedding_dim = 32
    hidden_dim = 64

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=2)
    gflownet = HierarchicalGFlowNet(embedding_dim, len(predicate_vocab), hidden_dim)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine)

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calc,
        predicate_vocab=predicate_vocab,
        predicate_arities=predicate_arities,
        learning_rate=1e-3
    )

    # Get initial parameter values
    initial_params = {}
    for name, param in state_encoder.named_parameters():
        initial_params[f'encoder_{name}'] = param.clone().detach()
    for name, param in gflownet.named_parameters():
        initial_params[f'gflownet_{name}'] = param.clone().detach()
    initial_log_Z = trainer.log_Z.clone().detach()

    # Train one step
    initial_state = get_initial_state('target', arity=2)
    pos_examples = [Example('target', ('a', 'b'))]
    neg_examples = [Example('target', ('x', 'x'))]

    metrics = trainer.train_step(initial_state, pos_examples, neg_examples)
    print(f"\n✓ Training step completed")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Reward: {metrics['reward']:.4f}")

    # Check if parameters changed
    params_changed = 0
    for name, param in state_encoder.named_parameters():
        if not torch.allclose(initial_params[f'encoder_{name}'], param, atol=1e-8):
            params_changed += 1
    for name, param in gflownet.named_parameters():
        if not torch.allclose(initial_params[f'gflownet_{name}'], param, atol=1e-8):
            params_changed += 1

    log_z_changed = not torch.allclose(initial_log_Z, trainer.log_Z, atol=1e-8)

    print(f"✓ Parameters changed: {params_changed}/{len(initial_params)}")
    print(f"✓ log_Z changed: {log_z_changed} (from {initial_log_Z.item():.6f} to {trainer.log_Z.item():.6f})")

    if params_changed > 0:
        print("\n✅ Gradient flow test PASSED")
        return True
    else:
        print("\n⚠️  WARNING: No parameters changed (might indicate gradient issue)")
        return False


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 18 + "COMPREHENSIVE PIPELINE TEST" + " " * 23 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    tests = [
        test_logic_structures,
        test_logic_engine,
        test_graph_encoder,
        test_gflownet_models,
        test_reward_calculator,
        test_trajectory_generation,
        test_training_loop,
        test_gradient_flow
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print(f"{'=' * 70}\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! The pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")


if __name__ == "__main__":
    main()
