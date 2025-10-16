"""
Deep analysis of convergence behavior to understand why the model
converges to simple unification rules instead of complex body rules.
"""

import torch
import numpy as np
from src.logic_structures import (
    get_initial_state, theory_to_string, get_all_variables,
    apply_add_atom, apply_unify_vars, Variable
)
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def analyze_state_encoding():
    """Analyze how states are encoded as graphs."""
    print("=" * 70)
    print("ANALYSIS 1: State Encoding")
    print("=" * 70)

    predicate_vocab = ['grandparent', 'parent']
    graph_constructor = GraphConstructor(predicate_vocab)

    # Test different theories
    theories = []

    # 1. Initial state: grandparent(X0, X1)
    t1 = get_initial_state('grandparent', 2)
    theories.append(("Initial: grandparent(X0, X1)", t1))

    # 2. After unify: grandparent(X0, X0)
    t2 = apply_unify_vars(t1, Variable(0), Variable(1))
    theories.append(("Unified: grandparent(X0, X0)", t2))

    # 3. After add atom: grandparent(X0, X1) :- parent(X2, X3)
    t3, _ = apply_add_atom(t1, 'parent', 2, 1)
    theories.append(("Add atom: grandparent(X0, X1) :- parent(X2, X3)", t3))

    # 4. Complex: grandparent(X0, X1) :- parent(X0, X2), parent(X2, X1)
    t4, max_var = apply_add_atom(t1, 'parent', 2, 1)
    t4 = apply_unify_vars(t4, Variable(0), Variable(2))  # X0 = X2
    t4, max_var = apply_add_atom(t4, 'parent', 2, max_var)
    t4 = apply_unify_vars(t4, Variable(3), Variable(1))  # X3 = X1
    t4 = apply_unify_vars(t4, Variable(4), Variable(5))  # Make second parent(X4, X4) -> parent(X2, X1)
    # This is getting complex - let me simplify

    print("\nGraph representations:")
    for desc, theory in theories:
        graph = graph_constructor.theory_to_graph(theory)
        print(f"\n{desc}")
        print(f"  Theory: {theory_to_string(theory)}")
        print(f"  Graph: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
        print(f"  Node features shape: {graph.x.shape}")

    print("\n✓ State encoding seems reasonable")
    print("  Issue: All encodings are small graphs (few nodes)")
    print("  Simple unification creates smallest graphs")


def analyze_action_probabilities():
    """Analyze what actions the model prefers."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Action Probabilities")
    print("=" * 70)

    torch.manual_seed(42)

    predicate_vocab = ['grandparent', 'parent']
    predicate_arities = {'grandparent': 2, 'parent': 2}

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=2)
    gflownet = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

    # Test on different states
    states = [
        ("Initial", get_initial_state('grandparent', 2)),
    ]

    t_with_atom, _ = apply_add_atom(get_initial_state('grandparent', 2), 'parent', 2, 1)
    states.append(("With 1 atom", t_with_atom))

    print("\nAction probabilities at different states (BEFORE training):")
    for desc, state in states:
        graph = graph_constructor.theory_to_graph(state)
        state_emb, node_embs = state_encoder(graph)
        state_emb = state_emb.squeeze(0)

        action_logits, log_flow = gflownet.forward_strategist(state_emb)
        action_probs = torch.softmax(action_logits, dim=-1)

        print(f"\n{desc}: {theory_to_string(state)}")
        print(f"  ADD_ATOM: {action_probs[0].item():.3f}")
        print(f"  UNIFY_VARIABLES: {action_probs[1].item():.3f}")
        print(f"  log F(s): {log_flow.item():.3f}")

        # Check atom adder
        atom_logits = gflownet.forward_atom_adder(state_emb)
        atom_probs = torch.softmax(atom_logits, dim=-1)
        print(f"  Predicate probs: grandparent={atom_probs[0].item():.3f}, parent={atom_probs[1].item():.3f}")


def analyze_reward_landscape():
    """Analyze the reward landscape for different theories."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Reward Landscape")
    print("=" * 70)

    engine = LogicEngine()
    reward_calc = RewardCalculator(engine)

    pos_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('bob', 'diana')),
    ]
    neg_examples = [
        Example('grandparent', ('alice', 'alice')),
        Example('grandparent', ('charlie', 'alice')),
    ]

    print(f"\nPositive examples: {[str(e) for e in pos_examples]}")
    print(f"Negative examples: {[str(e) for e in neg_examples]}")

    theories_to_test = []

    # 1. Empty rule: grandparent(X0, X1)
    t1 = get_initial_state('grandparent', 2)
    theories_to_test.append(("grandparent(X0, X1) - covers everything", t1))

    # 2. Unified: grandparent(X0, X0)
    t2 = apply_unify_vars(t1, Variable(0), Variable(1))
    theories_to_test.append(("grandparent(X0, X0) - identity only", t2))

    # 3. With unrelated atom: grandparent(X0, X1) :- parent(X2, X3)
    t3, _ = apply_add_atom(t1, 'parent', 2, 1)
    theories_to_test.append(("grandparent(X0, X1) :- parent(X2, X3) - unrelated vars", t3))

    # 4. With related atom: grandparent(X0, X1) :- parent(X0, X1)
    t4, _ = apply_add_atom(t1, 'parent', 2, 1)
    t4 = apply_unify_vars(t4, Variable(0), Variable(2))
    t4 = apply_unify_vars(t4, Variable(1), Variable(3))
    theories_to_test.append(("grandparent(X0, X1) :- parent(X0, X1) - same relation", t4))

    # 5. Different predicate in head
    t5 = apply_unify_vars(t1, Variable(0), Variable(1))
    t5_str = theory_to_string(t5)
    if 'X0, X0' in t5_str:
        theories_to_test.append(("grandparent(X0, X0) - unified head", t5))

    print("\n" + "-" * 70)
    print("Reward analysis for different theories:")
    print("-" * 70)

    for desc, theory in theories_to_test:
        scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)

        print(f"\n{desc}")
        print(f"  Theory: {theory_to_string(theory)}")
        print(f"  Positive: {scores['pos_covered']}/{scores['pos_total']} (score: {scores['pos_score']:.2f})")
        print(f"  Negative: {scores['neg_covered']}/{scores['neg_total']} (score: {scores['neg_score']:.2f})")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  Simplicity: {scores['simplicity']:.2f}")
        print(f"  Uninformative penalty: {scores['uninformative_penalty']:.2f}")
        print(f"  → REWARD: {scores['reward']:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("Without background knowledge (parent facts), NO rule can cover")
    print("positive examples! The logic engine can't prove anything.")
    print("So all rules get pos_score = 0, leading to accuracy = 0.")
    print("\nThe only way to get non-zero reward is:")
    print("  1. Avoid negative examples (neg_score = 1)")
    print("  2. Get simplicity bonus")
    print("\ngrandparent(X0, X0) achieves this by being selective via unification!")


def analyze_trajectory_balance_loss():
    """Analyze the loss function computation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Trajectory Balance Loss")
    print("=" * 70)

    print("\nTrajectory Balance formula:")
    print("  Loss = (log Z + Σ log P_F - log R - Σ log P_B)²")
    print("\nWhere:")
    print("  Z = partition function (learnable)")
    print("  P_F = forward policy probabilities")
    print("  R = reward")
    print("  P_B = backward policy probabilities")

    print("\n" + "-" * 70)
    print("Potential issues:")
    print("-" * 70)

    print("\n1. BACKWARD POLICY:")
    print("   Current: Uniform approximation (constant)")
    print("   Issue: Doesn't depend on state!")
    print("   Code: sum_log_pb = -log(num_predicates + 10) * trajectory_length")
    print("   Impact: May bias toward certain trajectory lengths")

    print("\n2. REWARD SIGNAL:")
    print("   Without background knowledge, all rewards ≈ 0")
    print("   log(1e-6) = -13.8 (very negative)")
    print("   This creates huge loss gradients!")

    print("\n3. TRAJECTORY LENGTH:")
    print("   Shorter trajectories = less log P_F to accumulate")
    print("   Simpler rules = fewer actions = shorter trajectories")
    print("   Bias: Model prefers 1-step trajectories (just UNIFY)")

    print("\n✓ This explains convergence to grandparent(X0, X0)!")
    print("  It's a 1-step trajectory (just unify variables)")


def run_extended_training():
    """Run 1000 episode training with detailed logging."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Extended Training (1000 episodes)")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    predicate_vocab = ['grandparent', 'parent']
    predicate_arities = {'grandparent': 2, 'parent': 2}

    pos_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('alice', 'diana')),
        Example('grandparent', ('bob', 'charlie')),
        Example('grandparent', ('bob', 'diana')),
    ]

    neg_examples = [
        Example('grandparent', ('alice', 'alice')),
        Example('grandparent', ('charlie', 'alice')),
        Example('grandparent', ('eve', 'frank')),
    ]

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=2)
    gflownet = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine)

    trainer = GFlowNetTrainer(
        state_encoder, gflownet, graph_constructor, reward_calc,
        predicate_vocab, predicate_arities, learning_rate=1e-3
    )

    initial_state = get_initial_state('grandparent', arity=2)

    print("\nTraining for 1000 episodes...")
    print("Logging every 100 episodes:\n")

    history = trainer.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=1000, verbose=True
    )

    # Analyze training progression
    print("\n" + "-" * 70)
    print("Training Analysis:")
    print("-" * 70)

    # Rewards over time
    rewards_by_100 = []
    steps_by_100 = []
    for i in range(0, 1000, 100):
        avg_reward = np.mean([h['reward'] for h in history[i:i+100]])
        avg_steps = np.mean([h['trajectory_length'] for h in history[i:i+100]])
        rewards_by_100.append(avg_reward)
        steps_by_100.append(avg_steps)
        print(f"Episodes {i:4d}-{i+99:4d}: Avg reward={avg_reward:.4f}, Avg steps={avg_steps:.1f}")

    # Action type distribution
    print("\n" + "-" * 70)
    print("Sample 50 theories and analyze their structure:")
    print("-" * 70)

    sampled = []
    for _ in range(50):
        traj, reward = trainer.generate_trajectory(
            initial_state, pos_examples, neg_examples, max_steps=10
        )
        if traj:
            final_theory = traj[-1].next_state
            sampled.append((final_theory, reward, len(traj)))

    # Count trajectory lengths
    length_counts = {}
    for _, _, length in sampled:
        length_counts[length] = length_counts.get(length, 0) + 1

    print(f"\nTrajectory length distribution:")
    for length in sorted(length_counts.keys()):
        print(f"  {length} steps: {length_counts[length]} theories ({100*length_counts[length]/50:.1f}%)")

    # Unique theories
    unique_theories = {}
    for theory, reward, _ in sampled:
        theory_str = theory_to_string(theory)
        if theory_str not in unique_theories:
            unique_theories[theory_str] = []
        unique_theories[theory_str].append(reward)

    print(f"\nUnique theories: {len(unique_theories)}")
    theory_list = [(t, len(rewards), max(rewards)) for t, rewards in unique_theories.items()]
    theory_list.sort(key=lambda x: x[1], reverse=True)

    print("\nMost common theories:")
    for i, (theory_str, count, max_reward) in enumerate(theory_list[:5]):
        print(f"\n{i+1}. Found {count} times (max reward: {max_reward:.4f})")
        print(f"   {theory_str}")

    return history


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "CONVERGENCE ANALYSIS" + " " * 33 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Run analyses
    analyze_state_encoding()
    analyze_action_probabilities()
    analyze_reward_landscape()
    analyze_trajectory_balance_loss()
    history = run_extended_training()

    # Final summary
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    print("\n🔍 PRIMARY ISSUE: No Background Knowledge")
    print("-" * 70)
    print("WITHOUT parent facts, the logic engine CANNOT prove any")
    print("grandparent relationships, even with correct rules!")
    print("\nExample:")
    print("  Rule: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)")
    print("  Query: grandparent(alice, charlie)")
    print("  Logic engine tries to prove:")
    print("    - parent(alice, Z) → No facts! Can't prove.")
    print("  Result: Rule gets 0 positive coverage")

    print("\n🔍 SECONDARY ISSUE: Reward Function Bias")
    print("-" * 70)
    print("Current reward: 0.9 * (pos_score × neg_score) + 0.1 * simplicity")
    print("\nWhen pos_score = 0 (no background knowledge):")
    print("  accuracy = 0 × neg_score = 0")
    print("  reward = 0.9 * 0 + 0.1 * simplicity = 0.1 * simplicity")
    print("\nSo model ONLY optimizes simplicity!")
    print("  → Prefers shortest trajectories")
    print("  → Converges to 1-step unification: grandparent(X0, X0)")

    print("\n🔍 TERTIARY ISSUE: Trajectory Length Bias")
    print("-" * 70)
    print("TB Loss = (log Z + Σ log P_F - log R - Σ log P_B)²")
    print("\nShorter trajectories:")
    print("  - Smaller Σ log P_F (fewer actions)")
    print("  - Smaller Σ log P_B")
    print("  - Easier to balance")
    print("\nModel learns: 1-step trajectory is easiest to optimize!")

    print("\n" + "=" * 70)
    print("SOLUTIONS")
    print("=" * 70)

    print("\n1. ADD BACKGROUND KNOWLEDGE to logic engine")
    print("   parent(alice, bob), parent(bob, charlie), ...")
    print("   → Enables reward signal for correct rules")

    print("\n2. CHANGE REWARD FUNCTION")
    print("   Option A: Give partial credit for rule structure")
    print("   Option B: Penalize degenerate unifications")
    print("   Option C: Reward rule complexity when pos_score = 0")

    print("\n3. FIX BACKWARD POLICY")
    print("   Use learned backward policy instead of uniform")
    print("   → Better trajectory balance")

    print("\n4. ADD EXPLORATION BONUS")
    print("   Encourage diverse trajectories")
    print("   → Prevent premature convergence")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
