"""
Test the improved reward function that penalizes empty rules.
Uses a simpler task where learning is feasible.
"""

import torch
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def main():
    print("=" * 70)
    print("Testing Improved Reward Function")
    print("=" * 70)

    torch.manual_seed(789)

    # Simpler task: learn identity relation
    # target(X, X) - should be true when both arguments are same
    predicate_vocab = ['target']
    predicate_arities = {'target': 2}

    # Positive: same arguments
    pos_examples = [
        Example('target', ('a', 'a')),
        Example('target', ('b', 'b')),
        Example('target', ('c', 'c')),
    ]

    # Negative: different arguments
    neg_examples = [
        Example('target', ('a', 'b')),
        Example('target', ('b', 'c')),
        Example('target', ('x', 'y')),
    ]

    print("\n" + "-" * 70)
    print("Task: Learn identity relation - target(X, X)")
    print("-" * 70)
    print(f"\nPositive examples (same args): {len(pos_examples)}")
    for ex in pos_examples:
        print(f"  ✓ {ex}")

    print(f"\nNegative examples (diff args): {len(neg_examples)}")
    for ex in neg_examples:
        print(f"  ✗ {ex}")

    # Initialize
    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=2)
    gflownet = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

    engine = LogicEngine(max_depth=5)
    reward_calc = RewardCalculator(engine)

    trainer = GFlowNetTrainer(
        state_encoder, gflownet, graph_constructor, reward_calc,
        predicate_vocab, predicate_arities, learning_rate=5e-4
    )

    initial_state = get_initial_state('target', arity=2)

    print(f"\nInitial state: {theory_to_string(initial_state)}")

    # Show reward for different theories
    print("\n" + "-" * 70)
    print("Reward Comparison")
    print("-" * 70)

    # Empty rule
    scores_empty = reward_calc.get_detailed_scores(initial_state, pos_examples, neg_examples)
    print(f"\nEmpty rule: {theory_to_string(initial_state)}")
    print(f"  Covers: pos={scores_empty['pos_covered']}/{scores_empty['pos_total']}, "
          f"neg={scores_empty['neg_covered']}/{scores_empty['neg_total']}")
    print(f"  Accuracy: {scores_empty['accuracy']:.4f}, Reward: {scores_empty['reward']:.4f}")

    # Train
    print("\n" + "-" * 70)
    print("Training (500 episodes)")
    print("-" * 70 + "\n")

    history = trainer.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=500, verbose=True
    )

    # Analysis
    print("\n" + "-" * 70)
    print("Training Analysis")
    print("-" * 70)

    rewards = [h['reward'] for h in history]
    steps = [h['trajectory_length'] for h in history]

    print(f"\nAverage reward:")
    print(f"  First 50 episodes:  {sum(rewards[:50])/50:.4f}")
    print(f"  Last 50 episodes:   {sum(rewards[-50:])/50:.4f}")

    print(f"\nAverage trajectory length:")
    print(f"  First 50 episodes:  {sum(steps[:50])/50:.1f}")
    print(f"  Last 50 episodes:   {sum(steps[-50:])/50:.1f}")

    # Sample best theories
    print("\n" + "-" * 70)
    print("Sampling Best Theories")
    print("-" * 70)

    theories = []
    for i in range(30):
        trajectory, reward = trainer.generate_trajectory(
            initial_state, pos_examples, neg_examples, max_steps=10
        )
        if trajectory:
            final_theory = trajectory[-1].next_state
            theories.append((final_theory, reward))

    theories.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 theories:")
    for i, (theory, reward) in enumerate(theories[:5]):
        scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)
        print(f"\n{i+1}. Reward: {reward:.4f}")
        print(f"   {theory_to_string(theory)}")
        print(f"   Pos: {scores['pos_covered']}/{scores['pos_total']}, "
              f"Neg: {scores['neg_covered']}/{scores['neg_total']}, "
              f"Atoms: {scores['total_atoms']}")

    # Best theory
    best_theory, best_reward = theories[0]

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nBest theory found:")
    print(f"  {theory_to_string(best_theory)}")
    print(f"  Reward: {best_reward:.4f}")

    scores = reward_calc.get_detailed_scores(best_theory, pos_examples, neg_examples)
    print(f"\nDetailed evaluation:")
    print(f"  Positive coverage: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
    print(f"  Negative avoidance: {scores['neg_total']-scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")
    print(f"  Accuracy: {scores['accuracy']:.4f}")
    print(f"  Body atoms: {scores['total_atoms']}")
    print(f"  Uninformative penalty: {scores['uninformative_penalty']:.2f}")

    # Check if improved
    final_avg_reward = sum(rewards[-50:])/50
    if final_avg_reward > 0.8 or (scores['accuracy'] == 1.0 and scores['uninformative_penalty'] == 0.0):
        print("\n✅ SUCCESS: Model learned the correct rule!")
        print(f"   The rule target(X0, X0) correctly captures the identity relation")
    elif final_avg_reward > 0.01:
        print("\n✅ PROGRESS: Model is learning to generate better rules!")
    else:
        print("\n⚠️  Model is exploring but may need more training")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
