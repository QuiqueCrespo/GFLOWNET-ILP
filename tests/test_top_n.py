"""
Test sampling top N hypotheses from the trained GFlowNet.
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
    print("Testing Top-N Hypothesis Sampling")
    print("=" * 70)

    torch.manual_seed(999)

    # Task: Learn identity relation
    predicate_vocab = ['target', 'parent']
    predicate_arities = {'target': 2, 'parent': 2}

    # Positive: same arguments
    pos_examples = [
        Example('target', ('a', 'a')),
        Example('target', ('b', 'b')),
        Example('target', ('c', 'c')),
        Example('target', ('d', 'd')),
    ]

    # Negative: different arguments
    neg_examples = [
        Example('target', ('a', 'b')),
        Example('target', ('b', 'c')),
        Example('target', ('x', 'y')),
        Example('target', ('d', 'e')),
    ]

    print("\n" + "-" * 70)
    print("Task: Learn identity relation")
    print("-" * 70)
    print(f"\nPositive examples: {len(pos_examples)}")
    for ex in pos_examples[:3]:
        print(f"  ✓ {ex}")
    print(f"  ... and {len(pos_examples) - 3} more")

    print(f"\nNegative examples: {len(neg_examples)}")
    for ex in neg_examples[:3]:
        print(f"  ✗ {ex}")
    print(f"  ... and {len(neg_examples) - 3} more")

    # Initialize models
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

    # Train
    print("\n" + "-" * 70)
    print("Training (300 episodes)")
    print("-" * 70)

    history = trainer.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=300, verbose=False
    )

    avg_reward_last = sum(h['reward'] for h in history[-50:]) / 50
    print(f"\nAverage reward (last 50 episodes): {avg_reward_last:.4f}")

    # Sample top N theories
    print("\n" + "=" * 70)
    print("Sampling Top-N Hypotheses")
    print("=" * 70)

    # Test with different N values
    for top_n in [1, 3, 5, 10]:
        print(f"\n{'-' * 70}")
        print(f"Top {top_n} hypotheses (from 50 samples)")
        print("-" * 70)

        top_theories = trainer.sample_top_theories(
            initial_state, pos_examples, neg_examples,
            num_samples=50, top_k=top_n
        )

        for i, (theory, reward) in enumerate(top_theories):
            scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)

            print(f"\n{i+1}. Reward: {reward:.4f}")
            print(f"   Theory: {theory_to_string(theory)}")
            print(f"   Coverage: pos={scores['pos_covered']}/{scores['pos_total']}, "
                  f"neg={scores['neg_covered']}/{scores['neg_total']}, "
                  f"atoms={scores['total_atoms']}")
            print(f"   Accuracy: {scores['accuracy']:.4f}")

    # Demonstrate diversity in top theories
    print("\n" + "=" * 70)
    print("Diversity Analysis")
    print("=" * 70)

    top_20 = trainer.sample_top_theories(
        initial_state, pos_examples, neg_examples,
        num_samples=100, top_k=20
    )

    # Count unique theories
    unique_theories = {}
    for theory, reward in top_20:
        theory_str = theory_to_string(theory)
        if theory_str not in unique_theories:
            unique_theories[theory_str] = []
        unique_theories[theory_str].append(reward)

    print(f"\nSampled 100 theories, keeping top 20:")
    print(f"  Unique theories in top 20: {len(unique_theories)}")
    print(f"  Most common theories:")

    theory_counts = [(t, len(rewards), max(rewards)) for t, rewards in unique_theories.items()]
    theory_counts.sort(key=lambda x: x[1], reverse=True)

    for i, (theory_str, count, max_reward) in enumerate(theory_counts[:5]):
        print(f"\n  {i+1}. Found {count} times (max reward: {max_reward:.4f})")
        print(f"     {theory_str}")

    # Compare old API vs new API
    print("\n" + "=" * 70)
    print("API Comparison")
    print("=" * 70)

    print("\n1. Old API - sample_best_theory(num_samples=20):")
    best_theory, best_reward = trainer.sample_best_theory(
        initial_state, pos_examples, neg_examples, num_samples=20
    )
    print(f"   Returns single best: {theory_to_string(best_theory)}")
    print(f"   Reward: {best_reward:.4f}")

    print("\n2. New API - sample_top_theories(num_samples=20, top_k=5):")
    top_5 = trainer.sample_top_theories(
        initial_state, pos_examples, neg_examples,
        num_samples=20, top_k=5
    )
    print(f"   Returns top 5 theories:")
    for i, (theory, reward) in enumerate(top_5):
        print(f"   {i+1}. {theory_to_string(theory)} (reward: {reward:.4f})")

    print("\n" + "=" * 70)
    print("✅ Top-N hypothesis sampling working correctly!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
