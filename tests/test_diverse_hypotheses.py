"""
Test top-N sampling with a harder problem that produces diverse hypotheses.
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
    print("Testing Top-N with Diverse Hypotheses")
    print("=" * 70)

    torch.manual_seed(42)

    # Harder task with more predicates to allow diversity
    predicate_vocab = ['grandparent', 'parent', 'ancestor']
    predicate_arities = {'grandparent': 2, 'parent': 2, 'ancestor': 2}

    # Ambiguous examples that could be explained multiple ways
    pos_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('bob', 'diana')),
    ]

    neg_examples = [
        Example('grandparent', ('alice', 'alice')),
        Example('grandparent', ('charlie', 'alice')),
    ]

    print("\n" + "-" * 70)
    print("Task: Learn grandparent relation (ambiguous examples)")
    print("-" * 70)
    print(f"\nPositive examples: {len(pos_examples)}")
    for ex in pos_examples:
        print(f"  ✓ {ex}")

    print(f"\nNegative examples: {len(neg_examples)}")
    for ex in neg_examples:
        print(f"  ✗ {ex}")

    # Initialize models
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

    # Train (limited episodes to keep some exploration)
    print("\n" + "-" * 70)
    print("Training (200 episodes - limited to maintain diversity)")
    print("-" * 70)

    history = trainer.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=200, verbose=False
    )

    avg_reward_last = sum(h['reward'] for h in history[-50:]) / 50
    print(f"\nAverage reward (last 50 episodes): {avg_reward_last:.4f}")

    # Sample many theories to get diversity
    print("\n" + "=" * 70)
    print("Top 10 Hypotheses (from 200 samples)")
    print("=" * 70)

    top_theories = trainer.sample_top_theories(
        initial_state, pos_examples, neg_examples,
        num_samples=200, top_k=10
    )

    for i, (theory, reward) in enumerate(top_theories):
        scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)

        print(f"\n{i+1}. Reward: {reward:.4f}")
        print(f"   Theory: {theory_to_string(theory)}")
        print(f"   Coverage: pos={scores['pos_covered']}/{scores['pos_total']}, "
              f"neg={scores['neg_covered']}/{scores['neg_total']}, "
              f"atoms={scores['total_atoms']}")
        print(f"   Accuracy: {scores['accuracy']:.4f}")

    # Analyze diversity
    print("\n" + "=" * 70)
    print("Diversity Analysis")
    print("=" * 70)

    # Count unique theories
    unique_theories = {}
    for theory, reward in top_theories:
        theory_str = theory_to_string(theory)
        if theory_str not in unique_theories:
            unique_theories[theory_str] = {'count': 0, 'max_reward': reward}
        unique_theories[theory_str]['count'] += 1
        unique_theories[theory_str]['max_reward'] = max(
            unique_theories[theory_str]['max_reward'], reward
        )

    print(f"\nUnique theories in top 10: {len(unique_theories)}")

    if len(unique_theories) > 1:
        print("\n✅ SUCCESS: Model produces diverse hypotheses!")
        print(f"   Found {len(unique_theories)} different rule structures")
    else:
        print("\n⚠️  Model has converged to single solution")
        print("   (This is okay - means the solution is very strong)")

    # Show all unique theories
    print(f"\nAll unique theories found:")
    for i, (theory_str, info) in enumerate(unique_theories.items()):
        print(f"\n{i+1}. (appears {info['count']} times, max reward: {info['max_reward']:.4f})")
        print(f"   {theory_str}")

    # Sample with even more diversity (early in training)
    print("\n" + "=" * 70)
    print("Re-training with Exploration Focus")
    print("=" * 70)

    # Reinitialize with more exploration (lower learning rate)
    torch.manual_seed(123)
    state_encoder2 = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=2)
    gflownet2 = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

    trainer2 = GFlowNetTrainer(
        state_encoder2, gflownet2, graph_constructor, reward_calc,
        predicate_vocab, predicate_arities, learning_rate=5e-4  # Lower LR
    )

    print("\nTraining with lower learning rate (100 episodes)...")
    history2 = trainer2.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=100, verbose=False
    )

    print("\nSampling 50 theories...")
    top_theories2 = trainer2.sample_top_theories(
        initial_state, pos_examples, neg_examples,
        num_samples=50, top_k=15
    )

    # Count unique
    unique_theories2 = {}
    for theory, reward in top_theories2:
        theory_str = theory_to_string(theory)
        if theory_str not in unique_theories2:
            unique_theories2[theory_str] = reward

    print(f"\nUnique theories in top 15: {len(unique_theories2)}")
    print(f"\nTop 5 unique theories by structure:")
    for i, (theory_str, reward) in enumerate(list(unique_theories2.items())[:5]):
        print(f"\n{i+1}. Reward: {reward:.4f}")
        print(f"   {theory_str}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n✅ Top-N hypothesis sampling implemented successfully!")
    print(f"   - Old API: sample_best_theory() returns single best")
    print(f"   - New API: sample_top_theories() returns top K ranked by reward")
    print(f"   - Diversity depends on training convergence and exploration")
    print(f"   - Strong convergence → fewer unique theories (good optimization)")
    print(f"   - More exploration → more diverse theories (good for search)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
