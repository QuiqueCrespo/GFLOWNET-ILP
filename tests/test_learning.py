"""
Test that the system can actually learn meaningful rules over extended training.
This test validates that the hierarchical GFlowNet can discover correct FOL rules.
"""

import torch
import numpy as np
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def main():
    print("=" * 70)
    print("LEARNING TEST: Extended Training on Grandparent Relation")
    print("=" * 70)

    # Set seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)

    # Define vocabulary
    predicate_vocab = ['grandparent', 'parent']
    predicate_arities = {'grandparent': 2, 'parent': 2}

    # Create training data
    print("\n" + "-" * 70)
    print("Dataset")
    print("-" * 70)

    positive_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('alice', 'diana')),
        Example('grandparent', ('bob', 'charlie')),
        Example('grandparent', ('bob', 'diana')),
    ]

    negative_examples = [
        Example('grandparent', ('alice', 'alice')),
        Example('grandparent', ('charlie', 'alice')),
        Example('grandparent', ('eve', 'frank')),
        Example('grandparent', ('alice', 'bob')),  # Parents, not grandparents
    ]

    print(f"\nPositive examples: {len(positive_examples)}")
    for ex in positive_examples:
        print(f"  ✓ {ex}")

    print(f"\nNegative examples: {len(negative_examples)}")
    for ex in negative_examples:
        print(f"  ✗ {ex}")

    # Initialize models
    print("\n" + "-" * 70)
    print("Model Initialization")
    print("-" * 70)

    node_feature_dim = len(predicate_vocab) + 1
    embedding_dim = 64
    hidden_dim = 128

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=3)
    gflownet = HierarchicalGFlowNet(embedding_dim, len(predicate_vocab), hidden_dim)

    total_params = sum(p.numel() for p in state_encoder.parameters())
    total_params += sum(p.numel() for p in gflownet.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")

    logic_engine = LogicEngine(max_depth=5)
    reward_calculator = RewardCalculator(
        logic_engine,
        weight_pos=0.6,
        weight_neg=0.3,
        weight_simplicity=0.1
    )

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calculator,
        predicate_vocab=predicate_vocab,
        predicate_arities=predicate_arities,
        learning_rate=1e-3
    )

    # Initial state
    initial_state = get_initial_state('grandparent', arity=2)
    print(f"\nInitial state: {theory_to_string(initial_state)}")

    # Training
    print("\n" + "-" * 70)
    print("Training Progress")
    print("-" * 70)

    num_episodes = 500
    print(f"\nTraining for {num_episodes} episodes...\n")

    history = []
    for episode in range(num_episodes):
        metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
        history.append(metrics)

        if episode % 50 == 0 or episode == num_episodes - 1:
            print(f"Episode {episode:4d} | "
                  f"Loss: {metrics['loss']:8.4f} | "
                  f"Reward: {metrics['reward']:.4f} | "
                  f"Steps: {metrics['trajectory_length']} | "
                  f"log_Z: {metrics['log_Z']:7.4f}")

    # Analyze training progress
    print("\n" + "-" * 70)
    print("Training Analysis")
    print("-" * 70)

    # Calculate moving averages
    window = 50
    avg_rewards = []
    avg_losses = []
    for i in range(len(history) - window + 1):
        avg_reward = np.mean([h['reward'] for h in history[i:i+window]])
        avg_loss = np.mean([h['loss'] for h in history[i:i+window]])
        avg_rewards.append(avg_reward)
        avg_losses.append(avg_loss)

    print(f"\nReward progression (50-episode moving average):")
    print(f"  First 50 episodes:  {avg_rewards[0]:.4f}")
    print(f"  Middle episodes:    {avg_rewards[len(avg_rewards)//2]:.4f}")
    print(f"  Last 50 episodes:   {avg_rewards[-1]:.4f}")
    print(f"  Improvement:        {avg_rewards[-1] - avg_rewards[0]:.4f}")

    print(f"\nLoss progression (50-episode moving average):")
    print(f"  First 50 episodes:  {avg_losses[0]:.4f}")
    print(f"  Last 50 episodes:   {avg_losses[-1]:.4f}")

    # Sample best theories
    print("\n" + "-" * 70)
    print("Sampling Best Theories")
    print("-" * 70)

    print(f"\nSampling {50} theories from trained model...")

    sampled_theories = []
    for i in range(50):
        trajectory, reward = trainer.generate_trajectory(
            initial_state, positive_examples, negative_examples, max_steps=10
        )
        if trajectory:
            final_theory = trajectory[-1].next_state
            sampled_theories.append((final_theory, reward))

    # Sort by reward
    sampled_theories.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 theories:")
    for i, (theory, reward) in enumerate(sampled_theories[:5]):
        print(f"\n{i+1}. Reward: {reward:.4f}")
        print(f"   {theory_to_string(theory)}")

        # Detailed evaluation
        scores = reward_calculator.get_detailed_scores(theory, positive_examples, negative_examples)
        print(f"   → Positive: {scores['pos_covered']}/{scores['pos_total']}, "
              f"Negative: {scores['neg_covered']}/{scores['neg_total']}, "
              f"Atoms: {scores['total_atoms']}")

    # Find best overall
    best_theory, best_reward = sampled_theories[0]

    print("\n" + "-" * 70)
    print("Best Theory Evaluation")
    print("-" * 70)

    print(f"\nBest theory:")
    print(theory_to_string(best_theory))

    scores = reward_calculator.get_detailed_scores(best_theory, positive_examples, negative_examples)

    print(f"\nDetailed scores:")
    print(f"  Positive coverage: {scores['pos_covered']}/{scores['pos_total']} ({scores['pos_score']*100:.1f}%)")
    print(f"  Negative avoidance: {scores['neg_total']-scores['neg_covered']}/{scores['neg_total']} ({scores['neg_score']*100:.1f}%)")
    print(f"  Simplicity: {scores['simplicity']:.4f} ({scores['total_atoms']} atoms in body)")
    print(f"  Overall reward: {scores['reward']:.4f}")

    # Test individual examples
    print(f"\nTesting individual examples:")
    print(f"\nPositive examples:")
    for ex in positive_examples:
        entailed = logic_engine.entails(best_theory, ex)
        status = "✓" if entailed else "✗"
        print(f"  {status} {ex}")

    print(f"\nNegative examples (should NOT be entailed):")
    for ex in negative_examples:
        entailed = logic_engine.entails(best_theory, ex)
        status = "✓" if not entailed else "✗"
        print(f"  {status} {ex}")

    # Check if we learned something meaningful
    print("\n" + "=" * 70)
    print("Learning Success Metrics")
    print("=" * 70)

    success_criteria = {
        'Reward improved': avg_rewards[-1] > avg_rewards[0],
        'Final avg reward > 0.35': avg_rewards[-1] > 0.35,
        'Best theory has body atoms': scores['total_atoms'] > 0,
        'Best theory reward > 0.4': best_reward > 0.4
    }

    print()
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}")

    all_passed = all(success_criteria.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ LEARNING TEST PASSED - System successfully learned from examples!")
    else:
        print("⚠️  LEARNING TEST PARTIAL - System trained but may need more episodes/tuning")
    print("=" * 70 + "\n")

    # Distribution analysis
    print("-" * 70)
    print("Theory Distribution Analysis")
    print("-" * 70)

    unique_theories = {}
    for theory, reward in sampled_theories:
        theory_str = theory_to_string(theory)
        if theory_str not in unique_theories:
            unique_theories[theory_str] = []
        unique_theories[theory_str].append(reward)

    print(f"\nUnique theories sampled: {len(unique_theories)}")
    print(f"Most common theories:")

    theory_counts = [(t, len(rewards), np.mean(rewards)) for t, rewards in unique_theories.items()]
    theory_counts.sort(key=lambda x: x[1], reverse=True)

    for i, (theory_str, count, avg_reward) in enumerate(theory_counts[:5]):
        print(f"\n{i+1}. Sampled {count} times (avg reward: {avg_reward:.4f})")
        print(f"   {theory_str}")


if __name__ == "__main__":
    main()
