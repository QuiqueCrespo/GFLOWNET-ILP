"""
Simple example demonstrating the hierarchical GFlowNet for FOL rule generation.

This example learns a "grandparent" rule from examples:
  grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
"""

import torch
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def create_grandparent_dataset():
    """
    Create a simple dataset for learning the grandparent relation.

    Rules we want to learn:
      grandparent(X, Y) :- parent(X, Z), parent(Z, Y)

    Or a simpler variant.
    """
    # Positive examples: grandparent relationships
    positive_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('alice', 'diana')),
        Example('grandparent', ('bob', 'charlie')),
        Example('grandparent', ('bob', 'diana')),
    ]

    # Negative examples: not grandparent relationships
    negative_examples = [
        Example('grandparent', ('alice', 'alice')),  # Can't be your own grandparent
        Example('grandparent', ('charlie', 'alice')),  # Wrong direction
        Example('grandparent', ('eve', 'frank')),  # Unrelated people
    ]

    # Background knowledge (not used in this simple version, but could be)
    # These are facts the logic engine would need to know
    background_facts = [
        Example('parent', ('alice', 'eve')),
        Example('parent', ('bob', 'eve')),
        Example('parent', ('eve', 'charlie')),
        Example('parent', ('eve', 'diana')),
    ]

    return positive_examples, negative_examples, background_facts


def main():
    print("=" * 70)
    print("Hierarchical GFlowNet for FOL Rule Generation")
    print("=" * 70)
    print("\nLearning task: Derive 'grandparent' rules from examples\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define predicate vocabulary and arities
    predicate_vocab = ['grandparent', 'parent']
    predicate_arities = {
        'grandparent': 2,
        'parent': 2
    }

    # Create dataset
    positive_examples, negative_examples, background_facts = create_grandparent_dataset()

    print(f"Positive examples ({len(positive_examples)}):")
    for ex in positive_examples:
        print(f"  {ex}")

    print(f"\nNegative examples ({len(negative_examples)}):")
    for ex in negative_examples:
        print(f"  {ex}")

    # Initialize components
    node_feature_dim = len(predicate_vocab) + 1  # +1 for variable indicator
    embedding_dim = 64
    hidden_dim = 128

    print(f"\n{'-' * 70}")
    print("Initializing models...")
    print(f"{'-' * 70}")

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(node_feature_dim, embedding_dim, num_layers=2)
    gflownet = HierarchicalGFlowNet(embedding_dim, len(predicate_vocab), hidden_dim)

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

    # Create initial state
    initial_state = get_initial_state('grandparent', arity=2)
    print(f"\nInitial state:")
    print(f"  {theory_to_string(initial_state)}")

    # Train
    print(f"\n{'-' * 70}")
    print("Training...")
    print(f"{'-' * 70}\n")

    num_episodes = 10000
    history = trainer.train(
        initial_state=initial_state,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        num_episodes=num_episodes,
        verbose=True
    )

    # Sample best theory
    print(f"\n{'-' * 70}")
    print("Sampling best theory from trained model...")
    print(f"{'-' * 70}\n")

    best_theory, best_reward = trainer.sample_best_theory(
        initial_state=initial_state,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        num_samples=20
    )

    print(f"Best theory found (reward: {best_reward:.4f}):")
    print(theory_to_string(best_theory))

    # Evaluate the best theory
    print(f"\n{'-' * 70}")
    print("Detailed evaluation of best theory:")
    print(f"{'-' * 70}\n")

    detailed_scores = reward_calculator.get_detailed_scores(
        best_theory, positive_examples, negative_examples
    )

    print(f"Positive examples covered: {detailed_scores['pos_covered']}/{detailed_scores['pos_total']}")
    print(f"Positive score: {detailed_scores['pos_score']:.4f}")
    print(f"Negative examples covered: {detailed_scores['neg_covered']}/{detailed_scores['neg_total']}")
    print(f"Negative score: {detailed_scores['neg_score']:.4f}")
    print(f"Simplicity: {detailed_scores['simplicity']:.4f} ({detailed_scores['total_atoms']} atoms)")
    print(f"Overall reward: {detailed_scores['reward']:.4f}")

    # Show training progress
    print(f"\n{'-' * 70}")
    print("Training progress summary:")
    print(f"{'-' * 70}\n")

    avg_reward_first = sum(h['reward'] for h in history[:10]) / 10
    avg_reward_last = sum(h['reward'] for h in history[-10:]) / 10

    print(f"Average reward (first 10 episodes): {avg_reward_first:.4f}")
    print(f"Average reward (last 10 episodes): {avg_reward_last:.4f}")
    print(f"Improvement: {avg_reward_last - avg_reward_first:.4f}")

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
