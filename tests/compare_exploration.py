"""
Compare different exploration strategies for GFlowNet training.

Runs experiments with multiple exploration strategies and analyzes which performs best.
"""

import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

import torch
import numpy as np
from typing import List, Dict
import json

from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer
from src.exploration import (
    EntropyBonus, TemperatureSchedule, TrajectoryLengthBonus,
    EpsilonGreedy, CuriosityBonus, CombinedExploration,
    get_combined_strategy
)


def setup_problem():
    """Setup the grandparent problem."""
    # Background knowledge
    background_facts = [
        Example('parent', ('alice', 'bob')),
        Example('parent', ('bob', 'charlie')),
        Example('parent', ('eve', 'frank')),
        Example('parent', ('frank', 'grace')),
    ]

    # Training examples
    positive_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('eve', 'grace')),
    ]

    negative_examples = [
        Example('grandparent', ('alice', 'alice')),
        Example('grandparent', ('bob', 'bob')),
        Example('grandparent', ('alice', 'eve')),
        Example('grandparent', ('bob', 'frank')),
    ]

    # Predicate vocabulary
    predicate_vocab = ['parent']
    predicate_arities = {'parent': 2}

    return background_facts, positive_examples, negative_examples, predicate_vocab, predicate_arities


def create_trainer(exploration_strategy=None):
    """Create a fresh trainer instance."""
    background_facts, pos_ex, neg_ex, vocab, arities = setup_problem()

    # Initialize components
    logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
    reward_calc = RewardCalculator(logic_engine)
    graph_constructor = GraphConstructor(vocab)
    state_encoder = StateEncoder(
        node_feature_dim=len(vocab) + 1,  # predicates + variable type
        embedding_dim=32,
        num_layers=2
    )
    gflownet = HierarchicalGFlowNet(
        embedding_dim=32,
        num_predicates=len(vocab),
        hidden_dim=64
    )

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calc,
        predicate_vocab=vocab,
        predicate_arities=arities,
        learning_rate=1e-3,
        exploration_strategy=exploration_strategy
    )

    return trainer, pos_ex, neg_ex


def run_experiment(strategy_name: str, strategy, num_episodes: int = 1000):
    """Run training with a specific exploration strategy."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {strategy_name}")
    print(f"{'='*80}")

    if strategy:
        print(f"Strategy: {strategy}")
    else:
        print("Strategy: None (baseline)")

    # Create trainer
    trainer, pos_ex, neg_ex = create_trainer(strategy)
    initial_state = get_initial_state('grandparent', 2)

    # Track metrics
    rewards = []
    lengths = []
    final_theories = []

    # Training loop
    for episode in range(num_episodes):
        metrics = trainer.train_step(initial_state, pos_ex, neg_ex)
        rewards.append(metrics['reward'])
        lengths.append(metrics['trajectory_length'])

        if episode % 50 == 0:
            print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, "
                  f"length={metrics['trajectory_length']}, loss={metrics['loss']:.4f}")

    # Sample final theories
    print(f"\nSampling final theories...")
    for i in range(10):
        trajectory, reward = trainer.generate_trajectory(initial_state, pos_ex, neg_ex)
        if trajectory:
            final_theory = trajectory[-1].next_state
            theory_str = theory_to_string(final_theory)
            final_theories.append((theory_str, reward))

    # Analyze results
    results = {
        'strategy_name': strategy_name,
        'avg_reward_last_100': float(np.mean(rewards[-100:])),
        'avg_reward_last_50': float(np.mean(rewards[-50:])),
        'max_reward': float(np.max(rewards)),
        'avg_length_last_100': float(np.mean(lengths[-100:])),
        'rewards': [float(r) for r in rewards],
        'lengths': [int(l) for l in lengths],
        'final_theories': final_theories
    }

    # Print summary
    print(f"\n{'-'*80}")
    print(f"RESULTS:")
    print(f"  Average reward (last 100): {results['avg_reward_last_100']:.4f}")
    print(f"  Average reward (last 50): {results['avg_reward_last_50']:.4f}")
    print(f"  Max reward: {results['max_reward']:.4f}")
    print(f"  Average length (last 100): {results['avg_length_last_100']:.2f}")
    print(f"\nFinal sampled theories (unique):")
    unique_theories = {}
    for theory, reward in final_theories:
        if theory not in unique_theories:
            unique_theories[theory] = reward
    for theory, reward in sorted(unique_theories.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  [{reward:.4f}] {theory}")

    return results


def main():
    """Run all experiments and compare results."""
    print("="*80)
    print("EXPLORATION STRATEGY COMPARISON")
    print("="*80)

    # Define strategies to test
    strategies = [
        ("Baseline (No Exploration)", None),
        ("Entropy Bonus (α=0.01)", EntropyBonus(alpha=0.01, decay=0.9999)),
        ("Temperature Schedule (T=2.0→0.5)", TemperatureSchedule(T_init=2.0, T_final=0.5, decay_steps=150)),
        ("Trajectory Length Bonus (β=0.05)", TrajectoryLengthBonus(beta=0.05, decay=0.9995)),
        ("Epsilon Greedy (ε=0.2)", EpsilonGreedy(epsilon=0.2, decay=0.995, epsilon_min=0.05)),
        ("Curiosity Bonus", CuriosityBonus(bonus_atoms=0.1, bonus_diversity=0.05)),
        ("Combined Balanced", get_combined_strategy("balanced")),
        ("Combined Aggressive", get_combined_strategy("aggressive")),
    ]

    all_results = []

    for strategy_name, strategy in strategies:
        results = run_experiment(strategy_name, strategy, num_episodes=1000)
        all_results.append(results)

    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Strategy':<40} {'Avg(100)':>10} {'Avg(50)':>10} {'Max':>10} {'Len':>10}")
    print("-"*80)

    for result in sorted(all_results, key=lambda x: x['avg_reward_last_100'], reverse=True):
        print(f"{result['strategy_name']:<40} "
              f"{result['avg_reward_last_100']:>10.4f} "
              f"{result['avg_reward_last_50']:>10.4f} "
              f"{result['max_reward']:>10.4f} "
              f"{result['avg_length_last_100']:>10.2f}")

    # Save results
    output_file = '/Users/jq23948/GFLowNet-ILP/analysis/exploration_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Determine winner
    best = max(all_results, key=lambda x: x['avg_reward_last_100'])
    print(f"\n{'='*80}")
    print(f"WINNER: {best['strategy_name']}")
    print(f"  Average reward (last 100 episodes): {best['avg_reward_last_100']:.4f}")
    print(f"  Average reward (last 50 episodes): {best['avg_reward_last_50']:.4f}")
    print(f"  Max reward achieved: {best['max_reward']:.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
