"""
Test different flow assignment strategies.

Compares:
1. Baseline (standard TB loss)
2. Reward-weighted loss
3. Reward scaling (alpha=2.0)
4. Reward scaling (alpha=3.0)
5. Combined: weighted + scaling
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
from src.exploration import get_combined_strategy


def setup_problem():
    """Setup the grandparent problem."""
    background_facts = [
        Example('parent', ('alice', 'bob')),
        Example('parent', ('bob', 'charlie')),
        Example('parent', ('eve', 'frank')),
        Example('parent', ('frank', 'grace')),
    ]

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

    predicate_vocab = ['parent']
    predicate_arities = {'parent': 2}

    return background_facts, positive_examples, negative_examples, predicate_vocab, predicate_arities


def create_trainer(reward_weighted=False, reward_scale_alpha=1.0):
    """Create a fresh trainer instance."""
    background_facts, pos_ex, neg_ex, vocab, arities = setup_problem()

    logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
    reward_calc = RewardCalculator(logic_engine)
    graph_constructor = GraphConstructor(vocab)
    state_encoder = StateEncoder(
        node_feature_dim=len(vocab) + 1,
        embedding_dim=32,
        num_layers=2
    )
    gflownet = HierarchicalGFlowNet(
        embedding_dim=32,
        num_predicates=len(vocab),
        hidden_dim=64
    )

    # Use Combined Aggressive exploration
    exploration = get_combined_strategy("aggressive")

    trainer = GFlowNetTrainer(
        state_encoder=state_encoder,
        gflownet=gflownet,
        graph_constructor=graph_constructor,
        reward_calculator=reward_calc,
        predicate_vocab=vocab,
        predicate_arities=arities,
        learning_rate=1e-3,
        exploration_strategy=exploration,
        reward_weighted_loss=reward_weighted,
        reward_scale_alpha=reward_scale_alpha
    )

    return trainer, pos_ex, neg_ex


def run_experiment(name: str, reward_weighted: bool, reward_scale_alpha: float, num_episodes: int = 500):
    """Run training with specific flow assignment strategy."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*80}")
    print(f"Reward weighted: {reward_weighted}")
    print(f"Reward scale alpha: {reward_scale_alpha}")

    trainer, pos_ex, neg_ex = create_trainer(reward_weighted, reward_scale_alpha)
    initial_state = get_initial_state('grandparent', 2)

    rewards = []
    lengths = []
    high_reward_episodes = []

    for episode in range(num_episodes):
        metrics = trainer.train_step(initial_state, pos_ex, neg_ex)
        rewards.append(metrics['reward'])
        lengths.append(metrics['trajectory_length'])

        if metrics['reward'] > 0.5:
            high_reward_episodes.append(episode)

        if episode % 50 == 0:
            print(f"Episode {episode:3d}: reward={metrics['reward']:.4f}, "
                  f"length={metrics['trajectory_length']}, loss={metrics['loss']:.4f}")

    # Analyze results
    avg_reward_last_100 = float(np.mean(rewards[-100:]))
    avg_reward_first_100 = float(np.mean(rewards[:100]))
    max_reward = float(np.max(rewards))
    avg_length_last_100 = float(np.mean(lengths[-100:]))

    # Find convergence point
    convergence_episode = None
    for i in range(len(lengths)-50):
        if np.mean(lengths[i:i+50]) < 1.1:
            convergence_episode = i
            break

    results = {
        'name': name,
        'reward_weighted': reward_weighted,
        'reward_scale_alpha': reward_scale_alpha,
        'avg_reward_first_100': avg_reward_first_100,
        'avg_reward_last_100': avg_reward_last_100,
        'max_reward': max_reward,
        'avg_length_last_100': avg_length_last_100,
        'convergence_episode': convergence_episode,
        'num_high_reward_episodes': len(high_reward_episodes),
        'high_reward_episodes': high_reward_episodes[:20],  # First 20
        'rewards': [float(r) for r in rewards],
        'lengths': [int(l) for l in lengths]
    }

    print(f"\n{'-'*80}")
    print(f"RESULTS:")
    print(f"  Avg reward (first 100): {avg_reward_first_100:.4f}")
    print(f"  Avg reward (last 100):  {avg_reward_last_100:.4f}")
    print(f"  Max reward: {max_reward:.4f}")
    print(f"  Avg length (last 100): {avg_length_last_100:.2f}")
    print(f"  High-reward episodes (>0.5): {len(high_reward_episodes)}")
    if convergence_episode:
        print(f"  Converged to 1-step at episode: {convergence_episode}")
    else:
        print(f"  Did NOT converge to 1-step!")

    # Sample final theories
    print(f"\nSampling final theories...")
    final_theories = []
    for i in range(10):
        trajectory, reward = trainer.generate_trajectory(initial_state, pos_ex, neg_ex)
        if trajectory:
            final_theory = trajectory[-1].next_state
            theory_str = theory_to_string(final_theory)
            final_theories.append((theory_str, reward))

    unique_theories = {}
    for theory, reward in final_theories:
        if theory not in unique_theories:
            unique_theories[theory] = reward

    print(f"Final sampled theories (unique):")
    for theory, reward in sorted(unique_theories.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  [{reward:.4f}] {theory}")

    results['final_theories'] = list(unique_theories.items())

    return results


def main():
    """Run all flow assignment experiments."""
    print("="*80)
    print("FLOW ASSIGNMENT STRATEGY COMPARISON")
    print("="*80)

    strategies = [
        ("Baseline (Standard TB)", False, 1.0),
        ("Reward Weighted", True, 1.0),
        ("Reward Scaling (α=2.0)", False, 2.0),
        ("Reward Scaling (α=3.0)", False, 3.0),
        ("Weighted + Scaling (α=2.0)", True, 2.0),
    ]

    all_results = []

    for name, weighted, alpha in strategies:
        results = run_experiment(name, weighted, alpha, num_episodes=500)
        all_results.append(results)

    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Strategy':<35} {'Avg(100)':>10} {'Max':>10} {'HighR':>6} {'Conv':>6} {'Len':>6}")
    print("-"*80)

    for r in sorted(all_results, key=lambda x: x['avg_reward_last_100'], reverse=True):
        conv_str = str(r['convergence_episode']) if r['convergence_episode'] else "None"
        print(f"{r['name']:<35} "
              f"{r['avg_reward_last_100']:>10.4f} "
              f"{r['max_reward']:>10.4f} "
              f"{r['num_high_reward_episodes']:>6d} "
              f"{conv_str:>6s} "
              f"{r['avg_length_last_100']:>6.2f}")

    # Save results
    output_file = '/Users/jq23948/GFLowNet-ILP/analysis/flow_assignment_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Determine winner
    best = max(all_results, key=lambda x: x['avg_reward_last_100'])
    print(f"\n{'='*80}")
    print(f"WINNER: {best['name']}")
    print(f"  Average reward (last 100): {best['avg_reward_last_100']:.4f}")
    print(f"  Max reward: {best['max_reward']:.4f}")
    print(f"  High-reward episodes: {best['num_high_reward_episodes']}")
    if best['convergence_episode']:
        print(f"  Converged at episode: {best['convergence_episode']}")
    else:
        print(f"  Did NOT converge (maintained exploration!)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
