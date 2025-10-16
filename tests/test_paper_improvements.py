"""
Test paper-based improvements to GFlowNet.

Compares:
1. Baseline (standard TB loss)
2. Detailed Balance loss (DAG-GFlowNet paper)
3. Replay Buffer (off-policy learning)
4. Detailed Balance + Replay Buffer
5. All improvements combined (DB + Replay + Reward Weighting + No-Decay Exploration)
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
        Example('parent', ('diana', 'henry')),
        Example('parent', ('henry', 'irene')),
        Example('parent', ('grace', 'jack'))
    ]

    positive_examples = [
        Example('grandparent', ('alice', 'charlie')),
        Example('grandparent', ('eve', 'grace')),
        Example('grandparent', ('diana', 'irene')),
        Example('grandparent', ('frank', 'jack'))

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


def create_trainer(use_detailed_balance=False,
                  use_replay_buffer=False,
                  reward_weighted_loss=False,
                  use_no_decay_exploration=False):
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

    # Exploration strategy
    if use_no_decay_exploration:
        # No-decay exploration (all decay=1.0)
        from src.exploration import EntropyBonus, TrajectoryLengthBonus, TemperatureSchedule, CombinedExploration
        exploration = CombinedExploration([
            EntropyBonus(alpha=0.05, decay=1.0),  # No decay!
            TrajectoryLengthBonus(beta=0.1, decay=1.0),  # No decay!
            TemperatureSchedule(T_init=2.0, T_final=2.0)  # Constant T!
        ])
    else:
        # Standard aggressive exploration (with decay)
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
        use_detailed_balance=use_detailed_balance,
        use_replay_buffer=use_replay_buffer,
        reward_weighted_loss=reward_weighted_loss,
        replay_buffer_capacity=50,
        replay_probability=0.3
    )

    return trainer, pos_ex, neg_ex


def run_experiment(name: str,
                  use_detailed_balance: bool,
                  use_replay_buffer: bool,
                  reward_weighted_loss: bool,
                  use_no_decay_exploration: bool,
                  num_episodes: int = 1000):
    """Run training with specific configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*80}")
    print(f"Detailed Balance: {use_detailed_balance}")
    print(f"Replay Buffer: {use_replay_buffer}")
    print(f"Reward Weighted: {reward_weighted_loss}")
    print(f"No-Decay Exploration: {use_no_decay_exploration}")

    trainer, pos_ex, neg_ex = create_trainer(
        use_detailed_balance,
        use_replay_buffer,
        reward_weighted_loss,
        use_no_decay_exploration
    )
    initial_state = get_initial_state('grandparent', 2)

    rewards = []
    lengths = []
    high_reward_episodes = []
    replay_counts = 0

    for episode in range(num_episodes):
        metrics = trainer.train_step(initial_state, pos_ex, neg_ex)
        rewards.append(metrics['reward'])
        lengths.append(metrics['trajectory_length'])

        if metrics['reward'] > 0.5:
            high_reward_episodes.append(episode)

        if metrics.get('replay_used', False):
            replay_counts += 1

        if episode % 50 == 0:
            print(f"Episode {episode:3d}: reward={metrics['reward']:.4f}, "
                  f"length={metrics['trajectory_length']}, loss={metrics['loss']:.4f}")

    # Analyze results
    avg_reward_last_100 = float(np.mean(rewards[-100:]))
    avg_reward_first_100 = float(np.mean(rewards[:100]))
    max_reward = float(np.max(rewards))
    avg_length_last_100 = float(np.mean(lengths[-100:]))

    # Find convergence point (to 1-step degenerate)
    convergence_episode = None
    for i in range(len(lengths)-50):
        if np.mean(lengths[i:i+50]) < 1.1:
            convergence_episode = i
            break

    # Count high-reward episodes in different phases
    high_reward_first_100 = sum(1 for ep in high_reward_episodes if ep < 100)
    high_reward_100_200 = sum(1 for ep in high_reward_episodes if 100 <= ep < 200)
    high_reward_200_500 = sum(1 for ep in high_reward_episodes if 200 <= ep < 500)

    results = {
        'name': name,
        'use_detailed_balance': use_detailed_balance,
        'use_replay_buffer': use_replay_buffer,
        'reward_weighted_loss': reward_weighted_loss,
        'use_no_decay_exploration': use_no_decay_exploration,
        'avg_reward_first_100': avg_reward_first_100,
        'avg_reward_last_100': avg_reward_last_100,
        'max_reward': max_reward,
        'avg_length_last_100': avg_length_last_100,
        'convergence_episode': convergence_episode,
        'num_high_reward_episodes': len(high_reward_episodes),
        'high_reward_first_100': high_reward_first_100,
        'high_reward_100_200': high_reward_100_200,
        'high_reward_200_500': high_reward_200_500,
        'replay_count': replay_counts,
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
    print(f"    First 100: {high_reward_first_100}")
    print(f"    100-200: {high_reward_100_200}")
    print(f"    200-500: {high_reward_200_500}")
    if use_replay_buffer:
        print(f"  Replay buffer used: {replay_counts} times")
    if convergence_episode:
        print(f"  Converged to 1-step at episode: {convergence_episode}")
    else:
        print(f"  Did NOT converge to 1-step! ✓✓✓")

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
    """Run all paper-based improvement experiments."""
    print("="*80)
    print("PAPER-BASED IMPROVEMENTS COMPARISON")
    print("="*80)

    strategies = [
        # Baseline
        ("Baseline (TB Loss)", False, False, False, False),

        # Individual improvements
        ("Detailed Balance", True, False, False, False),
        ("Replay Buffer", False, True, False, False),
        ("Reward Weighted", False, False, True, False),
        ("No-Decay Exploration", False, False, False, True),

        # Combinations
        ("DB + Replay", True, True, False, False),
        ("Replay + Reward Weight", False, True, True, False),
        ("Replay + No-Decay", False, True, False, True),

        # Full solution
        ("All Combined", True, True, True, False),
    ]

    all_results = []

    for name, db, replay, weighted, no_decay in strategies:
        results = run_experiment(name, db, replay, weighted, no_decay, num_episodes=10000)
        all_results.append(results)

    # Compare results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Strategy':<30} {'Avg(100)':>10} {'Max':>10} {'HighR':>6} "
          f"{'Conv':>6} {'Len':>6} {'H100-200':>8}")
    print("-"*80)

    for r in sorted(all_results, key=lambda x: x['avg_reward_last_100'], reverse=True):
        conv_str = str(r['convergence_episode']) if r['convergence_episode'] else "None"
        print(f"{r['name']:<30} "
              f"{r['avg_reward_last_100']:>10.4f} "
              f"{r['max_reward']:>10.4f} "
              f"{r['num_high_reward_episodes']:>6d} "
              f"{conv_str:>6s} "
              f"{r['avg_length_last_100']:>6.2f} "
              f"{r['high_reward_100_200']:>8d}")

    # Save results
    output_file = '/Users/jq23948/GFLowNet-ILP/analysis/paper_improvements_results.json'
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
    print(f"    Episodes 100-200: {best['high_reward_100_200']}")
    print(f"    Episodes 200-500: {best['high_reward_200_500']}")
    if best['convergence_episode']:
        print(f"  Converged at episode: {best['convergence_episode']}")
    else:
        print(f"  Did NOT converge (maintained exploration!) ✓✓✓")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
