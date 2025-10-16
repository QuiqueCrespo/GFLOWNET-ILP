"""
Test combined improvements: Enhanced encoding + Reward shaping.

Compares two configurations:
1. Baseline: Original encoding + paper improvements (no reward shaping)
2. Combined: Enhanced encoding + reward shaping + paper improvements
"""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

import torch
import json
import numpy as np
from src.logic_structures import get_initial_state
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder import GraphConstructor, StateEncoder
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

# Problem setup (same as test_paper_improvements.py)
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

# Training config
num_episodes = 2000
embedding_dim = 32
num_gnn_layers = 2

print("="*80)
print("COMBINED IMPROVEMENTS TEST")
print("="*80)
print(f"\nConfiguration:")
print(f"  Episodes: {num_episodes}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  GNN layers: {num_gnn_layers}")
print(f"  Target: grandparent(X, Y)")
print(f"  Positive examples: {len(positive_examples)}")
print(f"  Negative examples: {len(negative_examples)}")
print(f"  Background facts: {len(background_facts)}")

results = {}

# ============================================================================
# Configuration 1: Baseline (Paper improvements only)
# ============================================================================
print("\n" + "="*80)
print("CONFIGURATION 1: BASELINE")
print("="*80)
print("  - Original graph encoding")
print("  - NO reward shaping penalties")
print("  - Paper improvements: Detailed balance + Replay buffer + Reward weighting")

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.0,  # No penalties
    self_loop_penalty=0.0
)
graph_constructor = GraphConstructor(predicate_vocab)
state_encoder = StateEncoder(
    node_feature_dim=len(predicate_vocab) + 1,
    embedding_dim=embedding_dim,
    num_layers=num_gnn_layers
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=embedding_dim,
    num_predicates=len(predicate_vocab),
    hidden_dim=64
)
exploration = get_combined_strategy("aggressive")

trainer = GFlowNetTrainer(
    state_encoder=state_encoder,
    gflownet=gflownet,
    graph_constructor=graph_constructor,
    reward_calculator=reward_calc,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    learning_rate=1e-3,
    exploration_strategy=exploration,
    use_detailed_balance=True,
    use_replay_buffer=True,
    replay_buffer_capacity=50,
    reward_weighted_loss=True,
    replay_probability=0.3
)

print(f"\nTraining baseline for {num_episodes} episodes...")
initial_state = get_initial_state('grandparent', 2)

baseline_rewards = []
baseline_lengths = []
for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
    baseline_rewards.append(metrics['reward'])
    baseline_lengths.append(metrics['trajectory_length'])

    if episode % 200 == 0:
        print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, length={metrics['trajectory_length']}")

results['baseline'] = {
    'final_avg_reward': float(np.mean(baseline_rewards[-100:])),
    'max_reward': float(np.max(baseline_rewards)),
    'high_reward_count': sum(1 for r in baseline_rewards if r > 0.8),
    'final_avg_length': float(np.mean(baseline_lengths[-100:])),
    'rewards': baseline_rewards,
    'lengths': baseline_lengths
}

print(f"\nBaseline Results:")
print(f"  Final avg reward (last 100): {results['baseline']['final_avg_reward']:.4f}")
print(f"  Max reward: {results['baseline']['max_reward']:.4f}")
print(f"  High-reward episodes (>0.8): {results['baseline']['high_reward_count']}")
print(f"  Final avg trajectory length: {results['baseline']['final_avg_length']:.2f}")

# ============================================================================
# Configuration 2: Combined (Enhanced encoding + Reward shaping)
# ============================================================================
print("\n" + "="*80)
print("CONFIGURATION 2: COMBINED IMPROVEMENTS")
print("="*80)
print("  - Enhanced graph encoding (rich features + attention pooling)")
print("  - Reward shaping penalties (disconnected: 0.2, self-loop: 0.3)")
print("  - Paper improvements: Detailed balance + Replay buffer + Reward weighting")

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,  # Penalties enabled
    self_loop_penalty=0.3
)
graph_constructor = EnhancedGraphConstructor(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=embedding_dim,
    num_layers=num_gnn_layers
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=embedding_dim,
    num_predicates=len(predicate_vocab),
    hidden_dim=64
)
exploration = get_combined_strategy("aggressive")

trainer = GFlowNetTrainer(
    state_encoder=state_encoder,
    gflownet=gflownet,
    graph_constructor=graph_constructor,
    reward_calculator=reward_calc,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    learning_rate=1e-3,
    exploration_strategy=exploration,
    use_detailed_balance=True,
    use_replay_buffer=True,
    replay_buffer_capacity=50,
    reward_weighted_loss=True,
    replay_probability=0.3
)

print(f"\nTraining with combined improvements for {num_episodes} episodes...")

combined_rewards = []
combined_lengths = []
for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
    combined_rewards.append(metrics['reward'])
    combined_lengths.append(metrics['trajectory_length'])

    if episode % 200 == 0:
        print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, length={metrics['trajectory_length']}")

results['combined'] = {
    'final_avg_reward': float(np.mean(combined_rewards[-100:])),
    'max_reward': float(np.max(combined_rewards)),
    'high_reward_count': sum(1 for r in combined_rewards if r > 0.8),
    'final_avg_length': float(np.mean(combined_lengths[-100:])),
    'rewards': combined_rewards,
    'lengths': combined_lengths
}

print(f"\nCombined Results:")
print(f"  Final avg reward (last 100): {results['combined']['final_avg_reward']:.4f}")
print(f"  Max reward: {results['combined']['max_reward']:.4f}")
print(f"  High-reward episodes (>0.8): {results['combined']['high_reward_count']}")
print(f"  Final avg trajectory length: {results['combined']['final_avg_length']:.2f}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

comparison_table = [
    ["Metric", "Baseline", "Combined"],
    ["Final Avg Reward",
     f"{results['baseline']['final_avg_reward']:.4f}",
     f"{results['combined']['final_avg_reward']:.4f}"],
    ["Max Reward",
     f"{results['baseline']['max_reward']:.4f}",
     f"{results['combined']['max_reward']:.4f}"],
    ["High-Reward Episodes",
     str(results['baseline']['high_reward_count']),
     str(results['combined']['high_reward_count'])],
    ["Avg Trajectory Length",
     f"{results['baseline']['final_avg_length']:.2f}",
     f"{results['combined']['final_avg_length']:.2f}"]
]

print()
for row in comparison_table:
    print(f"  {row[0]:<25} {row[1]:<15} {row[2]:<15}")

# Calculate improvements
print("\n" + "-"*80)
print("IMPROVEMENTS OVER BASELINE")
print("-"*80)

if results['baseline']['final_avg_reward'] > 0:
    improvement = (
        (results['combined']['final_avg_reward'] - results['baseline']['final_avg_reward'])
        / results['baseline']['final_avg_reward'] * 100
    )
    print(f"\nAvg reward improvement: {improvement:+.1f}%")
else:
    print(f"\nAvg reward: baseline={results['baseline']['final_avg_reward']:.4f}, combined={results['combined']['final_avg_reward']:.4f}")

print(f"High-reward episode increase: {results['combined']['high_reward_count'] - results['baseline']['high_reward_count']:+d}")

# Save results
output_file = 'analysis/combined_improvements_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to: {output_file}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Expected outcomes:
1. Enhanced encoding should help model learn structural patterns faster
2. Reward shaping should reduce pathological rules in replay buffer
3. Combined improvements should achieve higher quality rules

The test validates whether architectural improvements (enhanced encoding) and
reward engineering (structural penalties) provide additive benefits beyond
the paper-based algorithmic improvements (detailed balance, replay buffer).
""")
