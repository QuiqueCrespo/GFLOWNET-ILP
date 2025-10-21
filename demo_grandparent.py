import sys

sys.path.insert(0, '/Users/jq23948/Documents/GFLOWNET-ILP')

import numpy as np
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy
from src.visualization import TrainingVisualizer



# Problem setup
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
    Example('grandparent', ('eve', 'frank')),
]

predicate_vocab = ['parent']
predicate_arities = {'parent': 2}



# Save configuration
config = {
    'problem': 'grandparent',
    'predicate_vocab': predicate_vocab,
    'predicate_arities': predicate_arities,

    'logic_engine_max_depth': 10,
    'num_episodes': 1000,
    'embedding_dim': 32,
    'hidden_dim': 64,
    'num_layers_encoder': 2,
    'learning_rate': 1e-4,
    'max_body_length': 4,

    'use_sophisticated_backward': True,

    'use_f1': True,
    'weight_precision': 0.5,
    'weight_recall': 0.5,
    'weight_simplicity': 0.01,
    'disconnected_var_penalty': 0.2,
    'self_loop_penalty': 0.3,
    'free_var_penalty': 1.0,

    'use_detailed_balance': True,

    'use_replay_buffer': True,
    'replay_probability': 0.5,
    'replay_buffer_capacity': 50,
    'buffer_reward_threshold': 0.5,

    'reward_weighted_loss': False,

    'num_background_facts': len(background_facts),
    'num_positive_examples': len(positive_examples),
    'num_negative_examples': len(negative_examples),
}


print("="*80)
print("METHOD DEMONSTRATION")
print("="*80)
print("\nGoal: Learn grandparent(X, Y) rule from examples")
print(f"\nBackground Knowledge ({len(background_facts)} facts):")
for fact in background_facts:
    print(f"  {fact.predicate_name}({', '.join(fact.args)})")

print(f"\nPositive Examples ({len(positive_examples)}):")
for ex in positive_examples:
    print(f"  {ex.predicate_name}({', '.join(ex.args)})")

print(f"\nNegative Examples ({len(negative_examples)}):")
for ex in negative_examples:
    print(f"  {ex.predicate_name}({', '.join(ex.args)})")

# Setup enhanced method
print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)
print("✓ Prolog-style logic engine")
print("✓ Enhanced graph encoding (rich features + attention pooling)")
print("✓ Confusion matrix reward (Precision: 0.5, Recall: 0.5, F1-score mode)")
print("✓ Structural penalties (disconnected: 0.2, self-loop: 0.3, free-var: 1.0)")
print("✓ GFlowNet improvements (detailed balance + replay buffer)")
print("✓ Sophisticated backward policy (learned action-specific probabilities)")


logic_engine = LogicEngine(max_depth=config['logic_engine_max_depth'], background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    weight_precision=config['weight_precision'],      # Penalize false positives (covering negatives)
    weight_recall=config["weight_recall"],          # Penalize false negatives (missing positives)
    weight_simplicity=config['weight_simplicity'],      # Small penalty for longer rules
    disconnected_var_penalty=config['disconnected_var_penalty'],
    self_loop_penalty= config['self_loop_penalty'],        # Moderate penalty for self-loops
    free_var_penalty=config['free_var_penalty'],
    use_f1=config['use_f1']                 # Use F1-score for balanced precision-recall
)
graph_constructor = EnhancedGraphConstructor(config['predicate_vocab'])
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(config['predicate_vocab']),
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers_encoder']
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=config['embedding_dim'],
    num_predicates=len(config['predicate_vocab']),
    hidden_dim=config['hidden_dim'],
    use_sophisticated_backward=config['use_sophisticated_backward'],
    predicate_vocab=config['predicate_vocab']
)


# exploration = get_combined_strategy("aggressive")


trainer = GFlowNetTrainer(
    state_encoder=state_encoder,
    gflownet=gflownet,
    graph_constructor=graph_constructor,
    reward_calculator=reward_calc,
    predicate_vocab=config['predicate_vocab'],
    predicate_arities=config['predicate_arities'],
    learning_rate=config['learning_rate'],
    exploration_strategy=None,  # No exploration strategy for demo
    use_detailed_balance=config['use_detailed_balance'],
    use_replay_buffer=config['use_replay_buffer'],
    replay_buffer_capacity=config['replay_buffer_capacity'],
    reward_weighted_loss=config['reward_weighted_loss'],
    replay_probability=config['replay_probability'],
    max_body_length=config['max_body_length'],
    buffer_reward_threshold=config['buffer_reward_threshold']
)

# Initialize visualizer
visualizer = TrainingVisualizer(
    experiment_name=config['problem'],
    output_dir="results"
)


visualizer.save_config(config)

# Training
num_episodes = config['num_episodes']
initial_state = get_initial_state('grandparent', 2)

print(f"\n" + "="*80)
print(f"TRAINING ({num_episodes} episodes)")
print("="*80)

rewards = []
discovered_rules = {}  # Rule string -> (reward, episode, scores)
recent_rules = []  # Track last 50 rules for analysis

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
    if metrics:
        rewards.append(metrics['reward'])

        # Record metrics with visualizer
        visualizer.record_episode(episode, metrics)

        # Sample trajectories periodically to see what rules are being found
        if episode % 10 == 0:
            trajectory, reward = trainer.generate_trajectory(
                initial_state, positive_examples, negative_examples
            )
            theory = trajectory[-1].next_state if trajectory else initial_state
            rule_str = theory_to_string(theory)

            scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

            # Record with visualizer
            visualizer.record_rule(rule_str, reward, episode, scores)

            # Add detailed metrics to visualizer
            visualizer.record_episode(episode, {
                **metrics,
                'precision': scores['precision'],
                'recall': scores['recall'],
                'f1_score': scores['f1_score'],
                'accuracy': scores['accuracy']
            })

            discovered_rules[rule_str] = (reward, episode, scores)

            recent_rules.append((rule_str, reward, episode, scores))
            if len(recent_rules) > 50:
                recent_rules.pop(0)

        if episode % 100 == 0 and recent_rules:
            latest_rule, latest_reward, _, _ = recent_rules[-1]
            print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, length={metrics['trajectory_length']}")
            print(f"  Latest sampled rule: {latest_rule}")

# Analysis
print("\n" + "="*80)
print("TRAINING RESULTS")
print("="*80)

if rewards:
    final_avg_reward = np.mean(rewards[-100:]) if len(rewards) > 100 else np.mean(rewards)
    max_reward = np.max(rewards)
    high_reward_count = sum(1 for r in rewards if r > 0.8)

    print(f"\nFinal avg reward (last 100): {final_avg_reward:.4f}")
    print(f"Max reward: {max_reward:.4f}")
    print(f"High-reward episodes (>0.8): {high_reward_count}")
else:
    print("No training data was generated.")

print(f"Unique rules discovered: {len(discovered_rules)}")


# Show discovered rules sorted by reward
print("\n" + "="*80)
print("TOP DISCOVERED RULES")
print("="*80)

sorted_rules = sorted(discovered_rules.items(), key=lambda x: x[1][0], reverse=True)

print("\nShowing top 10 rules by reward:\n")

for i, (rule_str, (reward, episode, scores)) in enumerate(sorted_rules[:10], 1):
    pos_total = scores['TP'] + scores['FN']
    neg_total = scores['FP'] + scores['TN']
    print(f"{i}. [Reward: {scores['reward']:.4f}] {rule_str}")
    print(f"   Discovered at Episode: {episode}")
    print(f"   Confusion Matrix: TP={scores['TP']}, FN={scores['FN']}, FP={scores['FP']}, TN={scores['TN']}")
    print(f"   Coverage: {scores['TP']}/{pos_total} positives, "
          f"{scores['FP']}/{neg_total} negatives")
    print(f"   Metrics: Precision={scores['precision']:.4f}, Recall={scores['recall']:.4f}, F1={scores['f1_score']:.4f}")
    print(f"   Penalties: Disconnected={scores['num_disconnected_vars']} (-{scores['disconnected_penalty']:.2f}), "
          f"Self-loops={scores['num_self_loops']} (-{scores['self_loop_penalty']:.2f}), "
          f"Free-vars={scores['num_free_vars']} (-{scores['free_var_penalty']:.2f})")
    print()


# Analyze replay buffer
print("="*80)
print("REPLAY BUFFER ANALYSIS")
print("="*80)

if trainer.replay_buffer and len(trainer.replay_buffer.buffer) > 0:
    print(f"\nReplay buffer size: {len(trainer.replay_buffer.buffer)}")

    replay_rules = []
    for trajectory, reward in trainer.replay_buffer.buffer:
        theory = trajectory[-1].next_state
        rule_str = theory_to_string(theory)
        scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)
        replay_rules.append((rule_str, reward, scores))

    replay_rules.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 rules in replay buffer:\n")

    for i, (rule_str, reward, scores) in enumerate(replay_rules[:10], 1):
        pos_total = scores['TP'] + scores['FN']
        neg_total = scores['FP'] + scores['TN']
        print(f"{i}. [Reward: {reward:.4f}] {rule_str}")
        print(f"   Coverage: {scores['TP']}/{pos_total} positives, "
              f"{scores['FP']}/{neg_total} negatives")
        print(f"   Issues: {scores['num_disconnected_vars']} disconnected, "
              f"{scores['num_self_loops']} self-loops, "
              f"{scores['num_free_vars']} free-vars")
        print()

    # Quality statistics
    num_perfect = sum(1 for _, _, s in replay_rules if s['recall'] == 1.0 and s['FP'] == 0)
    num_disconnected = sum(1 for _, _, s in replay_rules if s['num_disconnected_vars'] > 0)
    num_self_loops = sum(1 for _, _, s in replay_rules if s['num_self_loops'] > 0)
    buffer_size = len(replay_rules)

    print("="*80)
    print("REPLAY BUFFER QUALITY STATISTICS")
    print("="*80)
    print(f"\nPerfect rules (100% recall, 0 false positives): {num_perfect}/{buffer_size} ({100*num_perfect/buffer_size:.1f}%)")
    print(f"Rules with disconnected variables: {num_disconnected}/{buffer_size} ({100*num_disconnected/buffer_size:.1f}%)")
    print(f"Rules with self-loops: {num_self_loops}/{buffer_size} ({100*num_self_loops/buffer_size:.1f}%)")
else:
    print("Replay buffer is empty.")

# Generate all visualizations
visualizer.finalize()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Results saved to: {visualizer.run_dir}")
print("\nGenerated files:")
print("  - training_curves.png       : Reward and loss over time")
print("  - metrics_over_time.png     : Precision, recall, F1-score")
print("  - confusion_matrices.png    : Top rules' confusion matrices")
print("  - trajectory_lengths.png    : Trajectory length distribution")
print("  - best_rules.txt            : Top 20 discovered rules")
print("  - summary_dashboard.png     : Comprehensive overview")
print("  - config.json               : Training configuration")
print("="*80)