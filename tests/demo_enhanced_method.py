
# %%

"""
Demonstration of enhanced method (reward shaping + enhanced encoding).
Shows actual rules discovered during training.
"""
import sys
sys.path.insert(0, '/Users/jq23948/Documents/GFLowNet-ILP')

import numpy as np
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy
# %%

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
# %%

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

print("="*80)
print("ENHANCED METHOD DEMONSTRATION")
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
print("✓ Enhanced graph encoding (rich features + attention pooling)")
print("✓ Reward shaping penalties (disconnected: 0.2, self-loop: 0.3)")
print("✓ Paper improvements (detailed balance + replay buffer)")
print("✓ Sophisticated backward policy (learned action-specific probabilities)")

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3
)
graph_constructor = EnhancedGraphConstructor(predicate_vocab)
state_encoder = EnhancedStateEncoder(
    predicate_vocab_size=len(predicate_vocab),
    embedding_dim=32,
    num_layers=2
)
gflownet = HierarchicalGFlowNet(
    embedding_dim=32,
    num_predicates=len(predicate_vocab),
    hidden_dim=64,
    use_sophisticated_backward=False,  # Use sophisticated learned backward policy
    predicate_vocab=predicate_vocab  # Required for sophisticated backward policy
)
# exploration = get_combined_strategy("aggressive")
exploration = None

trainer = GFlowNetTrainer(
    state_encoder=state_encoder,
    gflownet=gflownet,
    graph_constructor=graph_constructor,
    reward_calculator=reward_calc,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    learning_rate=1e-6,
    exploration_strategy=exploration,
    use_detailed_balance=True,
    use_replay_buffer=True,
    replay_buffer_capacity=10,
    reward_weighted_loss=True,
    replay_probability=0.3,
    max_body_length=4,
    buffer_reward_threshold=0.7
)

# Training
num_episodes = 10000
initial_state = get_initial_state('grandparent', 2)

print(f"\n" + "="*80)
print(f"TRAINING ({num_episodes} episodes)")
print("="*80)

rewards = []
discovered_rules = {}  # Rule string -> (reward, episode, scores)
recent_rules = []  # Track last 50 rules for analysis

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
    rewards.append(metrics['reward'])

    # Extract final theory from most recent trajectory
    # Note: train_step doesn't return the trajectory, so we'll sample one
    if episode % 10 == 0:  # Sample every 10 episodes to avoid overhead
        from src.training import TrajectoryStep
        # Generate a trajectory to see what rules it's finding
        trajectory, reward = trainer.generate_trajectory(
            initial_state, positive_examples, negative_examples
        )
        theory = trajectory[-1].next_state if trajectory else initial_state
        rule_str = theory_to_string(theory)

        # Get detailed scores
        scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

        # Store best instance of each unique rule
        if rule_str not in discovered_rules or reward > discovered_rules[rule_str][0]:
            discovered_rules[rule_str] = (reward, episode, scores)

        # Keep track of recent rules
        recent_rules.append((rule_str, reward, episode, scores))
        if len(recent_rules) > 50:
            recent_rules.pop(0)

    if episode % 100 == 0 and recent_rules:
        latest_rule, latest_reward, _, _ = recent_rules[-1]
        print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, length={metrics['trajectory_length']}")
        print(f"  Latest sampled rule: {latest_rule}")

# Backward Policy Analysis
print("\n" + "="*80)
print("BACKWARD POLICY ANALYSIS")
print("="*80)

# Test backward policy on some sampled trajectories
print("\nTesting backward policy predictions on recent trajectories...")
backward_policy_stats = {
    'num_trajectories': 0,
    'avg_backward_log_prob': 0.0,
    'action_type_predictions': {'ADD_ATOM': 0, 'UNIFY_VARIABLES': 0, 'TERMINATE': 0},
    'total_steps': 0,
    'num_backward_computed': 0
}

# Sample 10 trajectories to analyze backward policy
for _ in range(10):
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positive_examples, negative_examples, max_steps=5
    )

    if not trajectory:
        continue

    backward_policy_stats['num_trajectories'] += 1

    for step in trajectory:
        # Track action type
        backward_policy_stats['action_type_predictions'][step.action_type] += 1
        backward_policy_stats['total_steps'] += 1

        # Skip backward probability computation for TERMINATE (no meaningful backward transition)
        if step.action_type == 'TERMINATE':
            continue

        # Encode next state
        graph_data = trainer.graph_constructor.theory_to_graph(step.next_state)
        next_state_embedding, node_embeddings = trainer.state_encoder(graph_data)
        next_state_embedding = next_state_embedding.squeeze(0)

        # Get variable embeddings
        from src.logic_structures import get_all_variables
        variables = get_all_variables(step.next_state)
        var_embeddings = node_embeddings[:len(variables)] if len(variables) > 0 else None

        # Compute backward log probability
        log_pb = trainer.gflownet.get_backward_log_probability(
            next_state_embedding,
            step.next_state,
            step.state,
            step.action_type,
            step.action_detail,
            var_embeddings
        )

        backward_policy_stats['avg_backward_log_prob'] += log_pb.item()
        backward_policy_stats['num_backward_computed'] += 1

if backward_policy_stats['num_backward_computed'] > 0:
    backward_policy_stats['avg_backward_log_prob'] /= backward_policy_stats['num_backward_computed']

    print(f"\nBackward Policy Statistics:")
    print(f"  Trajectories analyzed: {backward_policy_stats['num_trajectories']}")
    print(f"  Total steps: {backward_policy_stats['total_steps']}")
    print(f"  Backward probs computed: {backward_policy_stats['num_backward_computed']} (excluding TERMINATE)")
    print(f"  Avg backward log prob: {backward_policy_stats['avg_backward_log_prob']:.4f}")
    print(f"  Action type distribution:")
    print(f"    ADD_ATOM: {backward_policy_stats['action_type_predictions']['ADD_ATOM']} "
          f"({100*backward_policy_stats['action_type_predictions']['ADD_ATOM']/backward_policy_stats['total_steps']:.1f}%)")
    print(f"    UNIFY_VARIABLES: {backward_policy_stats['action_type_predictions']['UNIFY_VARIABLES']} "
          f"({100*backward_policy_stats['action_type_predictions']['UNIFY_VARIABLES']/backward_policy_stats['total_steps']:.1f}%)")
    print(f"    TERMINATE: {backward_policy_stats['action_type_predictions']['TERMINATE']} "
          f"({100*backward_policy_stats['action_type_predictions']['TERMINATE']/backward_policy_stats['total_steps']:.1f}%)")

# Check if using sophisticated backward policy
from src.gflownet_models import SophisticatedBackwardPolicy
import torch.nn.functional as F

if isinstance(trainer.gflownet.backward_policy, SophisticatedBackwardPolicy):
    print("\n✓ Using SophisticatedBackwardPolicy (learned action-specific probabilities)")
    print("  - Backward strategist: predicts action type")
    print("  - Backward atom remover: predicts which predicate was added")
    print("  - Backward variable splitter: predicts which variables were unified")

    # Test forward-backward alignment
    print("\n" + "="*80)
    print("FORWARD-BACKWARD POLICY ALIGNMENT TEST")
    print("="*80)
    print("\nTesting if backward policy correctly predicts forward actions...")

    alignment_stats = {
        'num_tests': 0,
        'strategist_correct': 0,
        'total_forward_log_prob': 0.0,
        'total_backward_log_prob': 0.0
    }

    # Sample a few trajectories and check alignment
    for _ in range(5):
        trajectory, reward = trainer.generate_trajectory(
            initial_state, positive_examples, negative_examples, max_steps=3
        )

        if not trajectory:
            continue

        for step in trajectory:
            # Skip TERMINATE actions (no meaningful backward transition)
            if step.action_type == 'TERMINATE':
                continue

            # Encode states
            graph_data_next = trainer.graph_constructor.theory_to_graph(step.next_state)
            next_state_embedding, node_embeddings = trainer.state_encoder(graph_data_next)
            next_state_embedding = next_state_embedding.squeeze(0)

            # Get backward strategist prediction
            backward_action_logits = trainer.gflownet.backward_policy.strategist(next_state_embedding)
            backward_action_probs = F.softmax(backward_action_logits, dim=-1)

            # Check if backward policy correctly predicts the action type
            predicted_action_idx = 0 if step.action_type == 'ADD_ATOM' else 1
            backward_confidence = backward_action_probs[predicted_action_idx].item()

            alignment_stats['num_tests'] += 1
            if backward_confidence > 0.5:  # Backward policy is confident about this action
                alignment_stats['strategist_correct'] += 1

            # Accumulate log probabilities
            alignment_stats['total_forward_log_prob'] += step.log_pf.item()

            # Get variable embeddings
            from src.logic_structures import get_all_variables
            variables = get_all_variables(step.next_state)
            var_embeddings = node_embeddings[:len(variables)] if len(variables) > 0 else None

            log_pb = trainer.gflownet.get_backward_log_probability(
                next_state_embedding,
                step.next_state,
                step.state,
                step.action_type,
                step.action_detail,
                var_embeddings
            )
            alignment_stats['total_backward_log_prob'] += log_pb.item()

    if alignment_stats['num_tests'] > 0:
        print(f"\nAlignment Test Results:")
        print(f"  Total transitions tested: {alignment_stats['num_tests']}")
        print(f"  Backward strategist accuracy: {alignment_stats['strategist_correct']}/{alignment_stats['num_tests']} "
              f"({100*alignment_stats['strategist_correct']/alignment_stats['num_tests']:.1f}%)")
        print(f"  Avg forward log prob: {alignment_stats['total_forward_log_prob']/alignment_stats['num_tests']:.4f}")
        print(f"  Avg backward log prob: {alignment_stats['total_backward_log_prob']/alignment_stats['num_tests']:.4f}")
        print(f"\nInterpretation:")
        print(f"  - Higher backward accuracy = backward policy learns to predict forward actions")
        print(f"  - Similar forward/backward log probs = good flow balance")
else:
    print("\n✓ Using UniformBackwardPolicy (uniform probabilities)")

# Analysis
print("\n" + "="*80)
print("TRAINING RESULTS")
print("="*80)

final_avg_reward = np.mean(rewards[-100:])
max_reward = np.max(rewards)
high_reward_count = sum(1 for r in rewards if r > 0.8)

print(f"\nFinal avg reward (last 100): {final_avg_reward:.4f}")
print(f"Max reward: {max_reward:.4f}")
print(f"High-reward episodes (>0.8): {high_reward_count}")
print(f"Unique rules discovered: {len(discovered_rules)}")

# Show discovered rules sorted by reward
print("\n" + "="*80)
print("TOP DISCOVERED RULES")
print("="*80)

sorted_rules = sorted(discovered_rules.items(), key=lambda x: x[1][0], reverse=True)

print("\nShowing top 10 rules by reward:\n")

for i, (rule_str, (reward, episode, scores)) in enumerate(sorted_rules[:10], 1):
    print(f"{i}. [{reward:.4f}] {rule_str}")
    print(f"   Episode: {episode}")
    print(f"   Coverage: {scores['pos_covered']}/{scores['pos_total']} pos, "
          f"{scores['neg_covered']}/{scores['neg_total']} neg")
    print(f"   Accuracy: {scores['accuracy']:.4f}")
    print(f"   Disconnected vars: {scores['num_disconnected_vars']} (penalty: -{scores['disconnected_penalty']:.2f})")
    print(f"   Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
    print(f"   Base reward: {scores['reward']:.4f}")
    print()

# Analyze replay buffer
print("="*80)
print("REPLAY BUFFER ANALYSIS")
print("="*80)

if trainer.replay_buffer and len(trainer.replay_buffer.buffer) > 0:
    print(f"\nReplay buffer size: {len(trainer.replay_buffer.buffer)}")

    # Get rules in replay buffer
    replay_rules = []
    for trajectory, reward in trainer.replay_buffer.buffer:
        theory = trajectory[-1].next_state
        rule_str = theory_to_string(theory)
        scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)
        replay_rules.append((rule_str, reward, scores))

    # Sort by reward
    replay_rules.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 rules in replay buffer:\n")

    for i, (rule_str, reward, scores) in enumerate(replay_rules[:10], 1):
        print(f"{i}. [{reward:.4f}] {rule_str}")
        print(f"   Coverage: {scores['pos_covered']}/{scores['pos_total']} pos, "
              f"{scores['neg_covered']}/{scores['neg_total']} neg")
        print(f"   Issues: {scores['num_disconnected_vars']} disconnected, "
              f"{scores['num_self_loops']} self-loops")
        print(f"   Base reward: {scores['reward']:.4f}")
        print()

    # Quality statistics
    num_perfect = sum(1 for _, _, s in replay_rules if s['pos_score'] == 1.0 and s['neg_score'] == 1.0)
    num_disconnected = sum(1 for _, _, s in replay_rules if s['num_disconnected_vars'] > 0)
    num_self_loops = sum(1 for _, _, s in replay_rules if s['num_self_loops'] > 0)

    print("="*80)
    print("REPLAY BUFFER QUALITY STATISTICS")
    print("="*80)
    print(f"\nPerfect rules (100% pos, 0% neg): {num_perfect}/{len(replay_rules)} ({100*num_perfect/len(replay_rules):.1f}%)")
    print(f"Rules with disconnected variables: {num_disconnected}/{len(replay_rules)} ({100*num_disconnected/len(replay_rules):.1f}%)")
    print(f"Rules with self-loops: {num_self_loops}/{len(replay_rules)} ({100*num_self_loops/len(replay_rules):.1f}%)")

