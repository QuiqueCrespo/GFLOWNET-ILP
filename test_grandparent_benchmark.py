"""Test benchmark pipeline with grandparent example."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

print("="*80)
print("GRANDPARENT BENCHMARK TEST")
print("="*80)

# Background knowledge
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
    Example('parent', ('charlie', 'diana')),
    Example('parent', ('eve', 'frank'))
]

# Positive examples
positive_examples = [
    Example('grandparent', ('alice', 'charlie')),
    Example('grandparent', ('bob', 'diana'))
]

# Negative examples
negative_examples = [
    Example('grandparent', ('alice', 'alice')),
    Example('grandparent', ('bob', 'bob')),
    Example('grandparent', ('alice', 'bob')),  # parent, not grandparent
    Example('grandparent', ('eve', 'charlie'))  # no connection
]

print(f"\nBackground knowledge:")
for fact in background_facts:
    print(f"  {fact}")

print(f"\nPositive examples (should be proven):")
for ex in positive_examples:
    print(f"  {ex}")

print(f"\nNegative examples (should NOT be proven):")
for ex in negative_examples:
    print(f"  {ex}")

# Setup
target_predicate = 'grandparent'
target_arity = 2

predicate_vocab = ['parent']
predicate_arities = {'parent': 2}

logic_engine = LogicEngine(max_depth=5, background_facts=background_facts)
reward_calc = RewardCalculator(
    logic_engine,
    disconnected_var_penalty=0.2,
    self_loop_penalty=0.3,
    free_var_penalty=1.0
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
    replay_buffer_capacity=20,
    replay_probability=0.3,
    reward_weighted_loss=True
)

initial_state = get_initial_state(target_predicate, target_arity)

# Training
print("\n" + "="*80)
print("TRAINING")
print("="*80)

num_episodes = 1000
print(f"\nRunning {num_episodes} episodes...")

best_reward = -float('inf')
best_rule = None
rewards = []

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)
    rewards.append(metrics['reward'])

    if episode % 50 == 0:
        trajectory, reward = trainer.generate_trajectory(
            initial_state,
            positive_examples,
            negative_examples
        )

        if trajectory and reward > best_reward:
            best_reward = reward
            best_rule = trajectory[-1].next_state

        if episode % 100 == 0:
            avg = sum(rewards[-50:])/min(50, len(rewards))
            print(f"  Episode {episode}: avg_reward={avg:.4f}, best_reward={best_reward:.4f}")

# Results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

if best_rule:
    rule = best_rule[0]
    scores = reward_calc.get_detailed_scores(best_rule, positive_examples, negative_examples)

    print(f"\nBest rule found:")
    print(f"  Rule: {theory_to_string(best_rule)}")
    print(f"\nReward breakdown:")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Accuracy: {scores['accuracy']:.4f}")
    print(f"  Simplicity: {scores['simplicity']:.4f}")

    print(f"\nCoverage:")
    print(f"  Positive: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
    print(f"  Negative: {scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")

    print(f"\nStructural checks:")
    print(f"  Disconnected vars: {scores['num_disconnected_vars']} (penalty: -{scores['disconnected_penalty']:.2f})")
    print(f"  Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
    print(f"  Free vars: {scores.get('num_free_vars', 0)} (penalty: -{scores.get('free_var_penalty', 0):.2f})")

    # Test individual examples
    print(f"\nDetailed proof test:")
    print(f"  Positive examples:")
    for ex in positive_examples:
        proven = logic_engine.entails(best_rule, ex)
        status = "✓" if proven else "✗"
        print(f"    {status} {ex}: {proven}")

    print(f"  Negative examples:")
    for ex in negative_examples:
        proven = logic_engine.entails(best_rule, ex)
        status = "✗" if proven else "✓"
        print(f"    {status} {ex}: {proven}")

    # Check head variables in body
    head_vars = set(arg for arg in rule.head.args
                   if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable')
    body_vars = set()
    for atom in rule.body:
        for arg in atom.args:
            if hasattr(arg, '__class__') and arg.__class__.__name__ == 'Variable':
                body_vars.add(arg)

    free_vars = head_vars - body_vars

    print(f"\nVariable analysis:")
    print(f"  Head variables: {head_vars}")
    print(f"  Body variables: {body_vars}")
    print(f"  Free variables: {free_vars}")

    if not free_vars and head_vars.issubset(body_vars):
        print(f"  ✓ Safety condition satisfied: all head vars in body")
    else:
        print(f"  ✗ Safety condition violated: head vars not in body")

    # Determine if perfect
    perfect = scores['pos_covered'] == len(positive_examples) and scores['neg_covered'] == 0
    if perfect:
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS: Found perfect rule!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"  Partial solution - coverage not perfect")
        print(f"{'='*80}")
else:
    print("\n✗ No rule found")
