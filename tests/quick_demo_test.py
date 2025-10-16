"""Quick test of demo_enhanced_method with fewer episodes."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

import numpy as np
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

# Problem setup
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
]

positive_examples = [
    Example('grandparent', ('alice', 'charlie')),
]

negative_examples = [
    Example('grandparent', ('alice', 'alice')),
    Example('grandparent', ('bob', 'bob')),
]

predicate_vocab = ['parent']
predicate_arities = {'parent': 2}

print("="*80)
print("QUICK ENHANCED METHOD TEST")
print("="*80)

# Setup enhanced method
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
    replay_probability=0.3
)

# Training
num_episodes = 100
initial_state = get_initial_state('grandparent', 2)

print(f"\nTraining for {num_episodes} episodes...")

discovered_rules = {}

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

    # Sample every 10 episodes
    if episode % 10 == 0:
        trajectory, reward = trainer.generate_trajectory(
            initial_state, positive_examples, negative_examples
        )
        theory = trajectory[-1].next_state if trajectory else initial_state
        rule_str = theory_to_string(theory)
        scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

        # Check for free variables
        rule = theory[0]
        head_vars = set(rule.head.args)
        body_vars = set()
        for atom in rule.body:
            body_vars.update(atom.args)
        free_vars = head_vars - body_vars

        if rule_str not in discovered_rules:
            discovered_rules[rule_str] = {
                'reward': reward,
                'episode': episode,
                'scores': scores,
                'free_vars': len(free_vars)
            }

        if episode % 20 == 0:
            print(f"Episode {episode}: reward={metrics['reward']:.4f}")
            print(f"  Rule: {rule_str}")
            print(f"  Free vars: {len(free_vars)}")

print("\n" + "="*80)
print("DISCOVERED RULES")
print("="*80)

# Check for free variables in discovered rules
rules_with_free_vars = 0

for rule_str, info in discovered_rules.items():
    if info['free_vars'] > 0:
        rules_with_free_vars += 1
        print(f"\n⚠️  FREE VARIABLES FOUND:")
        print(f"  Rule: {rule_str}")
        print(f"  Free vars: {info['free_vars']}")
        print(f"  Episode: {info['episode']}")

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total unique rules discovered: {len(discovered_rules)}")
print(f"Rules with free variables: {rules_with_free_vars}")

if rules_with_free_vars == 0:
    print("\n✓ SUCCESS: No free variables in any discovered rules!")
else:
    print(f"\n✗ FAILURE: {rules_with_free_vars} rules have free variables!")
