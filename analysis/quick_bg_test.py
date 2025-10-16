"""Quick test with background knowledge - shorter version."""

import torch
from src.logic_structures import get_initial_state, theory_to_string, Variable, Atom, Rule
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer

torch.manual_seed(42)

predicate_vocab = ['grandparent', 'parent']
predicate_arities = {'grandparent': 2, 'parent': 2}

# Background knowledge
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie')),
]

pos_examples = [Example('grandparent', ('alice', 'charlie'))]
neg_examples = [Example('grandparent', ('alice', 'alice'))]

print("Testing with background knowledge")
print(f"Background: {background_facts}")
print(f"Positive: {pos_examples}")
print(f"Negative: {neg_examples}")

# Verify correct rule works
v0, v1, v2 = Variable(0), Variable(1), Variable(2)
correct_rule = Rule(
    head=Atom('grandparent', (v0, v1)),
    body=[Atom('parent', (v0, v2)), Atom('parent', (v2, v1))]
)
correct_theory = [correct_rule]

engine = LogicEngine(max_depth=3, background_facts=background_facts)
print(f"\nCorrect rule: {theory_to_string(correct_theory)}")

result_pos = engine.entails(correct_theory, pos_examples[0])
result_neg = engine.entails(correct_theory, neg_examples[0])
print(f"Positive test: {result_pos} (should be True)")
print(f"Negative test: {result_neg} (should be False)")

if result_pos and not result_neg:
    print("✅ Logic engine working correctly!")
else:
    print("❌ Logic engine has issues")
    exit(1)

# Train
graph_constructor = GraphConstructor(predicate_vocab)
state_encoder = StateEncoder(len(predicate_vocab) + 1, 32, num_layers=2)
gflownet = HierarchicalGFlowNet(32, len(predicate_vocab), 64)

reward_calc = RewardCalculator(engine)
trainer = GFlowNetTrainer(
    state_encoder, gflownet, graph_constructor, reward_calc,
    predicate_vocab, predicate_arities, learning_rate=1e-3
)

initial_state = get_initial_state('grandparent', arity=2)

print(f"\n Training 200 episodes...")
history = trainer.train(initial_state, pos_examples, neg_examples,
                       num_episodes=200, verbose=False)

print(f"\nFinal 10 episodes avg reward: {sum(h['reward'] for h in history[-10:])/10:.4f}")

# Sample theories
top_5 = trainer.sample_top_theories(
    initial_state, pos_examples, neg_examples,
    num_samples=20, top_k=5
)

print("\nTop 5 theories:")
for i, (theory, reward) in enumerate(top_5):
    scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)
    print(f"{i+1}. {theory_to_string(theory)}")
    print(f"   Reward: {reward:.4f}, Pos: {scores['pos_covered']}/{scores['pos_total']}, "
          f"Neg: {scores['neg_covered']}/{scores['neg_total']}")
