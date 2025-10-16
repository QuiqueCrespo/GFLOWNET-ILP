"""
Test GFlowNet with background knowledge to enable proper learning.
"""

import torch
import numpy as np
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.graph_encoder import GraphConstructor, StateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.reward import RewardCalculator
from src.training import GFlowNetTrainer


def main():
    print("=" * 70)
    print("GFlowNet Training WITH Background Knowledge")
    print("=" * 70)

    torch.manual_seed(123)
    np.random.seed(123)

    # Define vocabulary
    predicate_vocab = ['grandparent', 'parent']
    predicate_arities = {'grandparent': 2, 'parent': 2}

    # Background knowledge: parent relationships
    background_facts = [
        Example('parent', ('alice', 'bob')),
        Example('parent', ('alice', 'carol')),
        Example('parent', ('bob', 'charlie')),
        Example('parent', ('bob', 'diana')),
        Example('parent', ('carol', 'eve')),
        Example('parent', ('carol', 'frank')),
    ]

    print("\n" + "-" * 70)
    print("Background Knowledge (parent facts):")
    print("-" * 70)
    for fact in background_facts:
        print(f"  {fact}")

    # Positive examples: grandparent relationships
    # These should be derivable from correct rules + background facts
    pos_examples = [
        Example('grandparent', ('alice', 'charlie')),  # alice->bob->charlie
        Example('grandparent', ('alice', 'diana')),    # alice->bob->diana
        Example('grandparent', ('alice', 'eve')),      # alice->carol->eve
        Example('grandparent', ('alice', 'frank')),    # alice->carol->frank
    ]

    # Negative examples: NOT grandparent relationships
    neg_examples = [
        Example('grandparent', ('alice', 'alice')),    # self
        Example('grandparent', ('charlie', 'alice')),  # reverse
        Example('grandparent', ('bob', 'alice')),      # child to parent
        Example('grandparent', ('charlie', 'diana')),  # siblings
    ]

    print("\n" + "-" * 70)
    print("Training Examples:")
    print("-" * 70)
    print(f"\nPositive (should be grandparent): {len(pos_examples)}")
    for ex in pos_examples:
        print(f"  ✓ {ex}")

    print(f"\nNegative (should NOT be grandparent): {len(neg_examples)}")
    for ex in neg_examples:
        print(f"  ✗ {ex}")

    # Verify correct rule would work
    print("\n" + "-" * 70)
    print("Verification: Can correct rule derive positive examples?")
    print("-" * 70)

    from logic_structures import Variable, Atom, Rule

    # Correct rule: grandparent(X, Y) :- parent(X, Z), parent(Z, Y)
    v0, v1, v2 = Variable(0), Variable(1), Variable(2)
    correct_rule = Rule(
        head=Atom('grandparent', (v0, v1)),
        body=[
            Atom('parent', (v0, v2)),
            Atom('parent', (v2, v1))
        ]
    )
    correct_theory = [correct_rule]

    print(f"\nCorrect rule: {theory_to_string(correct_theory)}")

    engine_with_bg = LogicEngine(max_depth=10, background_facts=background_facts)

    print("\nTesting correct rule:")
    for ex in pos_examples:
        result = engine_with_bg.entails(correct_theory, ex)
        status = "✓" if result else "✗"
        print(f"  {status} {ex}: {result}")

    print("\nTesting on negatives (should be False):")
    for ex in neg_examples:
        result = engine_with_bg.entails(correct_theory, ex)
        status = "✓" if not result else "✗"
        print(f"  {status} {ex}: {result}")

    # Initialize models
    print("\n" + "-" * 70)
    print("Training GFlowNet")
    print("-" * 70)

    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=3)
    gflownet = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

    reward_calc = RewardCalculator(engine_with_bg)

    trainer = GFlowNetTrainer(
        state_encoder, gflownet, graph_constructor, reward_calc,
        predicate_vocab, predicate_arities, learning_rate=1e-3
    )

    initial_state = get_initial_state('grandparent', arity=2)

    print(f"\nInitial state: {theory_to_string(initial_state)}")
    print(f"\nTraining for 1000 episodes...\n")

    history = trainer.train(
        initial_state, pos_examples, neg_examples,
        num_episodes=1000, verbose=True
    )

    # Analysis
    print("\n" + "=" * 70)
    print("Training Analysis")
    print("=" * 70)

    # Rewards over time
    print("\nReward progression:")
    for i in range(0, 1000, 100):
        avg_reward = np.mean([h['reward'] for h in history[i:i+100]])
        avg_steps = np.mean([h['trajectory_length'] for h in history[i:i+100]])
        print(f"  Episodes {i:4d}-{i+99:4d}: Avg reward={avg_reward:.4f}, Avg steps={avg_steps:.1f}")

    # Sample best theories
    print("\n" + "-" * 70)
    print("Top 10 Learned Theories")
    print("-" * 70)

    top_theories = trainer.sample_top_theories(
        initial_state, pos_examples, neg_examples,
        num_samples=100, top_k=10
    )

    for i, (theory, reward) in enumerate(top_theories):
        scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)

        print(f"\n{i+1}. Reward: {reward:.4f}")
        print(f"   Theory: {theory_to_string(theory)}")
        print(f"   Pos: {scores['pos_covered']}/{scores['pos_total']}, "
              f"Neg: {scores['neg_covered']}/{scores['neg_total']}, "
              f"Atoms: {scores['total_atoms']}")
        print(f"   Accuracy: {scores['accuracy']:.4f}")

    # Test best theory
    best_theory, best_reward = top_theories[0]

    print("\n" + "=" * 70)
    print("Best Theory Evaluation")
    print("=" * 70)

    print(f"\nBest theory: {theory_to_string(best_theory)}")
    print(f"Reward: {best_reward:.4f}")

    scores = reward_calc.get_detailed_scores(best_theory, pos_examples, neg_examples)

    print(f"\nDetailed evaluation:")
    print(f"  Positive coverage: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
    print(f"  Negative avoidance: {scores['neg_total']-scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")
    print(f"  Accuracy: {scores['accuracy']:.4f}")
    print(f"  Body atoms: {scores['total_atoms']}")

    print(f"\nTesting on individual examples:")
    print(f"\nPositive examples:")
    for ex in pos_examples:
        result = engine_with_bg.entails(best_theory, ex)
        status = "✓" if result else "✗"
        print(f"  {status} {ex}: {result}")

    print(f"\nNegative examples (should be False):")
    for ex in neg_examples:
        result = engine_with_bg.entails(best_theory, ex)
        status = "✓" if not result else "✗"
        print(f"  {status} {ex}: {result}")

    # Compare to correct rule
    print("\n" + "-" * 70)
    print("Comparison to Correct Rule")
    print("-" * 70)

    print(f"\nLearned:  {theory_to_string(best_theory)}")
    print(f"Correct:  {theory_to_string(correct_theory)}")

    if scores['accuracy'] > 0.9:
        print("\n✅ SUCCESS! Model learned a highly accurate rule!")
    elif scores['accuracy'] > 0.5:
        print("\n✅ PROGRESS! Model learned a partially correct rule!")
    else:
        print("\n⚠️  Model still struggling - may need more training")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
