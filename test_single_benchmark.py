"""Quick test of single benchmark with fixed logic engine."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from pathlib import Path
from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy
import re

# Parse simple prolog file
def parse_prolog_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    examples = []
    pattern = r'(\w+)\((.*?)\)\.'
    matches = re.findall(pattern, content)

    for pred, args_str in matches:
        args = tuple(arg.strip() for arg in args_str.split(','))
        examples.append(Example(pred, args))

    return examples

# Load first valid benchmark
examples_dir = Path('/Users/jq23948/GFLowNet-ILP/examples')
for benchmark_dir in sorted(examples_dir.iterdir()):
    if not benchmark_dir.is_dir():
        continue

    bk_file = benchmark_dir / 'bk.pl'
    exs_file = benchmark_dir / 'exs.pl'

    if not bk_file.exists() or not exs_file.exists():
        continue

    background_facts = parse_prolog_file(bk_file)

    if len(background_facts) == 0:
        continue

    # Parse examples
    positive_examples = []
    negative_examples = []

    with open(exs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\((\w+)\((.*?)\)\)', line)
                if match:
                    pred = match.group(1)
                    args_str = match.group(2)
                    args = tuple(arg.strip() for arg in args_str.split(','))
                    positive_examples.append(Example(pred, args))
            elif line.startswith('neg('):
                match = re.match(r'neg\((\w+)\((.*?)\)\)', line)
                if match:
                    pred = match.group(1)
                    args_str = match.group(2)
                    args = tuple(arg.strip() for arg in args_str.split(','))
                    negative_examples.append(Example(pred, args))

    if not positive_examples:
        continue

    print(f"Testing benchmark: {benchmark_dir.name}")
    print(f"  Background facts: {len(background_facts)}")
    print(f"  Positive examples: {len(positive_examples)}")
    print(f"  Negative examples: {len(negative_examples)}")

    # Setup
    target_predicate = positive_examples[0].predicate_name
    target_arity = len(positive_examples[0].args)

    predicate_vocab = list(set(ex.predicate_name for ex in background_facts))
    predicate_arities = {}
    for ex in background_facts:
        if ex.predicate_name not in predicate_arities:
            predicate_arities[ex.predicate_name] = len(ex.args)

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

    # Run 50 episodes
    print("\nTraining for 50 episodes...")
    best_reward = -float('inf')
    best_rule = None

    for episode in range(50):
        metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

        if episode % 10 == 9:
            trajectory, reward = trainer.generate_trajectory(
                initial_state,
                positive_examples,
                negative_examples
            )

            if trajectory and reward > best_reward:
                best_reward = reward
                best_rule = trajectory[-1].next_state

    if best_rule:
        rule = best_rule[0]
        scores = reward_calc.get_detailed_scores(best_rule, positive_examples, negative_examples)
        calculated_reward = reward_calc.calculate_reward(best_rule, positive_examples, negative_examples)

        print(f"\nBest rule found:")
        print(f"  Rule: {theory_to_string(best_rule)}")
        print(f"  Best reward (from training): {best_reward:.4f}")
        print(f"  Calculated reward (now): {calculated_reward:.6f}")
        print(f"  Positive coverage: {scores['pos_covered']}/{scores['pos_total']} ({100*scores['pos_score']:.1f}%)")
        print(f"  Negative coverage: {scores['neg_covered']}/{scores['neg_total']} ({100*scores['neg_score']:.1f}%)")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  Simplicity: {scores['simplicity']:.4f}")
        print(f"  Disconnected vars: {scores['num_disconnected_vars']} (penalty: -{scores['disconnected_penalty']:.2f})")
        print(f"  Self-loops: {scores['num_self_loops']} (penalty: -{scores['self_loop_penalty']:.2f})")
        print(f"  Free vars: {scores.get('num_free_vars', 0)} (penalty: -{scores.get('free_var_penalty', 0):.2f})")

    break  # Only test first valid benchmark
