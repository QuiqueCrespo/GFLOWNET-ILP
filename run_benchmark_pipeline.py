"""
Benchmark pipeline to test the enhanced GFlowNet-ILP method on all examples.
"""
import sys
import os
import time
from pathlib import Path
import json

sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

def parse_prolog_file(filepath):
    """Parse Prolog files to extract examples."""
    with open(filepath, 'r') as f:
        content = f.read()

    examples = []
    # Simple parser for Prolog facts like: predicate(arg1, arg2, ...).
    import re
    pattern = r'(\w+)\((.*?)\)\.'
    matches = re.findall(pattern, content)

    for pred, args_str in matches:
        # Split arguments by comma, strip whitespace
        args = tuple(arg.strip() for arg in args_str.split(','))
        examples.append(Example(pred, args))

    return examples

def load_benchmark(benchmark_dir):
    """Load a benchmark from directory."""
    bk_file = benchmark_dir / 'bk.pl'
    exs_file = benchmark_dir / 'exs.pl'

    if not bk_file.exists() or not exs_file.exists():
        return None

    # Parse background knowledge
    background_facts = parse_prolog_file(bk_file)

    # Parse examples (pos() and neg() wrap the actual examples)
    all_examples = parse_prolog_file(exs_file)

    # Extract actual examples from pos() and neg() wrappers
    positive_examples = []
    negative_examples = []

    import re
    with open(exs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                # Extract inner predicate: pos(ancestor(X, Y)) -> ancestor(X, Y)
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
        return None

    # Get target predicate from first positive example
    target_predicate = positive_examples[0].predicate_name
    target_arity = len(positive_examples[0].args)

    # Get vocabulary from background knowledge
    predicate_vocab = list(set(ex.predicate_name for ex in background_facts))
    predicate_arities = {}
    for ex in background_facts:
        if ex.predicate_name not in predicate_arities:
            predicate_arities[ex.predicate_name] = len(ex.args)

    if not predicate_vocab:
        return None

    return {
        'background_facts': background_facts,
        'positive_examples': positive_examples,
        'negative_examples': negative_examples if negative_examples else [],
        'predicate_vocab': predicate_vocab,
        'predicate_arities': predicate_arities,
        'target_predicate': target_predicate,
        'target_arity': target_arity
    }

def run_benchmark(benchmark_name, benchmark_data, num_episodes=100, timeout=300):
    """Run enhanced method on a benchmark."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"{'='*80}")

    try:
        # Setup
        logic_engine = LogicEngine(
            max_depth=5,
            background_facts=benchmark_data['background_facts']
        )

        reward_calc = RewardCalculator(
            logic_engine,
            disconnected_var_penalty=0.2,
            self_loop_penalty=0.3,
            free_var_penalty=1.0
        )

        graph_constructor = EnhancedGraphConstructor(benchmark_data['predicate_vocab'])
        state_encoder = EnhancedStateEncoder(
            predicate_vocab_size=len(benchmark_data['predicate_vocab']),
            embedding_dim=32,
            num_layers=2
        )

        gflownet = HierarchicalGFlowNet(
            embedding_dim=32,
            num_predicates=len(benchmark_data['predicate_vocab']),
            hidden_dim=64
        )

        exploration = get_combined_strategy("aggressive")

        trainer = GFlowNetTrainer(
            state_encoder=state_encoder,
            gflownet=gflownet,
            graph_constructor=graph_constructor,
            reward_calculator=reward_calc,
            predicate_vocab=benchmark_data['predicate_vocab'],
            predicate_arities=benchmark_data['predicate_arities'],
            learning_rate=1e-3,
            exploration_strategy=exploration,
            use_detailed_balance=True,
            use_replay_buffer=True,
            replay_buffer_capacity=20,
            replay_probability=0.3,
            reward_weighted_loss=True
        )

        initial_state = get_initial_state(
            benchmark_data['target_predicate'],
            benchmark_data['target_arity']
        )

        # Training
        start_time = time.time()
        rewards = []
        best_rule = None
        best_rule_theory = None  # Store actual theory object
        best_reward = -float('inf')
        best_true_reward = -float('inf')

        for episode in range(num_episodes):
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"  ⚠ Timeout after {episode} episodes")
                break

            metrics = trainer.train_step(
                initial_state,
                benchmark_data['positive_examples'],
                benchmark_data['negative_examples']
            )
            rewards.append(metrics['reward'])

            # Sample best rule every 10 episodes based on TRUE reward (no exploration bonus)
            if episode % 10 == 0:
                trajectory, training_reward = trainer.generate_trajectory(
                    initial_state,
                    benchmark_data['positive_examples'],
                    benchmark_data['negative_examples']
                )

                if trajectory:
                    theory = trajectory[-1].next_state

                    # Calculate TRUE reward (without exploration bonus)
                    true_reward = reward_calc.calculate_reward(
                        theory,
                        benchmark_data['positive_examples'],
                        benchmark_data['negative_examples']
                    )

                    # Select best rule based on TRUE reward, not training reward
                    if true_reward > best_true_reward:
                        best_true_reward = true_reward
                        best_reward = training_reward  # Keep for comparison
                        best_rule_theory = theory  # Store the theory object
                        best_rule = theory_to_string(theory)

        elapsed_time = time.time() - start_time

        # Check for free variables and get detailed scores for final best rule
        free_var_count = 0
        pos_coverage = 0.0
        neg_coverage = 0.0
        pos_covered = 0
        neg_covered = 0

        if best_rule_theory:
            # Use the stored theory object (don't regenerate - stochastic!)
            theory = best_rule_theory
            rule = theory[0]

            # Check free variables
            head_vars = set(rule.head.args)
            body_vars = set()
            for atom in rule.body:
                body_vars.update(atom.args)
            free_vars = head_vars - body_vars
            free_var_count = len(free_vars)

            # Get detailed scores for coverage
            scores = reward_calc.get_detailed_scores(
                theory,
                benchmark_data['positive_examples'],
                benchmark_data['negative_examples']
            )

            pos_covered = scores['pos_covered']
            neg_covered = scores['neg_covered']
            pos_total = scores['pos_total']
            neg_total = scores['neg_total']

            pos_coverage = pos_covered / pos_total if pos_total > 0 else 0.0
            neg_coverage = neg_covered / neg_total if neg_total > 0 else 0.0

        result = {
            'benchmark': benchmark_name,
            'status': 'completed',
            'num_episodes': len(rewards),
            'elapsed_time': elapsed_time,
            'final_avg_reward': sum(rewards[-10:]) / min(10, len(rewards)),
            'max_reward': max(rewards) if rewards else 0,
            'best_rule': best_rule,
            'best_reward': best_reward,
            'true_reward': best_true_reward,
            'free_variables': free_var_count,
            'pos_coverage': pos_coverage,
            'neg_coverage': neg_coverage,
            'pos_covered': pos_covered,
            'neg_covered': neg_covered,
            'num_background_facts': len(benchmark_data['background_facts']),
            'num_positive': len(benchmark_data['positive_examples']),
            'num_negative': len(benchmark_data['negative_examples'])
        }

        print(f"  Status: {result['status']}")
        print(f"  Episodes: {result['num_episodes']}")
        print(f"  Time: {elapsed_time:.1f}s")
        print(f"  Training reward (with bonus): {best_reward:.4f}")
        print(f"  True reward (no bonus): {best_true_reward:.6f}")
        print(f"  Coverage: {pos_covered}/{result['num_positive']} pos ({100*pos_coverage:.1f}%), {neg_covered}/{result['num_negative']} neg ({100*neg_coverage:.1f}%)")
        print(f"  Best rule: {best_rule}")
        print(f"  Free variables: {free_var_count}")

        return result

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return {
            'benchmark': benchmark_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    """Run pipeline on all benchmarks."""
    examples_dir = Path('/Users/jq23948/GFLowNet-ILP/examples')

    # Get all benchmark directories
    benchmarks = [d for d in examples_dir.iterdir() if d.is_dir()]
    benchmarks = sorted(benchmarks)

    print("="*80)
    print("GFLOWNET-ILP BENCHMARK PIPELINE")
    print("="*80)
    print(f"\nFound {len(benchmarks)} benchmarks")
    print(f"Testing enhanced method with:")
    print("  - Action mask (prevents ADD_ATOM at max body length)")
    print("  - Free variable constraint (no free vars in terminal states)")
    print("  - Reward shaping (penalties for structural issues)")
    print("  - Enhanced encoding (rich features + attention)")

    # Select subset for quick testing (first 10)
    test_benchmarks = benchmarks[:10]

    print(f"\nRunning on first {len(test_benchmarks)} benchmarks (quick test)...")
    print("To run all, modify test_benchmarks = benchmarks\n")

    results = []

    for benchmark_dir in test_benchmarks:
        benchmark_data = load_benchmark(benchmark_dir)

        if benchmark_data is None:
            print(f"\n⊘ Skipping {benchmark_dir.name} (missing files)")
            continue

        if len(benchmark_data['background_facts']) == 0:
            print(f"\n⊘ Skipping {benchmark_dir.name} (no background knowledge)")
            continue

        result = run_benchmark(
            benchmark_dir.name,
            benchmark_data,
            num_episodes=1000,
            timeout=300  # 5 minutes per benchmark
        )
        results.append(result)

    # Summary report
    print("\n" + "="*80)
    print("PIPELINE RESULTS SUMMARY")
    print("="*80)

    completed = [r for r in results if r['status'] == 'completed']
    errors = [r for r in results if r['status'] == 'error']

    print(f"\nTotal benchmarks tested: {len(results)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Errors: {len(errors)}")

    if completed:
        print(f"\n{'='*80}")
        print("COMPLETED BENCHMARKS")
        print(f"{'='*80}")

        for r in completed:
            print(f"\n{r['benchmark']}:")
            print(f"  Episodes: {r['num_episodes']}")
            print(f"  Time: {r['elapsed_time']:.1f}s")
            print(f"  Training reward: {r['best_reward']:.4f} | True reward: {r['true_reward']:.6f}")
            print(f"  Coverage: {r['pos_covered']}/{r['num_positive']} pos ({100*r['pos_coverage']:.1f}%), {r['neg_covered']}/{r['num_negative']} neg ({100*r['neg_coverage']:.1f}%)")
            print(f"  Free variables: {r['free_variables']}")
            print(f"  Best rule: {r['best_rule'][:80]}...")

        # Statistics
        avg_reward = sum(r['best_reward'] for r in completed) / len(completed)
        total_free_vars = sum(r['free_variables'] for r in completed)
        avg_pos_coverage = sum(r['pos_coverage'] for r in completed) / len(completed)
        avg_neg_coverage = sum(r['neg_coverage'] for r in completed) / len(completed)

        # Count perfect rules (100% pos, 0% neg)
        perfect_rules = sum(1 for r in completed if r['pos_coverage'] == 1.0 and r['neg_coverage'] == 0.0)

        print(f"\n{'='*80}")
        print("STATISTICS")
        print(f"{'='*80}")
        print(f"Average best reward: {avg_reward:.4f}")
        print(f"Average positive coverage: {100*avg_pos_coverage:.1f}%")
        print(f"Average negative coverage: {100*avg_neg_coverage:.1f}%")
        print(f"Perfect rules (100% pos, 0% neg): {perfect_rules}/{len(completed)}")
        print(f"Total free variables: {total_free_vars}")
        print(f"Free variable rate: {100 * total_free_vars / len(completed) if completed else 0:.1f}%")

        if total_free_vars == 0:
            print("\n✓ SUCCESS: No free variables in any benchmark!")
        else:
            print(f"\n⚠ {total_free_vars} benchmarks have free variables")

        if perfect_rules > 0:
            print(f"✓ {perfect_rules} benchmarks achieved perfect classification!")

    # Save results to JSON
    output_file = Path('/Users/jq23948/GFLowNet-ILP/benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
