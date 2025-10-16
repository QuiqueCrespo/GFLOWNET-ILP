"""Inspect generated rules in detail to check for free variables."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import get_initial_state, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

# Setup
background_facts = [
    Example('parent', ('alice', 'bob')),
    Example('parent', ('bob', 'charlie'))
]
positive_examples = [Example('grandparent', ('alice', 'charlie'))]
negative_examples = [Example('grandparent', ('alice', 'alice'))]

predicate_vocab = ['parent']
predicate_arities = {'parent': 2}

logic_engine = LogicEngine(background_facts=background_facts)
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
    exploration_strategy=exploration
)

print("="*80)
print("DETAILED RULE INSPECTION")
print("="*80)
print("\nGenerating 20 rules and inspecting each for free variables...\n")

initial_state = get_initial_state('grandparent', 2)

for i in range(20):
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positive_examples, negative_examples
    )

    if trajectory:
        final_state = trajectory[-1].next_state
        rule = final_state[0]

        # Manual free variable check
        head_vars = set(rule.head.args)
        body_vars = set()
        for atom in rule.body:
            body_vars.update(atom.args)
        free_vars = head_vars - body_vars

        # Get detailed scores including free var penalty
        scores = reward_calc.get_detailed_scores(final_state, positive_examples, negative_examples)

        print(f"Rule {i+1}:")
        print(f"  {theory_to_string(final_state)}")
        print(f"  Head variables: {{{', '.join(f'X{v.id}' for v in sorted(head_vars, key=lambda x: x.id))}}}")
        print(f"  Body variables: {{{', '.join(f'X{v.id}' for v in sorted(body_vars, key=lambda x: x.id))}}}")
        print(f"  Free variables: {{{', '.join(f'X{v.id}' for v in sorted(free_vars, key=lambda x: x.id))}}}")
        print(f"  Free var count (calc): {len(free_vars)}")
        print(f"  Free var count (reward): {scores.get('num_free_vars', 'N/A')}")
        print(f"  Free var penalty: {scores.get('free_var_penalty', 'N/A')}")
        print(f"  Base reward: {scores['reward']:.4f}")

        # Check for mismatch
        if len(free_vars) != scores.get('num_free_vars', 0):
            print(f"  ⚠️  MISMATCH: Manual count={len(free_vars)}, Reward count={scores.get('num_free_vars', 0)}")

        # Check if any head variable is truly missing from body
        for head_var in head_vars:
            if head_var not in body_vars:
                print(f"  ⚠️  X{head_var.id} is in head but NOT in body")
                # Check if it appears anywhere in body atoms
                appears_in_body = False
                for atom in rule.body:
                    if head_var in atom.args:
                        appears_in_body = True
                        break
                if appears_in_body:
                    print(f"      ERROR: X{head_var.id} DOES appear in body! Detection bug!")

        print()

print("="*80)
print("ANALYSIS")
print("="*80)
print("""
This script checks for:
1. Correct free variable detection
2. Consistency between manual calculation and reward calculator
3. Any variables truly missing from body that aren't flagged

If you see a rule with a variable in the head that doesn't appear in the
body, it should be flagged as a free variable.
""")
