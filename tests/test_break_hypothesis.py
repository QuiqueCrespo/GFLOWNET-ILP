"""Test hypothesis that break statements cause premature exit."""
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
print("BREAK STATEMENT HYPOTHESIS TEST")
print("="*80)
print("\nHypothesis: When UNIFY_VARIABLES action fails (no valid pairs, etc.),")
print("the break statement causes premature exit with free variables.\n")

initial_state = get_initial_state('grandparent', 2)

# Generate 20 trajectories and analyze
break_cases = 0
free_var_cases = 0
both_cases = 0

for i in range(20):
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positive_examples, negative_examples
    )

    if trajectory:
        final_state = trajectory[-1].next_state
        rule = final_state[0]

        # Check for free variables
        head_vars = set(rule.head.args)
        body_vars = set()
        for atom in rule.body:
            body_vars.update(atom.args)
        free_vars = head_vars - body_vars

        has_free_vars = len(free_vars) > 0

        # Analyze trajectory for signs of premature exit
        # If trajectory is very short and has free vars, likely hit a break
        trajectory_length = len(trajectory)
        likely_break = trajectory_length < 5 and has_free_vars

        if has_free_vars:
            free_var_cases += 1
        if likely_break:
            break_cases += 1
            if i < 5:  # Show first 5 examples
                print(f"Trajectory {i}:")
                print(f"  Final state: {theory_to_string(final_state)}")
                print(f"  Length: {trajectory_length} steps")
                print(f"  Free vars: {', '.join(f'X{v.id}' for v in free_vars)}")
                print(f"  Last action: {trajectory[-1].action_type}")
                print()

        if likely_break and has_free_vars:
            both_cases += 1

print("="*80)
print("RESULTS")
print("="*80)
print(f"\nOut of 20 trajectories:")
print(f"  Trajectories with free variables: {free_var_cases}")
print(f"  Likely premature breaks: {break_cases}")
print(f"  Both (free vars + premature): {both_cases}")

if both_cases > 15:
    print("\n✓ HYPOTHESIS CONFIRMED:")
    print("  Break statements are causing premature exits with free variables.")
elif both_cases > 5:
    print("\n⚠ HYPOTHESIS PARTIALLY CONFIRMED:")
    print("  Some break statements causing premature exits.")
else:
    print("\n✗ HYPOTHESIS REJECTED:")
    print("  Break statements are not the main cause.")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)
print("""
The break statements in lines 258, 265, 275 of src/training.py cause
premature termination when:
1. No valid variable pairs exist
2. Less than 2 variables exist
3. No pair logits are generated

Instead of breaking (which treats it as terminal), we should:
Option A: Continue the loop and force ADD_ATOM action
Option B: Treat UNIFY_VARIABLES failure as a non-action (no trajectory step)

Recommended fix: Change break to continue, but only allow ADD_ATOM action
in the next iteration.
""")
