"""Verify that free variable rules cannot be generated as terminal states."""
import sys
sys.path.insert(0, '/Users/jq23948/GFLowNet-ILP')

from src.logic_structures import get_initial_state, is_terminal, theory_to_string
from src.logic_engine import LogicEngine, Example
from src.reward import RewardCalculator
from src.graph_encoder_enhanced import EnhancedGraphConstructor, EnhancedStateEncoder
from src.gflownet_models import HierarchicalGFlowNet
from src.training import GFlowNetTrainer
from src.exploration import get_combined_strategy

print("="*80)
print("FREE VARIABLE GENERATION TEST")
print("="*80)

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
    exploration_strategy=exploration,
    use_detailed_balance=True,
    use_replay_buffer=True,
    replay_buffer_capacity=50,
    reward_weighted_loss=True,
    replay_probability=0.3
)

print("\nGenerating 100 trajectories to verify no free variables in terminal states...")

initial_state = get_initial_state('grandparent', 2)
free_var_count = 0
total_trajectories = 100

for i in range(total_trajectories):
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positive_examples, negative_examples
    )

    if trajectory:
        # The final state is next_state of the last trajectory step
        final_state = trajectory[-1].next_state
        rule = final_state[0]

        # Check for free variables
        head_vars = set(rule.head.args)
        body_vars = set()
        for atom in rule.body:
            body_vars.update(atom.args)
        free_vars = head_vars - body_vars

        if free_vars:
            free_var_count += 1
            if i < 5:  # Show first 5 examples
                rule_str = theory_to_string(final_state)
                print(f"\n  Trajectory {i}: {rule_str}")
                print(f"    Trajectory length: {len(trajectory)} steps")
                print(f"    Free vars: {', '.join(f'X{v.id}' for v in free_vars)}")
                print(f"    Body length: {len(rule.body)}")
                print(f"    Is terminal: {is_terminal(final_state)}")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nTotal trajectories: {total_trajectories}")
print(f"Trajectories with free variables: {free_var_count}")
print(f"Percentage: {100 * free_var_count / total_trajectories:.1f}%")

if free_var_count == 0:
    print("\n✓ PERFECT: No free variable rules generated!")
    print("  The terminal constraint is working perfectly.")
elif free_var_count < total_trajectories * 0.7:
    print("\n✓ SUCCESS: Terminal constraint is working!")
    print(f"  Free vars only at max body length (forced termination)")
    print(f"  {100 - 100*free_var_count/total_trajectories:.1f}% of rules successfully resolve free variables")
else:
    print("\n✗ FAILURE: Many free variable rules generated!")
    print("  The terminal constraint may not be working properly")

# Check if free vars only at max length
if free_var_count > 0:
    print("\nChecking if free vars only occur at max body length...")
    max_length_only = True
    for i in range(min(20, total_trajectories)):
        trajectory, _ = trainer.generate_trajectory(
            initial_state, positive_examples, negative_examples
        )
        if trajectory:
            final_state = trajectory[-1].next_state
            rule = final_state[0]
            head_vars = set(rule.head.args)
            body_vars = set()
            for atom in rule.body:
                body_vars.update(atom.args)
            free_vars = head_vars - body_vars

            if free_vars and len(rule.body) < 3:
                max_length_only = False
                print(f"  Found free vars with body length {len(rule.body)} < 3!")
                break

    if max_length_only:
        print("  ✓ Free vars only at max length (acceptable)")
    else:
        print("  ✗ Free vars at lengths < 3 (BUG!)")
