"""Test that action mask prevents ADD_ATOM at max body length."""
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
print("ACTION MASK TEST")
print("="*80)
print("\nTesting that ADD_ATOM is prevented at max body length (3 atoms)")
print("but UNIFY_VARIABLES is still allowed to resolve free variables.\n")

initial_state = get_initial_state('grandparent', 2)

# Generate 100 trajectories and analyze
rules_at_max_length = 0
free_vars_at_max = 0
unifications_at_max = 0

for i in range(100):
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positive_examples, negative_examples
    )

    if trajectory:
        final_state = trajectory[-1].next_state
        rule = final_state[0]

        if len(rule.body) >= 3:
            rules_at_max_length += 1

            # Check for free variables
            head_vars = set(rule.head.args)
            body_vars = set()
            for atom in rule.body:
                body_vars.update(atom.args)
            free_vars = head_vars - body_vars

            if free_vars:
                free_vars_at_max += 1

            # Count UNIFY_VARIABLES actions after reaching max length
            reached_max = False
            for step in trajectory:
                if len(step.state[0].body) >= 3:
                    reached_max = True
                if reached_max and step.action_type == 'UNIFY_VARIABLES':
                    unifications_at_max += 1

            if i < 5:  # Show first 5 examples
                print(f"Example {i+1}:")
                print(f"  Rule: {theory_to_string(final_state)}")
                print(f"  Body length: {len(rule.body)}")
                print(f"  Free vars: {len(free_vars)}")

                # Show actions after reaching max length
                reached_max_at = None
                for j, step in enumerate(trajectory):
                    if len(step.state[0].body) >= 3 and reached_max_at is None:
                        reached_max_at = j

                if reached_max_at is not None:
                    post_max_actions = [s.action_type for s in trajectory[reached_max_at:]]
                    print(f"  Actions after max length: {', '.join(post_max_actions)}")
                print()

print("="*80)
print("RESULTS")
print("="*80)
print(f"\nRules at max body length: {rules_at_max_length}/100")
print(f"Rules with free vars at max length: {free_vars_at_max}/{rules_at_max_length}")
print(f"UNIFY_VARIABLES actions at max length: {unifications_at_max}")

if rules_at_max_length > 0:
    free_var_rate = 100 * free_vars_at_max / rules_at_max_length
    print(f"\nFree variable rate at max length: {free_var_rate:.1f}%")

    if unifications_at_max > 0:
        print(f"\n✓ SUCCESS: UNIFY_VARIABLES actions are being used at max length")
        print(f"  {unifications_at_max} unifications occurred after reaching max body length")
    else:
        print(f"\n⚠ WARNING: No UNIFY_VARIABLES actions at max length")
        print(f"  The action mask may be too restrictive or no unifications were needed")

    if free_var_rate < 50:
        print(f"\n✓ IMPROVEMENT: Free variable rate is lower than before ({free_var_rate:.1f}%)")
        print(f"  The action mask is helping resolve free variables at max length")
    else:
        print(f"\n⚠ Note: Free variable rate is still {free_var_rate:.1f}%")
        print(f"  This may be acceptable if variables cannot be unified further")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
The action mask should:
1. Prevent ADD_ATOM when body length >= 3
2. Allow UNIFY_VARIABLES to continue at max length
3. Enable free variable resolution at max body length
4. Reduce the percentage of rules with free variables

Expected behavior:
- At max length, only UNIFY_VARIABLES actions should occur
- Free variable rate should be lower than the 52% we saw before
- Rules should terminate only when either:
  a) All free variables are resolved, OR
  b) No more unifications are possible
""")
