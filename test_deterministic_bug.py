"""
Quick test to demonstrate the deterministic sampling bug.
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_deterministic_sampling_bug(trainer, initial_state, pos_examples, neg_examples):
    """
    Demonstrates that stochastic=False still produces different trajectories.

    THIS SHOULD FAIL with current code (bug exists).
    SHOULD PASS after fix is applied.
    """

    print("\n" + "="*70)
    print("TESTING DETERMINISTIC SAMPLING")
    print("="*70 + "\n")

    # Generate 3 trajectories with SAME seed + stochastic=False
    trajectories = []

    for run in range(3):
        print(f"Run {run + 1}:")

        # Reset seed to SAME value
        set_seed(42)

        # Generate trajectory with stochastic=FALSE
        traj, reward = trainer.generate_trajectory(
            initial_state, pos_examples, neg_examples,
            stochastic=False  # Should be deterministic!
        )

        trajectories.append(traj)

        # Print trajectory details
        print(f"  Trajectory length: {len(traj)}")
        for i, step in enumerate(traj):
            if step.action_type == 'ADD_ATOM':
                print(f"    Step {i}: {step.action_type} → {step.action_detail}")
            elif step.action_type == 'UNIFY_VARIABLES':
                var1, var2 = step.action_detail
                print(f"    Step {i}: {step.action_type} → (X{var1.id}, X{var2.id})")
            else:
                print(f"    Step {i}: {step.action_type}")
        print(f"  Reward: {reward:.4f}\n")

    # Check if all trajectories are identical
    print("="*70)
    print("CHECKING FOR DETERMINISM")
    print("="*70 + "\n")

    all_identical = True

    # Compare trajectory 1 with trajectory 2
    if len(trajectories[0]) != len(trajectories[1]):
        print(f"❌ FAIL: Trajectory lengths differ!")
        print(f"   Run 1: {len(trajectories[0])} steps")
        print(f"   Run 2: {len(trajectories[1])} steps")
        all_identical = False
    else:
        for i, (step1, step2) in enumerate(zip(trajectories[0], trajectories[1])):
            if step1.action_type != step2.action_type:
                print(f"❌ FAIL: Step {i} action type differs!")
                print(f"   Run 1: {step1.action_type}")
                print(f"   Run 2: {step2.action_type}")
                all_identical = False
                break

            if step1.action_detail != step2.action_detail:
                print(f"❌ FAIL: Step {i} action detail differs!")
                print(f"   Run 1: {step1.action_detail}")
                print(f"   Run 2: {step2.action_detail}")
                print(f"\n   This is the BUG: stochastic=False should produce")
                print(f"   identical trajectories, but predicate/pair selection")
                print(f"   is still random because stochastic parameter not passed!")
                all_identical = False
                break

    if all_identical:
        print("✅ PASS: All trajectories are identical!")
        print("   Deterministic sampling is working correctly.")
    else:
        print("\n🔴 BUG CONFIRMED: Deterministic sampling produces different results!")
        print("\n   Root cause: The stochastic parameter is not passed to")
        print("   _handle_action_add_atom and _handle_action_unify_vars")
        print("\n   See DETERMINISTIC_SAMPLING_BUG.md for fix.")

    print("\n" + "="*70 + "\n")

    return all_identical


if __name__ == "__main__":
    print("Deterministic Sampling Bug Test")
    print("================================\n")
    print("This test demonstrates the bug where stochastic=False")
    print("still produces different trajectories due to random")
    print("predicate and variable pair selection.\n")
    print("Usage:")
    print("  from test_deterministic_bug import test_deterministic_sampling_bug")
    print("  passed = test_deterministic_sampling_bug(trainer, init_state, pos_ex, neg_ex)")
    print("\nExpected result (before fix): FAIL")
    print("Expected result (after fix):  PASS")
