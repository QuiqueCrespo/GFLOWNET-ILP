"""
Verify that GNN (state_encoder) parameters are being trained during GFlowNet training.

Run this after creating a trainer to ensure gradients flow through the encoder.
"""

import torch


def verify_encoder_is_trainable(trainer):
    """
    Verify that the state encoder parameters are:
    1. Included in the optimizer
    2. Set to require gradients
    3. Actually receive gradients during training

    Args:
        trainer: GFlowNetTrainer instance
    """
    print("=" * 80)
    print("VERIFYING GNN ENCODER IS TRAINABLE")
    print("=" * 80)

    # Check 1: Are encoder parameters in the optimizer?
    print("\n1. Checking if encoder parameters are in optimizer...")

    optimizer_params = set(trainer.optimizer.param_groups[0]['params'])
    encoder_params = list(trainer.state_encoder.parameters())

    encoder_params_in_opt = sum(1 for p in encoder_params if p in optimizer_params)

    print(f"   - Encoder has {len(encoder_params)} parameters")
    print(f"   - Optimizer has {len(optimizer_params)} total parameters")
    print(f"   - {encoder_params_in_opt}/{len(encoder_params)} encoder params in optimizer")

    if encoder_params_in_opt == len(encoder_params):
        print("   ✅ All encoder parameters are in optimizer")
    else:
        print(f"   ❌ WARNING: Only {encoder_params_in_opt}/{len(encoder_params)} encoder params in optimizer!")
        return False

    # Check 2: Do encoder parameters require gradients?
    print("\n2. Checking if encoder parameters require gradients...")

    params_requiring_grad = [p for p in encoder_params if p.requires_grad]

    print(f"   - {len(params_requiring_grad)}/{len(encoder_params)} parameters require gradients")

    if len(params_requiring_grad) == len(encoder_params):
        print("   ✅ All encoder parameters require gradients")
    else:
        print(f"   ❌ WARNING: Only {len(params_requiring_grad)}/{len(encoder_params)} params require grad!")
        frozen = [i for i, p in enumerate(encoder_params) if not p.requires_grad]
        print(f"      Frozen parameter indices: {frozen[:5]}...")
        return False

    # Check 3: Test if gradients actually flow during a dummy training step
    print("\n3. Testing gradient flow with dummy training step...")

    # Store original parameter values
    original_values = [p.clone().detach() for p in encoder_params]

    # Do a dummy training step
    from src.logic_structures import get_initial_state

    initial_state = get_initial_state('grandparent', 2)

    # Create dummy examples
    from src.logic_structures import Rule, Atom, Variable
    dummy_positive = [
        ({'grandparent': [(0, 0), (0, 1), (1, 1)]},
         {'parent': [(0, 0)]}),
    ]
    dummy_negative = [
        ({'grandparent': [(0, 1)]},
         {'parent': []}),
    ]

    # Generate trajectory
    trajectory, reward = trainer.generate_trajectory(
        initial_state, dummy_positive, dummy_negative
    )

    if not trajectory:
        print("   ⚠️  Could not generate trajectory for testing")
        return True  # Can't test but configuration looks OK

    # Compute loss
    loss = trainer.compute_trajectory_balance_loss(trajectory, reward)

    # Backward
    trainer.optimizer.zero_grad()
    loss.backward()

    # Check if encoder parameters have gradients
    params_with_grad = 0
    grad_norms = []

    for i, param in enumerate(encoder_params):
        if param.grad is not None:
            params_with_grad += 1
            grad_norms.append(param.grad.norm().item())

    print(f"   - {params_with_grad}/{len(encoder_params)} encoder params received gradients")

    if params_with_grad == 0:
        print("   ❌ CRITICAL: No gradients flowing through encoder!")
        return False
    elif params_with_grad < len(encoder_params):
        print(f"   ⚠️  WARNING: Only {params_with_grad}/{len(encoder_params)} params got gradients")
        print(f"      (This might be OK if some layers aren't used)")
    else:
        print("   ✅ All encoder parameters received gradients")

    # Check gradient magnitudes
    if grad_norms:
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)

        print(f"\n   Gradient magnitude statistics:")
        print(f"   - Mean: {avg_grad:.6f}")
        print(f"   - Max:  {max_grad:.6f}")
        print(f"   - Min:  {min_grad:.6f}")

        if avg_grad < 1e-8:
            print("   ⚠️  WARNING: Gradients are very small (might be vanishing)")
        elif avg_grad > 1e2:
            print("   ⚠️  WARNING: Gradients are very large (might be exploding)")
        else:
            print("   ✅ Gradient magnitudes look reasonable")

    # Apply optimizer step and check if parameters actually change
    print("\n4. Testing if optimizer actually updates encoder parameters...")

    trainer.optimizer.step()

    # Check if parameters changed
    params_changed = 0
    max_change = 0.0

    for orig, current in zip(original_values, encoder_params):
        diff = (current - orig).abs().max().item()
        if diff > 1e-10:  # Allow for numerical precision
            params_changed += 1
            max_change = max(max_change, diff)

    print(f"   - {params_changed}/{len(encoder_params)} encoder params changed after optimizer.step()")
    print(f"   - Max parameter change: {max_change:.6e}")

    if params_changed == 0:
        print("   ❌ CRITICAL: Optimizer didn't update any encoder parameters!")
        return False
    elif params_changed < len(encoder_params) * 0.5:
        print(f"   ⚠️  WARNING: Less than half of params changed")
    else:
        print("   ✅ Encoder parameters are being updated")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    if params_with_grad > 0 and params_changed > 0:
        print("\n✅ SUCCESS: GNN encoder IS being trained during GFlowNet training!")
        print("\nSummary:")
        print(f"  • Encoder has {len(encoder_params)} parameters")
        print(f"  • {encoder_params_in_opt} params in optimizer")
        print(f"  • {params_with_grad} params receive gradients")
        print(f"  • {params_changed} params updated by optimizer")
        print(f"  • Average gradient: {avg_grad:.6e}")
        return True
    else:
        print("\n❌ PROBLEM: GNN encoder is NOT being properly trained!")
        print("\nThis means pre-training will be wasted. Check your setup.")
        return False


def compare_encoder_before_after_training(trainer, num_steps=10):
    """
    Compare encoder outputs before and after a few training steps.
    If encoder is being trained, outputs should change.
    """
    print("\n" + "=" * 80)
    print("TESTING ENCODER CHANGE OVER TRAINING")
    print("=" * 80)

    from src.logic_structures import get_initial_state

    # Create test state
    test_state = get_initial_state('grandparent', 2)

    # Get initial embedding
    graph_data = trainer.graph_constructor.theory_to_graph(test_state)
    with torch.no_grad():
        initial_emb, _ = trainer.state_encoder(graph_data)
        initial_emb = initial_emb.clone()

    print(f"\nInitial embedding (first 5 dims): {initial_emb[0, :5].tolist()}")

    # Create dummy examples
    from src.logic_structures import Rule, Atom, Variable
    dummy_positive = [
        ({'grandparent': [(0, 0), (0, 1), (1, 1)]},
         {'parent': [(0, 0)]}),
    ]
    dummy_negative = [
        ({'grandparent': [(0, 1)]},
         {'parent': []}),
    ]

    # Do a few training steps
    print(f"\nRunning {num_steps} training steps...")

    for i in range(num_steps):
        metrics = trainer.train_step(test_state, dummy_positive, dummy_negative)
        if i % 3 == 0:
            print(f"  Step {i}: loss={metrics.get('loss', 0):.4f}")

    # Get final embedding
    with torch.no_grad():
        final_emb, _ = trainer.state_encoder(graph_data)

    print(f"\nFinal embedding (first 5 dims):   {final_emb[0, :5].tolist()}")

    # Compare
    diff = (final_emb - initial_emb).abs().mean().item()
    max_diff = (final_emb - initial_emb).abs().max().item()

    print(f"\nEmbedding change:")
    print(f"  - Mean absolute difference: {diff:.6e}")
    print(f"  - Max absolute difference:  {max_diff:.6e}")

    if diff > 1e-6:
        print(f"\n✅ Encoder embeddings CHANGED during training (good!)")
    else:
        print(f"\n❌ Encoder embeddings did NOT change (encoder not being trained!)")

    return diff > 1e-6


if __name__ == "__main__":
    print("This script should be run from a notebook or script that has created a trainer.")
    print("\nUsage:")
    print("  from verify_gnn_training import verify_encoder_is_trainable")
    print("  verify_encoder_is_trainable(trainer)")
