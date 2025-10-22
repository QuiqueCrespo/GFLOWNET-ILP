"""
Quick verification cell for Jupyter notebook.

Copy this into a cell AFTER creating the trainer to verify GNN is being trained.
"""

# ============================================================================
# VERIFY GNN IS BEING TRAINED
# ============================================================================

print("=" * 80)
print("VERIFYING GNN ENCODER IS TRAINABLE")
print("=" * 80)

# 1. Check optimizer contains encoder parameters
print("\n1. Checking optimizer configuration...")

optimizer_params = set(trainer.optimizer.param_groups[0]['params'])
encoder_params = list(trainer.state_encoder.parameters())
gflownet_params = list(trainer.gflownet.parameters())

encoder_in_opt = sum(1 for p in encoder_params if p in optimizer_params)
gflownet_in_opt = sum(1 for p in gflownet_params if p in optimizer_params)

print(f"   - State encoder: {len(encoder_params)} params, {encoder_in_opt} in optimizer")
print(f"   - GFlowNet:      {len(gflownet_params)} params, {gflownet_in_opt} in optimizer")
print(f"   - Total in optimizer: {len(optimizer_params)} params")

if encoder_in_opt == len(encoder_params):
    print("   ✅ All encoder parameters are in optimizer")
else:
    print(f"   ❌ WARNING: Only {encoder_in_opt}/{len(encoder_params)} encoder params in optimizer!")

# 2. Check all params require gradients
print("\n2. Checking gradient requirements...")

encoder_trainable = sum(1 for p in encoder_params if p.requires_grad)
gflownet_trainable = sum(1 for p in gflownet_params if p.requires_grad)

print(f"   - Encoder: {encoder_trainable}/{len(encoder_params)} require grad")
print(f"   - GFlowNet: {gflownet_trainable}/{len(gflownet_params)} require grad")

if encoder_trainable == len(encoder_params):
    print("   ✅ All encoder parameters are trainable")
else:
    print(f"   ❌ WARNING: Only {encoder_trainable}/{len(encoder_params)} encoder params trainable!")

# 3. Test gradient flow
print("\n3. Testing gradient flow with one training step...")

# Store initial parameter values
initial_encoder_params = [p.clone().detach() for p in encoder_params]
initial_gflownet_params = [p.clone().detach() for p in gflownet_params]

# Do one training step
metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

# Check if parameters changed
encoder_changed = sum(
    1 for orig, curr in zip(initial_encoder_params, encoder_params)
    if (curr - orig).abs().max().item() > 1e-10
)

gflownet_changed = sum(
    1 for orig, curr in zip(initial_gflownet_params, gflownet_params)
    if (curr - orig).abs().max().item() > 1e-10
)

print(f"   - Encoder: {encoder_changed}/{len(encoder_params)} params changed")
print(f"   - GFlowNet: {gflownet_changed}/{len(gflownet_params)} params changed")
print(f"   - Training loss: {metrics.get('loss', 0):.4f}")

if encoder_changed > 0:
    print("   ✅ Encoder parameters ARE being updated!")
else:
    print("   ❌ WARNING: Encoder parameters NOT changing!")

# 4. Calculate max parameter changes
print("\n4. Magnitude of parameter updates...")

encoder_max_change = max(
    (curr - orig).abs().max().item()
    for orig, curr in zip(initial_encoder_params, encoder_params)
)

gflownet_max_change = max(
    (curr - orig).abs().max().item()
    for orig, curr in zip(initial_gflownet_params, gflownet_params)
)

print(f"   - Encoder max change:  {encoder_max_change:.6e}")
print(f"   - GFlowNet max change: {gflownet_max_change:.6e}")

if encoder_max_change > 1e-10:
    print("   ✅ Encoder is being updated by optimizer")
else:
    print("   ❌ WARNING: Encoder updates too small or zero!")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_good = (
    encoder_in_opt == len(encoder_params) and
    encoder_trainable == len(encoder_params) and
    encoder_changed > 0 and
    encoder_max_change > 1e-10
)

if all_good:
    print("\n✅ SUCCESS: GNN encoder IS being trained during GFlowNet training!")
    print("\nWhat this means:")
    print("  • Contrastive pre-training initializes encoder with good representations")
    print("  • GFlowNet training continues to refine those representations")
    print("  • Encoder and policy co-evolve to maximize task-specific rewards")
    print("\nThis is the CORRECT setup!")
else:
    print("\n❌ PROBLEM: GNN encoder may NOT be properly trained!")
    print("\nPlease check:")
    print("  • Optimizer configuration in training.py")
    print("  • Whether encoder was accidentally frozen")
    print("  • Learning rate might be too small")

print("=" * 80)
