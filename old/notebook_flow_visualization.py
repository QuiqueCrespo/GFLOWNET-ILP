"""
Notebook Integration: Flow Visualization

Add this to your Demo_ILP.ipynb to visualize flow evolution during training.

Usage:
1. Add initialization cell BEFORE training loop
2. Add recording calls DURING training loop
3. Add visualization cell AFTER training completes
"""

# ============================================================================
# CELL 1: Initialize Flow Visualizer (add BEFORE training loop)
# ============================================================================

from src.flow_visualization import FlowVisualizer

print("="*80)
print("INITIALIZING FLOW VISUALIZER")
print("="*80)

# Initialize visualizer
flow_viz = FlowVisualizer(
    trainer=trainer,
    target_predicate='grandparent',
    arity=2,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    positive_examples=positive_examples,
    negative_examples=negative_examples,
    max_depth=4,  # Explore states 4 steps from origin
    output_dir=f"{visualizer.run_dir}/flow_viz"
)

print(f"\nFlow Visualizer Configuration:")
print(f"  Target depth: {flow_viz.max_depth}")
print(f"  States to track: {len(flow_viz.target_states)}")
print(f"  Output directory: {flow_viz.output_dir}")
print("\nFlow tracking will help diagnose if the model is learning correct flow values!")
print("="*80 + "\n")


# ============================================================================
# CELL 2: Modify Training Loop (REPLACE your existing training loop)
# ============================================================================

# Training
num_episodes = config['num_episodes']
initial_state = get_initial_state('grandparent', 2)

print(f"\n" + "="*80)
print(f"TRAINING ({num_episodes} episodes)")
print("="*80)

# Record initial flow predictions (episode 0, before any training)
print("\nRecording initial flow snapshot (episode 0)...")
flow_viz.record_snapshot(episode=0)
print("  Done!\n")

rewards = []
discovered_rules = {}  # Rule string -> (reward, episode, scores)
recent_rules = []  # Track last 50 rules for analysis

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

    if metrics:
        rewards.append(metrics['reward'])

        # Record metrics with visualizer
        visualizer.record_episode(episode, metrics)

        # Sample trajectories periodically to see what rules are being found
        if episode % 10 == 0:
            trajectory, reward = trainer.generate_trajectory(
                initial_state, positive_examples, negative_examples
            )
            theory = trajectory[-1].next_state if trajectory else initial_state
            rule_str = theory_to_string(theory)

            scores = reward_calc.get_detailed_scores(theory, positive_examples, negative_examples)

            # Record with visualizer
            visualizer.record_rule(rule_str, reward, episode, scores)

            # Add detailed metrics to visualizer
            visualizer.record_episode(episode, {
                **metrics,
                'precision': scores['precision'],
                'recall': scores['recall'],
                'f1_score': scores['f1_score'],
                'accuracy': scores['accuracy']
            })

            discovered_rules[rule_str] = (reward, episode, scores)

            recent_rules.append((rule_str, reward, episode, scores))
            if len(recent_rules) > 100:
                recent_rules.pop(0)

        # RECORD FLOW SNAPSHOTS periodically
        if episode % 100 == 0 and episode > 0:
            print(f"\n  📊 Recording flow snapshot at episode {episode}...")
            flow_viz.record_snapshot(episode=episode)

        if episode % 100 == 0 and recent_rules:
            mean_reward = np.mean(rewards[-100:])
            print(f"\n--- Episode {episode:4d}: Mean Reward (last 100 episodes) = {mean_reward:.4f} ---")
            latest_rule, latest_reward, _, _ = recent_rules[-1]
            print(f"Episode {episode:4d}: reward={metrics['reward']:.4f}, length={metrics['trajectory_length']}")
            print(f"  Latest sampled rule: {latest_rule}")

# Record final flow snapshot
print(f"\n\n📊 Recording final flow snapshot (episode {num_episodes})...")
flow_viz.record_snapshot(episode=num_episodes)
print("  Done!\n")


# ============================================================================
# CELL 3: Generate Flow Visualizations (add AFTER training completes)
# ============================================================================

print("\n" + "="*80)
print("GENERATING FLOW VISUALIZATIONS")
print("="*80)

print("\n1. Generating comprehensive flow evolution plot...")
flow_viz.plot_flow_evolution()

print("\n2. Generating individual state trajectories...")
flow_viz.plot_state_trajectories(num_states=15)

print("\n3. Generating flow analysis report...")
flow_viz.generate_report()

print("\n" + "="*80)
print("FLOW VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {flow_viz.output_dir}/")
print("\nGenerated files:")
print(f"  - flow_evolution_depth{flow_viz.max_depth}.png       : Main analysis dashboard")
print(f"  - state_trajectories_depth{flow_viz.max_depth}.png   : Individual state flow trajectories")
print(f"  - flow_report_depth{flow_viz.max_depth}.txt          : Detailed text report")
print("\nKey metrics to check:")
print("  1. Flow-Reward Correlation (should increase over training)")
print("  2. Flow values for top states (should be higher than bottom states)")
print("  3. Correlation evolution (target: >0.5 by end of training)")
print("="*80 + "\n")


# ============================================================================
# OPTIONAL: Diagnostic Cell (add if flow learning seems broken)
# ============================================================================

print("="*80)
print("FLOW LEARNING DIAGNOSTICS")
print("="*80)

# Check log_Z value
print(f"\nlog_Z value: {trainer.log_Z.item():.4f}")
if trainer.log_Z.item() > 10.0:
    print("  ⚠️  WARNING: log_Z is very large (>10), may be compensating instead of learning")
elif trainer.log_Z.item() < -10.0:
    print("  ⚠️  WARNING: log_Z is very negative (<-10), may indicate numerical issues")
else:
    print("  ✓ log_Z in reasonable range")

# Check if forward_flow network has gradients
print("\nForward flow network gradient norms:")
has_gradients = False
for name, param in trainer.gflownet.forward_flow_net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: {grad_norm:.6f}")
        if grad_norm > 1e-6:
            has_gradients = True
    else:
        print(f"  {name}: NO GRADIENT ❌")

if has_gradients:
    print("\n  ✓ Flow network is receiving gradients")
else:
    print("\n  ❌ WARNING: Flow network has no gradients or very small gradients")
    print("     This means the flow network is not learning!")
    print("     Possible causes:")
    print("       - Using Trajectory Balance with log_Z compensating")
    print("       - Learning rate too small")
    print("       - Gradients vanishing")

# Check reward scaling
print(f"\nReward scaling alpha: {trainer.reward_scale_alpha}")
if trainer.reward_scale_alpha > 5.0:
    print("  ⚠️  WARNING: Large alpha may cause numerical issues")
    print(f"     Example: reward=0.8 → scaled=(0.8)^{trainer.reward_scale_alpha} = {0.8**trainer.reward_scale_alpha:.2e}")
    print("     Consider reducing alpha to 1.0 or 2.0")
else:
    print("  ✓ Reward scaling in reasonable range")

# Check recent flow correlation
if len(flow_viz.episode_snapshots) >= 2:
    final_snap = flow_viz.episode_snapshots[-1]
    flows = list(final_snap['flows'].values())
    rewards = list(final_snap['rewards'].values())
    correlation = np.corrcoef(rewards, flows)[0, 1]

    print(f"\nFinal flow-reward correlation: {correlation:.4f}")
    if correlation > 0.5:
        print("  ✓ Strong correlation - flow learning is working!")
    elif correlation > 0.3:
        print("  ⚠️  Moderate correlation - some learning, but could be better")
    else:
        print("  ❌ Weak correlation - flow learning is NOT working properly")
        print("     Review diagnostics above to identify the issue")

print("="*80 + "\n")


# ============================================================================
# OPTIONAL: Quick Flow Check Function (run anytime during debugging)
# ============================================================================

def quick_flow_check():
    """Quick diagnostic to check if flow learning is working."""
    import torch
    from src.logic_structures import get_initial_state

    print("\n" + "="*60)
    print("QUICK FLOW CHECK")
    print("="*60)

    # Get a few test states
    test_states = [
        get_initial_state('grandparent', 2),  # Initial state
        flow_viz.target_states[0][0] if flow_viz.target_states else None,  # A depth-4 state
    ]

    for i, state in enumerate(test_states):
        if state is None:
            continue

        # Get flow prediction
        with torch.no_grad():
            graph_data = trainer.graph_constructor.theory_to_graph(state)
            state_embedding, _ = trainer.state_encoder(graph_data)
            state_embedding = state_embedding.squeeze(0)
            log_F = trainer.gflownet.forward_flow(state_embedding)

        # Get reward
        reward = trainer.reward_calculator.calculate_reward(
            state, positive_examples, negative_examples
        )

        # Display
        state_str = theory_to_string(state)
        print(f"\nState {i+1}: {state_str[:60]}...")
        print(f"  Predicted log F(s): {log_F.item():+.4f}")
        print(f"  Actual reward R(s): {reward:.4f}")

    print("="*60 + "\n")

# To use: call quick_flow_check() anytime
# quick_flow_check()
