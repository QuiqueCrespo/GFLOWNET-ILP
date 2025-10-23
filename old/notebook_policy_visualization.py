"""
Notebook Integration: Policy Convergence Visualization

Add this to your Demo_ILP.ipynb to visualize forward and backward policy convergence.
"""

# ============================================================================
# CELL 1: Initialize Policy Visualizer (add BEFORE training loop)
# ============================================================================

from src.policy_convergence_visualization import PolicyConvergenceVisualizer

print("="*80)
print("INITIALIZING POLICY CONVERGENCE VISUALIZER")
print("="*80)

# Initialize visualizer
policy_viz = PolicyConvergenceVisualizer(
    trainer=trainer,
    target_predicate='grandparent',
    arity=2,
    predicate_vocab=predicate_vocab,
    predicate_arities=predicate_arities,
    output_dir=f"{visualizer.run_dir}/policy_viz"
)

print(f"\nPolicy Visualizer Configuration:")
print(f"  Test states: {len(policy_viz.test_states)}")
print(f"  Output directory: {policy_viz.output_dir}")
print("\nThis will track how forward and backward policies converge!")
print("="*80 + "\n")


# ============================================================================
# CELL 2: Add Recording to Training Loop
# ============================================================================

# Training
num_episodes = config['num_episodes']
initial_state = get_initial_state('grandparent', 2)

print(f"\n" + "="*80)
print(f"TRAINING ({num_episodes} episodes)")
print("="*80)

# Record initial policies (episode 0)
print("\nRecording initial policy snapshot (episode 0)...")
policy_viz.record_snapshot(episode=0)
print("  Done!\n")

rewards = []
discovered_rules = {}
recent_rules = []

for episode in range(num_episodes):
    metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

    if metrics:
        rewards.append(metrics['reward'])

        # ... existing visualization code ...

        # RECORD POLICY SNAPSHOTS periodically
        if episode % 100 == 0 and episode > 0:
            print(f"\n  📊 Recording policy snapshot at episode {episode}...")
            policy_viz.record_snapshot(episode=episode)

        if episode % 100 == 0 and recent_rules:
            mean_reward = np.mean(rewards[-100:])
            print(f"\n--- Episode {episode:4d}: Mean Reward (last 100) = {mean_reward:.4f} ---")

# Record final policy snapshot
print(f"\n\n📊 Recording final policy snapshot (episode {num_episodes})...")
policy_viz.record_snapshot(episode=num_episodes)
print("  Done!\n")


# ============================================================================
# CELL 3: Generate Policy Visualizations (add AFTER training)
# ============================================================================

print("\n" + "="*80)
print("GENERATING POLICY CONVERGENCE VISUALIZATIONS")
print("="*80)

print("\n1. Generating forward strategist convergence plot...")
policy_viz.plot_strategist_convergence()

print("\n2. Generating backward strategist convergence plot...")
policy_viz.plot_backward_strategist_convergence()

print("\n3. Generating policy consistency analysis...")
policy_viz.plot_policy_consistency()

print("\n4. Generating atom adder convergence plot...")
policy_viz.plot_atom_adder_convergence()

print("\n5. Generating variable unifier entropy plot...")
policy_viz.plot_variable_unifier_entropy()

print("\n6. Generating comprehensive policy dashboard...")
policy_viz.plot_comprehensive_dashboard()

print("\n7. Generating policy convergence report...")
policy_viz.generate_report()

print("\n" + "="*80)
print("POLICY VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {policy_viz.output_dir}/")
print("\nGenerated files:")
print("  - forward_strategist_convergence.png   : How forward policy evolves")
print("  - backward_strategist_convergence.png  : How backward policy evolves")
print("  - policy_consistency.png               : Forward vs Backward comparison")
print("  - atom_adder_convergence.png           : Predicate selection evolution")
print("  - variable_unifier_entropy.png         : Variable pair selection entropy")
print("  - policy_dashboard.png                 : Comprehensive overview")
print("  - policy_convergence_report.txt        : Detailed text report")
print("\nKey metrics to check:")
print("  1. Do forward/backward policies converge? (divergence < 0.1)")
print("  2. Does strategist become more deterministic over time?")
print("  3. Is atom adder learning predicate preferences?")
print("  4. Is variable unifier entropy decreasing (becoming more deterministic)?")
print("="*80 + "\n")


# ============================================================================
# OPTIONAL: Quick Policy Check (run anytime)
# ============================================================================

def quick_policy_check():
    """Quick check of current policy state."""
    import torch
    import torch.nn.functional as F
    from src.logic_structures import get_initial_state, apply_add_atom, get_all_variables

    print("\n" + "="*60)
    print("QUICK POLICY CHECK")
    print("="*60)

    # Test on initial state + 1 atom
    state = get_initial_state('grandparent', 2)
    state, _ = apply_add_atom(state, 'parent', 2, 1)

    with torch.no_grad():
        # Get embeddings
        graph = trainer.graph_constructor.theory_to_graph(state)
        state_emb, node_embs = trainer.state_encoder(graph)
        state_emb = state_emb.squeeze(0)

        # Forward strategist
        fwd_logits = trainer.gflownet.forward_strategist(state_emb)
        fwd_probs = F.softmax(fwd_logits, dim=-1).numpy()

        # Backward strategist
        back_logits = trainer.gflownet.forward_backward_policy(state_emb)
        back_probs = F.softmax(back_logits, dim=-1).numpy()

        print("\nForward Strategist:")
        print(f"  P(ADD_ATOM):        {fwd_probs[0]:.4f}")
        print(f"  P(UNIFY_VARIABLES): {fwd_probs[1]:.4f}")
        print(f"  P(TERMINATE):       {fwd_probs[2]:.4f}")

        print("\nBackward Strategist:")
        print(f"  P(ADD_ATOM):        {back_probs[0]:.4f}")
        print(f"  P(UNIFY_VARIABLES): {back_probs[1]:.4f}")

        print("\nDivergence:")
        print(f"  |P_F(ADD) - P_B(ADD)|:    {abs(fwd_probs[0] - back_probs[0]):.4f}")
        print(f"  |P_F(UNIFY) - P_B(UNIFY)|: {abs(fwd_probs[1] - back_probs[1]):.4f}")

        if abs(fwd_probs[0] - back_probs[0]) < 0.1 and abs(fwd_probs[1] - back_probs[1]) < 0.1:
            print("\n  ✓ Policies are consistent (divergence < 0.1)")
        else:
            print("\n  ⚠️  Policies are diverging (divergence >= 0.1)")

    print("="*60 + "\n")

# To use: call quick_policy_check() anytime during/after training
# quick_policy_check()
