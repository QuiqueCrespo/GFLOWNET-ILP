"""
Diagnostic script to detect replay buffer overfitting.

This script checks if the model is:
1. Learning replayed trajectories well (low loss on buffer)
2. But failing to generalize (low reward on new samples)

Run this during or after training to diagnose the issue.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from src.training import GFlowNetTrainer
from src.logic_structures import Theory
from src.logic_engine import Example


def diagnose_replay_overfitting(
    trainer: GFlowNetTrainer,
    initial_state: Theory,
    pos_examples: List[Example],
    neg_examples: List[Example],
    num_samples: int = 10
) -> Dict:
    """
    Check if model is overfitting to replay buffer.

    Args:
        trainer: The GFlowNet trainer
        initial_state: Starting state for trajectories
        pos_examples: Positive examples
        neg_examples: Negative examples
        num_samples: Number of on-policy samples to generate

    Returns:
        Dictionary with diagnostic metrics
    """

    print(f"\n{'='*70}")
    print(f"REPLAY BUFFER OVERFITTING DIAGNOSTIC")
    print(f"{'='*70}\n")

    # --- 1. Replay Buffer Statistics ---
    print(f"📊 Replay Buffer Statistics:")
    print(f"   Buffer size: {len(trainer.replay_buffer) if trainer.replay_buffer else 0}")
    print(f"   Buffer capacity: {trainer.replay_buffer.capacity if trainer.replay_buffer else 'N/A'}")
    print(f"   Replay probability: {trainer.replay_probability}")
    print(f"   Reward threshold: {trainer.buffer_reward_threshold}")

    if not trainer.replay_buffer or len(trainer.replay_buffer) == 0:
        print(f"\n⚠️  Replay buffer is empty - cannot diagnose overfitting")
        return {}

    # --- 2. Sample from Replay Buffer ---
    print(f"\n🔄 Sampling from Replay Buffer...")

    buffer_losses = []
    buffer_rewards = []

    for _ in range(min(10, len(trainer.replay_buffer))):
        buffer_traj, buffer_reward = trainer.replay_buffer.sample(1)[0]

        # Recompute loss with current model
        buffer_loss = trainer.compute_trajectory_balance_loss(buffer_traj, buffer_reward)

        buffer_losses.append(buffer_loss.item())
        buffer_rewards.append(buffer_reward)

    avg_buffer_loss = np.mean(buffer_losses)
    avg_buffer_reward = np.mean(buffer_rewards)
    std_buffer_loss = np.std(buffer_losses)
    std_buffer_reward = np.std(buffer_rewards)

    print(f"   Avg buffer loss:   {avg_buffer_loss:.4f} ± {std_buffer_loss:.4f}")
    print(f"   Avg buffer reward: {avg_buffer_reward:.4f} ± {std_buffer_reward:.4f}")

    # --- 3. Generate On-Policy Samples ---
    print(f"\n🎲 Generating On-Policy Samples (n={num_samples})...")

    onpolicy_losses = []
    onpolicy_rewards = []
    trajectory_lengths = []

    for i in range(num_samples):
        traj, reward = trainer.generate_trajectory(
            initial_state, pos_examples, neg_examples, stochastic=True
        )

        if traj:
            loss = trainer.compute_trajectory_balance_loss(traj, reward)
            onpolicy_losses.append(loss.item())
            onpolicy_rewards.append(reward)
            trajectory_lengths.append(len(traj))

    if not onpolicy_losses:
        print(f"   ⚠️  No valid on-policy trajectories generated")
        return {}

    avg_onpolicy_loss = np.mean(onpolicy_losses)
    avg_onpolicy_reward = np.mean(onpolicy_rewards)
    std_onpolicy_loss = np.std(onpolicy_losses)
    std_onpolicy_reward = np.std(onpolicy_rewards)

    print(f"   Avg on-policy loss:   {avg_onpolicy_loss:.4f} ± {std_onpolicy_loss:.4f}")
    print(f"   Avg on-policy reward: {avg_onpolicy_reward:.4f} ± {std_onpolicy_reward:.4f}")
    print(f"   Avg trajectory length: {np.mean(trajectory_lengths):.1f}")

    # --- 4. Compute Gaps ---
    loss_gap = avg_onpolicy_loss - avg_buffer_loss
    reward_gap = avg_buffer_reward - avg_onpolicy_reward

    print(f"\n📈 Performance Gaps:")
    print(f"   Loss gap (on-policy - buffer):     {loss_gap:+.4f}")
    print(f"   Reward gap (buffer - on-policy):   {reward_gap:+.4f}")

    # --- 5. Diagnosis ---
    print(f"\n🔍 Diagnosis:")

    overfitting_detected = False
    issues = []

    # Check 1: Low buffer loss but high on-policy loss
    if avg_buffer_loss < 0.5 and avg_onpolicy_loss > 2.0:
        overfitting_detected = True
        issues.append("Low loss on buffer but high loss on new samples")
        print(f"   ⚠️  ISSUE 1: Model has learned buffer trajectories well")
        print(f"               but struggles with new trajectories")

    # Check 2: High buffer reward but low on-policy reward
    if avg_buffer_reward > 0.6 and avg_onpolicy_reward < 0.3:
        overfitting_detected = True
        issues.append("High reward in buffer but low reward on new samples")
        print(f"   ⚠️  ISSUE 2: Buffer contains high-reward trajectories")
        print(f"               but model doesn't sample similar ones")

    # Check 3: Large reward gap
    if reward_gap > 0.4:
        overfitting_detected = True
        issues.append(f"Large reward gap ({reward_gap:.2f})")
        print(f"   ⚠️  ISSUE 3: Significant reward gap between buffer and on-policy")
        print(f"               Model may be memorizing rather than generalizing")

    # Check 4: Small buffer size
    if len(trainer.replay_buffer) < 100:
        issues.append(f"Small buffer size ({len(trainer.replay_buffer)})")
        print(f"   ⚠️  ISSUE 4: Replay buffer is small (size={len(trainer.replay_buffer)})")
        print(f"               May not cover enough of state space")

    # Check 5: High replay probability
    if trainer.replay_probability > 0.4:
        issues.append(f"High replay probability ({trainer.replay_probability})")
        print(f"   ⚠️  ISSUE 5: High replay probability ({trainer.replay_probability})")
        print(f"               Training may be dominated by buffer")

    if not overfitting_detected:
        print(f"   ✅ No significant overfitting detected")
    else:
        print(f"\n🚨 OVERFITTING TO REPLAY BUFFER DETECTED!")

    # --- 6. Recommendations ---
    if overfitting_detected:
        print(f"\n💡 Recommendations:")

        if len(trainer.replay_buffer) < 100:
            print(f"   1. Increase buffer capacity:")
            print(f"      replay_buffer_capacity = 500  # Currently: {trainer.replay_buffer.capacity}")

        if trainer.replay_probability > 0.3:
            print(f"   2. Decrease replay probability:")
            print(f"      replay_probability = 0.1  # Currently: {trainer.replay_probability}")

        if trainer.buffer_reward_threshold > 0.6:
            print(f"   3. Lower reward threshold for buffer:")
            print(f"      buffer_reward_threshold = 0.5  # Currently: {trainer.buffer_reward_threshold}")

        print(f"   4. Add diversity metric when adding to buffer")
        print(f"   5. Use data augmentation on replayed trajectories")
        print(f"   6. Consider using Detailed Balance instead of TB")

    # --- 7. Additional Metrics ---
    print(f"\n📊 Additional Metrics:")
    print(f"   log_Z value: {trainer.log_Z.item():.4f}")

    # Estimate state space coverage
    if trainer.replay_buffer:
        total_states_in_buffer = sum(
            len(traj) for traj, _ in trainer.replay_buffer.buffer
        )
        print(f"   Estimated states in buffer: {total_states_in_buffer}")
        print(f"   Avg states per trajectory: {total_states_in_buffer / len(trainer.replay_buffer):.1f}")

    print(f"\n{'='*70}\n")

    # Return results
    return {
        'overfitting_detected': overfitting_detected,
        'issues': issues,
        'buffer_loss': avg_buffer_loss,
        'buffer_reward': avg_buffer_reward,
        'onpolicy_loss': avg_onpolicy_loss,
        'onpolicy_reward': avg_onpolicy_reward,
        'loss_gap': loss_gap,
        'reward_gap': reward_gap,
        'buffer_size': len(trainer.replay_buffer) if trainer.replay_buffer else 0,
        'log_Z': trainer.log_Z.item()
    }


def track_overfitting_over_training(
    trainer: GFlowNetTrainer,
    initial_state: Theory,
    pos_examples: List[Example],
    neg_examples: List[Example],
    num_episodes: int = 500,
    check_every: int = 50
) -> List[Dict]:
    """
    Track overfitting metrics during training.

    Returns:
        List of diagnostic results at checkpoints
    """

    history = []

    for episode in range(num_episodes):
        # Regular training step
        metrics = trainer.train_step(initial_state, pos_examples, neg_examples)

        # Periodic diagnostic
        if episode % check_every == 0 or episode == num_episodes - 1:
            print(f"\n--- Episode {episode} ---")

            diagnostic = diagnose_replay_overfitting(
                trainer, initial_state, pos_examples, neg_examples, num_samples=5
            )

            diagnostic['episode'] = episode
            diagnostic['train_metrics'] = metrics

            history.append(diagnostic)

    return history


def plot_overfitting_trends(history: List[Dict], save_path: Optional[str] = None):
    """
    Plot overfitting trends over training.

    Args:
        history: List of diagnostic results from track_overfitting_over_training
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - cannot plot")
        return

    episodes = [h['episode'] for h in history]
    buffer_rewards = [h['buffer_reward'] for h in history if 'buffer_reward' in h]
    onpolicy_rewards = [h['onpolicy_reward'] for h in history if 'onpolicy_reward' in h]
    reward_gaps = [h['reward_gap'] for h in history if 'reward_gap' in h]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Rewards over time
    axes[0, 0].plot(episodes, buffer_rewards, 'b-', label='Buffer Reward', linewidth=2)
    axes[0, 0].plot(episodes, onpolicy_rewards, 'r-', label='On-Policy Reward', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Buffer vs On-Policy Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Reward gap over time
    axes[0, 1].plot(episodes, reward_gaps, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.4, color='r', linestyle='--', label='Warning threshold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward Gap')
    axes[0, 1].set_title('Generalization Gap (Buffer - On-Policy)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Buffer size over time
    buffer_sizes = [h['buffer_size'] for h in history]
    axes[1, 0].plot(episodes, buffer_sizes, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Buffer Size')
    axes[1, 0].set_title('Replay Buffer Size')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: log_Z over time
    log_Zs = [h['log_Z'] for h in history if 'log_Z' in h]
    axes[1, 1].plot(episodes, log_Zs, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('log_Z')
    axes[1, 1].set_title('Log Partition Function')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Replay Buffer Overfitting Diagnostic Tool")
    print("=========================================\n")
    print("This script diagnoses if your GFlowNet model is overfitting to the replay buffer.")
    print("\nUsage:")
    print("  from diagnose_replay_overfitting import diagnose_replay_overfitting")
    print("  results = diagnose_replay_overfitting(trainer, initial_state, pos_ex, neg_ex)")
    print("\nOr track during training:")
    print("  from diagnose_replay_overfitting import track_overfitting_over_training")
    print("  history = track_overfitting_over_training(trainer, initial_state, pos_ex, neg_ex)")
