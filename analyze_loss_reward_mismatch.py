"""
Analyze the mismatch between loss minimization and reward optimization.

Diagnoses the "zero flow problem" where the model learns to assign
low/zero flow everywhere to minimize loss, but doesn't find good rules.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class LossRewardAnalyzer:
    """Analyzes the relationship between loss and reward during training."""

    def __init__(self, trainer):
        self.trainer = trainer

    def analyze_flow_values(self, states, num_samples=20):
        """
        Analyze flow values (log_Z, log_pf) across different states.

        Checks for "zero flow problem" where all flows converge to low values.
        """
        print("=" * 80)
        print("FLOW VALUE ANALYSIS")
        print("=" * 80)

        log_pf_values = []
        log_Z_value = self.trainer.log_Z.item()

        print(f"\nLearned log Z (partition function): {log_Z_value:.4f}")
        print(f"  → Z = exp(log Z) = {np.exp(log_Z_value):.4f}")

        print(f"\nSampling {num_samples} trajectories and analyzing flow values...")

        for i in range(num_samples):
            trajectory, reward = self.trainer.generate_trajectory(
                states['initial'],
                states['positives'],
                states['negatives']
            )

            if trajectory:
                # Collect log_pf for each step
                for step in trajectory:
                    log_pf_values.append(step.log_pf.item())

        if not log_pf_values:
            print("\n✗ No trajectories generated!")
            return

        log_pf_values = np.array(log_pf_values)

        print(f"\nLog P_F (forward probability) statistics:")
        print(f"  - Mean:   {np.mean(log_pf_values):.4f}")
        print(f"  - Std:    {np.std(log_pf_values):.4f}")
        print(f"  - Min:    {np.min(log_pf_values):.4f}")
        print(f"  - Max:    {np.max(log_pf_values):.4f}")
        print(f"  - Median: {np.median(log_pf_values):.4f}")

        # Check for zero flow problem
        mean_log_pf = np.mean(log_pf_values)
        if mean_log_pf < -10:
            print("\n⚠️  WARNING: Very low forward probabilities detected!")
            print("   This suggests the model may be assigning near-zero flow.")
            zero_flow_problem = True
        else:
            print("\n✓ Forward probabilities seem reasonable.")
            zero_flow_problem = False

        # Visualize distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(log_pf_values, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(mean_log_pf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_log_pf:.2f}')
        axes[0].set_xlabel('log P_F', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Forward Log Probabilities', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot
        axes[1].boxplot(log_pf_values, vert=True)
        axes[1].axhline(log_Z_value, color='blue', linestyle='--', linewidth=2, label=f'log Z: {log_Z_value:.2f}')
        axes[1].set_ylabel('log P_F', fontsize=12)
        axes[1].set_title('Forward Log Probabilities (Box Plot)', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        return fig, zero_flow_problem, {
            'log_Z': log_Z_value,
            'mean_log_pf': mean_log_pf,
            'std_log_pf': np.std(log_pf_values),
            'log_pf_values': log_pf_values
        }

    def analyze_gradient_magnitudes(self, states, num_samples=10):
        """
        Analyze gradient magnitudes to check if learning is happening.

        Very small gradients suggest the model has converged (possibly to a bad solution).
        """
        print("\n" + "=" * 80)
        print("GRADIENT MAGNITUDE ANALYSIS")
        print("=" * 80)

        # Store gradients
        gradient_norms = defaultdict(list)

        for i in range(num_samples):
            # Generate trajectory
            trajectory, reward = self.trainer.generate_trajectory(
                states['initial'],
                states['positives'],
                states['negatives']
            )

            if not trajectory:
                continue

            # Compute loss
            loss = self.trainer.compute_trajectory_balance_loss(trajectory, reward)

            # Zero gradients
            self.trainer.optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Collect gradient norms for each parameter
            for name, param in self.trainer.gflownet.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms[name].append(grad_norm)

            # Also collect for log_Z
            if self.trainer.log_Z.grad is not None:
                gradient_norms['log_Z'].append(self.trainer.log_Z.grad.norm().item())

        if not gradient_norms:
            print("\n✗ No gradients collected!")
            return

        # Analyze
        print(f"\nGradient norms (averaged over {num_samples} samples):\n")

        vanishing_grads = []
        for name, norms in gradient_norms.items():
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)

            # Truncate long parameter names
            short_name = name if len(name) <= 40 else name[:37] + '...'
            print(f"  {short_name:45s}: {mean_norm:.6f} ± {std_norm:.6f}")

            if mean_norm < 1e-6:
                vanishing_grads.append(name)

        if vanishing_grads:
            print(f"\n⚠️  WARNING: Vanishing gradients detected in {len(vanishing_grads)} parameters:")
            for name in vanishing_grads[:5]:  # Show first 5
                print(f"     - {name}")
            if len(vanishing_grads) > 5:
                print(f"     ... and {len(vanishing_grads) - 5} more")
        else:
            print("\n✓ No vanishing gradients detected.")

        return gradient_norms

    def analyze_reward_distribution(self, states, num_samples=100):
        """Analyze the distribution of rewards from sampled trajectories."""
        print("\n" + "=" * 80)
        print("REWARD DISTRIBUTION ANALYSIS")
        print("=" * 80)

        print(f"\nSampling {num_samples} trajectories...")

        rewards = []
        trajectory_lengths = []

        for i in range(num_samples):
            trajectory, reward = self.trainer.generate_trajectory(
                states['initial'],
                states['positives'],
                states['negatives']
            )
            rewards.append(reward)
            trajectory_lengths.append(len(trajectory))

        rewards = np.array(rewards)
        trajectory_lengths = np.array(trajectory_lengths)

        print(f"\nReward statistics:")
        print(f"  - Mean:     {np.mean(rewards):.4f}")
        print(f"  - Std:      {np.std(rewards):.4f}")
        print(f"  - Min:      {np.min(rewards):.4f}")
        print(f"  - Max:      {np.max(rewards):.4f}")
        print(f"  - Median:   {np.median(rewards):.4f}")
        print(f"  - % Zero:   {100 * np.sum(rewards < 1e-6) / len(rewards):.1f}%")
        print(f"  - % > 0.5:  {100 * np.sum(rewards > 0.5) / len(rewards):.1f}%")
        print(f"  - % > 0.8:  {100 * np.sum(rewards > 0.8) / len(rewards):.1f}%")

        print(f"\nTrajectory length statistics:")
        print(f"  - Mean:     {np.mean(trajectory_lengths):.2f}")
        print(f"  - Std:      {np.std(trajectory_lengths):.2f}")
        print(f"  - Min:      {np.min(trajectory_lengths)}")
        print(f"  - Max:      {np.max(trajectory_lengths)}")

        # Check for reward mismatch problem
        if np.mean(rewards) < 0.1:
            print("\n⚠️  WARNING: Very low average reward!")
            print("   The policy is not finding good rules.")
            low_reward_problem = True
        else:
            print("\n✓ Policy is finding some good rules.")
            low_reward_problem = False

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Reward distribution
        axes[0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rewards):.3f}')
        axes[0].set_xlabel('Reward', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Sampled Rewards', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Scatter: trajectory length vs reward
        axes[1].scatter(trajectory_lengths, rewards, alpha=0.5)
        axes[1].set_xlabel('Trajectory Length', fontsize=12)
        axes[1].set_ylabel('Reward', fontsize=12)
        axes[1].set_title('Reward vs Trajectory Length', fontsize=14)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        return fig, low_reward_problem, {
            'rewards': rewards,
            'trajectory_lengths': trajectory_lengths
        }

    def diagnose_zero_flow_problem(self, states, num_samples=50):
        """
        Comprehensive diagnosis of the zero flow problem.

        Returns a diagnostic report with actionable recommendations.
        """
        print("\n" + "=" * 80)
        print("ZERO FLOW PROBLEM DIAGNOSIS")
        print("=" * 80)

        # Run all analyses
        flow_fig, zero_flow, flow_stats = self.analyze_flow_values(states, num_samples)
        reward_fig, low_reward, reward_stats = self.analyze_reward_distribution(states, num_samples)
        gradient_norms = self.analyze_gradient_magnitudes(states, num_samples // 5)

        # Diagnose
        print("\n" + "=" * 80)
        print("DIAGNOSIS SUMMARY")
        print("=" * 80)

        problems = []
        if zero_flow:
            problems.append("Zero Flow Problem")
        if low_reward:
            problems.append("Low Reward Problem")

        if problems:
            print(f"\n⚠️  PROBLEMS DETECTED: {', '.join(problems)}")
        else:
            print("\n✓ No major problems detected!")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if zero_flow and low_reward:
            print("\n1. CRITICAL: Model has collapsed to zero flow everywhere")
            print("   → The loss function is minimized by assigning low probabilities")
            print("   → But this doesn't help find good rules")
            print("\n   Recommended fixes:")
            print("   a) Increase learning rate (currently using default)")
            print("   b) Use reward-weighted loss (weight high-reward trajectories more)")
            print("   c) Add entropy regularization to encourage exploration")
            print("   d) Use reward scaling: R^alpha with alpha > 1 to amplify differences")
            print("   e) Check if embedding encoder is learning (see embedding analysis)")

        elif zero_flow:
            print("\n1. Zero flow detected, but rewards are okay")
            print("   → The model may be under-confident")
            print("   → Consider adjusting temperature or using entropy regularization")

        elif low_reward:
            print("\n1. Low rewards detected, but flow values seem okay")
            print("   → The policy is exploring but not finding good solutions")
            print("   → Recommended fixes:")
            print("   a) Increase exploration (use exploration strategy)")
            print("   b) Increase max_body_length if rules are too simple")
            print("   c) Check reward function - are good rules being rewarded?")
            print("   d) Check if encoder can distinguish good from bad rules")

        else:
            print("\n✓ Training appears healthy!")
            print("  Continue monitoring loss and reward metrics.")

        print("\n" + "=" * 80)

        return {
            'zero_flow_problem': zero_flow,
            'low_reward_problem': low_reward,
            'flow_stats': flow_stats,
            'reward_stats': reward_stats,
            'gradient_norms': gradient_norms,
            'flow_fig': flow_fig,
            'reward_fig': reward_fig
        }


def main():
    """Example usage."""
    print("=" * 80)
    print("LOSS VS REWARD MISMATCH ANALYZER")
    print("=" * 80)
    print("\nThis tool diagnoses the 'zero flow problem' in GFlowNet training.")
    print("See Demo_ILP.ipynb for integration with training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
