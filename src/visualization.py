"""
Visualization tools for GFlowNet training.
Creates comprehensive plots and saves them to organized folders.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """
    Visualizes GFlowNet training progress and saves plots to disk.
    
    Creates a timestamped folder structure:
    results/
      └── run_YYYYMMDD_HHMMSS/
          ├── training_curves.png
          ├── metrics_over_time.png
          ├── confusion_matrices.png
          ├── trajectory_lengths.png
          ├── best_rules.txt
          ├── summary_dashboard.png
          └── config.json
    """
    
    def __init__(self, experiment_name: str = "gflownet_ilp", 
                 output_dir: str = "results"):
        """
        Initialize visualizer.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Base directory for saving results
        """
        self.experiment_name = experiment_name
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to: {self.run_dir}")
        
        # Storage for metrics
        self.metrics_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'trajectory_lengths': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
        }

        # Separate tracking for episodes with detailed metrics
        self.detailed_metrics_episodes = []

        self.best_rules = []  # List of (rule_str, reward, episode, scores)
        
    def record_episode(self, episode: int, metrics: Dict):
        """
        Record metrics from a training episode.

        Args:
            episode: Episode number
            metrics: Dictionary containing episode metrics
        """
        self.metrics_history['episodes'].append(episode)
        self.metrics_history['rewards'].append(metrics.get('on_policy_reward', 0))
        self.metrics_history['losses'].append(metrics.get('loss', 0))
        self.metrics_history['trajectory_lengths'].append(
            metrics.get('trajectory_length', 0)
        )

        # Optional detailed scores - track episodes separately
        has_detailed_metrics = False
        if 'precision' in metrics:
            self.metrics_history['precision'].append(metrics['precision'])
            has_detailed_metrics = True
        if 'recall' in metrics:
            self.metrics_history['recall'].append(metrics['recall'])
            has_detailed_metrics = True
        if 'f1_score' in metrics:
            self.metrics_history['f1_score'].append(metrics['f1_score'])
            has_detailed_metrics = True
        if 'accuracy' in metrics:
            self.metrics_history['accuracy'].append(metrics['accuracy'])
            has_detailed_metrics = True

        # Track episode number for detailed metrics
        if has_detailed_metrics and (not self.detailed_metrics_episodes or
                                      self.detailed_metrics_episodes[-1] != episode):
            self.detailed_metrics_episodes.append(episode)
    
    def record_rule(self, rule_str: str, reward: float, episode: int, 
                   scores: Dict):
        """Record a discovered rule with its scores."""
        self.best_rules.append((rule_str, reward, episode, scores))
    
    def plot_training_curves(self, window_size: int = 100):
        """
        Plot reward and loss curves over training.
        
        Args:
            window_size: Size of moving average window
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = np.array(self.metrics_history['episodes'])
        rewards = np.array(self.metrics_history['rewards'])
        losses = np.array(self.metrics_history['losses'])
        
        # Plot rewards
        ax1.plot(episodes, rewards, alpha=0.3, label='Raw Reward', color='blue')
        if len(rewards) >= window_size:
            smoothed_rewards = self._moving_average(rewards, window_size)
            ax1.plot(episodes[window_size-1:], smoothed_rewards, 
                    label=f'Moving Avg ({window_size})', color='darkblue', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(episodes, losses, alpha=0.3, label='Raw Loss', color='red')
        if len(losses) >= window_size:
            smoothed_losses = self._moving_average(losses, window_size)
            ax2.plot(episodes[window_size-1:], smoothed_losses,
                    label=f'Moving Avg ({window_size})', color='darkred', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves to {self.run_dir / 'training_curves.png'}")
    
    def plot_metrics_over_time(self, window_size: int = 100):
        """Plot precision, recall, F1-score over time."""
        if not self.metrics_history['precision']:
            return  # No detailed metrics recorded

        fig, ax = plt.subplots(figsize=(12, 6))

        # Use episodes where detailed metrics were recorded
        episodes = np.array(self.detailed_metrics_episodes)
        precision = np.array(self.metrics_history['precision'])
        recall = np.array(self.metrics_history['recall'])
        f1 = np.array(self.metrics_history['f1_score'])

        # Plot raw metrics
        ax.plot(episodes, precision, alpha=0.3, color='green', label='Precision (raw)')
        ax.plot(episodes, recall, alpha=0.3, color='blue', label='Recall (raw)')
        ax.plot(episodes, f1, alpha=0.3, color='purple', label='F1-Score (raw)')

        # Plot smoothed metrics
        if len(precision) >= window_size:
            ax.plot(episodes[window_size-1:],
                   self._moving_average(precision, window_size),
                   color='darkgreen', linewidth=2, label='Precision (smoothed)')
            ax.plot(episodes[window_size-1:],
                   self._moving_average(recall, window_size),
                   color='darkblue', linewidth=2, label='Recall (smoothed)')
            ax.plot(episodes[window_size-1:],
                   self._moving_average(f1, window_size),
                   color='darkviolet', linewidth=2, label='F1-Score (smoothed)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics over Training')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metrics plot to {self.run_dir / 'metrics_over_time.png'}")
    
    def plot_confusion_matrices(self, num_rules: int = 6):
        """
        Plot confusion matrices for top rules.
        
        Args:
            num_rules: Number of top rules to show
        """
        if not self.best_rules:
            return
        
        # Sort by reward and take top N
        sorted_rules = sorted(self.best_rules, key=lambda x: x[1], reverse=True)
        top_rules = sorted_rules[:num_rules]
        
        # Create subplot grid
        rows = (num_rules + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten() if num_rules > 1 else [axes]
        
        for idx, (rule_str, reward, episode, scores) in enumerate(top_rules):
            if idx >= len(axes):
                break
            
            # Create confusion matrix
            cm = np.array([
                [scores['TP'], scores['FN']],
                [scores['FP'], scores['TN']]
            ])
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Entailed', 'Not Entailed'],
                       yticklabels=['Positive', 'Negative'],
                       cbar=False)
            
            # Title with truncated rule
            short_rule = rule_str[:50] + '...' if len(rule_str) > 50 else rule_str
            axes[idx].set_title(
                f"Rule {idx+1} (R={reward:.3f}, Ep={episode})\n{short_rule}",
                fontsize=9
            )
        
        # Hide unused subplots
        for idx in range(len(top_rules), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrices to {self.run_dir / 'confusion_matrices.png'}")
    
    def plot_trajectory_lengths(self):
        """Plot distribution of trajectory lengths."""
        lengths = self.metrics_history['trajectory_lengths']
        if not lengths:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(lengths):.1f}')
        ax1.axvline(np.median(lengths), color='green', linestyle='--',
                   label=f'Median: {np.median(lengths):.1f}')
        ax1.set_xlabel('Trajectory Length')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trajectory Lengths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Over time
        episodes = self.metrics_history['episodes']
        ax2.scatter(episodes, lengths, alpha=0.3, s=10, color='skyblue')
        if len(lengths) >= 100:
            smoothed = self._moving_average(np.array(lengths), 100)
            ax2.plot(episodes[99:], smoothed, color='darkblue', linewidth=2,
                    label='Moving Avg (100)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Trajectory Length')
        ax2.set_title('Trajectory Length over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'trajectory_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved trajectory lengths to {self.run_dir / 'trajectory_lengths.png'}")
    
    def save_best_rules(self, num_rules: int = 20):
        """Save best discovered rules to text file."""
        if not self.best_rules:
            return
        
        # Sort by reward
        sorted_rules = sorted(self.best_rules, key=lambda x: x[1], reverse=True)
        
        with open(self.run_dir / 'best_rules.txt', 'w') as f:
            f.write(f"Top {num_rules} Rules Discovered\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (rule_str, reward, episode, scores) in enumerate(sorted_rules[:num_rules], 1):
                f.write(f"Rule {idx}: (Reward: {reward:.4f}, Episode: {episode})\n")
                f.write(f"{rule_str}\n")
                f.write(f"  TP={scores['TP']}, FP={scores['FP']}, "
                       f"TN={scores['TN']}, FN={scores['FN']}\n")
                f.write(f"  Precision={scores['precision']:.3f}, "
                       f"Recall={scores['recall']:.3f}, "
                       f"F1={scores['f1_score']:.3f}\n")
                f.write("\n")
        
        print(f"✓ Saved best rules to {self.run_dir / 'best_rules.txt'}")
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        episodes = np.array(self.metrics_history['episodes'])
        rewards = np.array(self.metrics_history['rewards'])
        losses = np.array(self.metrics_history['losses'])
        
        # 1. Reward curve (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(episodes, rewards, alpha=0.3, color='blue')
        if len(rewards) >= 100:
            smoothed = self._moving_average(rewards, 100)
            ax1.plot(episodes[99:], smoothed, color='darkblue', linewidth=2)
        ax1.set_title('Reward over Training', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # 2. Statistics box (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = f"""
        Training Statistics
        {'='*25}
        Total Episodes: {len(episodes)}
        
        Reward:
          Mean: {np.mean(rewards):.4f}
          Max:  {np.max(rewards):.4f}
          Final: {rewards[-1]:.4f}
        
        Loss:
          Mean: {np.mean(losses):.4f}
          Final: {losses[-1]:.4f}
        
        Trajectory Length:
          Mean: {np.mean(self.metrics_history['trajectory_lengths']):.1f}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center')
        
        # 3. Loss curve (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episodes, losses, alpha=0.3, color='red')
        if len(losses) >= 100:
            smoothed = self._moving_average(losses, 100)
            ax3.plot(episodes[99:], smoothed, color='darkred', linewidth=2)
        ax3.set_title('Loss over Training')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics (middle center)
        if self.metrics_history['precision']:
            ax4 = fig.add_subplot(gs[1, 1])
            # Use episodes where detailed metrics were recorded
            detailed_episodes = np.array(self.detailed_metrics_episodes)
            precision = np.array(self.metrics_history['precision'])
            recall = np.array(self.metrics_history['recall'])
            f1 = np.array(self.metrics_history['f1_score'])

            if len(precision) >= 100:
                ax4.plot(detailed_episodes[99:], self._moving_average(precision, 100),
                        label='Precision', color='green', linewidth=2)
                ax4.plot(detailed_episodes[99:], self._moving_average(recall, 100),
                        label='Recall', color='blue', linewidth=2)
                ax4.plot(detailed_episodes[99:], self._moving_average(f1, 100),
                        label='F1', color='purple', linewidth=2)
            else:
                ax4.plot(detailed_episodes, precision, label='Precision', color='green')
                ax4.plot(detailed_episodes, recall, label='Recall', color='blue')
                ax4.plot(detailed_episodes, f1, label='F1', color='purple')
            
            ax4.set_title('Classification Metrics')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Score')
            ax4.set_ylim([0, 1.05])
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        # 5. Trajectory lengths histogram (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        lengths = self.metrics_history['trajectory_lengths']
        ax5.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_title('Trajectory Lengths')
        ax5.set_xlabel('Length')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Best rule confusion matrix (bottom, spans all)
        if self.best_rules:
            ax6 = fig.add_subplot(gs[2, :])
            sorted_rules = sorted(self.best_rules, key=lambda x: x[1], reverse=True)
            best_rule_str, best_reward, best_ep, best_scores = sorted_rules[0]
            
            cm = np.array([
                [best_scores['TP'], best_scores['FN']],
                [best_scores['FP'], best_scores['TN']]
            ])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                       xticklabels=['Entailed', 'Not Entailed'],
                       yticklabels=['Positive', 'Negative'],
                       cbar_kws={'label': 'Count'})
            
            short_rule = best_rule_str[:80] + '...' if len(best_rule_str) > 80 else best_rule_str
            ax6.set_title(
                f"Best Rule (Reward={best_reward:.4f}, Episode={best_ep})\n{short_rule}",
                fontsize=11, fontweight='bold'
            )
        
        plt.suptitle(f'Training Summary Dashboard - {self.experiment_name}',
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig(self.run_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved summary dashboard to {self.run_dir / 'summary_dashboard.png'}")
    
    def save_config(self, config: Dict):
        """Save training configuration to JSON file."""
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved configuration to {self.run_dir / 'config.json'}")
    
    def finalize(self):
        """Generate all plots and save everything."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        self.plot_training_curves()
        self.plot_metrics_over_time()
        self.plot_confusion_matrices()
        self.plot_trajectory_lengths()
        self.save_best_rules()
        self.create_summary_dashboard()
        
        print(f"\n✓ All visualizations saved to: {self.run_dir}")
        print("="*80)
    
    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
