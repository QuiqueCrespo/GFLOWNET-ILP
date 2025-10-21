"""
Visualize how embedding representations evolve during trajectory generation.

Shows how the encoder's representation of rules changes as they are constructed
step-by-step through the GFlowNet policy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from src.logic_structures import theory_to_string


class EmbeddingTrajectoryVisualizer:
    """Visualize embedding evolution across multiple trajectories."""

    def __init__(self, trainer, graph_constructor, state_encoder):
        self.trainer = trainer
        self.graph_constructor = graph_constructor
        self.state_encoder = state_encoder

    def collect_trajectory_embeddings(self, initial_state, positives, negatives,
                                     num_trajectories=10, max_steps=10):
        """
        Sample multiple trajectories and collect embeddings at each step.

        Returns:
            trajectories_data: List of dicts with trajectory info
        """
        print(f"Collecting embeddings from {num_trajectories} trajectories...")
        trajectories_data = []

        for traj_id in range(num_trajectories):
            # Generate trajectory
            trajectory, reward = self.trainer.generate_trajectory(
                initial_state, positives, negatives, max_steps=max_steps
            )

            if not trajectory:
                continue

            # Collect embeddings for each step
            embeddings = []
            states = []
            actions = []
            log_probs = []

            for step in trajectory:
                # Get embedding for this state
                graph_data = self.graph_constructor.theory_to_graph(step.state)
                state_embedding, _ = self.state_encoder(graph_data)
                emb = state_embedding.squeeze(0).detach().cpu().numpy()

                embeddings.append(emb)
                states.append(theory_to_string(step.state))
                actions.append(step.action_type)
                log_probs.append(step.log_pf.item())

            # Also get final state embedding
            if trajectory:
                final_state = trajectory[-1].next_state
                graph_data = self.graph_constructor.theory_to_graph(final_state)
                state_embedding, _ = self.state_encoder(graph_data)
                emb = state_embedding.squeeze(0).detach().cpu().numpy()

                embeddings.append(emb)
                states.append(theory_to_string(final_state))
                actions.append('FINAL')
                log_probs.append(0.0)

            trajectories_data.append({
                'id': traj_id,
                'embeddings': np.array(embeddings),
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'reward': reward,
                'length': len(trajectory)
            })

            if (traj_id + 1) % 5 == 0:
                print(f"  Collected {traj_id + 1}/{num_trajectories} trajectories")

        print(f"✓ Collected {len(trajectories_data)} trajectories\n")
        return trajectories_data

    def visualize_all(self, trajectories_data, output_dir='.', prefix='embedding_traj'):
        """Create all visualizations."""
        print("=" * 80)
        print("EMBEDDING TRAJECTORY VISUALIZATION")
        print("=" * 80)

        # 1. PCA visualization (2D)
        print("\n1. Creating PCA 2D visualization...")
        fig1 = self.plot_trajectories_pca_2d(trajectories_data)
        path1 = f'{output_dir}/{prefix}_pca_2d.png'
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path1}")

        # 2. PCA visualization (3D)
        print("\n2. Creating PCA 3D visualization...")
        fig2 = self.plot_trajectories_pca_3d(trajectories_data)
        path2 = f'{output_dir}/{prefix}_pca_3d.png'
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path2}")

        # 3. t-SNE visualization
        print("\n3. Creating t-SNE visualization...")
        fig3 = self.plot_trajectories_tsne(trajectories_data)
        path3 = f'{output_dir}/{prefix}_tsne.png'
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path3}")

        # 4. Embedding similarity heatmap
        print("\n4. Creating similarity heatmap...")
        fig4 = self.plot_similarity_heatmap(trajectories_data)
        path4 = f'{output_dir}/{prefix}_similarity.png'
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path4}")

        # 5. Embedding distance evolution
        print("\n5. Creating distance evolution plot...")
        fig5 = self.plot_distance_evolution(trajectories_data)
        path5 = f'{output_dir}/{prefix}_distance_evolution.png'
        fig5.savefig(path5, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path5}")

        # 6. Action-colored visualization
        print("\n6. Creating action-colored visualization...")
        fig6 = self.plot_by_action_type(trajectories_data)
        path6 = f'{output_dir}/{prefix}_by_action.png'
        fig6.savefig(path6, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {path6}")

        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("=" * 80)

        return [fig1, fig2, fig3, fig4, fig5, fig6]

    def plot_trajectories_pca_2d(self, trajectories_data):
        """Plot trajectories in 2D PCA space."""
        # Collect all embeddings
        all_embeddings = []
        trajectory_labels = []

        for traj in trajectories_data:
            all_embeddings.extend(traj['embeddings'])
            trajectory_labels.extend([traj['id']] * len(traj['embeddings']))

        all_embeddings = np.array(all_embeddings)

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)

        # Explained variance
        var_explained = pca.explained_variance_ratio_

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot each trajectory
        start_idx = 0
        colors = cm.tab20(np.linspace(0, 1, len(trajectories_data)))

        for traj_idx, traj in enumerate(trajectories_data):
            length = len(traj['embeddings'])
            traj_embeddings = embeddings_2d[start_idx:start_idx + length]

            # Plot trajectory path
            ax.plot(traj_embeddings[:, 0], traj_embeddings[:, 1],
                   color=colors[traj_idx], alpha=0.6, linewidth=2,
                   label=f"Traj {traj['id']} (R={traj['reward']:.3f})")

            # Plot points
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1],
                      c=colors[traj_idx], s=100, alpha=0.8, edgecolors='black', linewidth=1)

            # Mark start and end
            ax.scatter(traj_embeddings[0, 0], traj_embeddings[0, 1],
                      c=colors[traj_idx], s=300, marker='s', alpha=1.0,
                      edgecolors='black', linewidth=2, label=f'Start {traj["id"]}')
            ax.scatter(traj_embeddings[-1, 0], traj_embeddings[-1, 1],
                      c=colors[traj_idx], s=300, marker='*', alpha=1.0,
                      edgecolors='black', linewidth=2, label=f'End {traj["id"]}')

            # Annotate steps
            for i, (x, y) in enumerate(traj_embeddings):
                ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                           color='white', weight='bold')

            start_idx += length

        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title('Embedding Evolution in PCA Space\n(Each line = one trajectory)',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_trajectories_pca_3d(self, trajectories_data):
        """Plot trajectories in 3D PCA space."""
        # Collect all embeddings
        all_embeddings = []
        for traj in trajectories_data:
            all_embeddings.extend(traj['embeddings'])
        all_embeddings = np.array(all_embeddings)

        # Apply PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(all_embeddings)
        var_explained = pca.explained_variance_ratio_

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each trajectory
        start_idx = 0
        colors = cm.tab20(np.linspace(0, 1, len(trajectories_data)))

        for traj_idx, traj in enumerate(trajectories_data):
            length = len(traj['embeddings'])
            traj_embeddings = embeddings_3d[start_idx:start_idx + length]

            # Plot trajectory path
            ax.plot(traj_embeddings[:, 0], traj_embeddings[:, 1], traj_embeddings[:, 2],
                   color=colors[traj_idx], alpha=0.6, linewidth=2,
                   label=f"Traj {traj['id']} (R={traj['reward']:.3f})")

            # Plot points
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1], traj_embeddings[:, 2],
                      c=colors[traj_idx], s=50, alpha=0.8)

            # Mark start and end
            ax.scatter(traj_embeddings[0, 0], traj_embeddings[0, 1], traj_embeddings[0, 2],
                      c=colors[traj_idx], s=200, marker='s', edgecolors='black', linewidth=2)
            ax.scatter(traj_embeddings[-1, 0], traj_embeddings[-1, 1], traj_embeddings[-1, 2],
                      c=colors[traj_idx], s=200, marker='*', edgecolors='black', linewidth=2)

            start_idx += length

        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)', fontsize=10)
        ax.set_title('3D Embedding Evolution (PCA)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_trajectories_tsne(self, trajectories_data):
        """Plot trajectories in t-SNE space."""
        # Collect all embeddings
        all_embeddings = []
        for traj in trajectories_data:
            all_embeddings.extend(traj['embeddings'])
        all_embeddings = np.array(all_embeddings)

        # Apply t-SNE
        print("   Running t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings) - 1),
                   random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot each trajectory
        start_idx = 0
        colors = cm.tab20(np.linspace(0, 1, len(trajectories_data)))

        for traj_idx, traj in enumerate(trajectories_data):
            length = len(traj['embeddings'])
            traj_embeddings = embeddings_2d[start_idx:start_idx + length]

            # Plot trajectory path
            ax.plot(traj_embeddings[:, 0], traj_embeddings[:, 1],
                   color=colors[traj_idx], alpha=0.6, linewidth=2,
                   label=f"Traj {traj['id']} (R={traj['reward']:.3f})")

            # Plot points
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1],
                      c=colors[traj_idx], s=100, alpha=0.8, edgecolors='black', linewidth=1)

            # Mark start and end
            ax.scatter(traj_embeddings[0, 0], traj_embeddings[0, 1],
                      c=colors[traj_idx], s=300, marker='s', alpha=1.0,
                      edgecolors='black', linewidth=2)
            ax.scatter(traj_embeddings[-1, 0], traj_embeddings[-1, 1],
                      c=colors[traj_idx], s=300, marker='*', alpha=1.0,
                      edgecolors='black', linewidth=2)

            start_idx += length

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Embedding Evolution in t-SNE Space\n(Non-linear dimensionality reduction)',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        return fig

    def plot_similarity_heatmap(self, trajectories_data):
        """Plot similarity heatmap across all steps in all trajectories."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Collect all embeddings
        all_embeddings = []
        labels = []

        for traj in trajectories_data:
            for step_idx, emb in enumerate(traj['embeddings']):
                all_embeddings.append(emb)
                labels.append(f"T{traj['id']}_S{step_idx}")

        all_embeddings = np.array(all_embeddings)

        # Compute similarity matrix
        similarity = cosine_similarity(all_embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot heatmap
        im = ax.imshow(similarity, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)

        # Add trajectory separators
        cumsum = 0
        for traj in trajectories_data:
            cumsum += len(traj['embeddings'])
            ax.axhline(y=cumsum - 0.5, color='blue', linewidth=2, alpha=0.5)
            ax.axvline(x=cumsum - 0.5, color='blue', linewidth=2, alpha=0.5)

        ax.set_title('Embedding Similarity Across All Steps\n(Blue lines separate trajectories)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Step Index', fontsize=12)
        ax.set_ylabel('Step Index', fontsize=12)

        # Only show some labels
        step = max(1, len(labels) // 20)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=90, fontsize=8)
        ax.set_yticks(range(0, len(labels), step))
        ax.set_yticklabels(labels[::step], fontsize=8)

        plt.tight_layout()
        return fig

    def plot_distance_evolution(self, trajectories_data):
        """Plot how embedding distance from initial state evolves."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        colors = cm.tab20(np.linspace(0, 1, len(trajectories_data)))

        # Plot 1: Distance from initial state
        for traj_idx, traj in enumerate(trajectories_data):
            embeddings = traj['embeddings']
            initial_emb = embeddings[0]

            # Compute L2 distance from initial
            distances = [np.linalg.norm(emb - initial_emb) for emb in embeddings]

            steps = range(len(distances))
            axes[0].plot(steps, distances, marker='o', linewidth=2,
                        color=colors[traj_idx], alpha=0.7,
                        label=f"Traj {traj['id']} (R={traj['reward']:.3f})")

        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('L2 Distance from Initial State', fontsize=12)
        axes[0].set_title('How far does the embedding move from the initial state?',
                         fontsize=12, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(alpha=0.3)

        # Plot 2: Distance from previous step
        for traj_idx, traj in enumerate(trajectories_data):
            embeddings = traj['embeddings']

            # Compute L2 distance from previous step
            step_distances = [0.0]  # First step has no previous
            for i in range(1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[i-1])
                step_distances.append(dist)

            steps = range(len(step_distances))
            axes[1].plot(steps, step_distances, marker='s', linewidth=2,
                        color=colors[traj_idx], alpha=0.7,
                        label=f"Traj {traj['id']}")

        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('L2 Distance from Previous Step', fontsize=12)
        axes[1].set_title('How much does each action change the embedding?',
                         fontsize=12, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_by_action_type(self, trajectories_data):
        """Plot embeddings colored by action type."""
        # Collect all embeddings
        all_embeddings = []
        action_types = []

        for traj in trajectories_data:
            all_embeddings.extend(traj['embeddings'])
            action_types.extend(traj['actions'])

        all_embeddings = np.array(all_embeddings)

        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Define action colors
        action_color_map = {
            'ADD_ATOM': 'blue',
            'UNIFY_VARIABLES': 'green',
            'TERMINATE': 'red',
            'FINAL': 'gold'
        }

        # Plot by action type
        for action_type, color in action_color_map.items():
            mask = np.array([a == action_type for a in action_types])
            if mask.any():
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=color, label=action_type, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1)

        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Embeddings Colored by Action Type\n(Do different actions cluster separately?)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """Example usage."""
    print("Embedding Trajectory Visualizer")
    print("\nThis tool visualizes how embeddings evolve during trajectory generation.")
    print("See Demo_ILP.ipynb for integration with training.")


if __name__ == "__main__":
    main()
