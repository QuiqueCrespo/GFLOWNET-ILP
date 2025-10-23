"""
Standalone script to pretrain the GNN encoder.

Usage:
    python pretrain_encoder.py --steps 5000 --batch_size 32 --output pretrained_encoder.pt

This will pretrain the encoder using contrastive learning on randomly generated rules,
then save the weights for use in the main ILP training pipeline.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.graph_encoder import GraphConstructor, StateEncoder
from src.encoder_pretraining import EncoderPretrainer, visualize_augmentations


def plot_training_curves(history, save_path=None):
    """Plot pretraining metrics over time."""
    steps = list(range(len(history)))
    losses = [h['loss'] for h in history]
    accuracies = [h['accuracy'] for h in history]
    pos_sims = [h['avg_pos_sim'] for h in history]
    neg_sims = [h['avg_neg_sim'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(steps, losses, linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Contrastive Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(steps, accuracies, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Positive vs Negative Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    # Similarities
    axes[1, 0].plot(steps, pos_sims, label='Positive', linewidth=2, color='blue')
    axes[1, 0].plot(steps, neg_sims, label='Negative', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Average Similarities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Similarity gap
    gaps = [pos - neg for pos, neg in zip(pos_sims, neg_sims)]
    axes[1, 1].plot(steps, gaps, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Similarity Gap')
    axes[1, 1].set_title('Positive - Negative Similarity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Pretrain GNN encoder for ILP')

    # Pretraining parameters
    parser.add_argument('--steps', type=int, default=5000,
                       help='Number of pretraining steps (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for pretraining (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss (default: 0.5)')
    parser.add_argument('--num_negatives', type=int, default=4,
                       help='Number of negative samples per anchor (default: 4)')

    # Encoder architecture
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers (default: 3)')

    # Data generation
    parser.add_argument('--max_body_length', type=int, default=4,
                       help='Maximum body length for random rules (default: 4)')

    # Output
    parser.add_argument('--output', type=str, default='pretrained_encoder.pt',
                       help='Output path for pretrained weights (default: pretrained_encoder.pt)')
    parser.add_argument('--plot', type=str, default=None,
                       help='Path to save training curves plot (optional)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize augmentation examples before training')

    args = parser.parse_args()

    print("=" * 80)
    print("GNN ENCODER PRETRAINING FOR ILP")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Num negatives: {args.num_negatives}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  GNN layers: {args.num_layers}")
    print(f"  Max body length: {args.max_body_length}")
    print(f"  Output: {args.output}")
    print()

    # Define predicate vocabulary (8 predicates with arity 1 or 2)
    predicate_vocab = ['parent', 'child', 'male', 'female', 'adult', 'sibling', 'ancestor', 'friend']
    predicate_arities = {
        'parent': 2,
        'child': 2,
        'sibling': 2,
        'ancestor': 2,
        'friend': 2,
        'male': 1,
        'female': 1,
        'adult': 1
    }

    print(f"Predicate vocabulary ({len(predicate_vocab)} predicates):")
    for pred in predicate_vocab:
        print(f"  {pred}/{predicate_arities[pred]}")
    print()

    # Visualize augmentations if requested
    if args.visualize:
        visualize_augmentations(predicate_vocab, predicate_arities, num_examples=3)
        print()

    # Create graph constructor and encoder
    node_feature_dim = len(predicate_vocab) + 1  # One-hot predicates + variable indicator
    graph_constructor = GraphConstructor(predicate_vocab)
    state_encoder = StateEncoder(
        node_feature_dim=node_feature_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers
    )

    print(f"Created GNN encoder:")
    print(f"  Input dim: {node_feature_dim}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in state_encoder.parameters())}")
    print()

    # Create pretrainer
    pretrainer = EncoderPretrainer(
        state_encoder=state_encoder,
        graph_constructor=graph_constructor,
        predicate_vocab=predicate_vocab,
        predicate_arities=predicate_arities,
        learning_rate=args.lr,
        temperature=args.temperature,
        num_negatives=args.num_negatives
    )

    # Pretrain
    print("Starting pretraining...")
    print("-" * 80)
    history = pretrainer.pretrain(
        num_steps=args.steps,
        batch_size=args.batch_size,
        verbose=True,
        log_interval=100
    )
    print("-" * 80)

    # Print final statistics
    final_metrics = history[-1]
    print(f"\nFinal metrics:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Positive similarity: {final_metrics['avg_pos_sim']:.3f}")
    print(f"  Negative similarity: {final_metrics['avg_neg_sim']:.3f}")
    print(f"  Similarity gap: {final_metrics['avg_pos_sim'] - final_metrics['avg_neg_sim']:.3f}")
    print()

    # Save encoder
    pretrainer.save_pretrained_encoder(args.output)

    # Plot training curves
    if args.plot:
        plot_training_curves(history, save_path=args.plot)
    else:
        # Show plot without saving
        plot_training_curves(history)

    print("\nPretraining complete!")
    print(f"\nTo use this pretrained encoder in your ILP training:")
    print(f"  1. Load the encoder: state_encoder.load_state_dict(torch.load('{args.output}'))")
    print(f"  2. (Optional) Freeze encoder: for param in state_encoder.parameters(): param.requires_grad = False")
    print(f"  3. Train GFlowNet with pretrained encoder")


if __name__ == "__main__":
    main()
