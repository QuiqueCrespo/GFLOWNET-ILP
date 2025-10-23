"""
Dissimilarity Loss Integration for Demo_ILP.ipynb

Add this code BEFORE the training loop in your notebook.
Then replace trainer.train_step() with train_step_with_dissimilarity().
"""

import torch
import torch.nn.functional as F


def train_step_with_dissimilarity(trainer, initial_state, positives, negatives,
                                   dissimilarity_weight=0.1):
    """
    Training step with dissimilarity loss to maintain diverse embeddings.

    This wrapper adds a dissimilarity term to the GFlowNet loss to prevent
    embeddings from collapsing back to similarity during training.

    Args:
        trainer: Your GFlowNetTrainer instance
        initial_state: Initial state for trajectory
        positives: Positive examples
        negatives: Negative examples
        dissimilarity_weight: Weight for dissimilarity loss (0.05-0.2 recommended)

    Returns:
        metrics dict with loss breakdown
    """

    # Generate trajectory
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positives, negatives
    )

    if not trajectory:
        return {'loss': 0.0, 'reward': 0.0, 'trajectory_length': 0}

    # Compute GFlowNet loss
    if trainer.use_detailed_balance:
        gfn_loss = trainer.compute_detailed_balance_loss(trajectory, reward)
    else:
        gfn_loss = trainer.compute_trajectory_balance_loss(trajectory, reward)

    # Compute dissimilarity loss
    if len(trajectory) >= 2:
        # Collect embeddings from trajectory
        embeddings = []
        for step in trajectory:
            graph_data = trainer.graph_constructor.theory_to_graph(step.state)
            state_embedding, _ = trainer.state_encoder(graph_data)
            embeddings.append(state_embedding.squeeze(0))

        embeddings_tensor = torch.stack(embeddings)

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings_tensor, dim=-1)

        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)

        # Get off-diagonal similarities (exclude self-similarity)
        n = similarity_matrix.size(0)
        mask = 1 - torch.eye(n, device=similarity_matrix.device)
        avg_similarity = (similarity_matrix * mask).sum() / (n * (n - 1))

        # Dissimilarity loss = average similarity (minimizing this promotes dissimilarity)
        dissim_loss = avg_similarity
    else:
        dissim_loss = torch.tensor(0.0)

    # Combined loss
    total_loss = gfn_loss + dissimilarity_weight * dissim_loss

    # Optimize
    trainer.optimizer.zero_grad()
    total_loss.backward()
    trainer.optimizer.step()

    # Update replay buffer
    if trainer.replay_buffer is not None and reward > trainer.buffer_reward_threshold:
        trainer.replay_buffer.add(trajectory, reward)

    return {
        'loss': total_loss.item(),
        'gfn_loss': gfn_loss.item(),
        'dissimilarity_loss': dissim_loss.item() if isinstance(dissim_loss, torch.Tensor) else dissim_loss,
        'reward': reward,
        'trajectory_length': len(trajectory)
    }


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
STEP 1: Add the function above to a new cell BEFORE your training loop

STEP 2: Modify your training loop from:

    for episode in range(num_episodes):
        metrics = trainer.train_step(initial_state, positive_examples, negative_examples)

        if metrics:
            rewards.append(metrics['reward'])
            # ... rest of code ...

TO:

    for episode in range(num_episodes):
        metrics = train_step_with_dissimilarity(
            trainer,
            initial_state,
            positive_examples,
            negative_examples,
            dissimilarity_weight=0.1  # Adjust 0.05-0.2
        )

        if metrics:
            rewards.append(metrics['reward'])
            # ... rest of code ...

STEP 3: (Optional) Monitor dissimilarity during training:

    if episode % 100 == 0 and metrics:
        print(f"Episode {episode}:")
        print(f"  Total loss: {metrics['loss']:.4f}")
        print(f"  GFN loss:   {metrics['gfn_loss']:.4f}")
        print(f"  Dissim loss: {metrics['dissimilarity_loss']:.4f}")
        print(f"  Reward:     {metrics['reward']:.4f}")

EXPECTED BEHAVIOR:
- dissimilarity_loss should start around 0.65-0.75 (after pre-training)
- It may increase slightly during training (embeddings trying to collapse)
- Weight of 0.1 prevents collapse while maintaining GFN performance

TUNING THE WEIGHT:
- Too high (>0.3): GFN performance may degrade (reward decreases)
- Too low (<0.05): Embeddings may collapse back to high similarity
- Start with 0.1 and adjust based on results
"""


if __name__ == "__main__":
    print("=" * 80)
    print("DISSIMILARITY LOSS - NOTEBOOK INTEGRATION")
    print("=" * 80)
    print()
    print("This file contains:")
    print("  1. train_step_with_dissimilarity() - Drop-in replacement for trainer.train_step()")
    print("  2. Integration instructions for Demo_ILP.ipynb")
    print()
    print("Benefits:")
    print("  ✓ Prevents embedding collapse during GFlowNet training")
    print("  ✓ Maintains diversity achieved by contrastive pre-training")
    print("  ✓ No modifications to training.py required")
    print()
    print("Usage: Copy the function to a notebook cell before your training loop")
    print("=" * 80)
