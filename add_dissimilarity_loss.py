"""
Add dissimilarity term to the GFlowNet loss function.

Simple modification to promote diverse embeddings during training.
"""

import torch
import torch.nn.functional as F


def compute_dissimilarity_loss(embeddings, method='variance'):
    """
    Compute dissimilarity loss for a batch of embeddings.

    Lower loss = more dissimilar embeddings (GOOD)
    Higher loss = more similar embeddings (BAD)

    Args:
        embeddings: [batch_size, embed_dim] - embeddings from trajectory steps
        method: 'variance', 'cosine', or 'distance'

    Returns:
        loss: scalar tensor (minimize this to promote dissimilarity)
    """

    if len(embeddings) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    if method == 'variance':
        # Method 1: Maximize variance across embeddings
        # Higher variance = more spread out = more dissimilar
        mean_emb = embeddings.mean(dim=0)
        variance = ((embeddings - mean_emb) ** 2).mean()

        # Loss = negative variance (minimizing this maximizes variance)
        loss = -variance

    elif method == 'cosine':
        # Method 2: Minimize pairwise cosine similarities
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)

        # Get off-diagonal elements (don't include self-similarity)
        n = similarity_matrix.size(0)
        mask = 1 - torch.eye(n, device=similarity_matrix.device)
        off_diagonal_sim = (similarity_matrix * mask).sum() / (n * (n - 1))

        # Loss = average pairwise similarity (want this LOW)
        loss = off_diagonal_sim

    elif method == 'distance':
        # Method 3: Maximize pairwise L2 distances
        # Compute pairwise distances
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # [n, n, dim]
        distances = torch.norm(diff, dim=-1)  # [n, n]

        # Get off-diagonal distances
        n = distances.size(0)
        mask = 1 - torch.eye(n, device=distances.device)
        avg_distance = (distances * mask).sum() / (n * (n - 1))

        # Loss = negative distance (minimizing this maximizes distance)
        loss = -avg_distance

    else:
        raise ValueError(f"Unknown method: {method}")

    return loss


def modified_train_step(trainer, initial_state, positives, negatives,
                        dissimilarity_weight=0.1, dissimilarity_method='cosine'):
    """
    Modified training step with dissimilarity loss.

    Args:
        trainer: Your GFlowNetTrainer instance
        dissimilarity_weight: How much to weight dissimilarity (0.05-0.2 recommended)
        dissimilarity_method: 'variance', 'cosine', or 'distance'

    Returns:
        metrics dict with loss breakdown
    """

    # Generate trajectory
    trajectory, reward = trainer.generate_trajectory(
        initial_state, positives, negatives
    )

    if not trajectory:
        return {'loss': 0.0, 'reward': 0.0, 'trajectory_length': 0}

    # Compute original GFlowNet loss
    gfn_loss = trainer.compute_trajectory_balance_loss(trajectory, reward)

    # Collect embeddings from trajectory
    embeddings = []
    for step in trajectory:
        graph_data = trainer.graph_constructor.theory_to_graph(step.state)
        state_embedding, _ = trainer.state_encoder(graph_data)
        embeddings.append(state_embedding.squeeze(0))

    embeddings_tensor = torch.stack(embeddings)

    # Compute dissimilarity loss
    dissim_loss = compute_dissimilarity_loss(embeddings_tensor, method=dissimilarity_method)

    # Combined loss
    total_loss = gfn_loss + dissimilarity_weight * dissim_loss

    # Backprop and optimize
    trainer.optimizer.zero_grad()
    total_loss.backward()
    trainer.optimizer.step()

    return {
        'loss': total_loss.item(),
        'gfn_loss': gfn_loss.item(),
        'dissimilarity_loss': dissim_loss.item(),
        'reward': reward,
        'trajectory_length': len(trajectory)
    }


# Example integration into training loop
def example_training_loop():
    """
    Example showing how to integrate dissimilarity loss.
    """

    # Assume you have:
    # - trainer: GFlowNetTrainer instance
    # - initial_state, positives, negatives
    # - num_episodes

    print("Training with dissimilarity loss:")
    print("=" * 60)

    # for episode in range(num_episodes):
    #     # Use modified train step
    #     metrics = modified_train_step(
    #         trainer,
    #         initial_state,
    #         positives,
    #         negatives,
    #         dissimilarity_weight=0.1,      # 10% of total loss
    #         dissimilarity_method='cosine'  # or 'variance' or 'distance'
    #     )
    #
    #     if episode % 100 == 0:
    #         print(f"Episode {episode}:")
    #         print(f"  Total loss: {metrics['loss']:.4f}")
    #         print(f"  GFN loss:   {metrics['gfn_loss']:.4f}")
    #         print(f"  Dissim loss: {metrics['dissimilarity_loss']:.4f}")
    #         print(f"  Reward:     {metrics['reward']:.4f}")

    print("That's it! Just replace trainer.train_step() with modified_train_step()")


if __name__ == "__main__":
    print("=" * 80)
    print("DISSIMILARITY LOSS - ADD TO LOSS FUNCTION")
    print("=" * 80)
    print()

    print("Three methods to promote dissimilarity:")
    print()

    print("1. VARIANCE METHOD (maximize spread)")
    print("   Loss = -variance(embeddings)")
    print("   → Minimizing this spreads embeddings apart")
    print()

    print("2. COSINE METHOD (minimize similarity)")
    print("   Loss = average_pairwise_cosine_similarity")
    print("   → Minimizing this makes embeddings dissimilar")
    print("   ✓ RECOMMENDED - most stable")
    print()

    print("3. DISTANCE METHOD (maximize separation)")
    print("   Loss = -average_pairwise_L2_distance")
    print("   → Minimizing this pushes embeddings apart")
    print()

    print("=" * 80)
    print("USAGE")
    print("=" * 80)
    print()
    print("Just replace:")
    print("  metrics = trainer.train_step(initial_state, positives, negatives)")
    print()
    print("With:")
    print("  metrics = modified_train_step(")
    print("      trainer, initial_state, positives, negatives,")
    print("      dissimilarity_weight=0.1,  # 10% of total loss")
    print("      dissimilarity_method='cosine'")
    print("  )")
    print()
    print("=" * 80)
