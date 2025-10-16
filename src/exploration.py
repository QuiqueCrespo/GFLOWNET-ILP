"""
Exploration strategies for GFlowNet to encourage discovery of complex rules.

Implements multiple exploration mechanisms:
1. Entropy regularization
2. Temperature scheduling
3. Trajectory length bonus
4. ε-greedy exploration
5. Curiosity-driven exploration
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ExplorationStrategy:
    """Base class for exploration strategies."""

    def __init__(self, **kwargs):
        self.step_count = 0

    def modify_loss(self, base_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """Modify the loss to encourage exploration."""
        return base_loss

    def modify_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Modify action logits to encourage exploration."""
        return logits

    def modify_reward(self, reward: float, **kwargs) -> float:
        """Modify reward to encourage exploration."""
        return reward

    def step(self):
        """Update strategy state (e.g., decay parameters)."""
        self.step_count += 1


class EntropyBonus(ExplorationStrategy):
    """
    Add entropy bonus to encourage diverse action selection.

    Loss = TB_loss - α × H(π(a|s))

    where H(π) = -Σ π(a) log π(a) is the entropy of the action distribution.
    """

    def __init__(self, alpha: float = 0.01, decay: float = 0.9999):
        """
        Args:
            alpha: Entropy bonus coefficient
            decay: Decay rate per step (alpha *= decay each step)
        """
        super().__init__()
        self.alpha_init = alpha
        self.alpha = alpha
        self.decay = decay

    def modify_loss(self, base_loss: torch.Tensor, action_logits: torch.Tensor = None,
                   **kwargs) -> torch.Tensor:
        """Add negative entropy as bonus (reduces loss)."""
        if action_logits is None:
            return base_loss

        # Calculate entropy: H = -Σ p log p
        action_probs = F.softmax(action_logits, dim=-1)
        entropy = -(action_probs * F.log_softmax(action_logits, dim=-1)).sum(dim=-1)

        # Add entropy bonus (subtract from loss to encourage high entropy)
        return base_loss - self.alpha * entropy.mean()

    def step(self):
        """Decay alpha over time."""
        super().step()
        self.alpha *= self.decay

    def __str__(self):
        return f"EntropyBonus(α={self.alpha:.6f}, decay={self.decay})"


class TemperatureSchedule(ExplorationStrategy):
    """
    Temperature-based exploration: scale logits by temperature.

    π(a|s) = softmax(logits / T)

    High T → uniform (explore), Low T → peaked (exploit)
    """

    def __init__(self, T_init: float = 2.0, T_final: float = 0.5,
                 decay_steps: int = 1000):
        """
        Args:
            T_init: Initial temperature (high = more exploration)
            T_final: Final temperature (low = more exploitation)
            decay_steps: Number of steps to decay from T_init to T_final
        """
        super().__init__()
        self.T_init = T_init
        self.T_final = T_final
        self.decay_steps = decay_steps
        self.T = T_init

    def modify_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Scale logits by current temperature."""
        return logits / self.T

    def step(self):
        """Decay temperature linearly."""
        super().step()
        progress = min(self.step_count / self.decay_steps, 1.0)
        self.T = self.T_init + (self.T_final - self.T_init) * progress

    def __str__(self):
        return f"Temperature(T={self.T:.3f}, init={self.T_init}, final={self.T_final})"


class TrajectoryLengthBonus(ExplorationStrategy):
    """
    Reward longer trajectories to encourage complex rules.

    reward' = reward + β × trajectory_length
    """

    def __init__(self, beta: float = 0.05, decay: float = 0.9995):
        """
        Args:
            beta: Bonus per step in trajectory
            decay: Decay rate for beta
        """
        super().__init__()
        self.beta_init = beta
        self.beta = beta
        self.decay = decay

    def modify_reward(self, reward: float, trajectory_length: int = 0,
                     **kwargs) -> float:
        """Add bonus proportional to trajectory length."""
        return reward + self.beta * trajectory_length

    def step(self):
        """Decay beta over time."""
        super().step()
        self.beta *= self.decay

    def __str__(self):
        return f"TrajectoryBonus(β={self.beta:.6f}, decay={self.decay})"


class EpsilonGreedy(ExplorationStrategy):
    """
    ε-greedy exploration: random action with probability ε.
    """

    def __init__(self, epsilon: float = 0.1, decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Args:
            epsilon: Probability of random action
            decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        super().__init__()
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min

    def modify_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Mix uniform distribution with learned distribution.

        π' = (1-ε) × π + ε × uniform
        """
        if torch.rand(1).item() < self.epsilon:
            # Return uniform logits (all zeros → uniform after softmax)
            return torch.zeros_like(logits)
        return logits

    def step(self):
        """Decay epsilon."""
        super().step()
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)

    def __str__(self):
        return f"EpsilonGreedy(ε={self.epsilon:.4f}, min={self.epsilon_min})"


class CuriosityBonus(ExplorationStrategy):
    """
    Curiosity-driven exploration: bonus for novel states.

    Rewards visiting states with unusual features (e.g., many atoms, diverse predicates).
    """

    def __init__(self, bonus_atoms: float = 0.1, bonus_diversity: float = 0.05):
        """
        Args:
            bonus_atoms: Bonus per atom in body (encourages complexity)
            bonus_diversity: Bonus for using diverse predicates
        """
        super().__init__()
        self.bonus_atoms = bonus_atoms
        self.bonus_diversity = bonus_diversity

    def modify_reward(self, reward: float, num_atoms: int = 0,
                     num_unique_predicates: int = 0, **kwargs) -> float:
        """Add curiosity bonus."""
        atom_bonus = self.bonus_atoms * num_atoms
        diversity_bonus = self.bonus_diversity * num_unique_predicates
        return reward + atom_bonus + diversity_bonus

    def __str__(self):
        return f"Curiosity(atoms={self.bonus_atoms}, diversity={self.bonus_diversity})"


class CombinedExploration(ExplorationStrategy):
    """
    Combine multiple exploration strategies.
    """

    def __init__(self, strategies: list):
        """
        Args:
            strategies: List of ExplorationStrategy instances
        """
        super().__init__()
        self.strategies = strategies

    def modify_loss(self, base_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply all loss modifications."""
        loss = base_loss
        for strategy in self.strategies:
            loss = strategy.modify_loss(loss, **kwargs)
        return loss

    def modify_logits(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply all logit modifications."""
        modified = logits
        for strategy in self.strategies:
            modified = strategy.modify_logits(modified, **kwargs)
        return modified

    def modify_reward(self, reward: float, **kwargs) -> float:
        """Apply all reward modifications."""
        modified = reward
        for strategy in self.strategies:
            modified = strategy.modify_reward(modified, **kwargs)
        return modified

    def step(self):
        """Step all strategies."""
        super().step()
        for strategy in self.strategies:
            strategy.step()

    def __str__(self):
        strategies_str = ", ".join(str(s) for s in self.strategies)
        return f"Combined[{strategies_str}]"


# Predefined strategy configurations
def get_entropy_strategy(alpha: float = 0.01) -> EntropyBonus:
    """Get entropy bonus strategy."""
    return EntropyBonus(alpha=alpha, decay=0.9999)


def get_temperature_strategy(T_init: float = 2.0) -> TemperatureSchedule:
    """Get temperature schedule strategy."""
    return TemperatureSchedule(T_init=T_init, T_final=0.5, decay_steps=1000)


def get_length_bonus_strategy(beta: float = 0.05) -> TrajectoryLengthBonus:
    """Get trajectory length bonus strategy."""
    return TrajectoryLengthBonus(beta=beta, decay=0.9995)


def get_epsilon_greedy_strategy(epsilon: float = 0.1) -> EpsilonGreedy:
    """Get epsilon-greedy strategy."""
    return EpsilonGreedy(epsilon=epsilon, decay=0.995, epsilon_min=0.01)


def get_curiosity_strategy() -> CuriosityBonus:
    """Get curiosity bonus strategy."""
    return CuriosityBonus(bonus_atoms=0.1, bonus_diversity=0.05)


def get_combined_strategy(config: str = "balanced") -> CombinedExploration:
    """
    Get a predefined combined strategy.

    Args:
        config: One of "balanced", "aggressive", "conservative"
    """
    if config == "balanced":
        return CombinedExploration([
            EntropyBonus(alpha=0.01, decay=0.9999),
            TemperatureSchedule(T_init=1.5, T_final=0.5, decay_steps=1000),
        ])
    elif config == "aggressive":
        return CombinedExploration([
            EntropyBonus(alpha=0.05, decay=0.9998),
            TemperatureSchedule(T_init=3.0, T_final=0.5, decay_steps=1500),
            TrajectoryLengthBonus(beta=0.1, decay=0.999),
        ])
    elif config == "conservative":
        return CombinedExploration([
            EntropyBonus(alpha=0.005, decay=0.9999),
            TemperatureSchedule(T_init=1.2, T_final=0.6, decay_steps=800),
        ])
    else:
        raise ValueError(f"Unknown config: {config}")
