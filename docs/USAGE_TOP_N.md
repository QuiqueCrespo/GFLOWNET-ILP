# Top-N Hypothesis Sampling - Usage Guide

## Overview

The GFlowNet trainer now supports sampling multiple top hypotheses ranked by reward, not just the single best. This is useful for:
- Exploring alternative rule formulations
- Ensemble methods
- Understanding the learned distribution
- Beam search in program synthesis

## API

### New Method: `sample_top_theories()`

```python
def sample_top_theories(self,
                       initial_state: Theory,
                       positive_examples: List[Example],
                       negative_examples: List[Example],
                       num_samples: int = 10,
                       top_k: int = 5) -> List[Tuple[Theory, float]]:
    """
    Sample multiple theories and return the top K by reward.

    Args:
        initial_state: Starting theory state
        positive_examples: Positive training examples
        negative_examples: Negative training examples
        num_samples: Number of theories to sample from the model
        top_k: Number of top theories to return

    Returns:
        List of (theory, reward) tuples, sorted by reward (highest first)
    """
```

### Existing Method: `sample_best_theory()` (Updated)

Now uses `sample_top_theories()` internally and returns only the best.

```python
def sample_best_theory(self,
                      initial_state: Theory,
                      positive_examples: List[Example],
                      negative_examples: List[Example],
                      num_samples: int = 10) -> Tuple[Theory, float]:
    """
    Sample multiple theories and return the one with highest reward.

    For top-N hypotheses, use sample_top_theories() instead.
    """
```

## Usage Examples

### Example 1: Get Top-5 Hypotheses

```python
from training import GFlowNetTrainer

# ... (setup trainer, examples, etc.) ...

# Sample 50 theories and keep the top 5
top_5_theories = trainer.sample_top_theories(
    initial_state=initial_state,
    positive_examples=pos_examples,
    negative_examples=neg_examples,
    num_samples=50,
    top_k=5
)

# Iterate through results
for i, (theory, reward) in enumerate(top_5_theories):
    print(f"{i+1}. Reward: {reward:.4f}")
    print(f"   {theory_to_string(theory)}")
```

### Example 2: Compare Best vs Top-N

```python
# Old way - get single best
best_theory, best_reward = trainer.sample_best_theory(
    initial_state, pos_examples, neg_examples, num_samples=20
)
print(f"Best: {theory_to_string(best_theory)} (reward: {best_reward})")

# New way - get top 3
top_3 = trainer.sample_top_theories(
    initial_state, pos_examples, neg_examples,
    num_samples=20, top_k=3
)
for theory, reward in top_3:
    print(f"  {theory_to_string(theory)} (reward: {reward})")
```

### Example 3: Analyze Diversity

```python
# Sample many theories
top_20 = trainer.sample_top_theories(
    initial_state, pos_examples, neg_examples,
    num_samples=100, top_k=20
)

# Count unique structures
unique_theories = {}
for theory, reward in top_20:
    theory_str = theory_to_string(theory)
    if theory_str not in unique_theories:
        unique_theories[theory_str] = []
    unique_theories[theory_str].append(reward)

print(f"Unique theories in top 20: {len(unique_theories)}")
for theory_str, rewards in unique_theories.items():
    print(f"  {theory_str}")
    print(f"    Appears {len(rewards)} times, avg reward: {sum(rewards)/len(rewards):.4f}")
```

### Example 4: Ensemble Prediction

```python
# Get top-10 hypotheses for ensemble
top_10 = trainer.sample_top_theories(
    initial_state, pos_examples, neg_examples,
    num_samples=100, top_k=10
)

# Test on a new example
test_example = Example('target', ('x', 'y'))

votes = 0
for theory, reward in top_10:
    if logic_engine.entails(theory, test_example):
        votes += reward  # Weight by reward

ensemble_prediction = votes / sum(r for _, r in top_10)
print(f"Ensemble confidence: {ensemble_prediction:.2%}")
```

## Parameters

### `num_samples` - How Many to Sample

- **Low (10-20)**: Fast, but may miss good solutions
- **Medium (50-100)**: Good balance for most tasks
- **High (200+)**: Thorough search, slower but more diverse

### `top_k` - How Many to Return

- **1**: Just the best (use `sample_best_theory()` instead)
- **3-5**: Good for quick comparison of alternatives
- **10-20**: Good for ensemble methods
- **50+**: For thorough analysis of the learned distribution

## Performance Characteristics

### Time Complexity
- Sampling: O(num_samples × trajectory_length)
- Sorting: O(num_samples × log(num_samples))
- Total: Dominated by sampling

### Space Complexity
- Stores all sampled theories before sorting
- O(num_samples) memory

### Recommendations
```python
# For production (fast)
top_theories = trainer.sample_top_theories(
    ..., num_samples=20, top_k=5
)

# For research (thorough)
top_theories = trainer.sample_top_theories(
    ..., num_samples=200, top_k=20
)

# For ensemble (balanced)
top_theories = trainer.sample_top_theories(
    ..., num_samples=100, top_k=10
)
```

## Diversity vs Convergence

**High Diversity** (many unique theories in top-K):
- Early in training
- Lower learning rates
- More complex/ambiguous tasks
- Exploration bonus enabled

**Low Diversity** (few unique theories):
- Late in training (strong convergence)
- Higher learning rates
- Simple/clear tasks
- Strong optimum found

Both scenarios are valid:
- **Low diversity** = Model is confident (good for well-defined problems)
- **High diversity** = Model explores alternatives (good for ambiguous problems)

## Testing

Run the test suite:

```bash
# Basic top-N functionality
python test_top_n.py

# Diversity analysis
python test_diverse_hypotheses.py
```

## Full Example

```python
import torch
from logic_structures import get_initial_state, theory_to_string
from logic_engine import LogicEngine, Example
from graph_encoder import GraphConstructor, StateEncoder
from gflownet_models import HierarchicalGFlowNet
from reward import RewardCalculator
from training import GFlowNetTrainer

# Setup
predicate_vocab = ['target']
predicate_arities = {'target': 2}

pos_examples = [Example('target', ('a', 'a')), Example('target', ('b', 'b'))]
neg_examples = [Example('target', ('a', 'b')), Example('target', ('x', 'y'))]

graph_constructor = GraphConstructor(predicate_vocab)
state_encoder = StateEncoder(len(predicate_vocab) + 1, 64, num_layers=2)
gflownet = HierarchicalGFlowNet(64, len(predicate_vocab), 128)

engine = LogicEngine()
reward_calc = RewardCalculator(engine)

trainer = GFlowNetTrainer(
    state_encoder, gflownet, graph_constructor, reward_calc,
    predicate_vocab, predicate_arities, learning_rate=1e-3
)

# Train
initial_state = get_initial_state('target', arity=2)
trainer.train(initial_state, pos_examples, neg_examples, num_episodes=300)

# Get top-5 hypotheses
top_5 = trainer.sample_top_theories(
    initial_state, pos_examples, neg_examples,
    num_samples=50, top_k=5
)

# Display results
print("Top 5 Hypotheses:")
for i, (theory, reward) in enumerate(top_5):
    scores = reward_calc.get_detailed_scores(theory, pos_examples, neg_examples)
    print(f"\n{i+1}. Reward: {reward:.4f}")
    print(f"   Theory: {theory_to_string(theory)}")
    print(f"   Accuracy: {scores['accuracy']:.4f}")
    print(f"   Coverage: +{scores['pos_covered']}/{scores['pos_total']}, "
          f"-{scores['neg_covered']}/{scores['neg_total']}")
```

## Summary

✅ **Backward Compatible**: Existing code using `sample_best_theory()` still works
✅ **Flexible**: Control both sampling budget (`num_samples`) and output size (`top_k`)
✅ **Sorted**: Results always returned in descending reward order
✅ **Simple API**: Easy to use with clear parameter meanings
✅ **Efficient**: Minimal overhead compared to sampling just the best

The top-N hypothesis sampling enables more sophisticated use cases while maintaining simplicity for basic scenarios.
