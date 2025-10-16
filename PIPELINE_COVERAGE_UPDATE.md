# Pipeline Coverage Reporting Update

## Changes Made

Updated `run_benchmark_pipeline.py` to report positive and negative coverage proportions for each benchmark.

### New Metrics Added

1. **Positive Coverage** (`pos_coverage`): Proportion of positive examples covered by the best rule (0.0 to 1.0)
2. **Negative Coverage** (`neg_coverage`): Proportion of negative examples covered by the best rule (0.0 to 1.0)
3. **Positive Covered** (`pos_covered`): Absolute number of positive examples covered
4. **Negative Covered** (`neg_covered`): Absolute number of negative examples covered

### Ideal Rule Metrics
- **Perfect Rule:** 100% positive coverage, 0% negative coverage
- **Good Rule:** High positive coverage (>80%), low negative coverage (<20%)

## Updated Output Format

### Per-Benchmark Output
```
BENCHMARK: kinship-ancestor
================================================================================
  Status: completed
  Episodes: 10000
  Time: 15.2s
  Best reward: 0.8322
  Coverage: 3/3 pos (100.0%), 0/0 neg (0.0%)
  Best rule: ancestor(X0, X1) :- parent(X0, X2), parent(X2, X1).
  Free variables: 0
```

### Summary Statistics
```
================================================================================
STATISTICS
================================================================================
Average best reward: 0.832
Average positive coverage: 85.5%
Average negative coverage: 12.3%
Perfect rules (100% pos, 0% neg): 4/10
Total free variables: 0
Free variable rate: 0.0%

✓ SUCCESS: No free variables in any benchmark!
✓ 4 benchmarks achieved perfect classification!
```

## JSON Output Schema

Each result now includes:
```json
{
  "benchmark": "kinship-ancestor",
  "status": "completed",
  "best_reward": 0.8322,
  "pos_coverage": 1.0,
  "neg_coverage": 0.0,
  "pos_covered": 3,
  "neg_covered": 0,
  "num_positive": 3,
  "num_negative": 0,
  "free_variables": 0,
  ...
}
```

## Running the Pipeline

The pipeline is currently running with 10,000 episodes per benchmark (as modified by user).

**Command:**
```bash
python run_benchmark_pipeline.py 2>&1 | tee benchmark_pipeline_with_coverage.log
```

**Expected time:** ~10-15 minutes for 10 benchmarks with 10,000 episodes each

**Monitor progress:**
```bash
tail -f benchmark_pipeline_with_coverage.log
```

## Interpretation Guide

### Coverage Metrics

| Pos Coverage | Neg Coverage | Interpretation |
|--------------|--------------|----------------|
| 100% | 0% | Perfect rule ✓ |
| >80% | <20% | Good rule |
| >60% | <40% | Acceptable rule |
| <60% | >40% | Poor rule |

### Common Patterns

1. **High pos, low neg** (e.g., 90% pos, 10% neg)
   - Good generalization
   - Rule captures most positives
   - Few false positives

2. **High pos, high neg** (e.g., 90% pos, 80% neg)
   - Overgeneralization
   - Rule is too broad
   - Many false positives

3. **Low pos, low neg** (e.g., 30% pos, 5% neg)
   - Undergeneralization
   - Rule is too specific
   - Misses many true positives

4. **Perfect** (100% pos, 0% neg)
   - Ideal rule
   - Complete and consistent with examples

## Benefits

1. **Better evaluation:** Can assess rule quality beyond just reward score
2. **Identify issues:** Spot overgeneralization (high neg coverage) or undergeneralization (low pos coverage)
3. **Perfect rule detection:** Easily identify rules that achieve 100% pos, 0% neg
4. **Comparative analysis:** Compare coverage across different benchmarks

## Expected Results

Based on previous runs, we expect:
- Most benchmarks to achieve >70% positive coverage
- Low negative coverage (ideally <30%)
- Several benchmarks (3-5) to achieve perfect classification
- 0 free variables across all benchmarks (confirmed by action mask)
