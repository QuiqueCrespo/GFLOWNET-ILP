# GFlowNet-ILP: Hierarchical GFlowNet for FOL Rule Generation

A hierarchical GFlowNet implementation for generating First-Order Logic (FOL) rules using Trajectory Balance objective with multiple exploration strategies.

## 📁 Project Structure

```
GFLowNet-ILP/
├── src/                    # Core source code
│   ├── logic_structures.py # FOL data structures
│   ├── logic_engine.py     # Logic evaluation engine
│   ├── graph_encoder.py    # GNN state encoder
│   ├── gflownet_models.py  # GFlowNet models
│   ├── reward.py          # Reward functions
│   └── training.py        # Training loop
├── tests/                 # Test suite
├── analysis/              # Analysis scripts
├── docs/                  # Documentation
└── examples/             # Usage examples
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic pipeline test
python -m tests.test_pipeline

# Run example with background knowledge
python -m analysis.quick_bg_test

# Run convergence analysis
python -m analysis.analyze_convergence
```

## 📚 Documentation

- `docs/FINAL_DIAGNOSIS.md` - Complete convergence analysis
- `docs/ANALYSIS_SUMMARY.md` - Root cause analysis
- `docs/USAGE_TOP_N.md` - Top-N hypothesis sampling guide
- `docs/CHANGELOG.md` - Recent changes and improvements

## 🔬 Key Features

- **Hierarchical GFlowNet** with Strategist + Tacticians
- **Graph Neural Network** state encoding
- **Background knowledge** support for logic reasoning
- **Top-N hypothesis** sampling
- **Comprehensive testing** suite

## ⚙️ Recent Improvements

- ✅ Fixed logic engine for existential variables
- ✅ Added background knowledge support
- ✅ Improved reward function (multiplicative accuracy)
- ✅ Implemented top-N hypothesis sampling
- 🔄 **IN PROGRESS:** Exploration improvements

## 🎯 Current Status

The core system is fully functional. Working on exploration mechanisms to help discover complex multi-step rules.

See `docs/FINAL_DIAGNOSIS.md` for detailed analysis.
