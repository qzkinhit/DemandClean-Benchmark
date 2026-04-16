# Ablation Experiments

This directory contains two ablation experiment suites conducted on the **beers** dataset to analyze the design choices in DemandClean.

## Search Space Visualization (`search_space_beers/`)

Explores the structure of the data cleaning search space by exhaustively sampling 1,000 random cleaning strategy configurations.

### Key Findings

The feasible cleaning strategies form a **crescent-shaped region** in the Authenticity-Diversity plane, bounded by four extreme strategies:

| Strategy | Authenticity | Diversity | Description |
|----------|-------------|-----------|-------------|
| **NoFix** | Low | High | Keep all data as-is (maximum diversity, poor quality) |
| **FullFix** | High | High | Repair every error with ground truth (ideal but expensive) |
| **DeleteFix** | High | Low | Remove all erroneous rows (clean but small dataset) |
| **RelaxFix** | Medium | Medium | Replace errors with nearby estimates |

DemandClean (DemandFix) learns to position itself optimally within this space, balancing data quality against repair cost.

### Running

```bash
cd experiment/search_space_beers
python run_search_space.py
```

### Outputs

- `result/search_space_results.csv` — 1,000 sampled strategy configurations with metrics
- `result/search_space_scatter.png` — Scatter plot colored by performance
- `result/search_space_heatmap.png` — Grid heatmap of the search space
- `result/dqn_distribution.csv` — DQN agent's learned action distribution
- `result/extreme_points.json` — Coordinates of the four extreme strategies
- `result/action_pairs/` — Pairwise action probability heatmaps (6 combinations)

## Component Ablation (`ablation_beers/`)

Compares **17 strategy variants** to isolate the contribution of each design component.

### Strategies Compared

#### Deterministic Baselines (6)
| Strategy | Description |
|----------|-------------|
| NoFix | No cleaning |
| FullFix | Perfect repair (oracle) |
| DeleteFix | Delete all erroneous rows |
| DeleteAll | Delete rows with missing values |
| ReplaceAll | Replace all errors with nearby estimates |
| RelaxFix | Heuristic nearby replacement |

#### DQN Variants — Semi-Supervised (4)
| Strategy | Architecture | Phases |
|----------|-------------|--------|
| SemiSup_Single | Plain DQN | Single |
| SemiSup_TwoStage | Plain DQN | Two-stage |
| SemiSup_Dueling_Single | Dueling DQN | Single |
| SemiSup_Dueling_TwoStage | Dueling DQN | Two-stage |

#### DQN Variants — Fully Unsupervised (4)
| Strategy | Architecture | Phases |
|----------|-------------|--------|
| FullUnsup_Single | Plain DQN | Single |
| FullUnsup_TwoStage | Plain DQN | Two-stage |
| FullUnsup_Dueling_Single | Dueling DQN | Single |
| FullUnsup_Dueling_TwoStage | Dueling DQN | Two-stage |

#### Reference DQN (3)
| Strategy | Description |
|----------|-------------|
| DQN_Single | Basic single-stage DQN |
| DQN_TwoStage | Basic two-stage DQN |
| DemandFix | Full DemandClean (best configuration) |

### Running

```bash
cd experiment/ablation_beers
python run_ablation.py
```

### Outputs

- `result/*.png` — Per-strategy visualization
- `result/*.log` — Execution logs with detailed metrics
- `model/` — Pre-trained model weights for each variant
- `datasets/beers/` — Dataset used for ablation

## Data

Both experiments use the **beers** dataset (2,410 rows, 8 features, classification task). The `search_space_beers/datasets` directory is a symlink to `ablation_beers/datasets/beers` to avoid duplication.
