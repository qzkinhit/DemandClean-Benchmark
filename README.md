# DemandClean-Benchmark

A comprehensive benchmark for evaluating data cleaning methods in machine learning pipelines, featuring DemandClean — a demand-driven data cleaning framework based on deep reinforcement learning — alongside 14 established baseline methods evaluated on 9 standard datasets.

## Overview

Not all data errors are equally harmful to downstream ML tasks. Some errors barely affect model performance and can be safely tolerated, while others cause significant degradation and warrant the cost of repair. **DemandClean** leverages this insight through a DQN-based agent that learns per-error cleaning decisions (tolerate, repair, delete, or replace) by balancing downstream task improvement against repair cost.

This repository provides:

- **DemandClean framework**: the complete RL-based cleaning pipeline with 8 configuration variants
- **14 baseline methods**: from simple imputation to state-of-the-art learning-based cleaners (HoloClean, Raha+Baran, UniClean, Lopster, CTXPipe, etc.)
- **9 benchmark datasets**: spanning classification, regression, and clustering tasks
- **Pre-computed results**: all experiment outputs, evaluation metrics, and summary tables for direct comparison
- **One-click reproducibility**: shell scripts to re-run any experiment from scratch

## Repository Structure

```
DemandClean-Benchmark/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies for DemandClean
├── run.sh                      # One-click entry point for DemandClean
│
├── demandclean/                # Core DemandClean framework
│   ├── api/                    #   High-level API (DemandClean class)
│   ├── config/                 #   Configuration and enums
│   ├── core/                   #   RL components
│   │   ├── agents/             #     DQN agents (plain, dueling)
│   │   ├── environments/       #     Training/inference environments
│   │   └── state/              #     State extractors per task type
│   ├── detectors/              #   Error detectors (oracle, auto)
│   ├── inference/              #   Single-phase and two-phase inference
│   ├── models/                 #   ML model adapters (11 models)
│   ├── training/               #   DQN trainer with experience replay
│   ├── tools/                  #   Shapley and tolerance analysis
│   ├── utils/                  #   Logging, metrics, model I/O
│   └── tests/                  #   Unit and integration tests
│
├── data/                       # 9 benchmark datasets (clean/dirty pairs)
│
├── run_demandclean/            # Experiment execution scripts
│   ├── run.sh                  #   DemandClean runner (train + infer + eval)
│   ├── run_demandclean_base.py #   Core experiment pipeline
│   └── ...                     #   Report generation, plotting, etc.
│
├── tools/                      # Evaluation toolkit
│   ├── getScoreML.py           #   ML evaluation scoring engine
│   ├── eval_all_detection.py   #   Detection accuracy evaluation
│   └── ...                     #   Error injection, rule parsing, Snoopy
│
├── experiment/                 # Ablation studies
│   ├── search_space_beers/     #   Search space visualization (1000 samples)
│   └── ablation_beers/         #   Component ablation (17 strategies)
│
├── benchmark/                  # Baseline methods (Clean4MLBaseline)
│   ├── README.md               #   Baseline-specific documentation
│   ├── run_all.sh              #   One-click runner for all baselines
│   ├── Data/                   #   Datasets (clean/dirty CSV pairs)
│   ├── Methods/                #   14 method implementations
│   ├── MethodsRunScript/       #   Per-method run scripts
│   ├── tools/                  #   Baseline evaluation tools
│   ├── results/                #   Pre-computed baseline results
│   └── logs/                   #   Execution logs
│
└── results_and_logs/           # Consolidated experiment outputs
    ├── demandclean/            #   DemandClean results (9 datasets × versions)
    ├── reference_strategies/   #   NoFix/DeleteAll/ReplaceAll/RepairAll
    ├── replaceall_baseline/    #   ReplaceAll detailed evaluation
    ├── logs/                   #   Training and execution logs
    └── summary/                #   Cross-method comparison CSVs
```

> **Note on datasets**: The `data/` directory contains datasets in DemandClean's format (`*_index.csv` with index columns), while `benchmark/Data/` contains the same datasets in the baseline format (additional `clean.csv`/`dirty.csv` without index columns). Both copies are kept because the two systems expect different file layouts.

## Quick Start

### Prerequisites

- Python 3.8 or later
- ~2 GB disk space for datasets and pre-computed results
- (Optional) CUDA-compatible GPU for faster DQN training

### Installation

```bash
git clone https://github.com/<your-username>/DemandClean-Benchmark.git
cd DemandClean-Benchmark
pip install -r requirements.txt
```

### Run DemandClean

```bash
# Quick test (10 training episodes on beers dataset, v5 configuration)
bash run.sh --dataset beers --versions v5 --n_episodes 10

# Full run (300 episodes, all datasets, default v5)
bash run.sh --all_datasets --n_episodes 300

# Specific version and dataset
bash run.sh --dataset nasa --versions v3,v5,v7 --n_episodes 200
```

Results are saved to `results/demandclean/{dataset}/{version}/` and logs to `logs/demandclean/`.

### Run Baseline Methods

```bash
cd benchmark

# Install baseline dependencies
pip install -r requirements.txt

# Run all baselines sequentially
bash run_all.sh --version v1

# Run a specific baseline on a specific dataset
bash run_all.sh --version v1 --baseline raha_baran --dataset beers

# Run all baselines in parallel (uses GNU screen)
bash run_all.sh --version v1 --parallel
```

Results are saved to `benchmark/results/` and logs to `benchmark/logs/`.

## DemandClean Framework

### Architecture

DemandClean formulates data cleaning as a Markov Decision Process where a DQN agent processes each detected error and selects one of four actions:

| Action | Code | Effect | Cost |
|--------|------|--------|------|
| **No Action** | 0 | Tolerate the error (keep as-is) | 0 |
| **Repair** | 1 | Replace with ground-truth value | `repair_lambda` |
| **Delete** | 2 | Remove the entire row | 0 |
| **Replace** | 3 | Substitute with a nearby estimated value | 0 |

The agent observes an 8-dimensional state vector per error:

```
[error_type, feature_importance, distance_to_boundary, row_position,
 col_index, col_error_rate, sample_retention, variance_retention]
```

Training uses self-supervised error injection: the agent learns on artificially corrupted data without requiring pre-existing clean labels.

### Configuration Variants (v1–v8)

DemandClean supports 8 configuration variants combining three dimensions:

| Version | Detector | Agent Architecture | Inference Mode |
|---------|----------|--------------------|----------------|
| v1 | Oracle | Dueling DQN | Single-phase |
| v2 | Oracle | Dueling DQN | Two-phase |
| v3 | Oracle | Plain DQN | Single-phase |
| v4 | Oracle | Plain DQN | Two-phase |
| **v5** | **Auto** | **Dueling DQN** | **Single-phase** |
| v6 | Auto | Dueling DQN | Two-phase |
| v7 | Auto | Plain DQN | Single-phase |
| v8 | Auto | Plain DQN | Two-phase |

**v5** is the default and recommended configuration for practical use. It uses automatic error detection (RAHA + rule-based pipeline) and requires no oracle access.

- **Oracle detector**: uses ground-truth clean data to identify errors (for ablation studies only)
- **Auto detector**: 4-stage pipeline — missing value detection → RAHA → FD/CFD/DC rules → label noise rules
- **Dueling DQN**: separates value and advantage streams for better action differentiation
- **Two-phase inference**: first plans all actions, then executes them jointly (vs. sequential single-phase)

### Programmatic API

```python
from demandclean import DemandClean
from demandclean.config import DemandCleanConfig, TaskType

config = DemandCleanConfig(
    task_type=TaskType.CLASSIFICATION,
    n_episodes=300,
    repair_lambda=0.03,
)

dc = DemandClean(config)
dc.train(dirty_path="data/beers/dirty_index.csv",
         clean_path="data/beers/clean_index.csv",
         rules_path="data/beers/rules.txt",
         label_column="style")

cleaned_df = dc.infer(dirty_path="data/beers/dirty_index.csv")
```

## Baseline Methods

| Method | Venue | Approach | Ground Truth |
|--------|-------|----------|--------------|
| DoNothing | — | No cleaning (lower bound) | None |
| DeleteAll | — | Remove all rows with detected errors | None |
| RepairAll | — | Replace all errors with clean values (upper bound) | Full |
| SimpleImputer | — | Statistical imputation (mean/median/mode) | None |
| MLImputer | — | ML-based imputation (MICE/KNN/RF) | None |
| HoloClean | VLDB 2017 | Probabilistic graphical model with constraints | None |
| Horizon | VLDB 2021 | Dependency-driven pattern selection | None |
| Raha+Baran | SIGMOD'19 / VLDB'20 | Configuration-free detection + context-based repair | Iterative |
| UniClean | VLDB 2025 | Multi-signal fusion with workflow optimization | None |
| Lopster | VLDB 2024 | Latent space representation learning | None |
| ActiveClean | SIGMOD 2016 | Gradient-based sample prioritization | Iterative |
| BoostClean | SIGMOD 2017 | Detector-repair ensemble with boosting | Validation set |
| CTXPipe | SIGMOD 2024 | Context-aware RL pipeline generation | None |

See `benchmark/README.md` for detailed setup instructions per method, including external dependencies (PostgreSQL for HoloClean, Spark for UniClean, etc.).

## Datasets

| Dataset | Task | Target Column | Rows | Features | Error Types |
|---------|------|---------------|------|----------|-------------|
| adult | Classification | income | 45,222 | 15 | Rule violations, outliers |
| beers | Classification | style | 2,410 | 8 | Missing values, outliers |
| bike | Regression | cnt | 17,379 | 12 | Missing values, noise |
| breast_cancer | Classification | class | 699 | 31 | Missing values |
| har | Clustering | gt | 10,299 | 562 | Missing values, noise |
| mercedes | Regression | y | 4,209 | 377 | Missing values |
| nasa | Regression | sound_pressure_level | 1,503 | 6 | Missing values |
| smartfactory | Classification | labels | 7,484 | 6 | Missing values, outliers |
| soilmoisture | Regression | soil_moisture | 1,440 | 14 | Missing values |

Each dataset directory contains:
- `dirty_index.csv` — error-injected data with index column
- `clean_index.csv` — ground-truth clean data with index column
- `rules.txt` — functional dependency and domain constraint rules

## Experiments

### Main Evaluation

To reproduce the full comparison across all methods and datasets:

```bash
# 1. Run DemandClean (v5) on all datasets
bash run.sh --all_datasets --versions v5 --n_episodes 300

# 2. Run all baselines
cd benchmark && bash run_all.sh --version v1

# 3. Generate summary tables
python run_demandclean/generate_csv.py
```

### Ablation Studies

Two ablation experiments are included in `experiment/`:

**Search Space Visualization** (`experiment/search_space_beers/`):
Exhaustively samples 1,000 cleaning strategy configurations to map the search space structure. Demonstrates the crescent-shaped feasible region bounded by four extreme strategies (NoFix, FullFix, DeleteFix, RelaxFix).

```bash
cd experiment/search_space_beers
python run_search_space.py
```

**Component Ablation** (`experiment/ablation_beers/`):
Compares 17 strategy variants on the beers dataset, isolating the contribution of each component (detector type, network architecture, training mode, action space).

```bash
cd experiment/ablation_beers
python run_ablation.py
```

### Component Analysis

**Shapley Value Analysis**: quantifies the contribution of each action type and error type to cleaning performance.

**Tolerance Analysis**: measures how much model performance improves relative to the gap between dirty data and perfectly clean data.

## Pre-computed Results

All experiment outputs are provided in `results_and_logs/`:

| Directory | Contents |
|-----------|----------|
| `demandclean/` | DemandClean outputs for 9 datasets × multiple versions (models, cleaned CSVs, evaluation reports) |
| `reference_strategies/` | NoFix, DeleteAll, ReplaceAll, RepairAll evaluation results |
| `replaceall_baseline/` | Detailed ReplaceAll baseline with value estimation chain |
| `logs/` | Training and execution logs |
| `summary/` | Cross-method comparison CSVs (see below) |

### Summary Tables

| File | Rows | Description |
|------|------|-------------|
| `demandclean_all_versions_results.csv` | 943 | All DemandClean version results across datasets |
| `baseline_evaluation_results_merged.csv` | 2,997 | All baseline method evaluations merged |
| `detector_comparison.csv` | 10 | Detection accuracy (P/R/F1) per dataset |
| `detector_comparison_all.csv` | 118 | Detailed detection comparison across all methods |
| `tolerance_analysis_results.csv` | 86 | Prior and posterior tolerance metrics |
| `replaceall_evaluation_results.csv` | 47 | ReplaceAll detailed metrics |
| `shapley_analysis_results.csv` | 46 | Shapley value attribution results |

## Evaluation Metrics

### Data Quality Metrics
- **Accuracy**: correctly repaired cells / total repaired cells
- **Recall**: correctly repaired cells / total actual errors
- **F1 Score**: harmonic mean of accuracy and recall
- **EDR**: error detection rate
- **Hybrid Distance**: weighted combination of MSE (numerical) and Jaccard distance (categorical)

### Downstream Task Performance
- **Classification**: accuracy, weighted F1, precision, recall
- **Regression**: MSE, MAE, R²
- **Clustering**: silhouette score, adjusted Rand index

### Model Tolerance
- **Prior tolerance**: `P_cleaned / P_dirty` — relative improvement over doing nothing
- **Posterior tolerance**: `(P_cleaned - P_dirty) / (P_clean - P_dirty)` — fraction of the optimal gap recovered

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{anonymous2026demandclean,
  title={DemandClean: Demand-Driven Data Cleaning for Machine Learning via Deep Reinforcement Learning},
  author={Anonymous},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
