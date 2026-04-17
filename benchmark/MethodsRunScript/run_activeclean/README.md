# ActiveClean

## Overview

ActiveClean is a model-oriented iterative data cleaning method. It uses active learning to prioritize the dirty samples that most affect the downstream model.

**Paper**: [ActiveClean: Interactive Data Cleaning For Statistical Modeling](https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf) (VLDB 2016)

## Method Type

| Type | Description |
|------|-------------|
| **Type 3** | Iterative active learning, requires human labeling |
| Ground truth | On-demand (batch_size samples per round) |

## Core Idea

1. Train an initial model.
2. Use model gradients to identify the dirty samples with the largest impact on the model.
3. Ask the user to clean the selected samples.
4. Update the model and repeat steps 2-4 until convergence.

## How to Run

```bash
# Single-dataset run
python MethodsRunScript/run_activeclean/run_activeclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_activeclean \
    --output_path results/activeclean/ \
    --label_column style \
    --task_type classification \
    --batch_size 50 \
    --total_budget 500

# Batch run across all datasets
bash MethodsRunScript/run_activeclean/run.sh
```

## Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--dirty_path` | yes | Path to the dirty data | - |
| `--clean_path` | yes | Path to the clean data (simulated human labeling) | - |
| `--task_name` | yes | Task name | - |
| `--output_path` | no | Output directory | `results/activeclean/` |
| `--label_column` | yes | Label column name | - |
| `--task_type` | no | Task type | `classification` |
| `--batch_size` | no | Samples cleaned per round | `50` |
| `--total_budget` | no | Total cleaning budget | `10000` |
| `--models` | no | Evaluation model list | `rf lr` |

## Output Files

```
results/activeclean/{task_name}/
├── {task_name}_cleaned.csv          # Cleaned data
├── {task_name}_total_evaluation.txt # Full evaluation report (incl. ground-truth usage cost)
└── {task_name}.log                  # Run log
```

## Highlights

- **Model-aware optimization**: prioritizes samples that most improve the downstream model.
- **Budget control**: bounded cleaning budget caps human cost.
- **Iterative convergence**: keeps improving until model performance converges.

## Comparison with Related Methods

| Method | Cleaning Strategy | Ground Truth |
|--------|-------------------|--------------|
| ActiveClean | Gradient-guided selection | On-demand (Type 3) |
| BoostClean | Boosting-based selection | Validation set (Type 2) |
| Raha/Baran | Active learning | On-demand (Type 3) |
