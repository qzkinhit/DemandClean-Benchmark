# BoostClean Baseline

## Overview

BoostClean is a model-oriented cleaning method that uses a Boosting strategy to ensemble multiple detectors and repairers, automatically selecting the combination that best improves the downstream ML model.

**Paper**: [BoostClean: Automatic Error Detection and Repair for Machine Learning](https://arxiv.org/abs/1711.01299)

## Method Type

| Type | Description |
|------|-------------|
| **Type 2** | Requires a validation-set ground truth |
| Ground truth | Validation set (validation_ratio) |

## Core Idea

1. Define a suite of error detectors (missing value, outlier, type error, etc.).
2. Define a suite of repairers (delete, impute, rule-based repair, etc.).
3. Use Boosting to iteratively pick the best detector-repairer combination.
4. Optimize against validation-set model performance.

## How to Run

```bash
# Single dataset
python MethodsRunScript/run_boostclean/run_boostclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_boostclean \
    --output_path results/boostclean/ \
    --label_column style \
    --task_type classification \
    --boosting_rounds 5

# Batch run across all datasets
bash MethodsRunScript/run_boostclean/run.sh
```

## Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--dirty_path` | yes | Path to the dirty data | - |
| `--clean_path` | yes | Path to the clean data | - |
| `--task_name` | yes | Task name | - |
| `--output_path` | no | Output directory | `results/boostclean/` |
| `--label_column` | yes | Label column name | - |
| `--task_type` | no | Task type | `classification` |
| `--boosting_rounds` | no | Number of Boosting rounds | `5` |
| `--quantitative_thresh` | no | Numeric-outlier threshold | `10` |
| `--models` | no | Evaluation model list | `rf lr` |

## Output Files

```
results/boostclean/{task_name}/
├── {task_name}_cleaned.csv          # Cleaned data
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Comparison with ActiveClean

| Method | Selection Strategy | Ground Truth | Optimization Target |
|--------|--------------------|--------------|---------------------|
| ActiveClean | Gradient-guided | On-demand | Model loss |
| **BoostClean** | Boosting | Validation set | Validation set performance |

## Notes

- Requires a validation split; small datasets may suffer.
- More rounds usually yield better results but cost more compute.
- Well suited to scenarios with a clear downstream task.
