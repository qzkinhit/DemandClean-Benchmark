# MLImputer Baseline

## Overview

MLImputer is an ML-based missing-value imputer that predicts and fills missing values using MICE, KNN, or Random Forest.

**Implementation**: Built on scikit-learn's `IterativeImputer` and `KNNImputer`.

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Usage

```bash
# Use MICE (default)
python MethodsRunScript/run_mlimputer/run_mlimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_mlimputer \
    --output_path results/mlimputer/ \
    --method mice \
    --label_column style \
    --task_type classification

# Use KNN
python MethodsRunScript/run_mlimputer/run_mlimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_mlimputer_knn \
    --method knn

# Batch run on all datasets
bash MethodsRunScript/run_mlimputer/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (for evaluation) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/mlimputer/` |
| `--method` | No | Imputation method | `mice` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Supported Imputation Methods

| Method | Description | Characteristics |
|------|------|------|
| `mice` | Multiple Imputation by Chained Equations | Iterative, good quality |
| `knn` | K-Nearest Neighbors | Based on similar samples |
| `rf` | Random Forest | Uses a random forest for prediction |

## Output Files

```
results/mlimputer/{task_name}/
├── {task_name}_cleaned.csv          # Imputed data
├── {task_name}_total_evaluation.txt # Full evaluation report
└── {task_name}.log                  # Run log
```

## Comparison with SimpleImputer

| Method | Approach | Quality | Speed |
|------|------|------|------|
| SimpleImputer | Statistical filling | Fair | Fast |
| MLImputer | ML-based prediction | Better | Slower |

## Notes

- MICE can be slow on large datasets
- Handles missing values only — does not repair erroneous values
- Requires numeric encoding before use
