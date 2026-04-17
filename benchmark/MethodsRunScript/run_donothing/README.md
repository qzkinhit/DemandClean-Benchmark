# DoNothing Baseline

## Overview

DoNothing is the simplest baseline: it **performs no cleaning** and returns the original dirty data as-is. It establishes a lower bound on performance and validates the improvement provided by other cleaning methods.

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Usage

```bash
# Single dataset
python MethodsRunScript/run_donothing/run_donothing_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_donothing \
    --output_path results/donothing/ \
    --label_column style \
    --task_type classification \
    --models rf lr

# Batch run on all datasets
bash MethodsRunScript/run_donothing/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (for evaluation) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/donothing/` |
| `--index_attribute` | No | Index column name | `index` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Supported Task Types

| Task Type | Models |
|----------|------|
| classification | rf, lr, svm, knn, dt, gb |
| regression | rf, lr, ridge, lasso, knn, gb |
| clustering | kmeans, agglomerative |

## Output Files

```
results/donothing/{task_name}/
├── {task_name}_cleaned.csv          # Output data (identical to input)
├── {task_name}_total_evaluation.txt # Full evaluation report
└── {task_name}.log                  # Run log
```

## Purpose

- Establishes a lower bound on cleaning effectiveness
- Verifies whether other cleaning methods provide real improvement
- Benchmarks the downstream-task performance achievable on raw dirty data
