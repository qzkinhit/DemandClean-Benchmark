# SimpleImputer Baseline

## Overview

SimpleImputer is a scikit-learn-based missing-value imputer that fills missing values with simple statistics (mean, median, most frequent value, etc.).

**Paper**: Built-in method of scikit-learn.

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Usage

```bash
# Default strategy (mean imputation)
python MethodsRunScript/run_simpleimputer/run_simpleimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_simpleimputer \
    --output_path results/simpleimputer/ \
    --label_column style \
    --task_type classification

# Specify the imputation strategy
python MethodsRunScript/run_simpleimputer/run_simpleimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_simpleimputer \
    --strategy median

# Batch run on all datasets
bash MethodsRunScript/run_simpleimputer/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (for evaluation) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/simpleimputer/` |
| `--strategy` | No | Imputation strategy | `mean` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Supported Strategies

| Strategy | Description | Applicable Type |
|------|------|----------|
| `mean` | Mean imputation | Numeric columns |
| `median` | Median imputation | Numeric columns |
| `most_frequent` | Most-frequent imputation | All types |
| `constant` | Constant imputation | All types |

## Output Files

```
results/simpleimputer/{task_name}/
├── {task_name}_cleaned.csv          # Imputed data
├── {task_name}_total_evaluation.txt # Full evaluation report
└── {task_name}.log                  # Run log
```

## Key Features

- Simple and efficient, suitable for quick benchmarking
- Handles missing values only — does not repair erroneous values
- Numeric and categorical columns are handled separately
