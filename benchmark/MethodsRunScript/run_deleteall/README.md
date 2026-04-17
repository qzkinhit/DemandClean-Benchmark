# DeleteAll Baseline

## Overview

DeleteAll is a deletion-based baseline that "cleans" data by dropping rows that contain problems. Two modes are supported.

## Method Types

| Mode | Type | Description |
|------|------|------|
| drop_missing | **Type 1** | Drops rows containing missing values, fully automatic |
| drop_errors | **Type 2** | Drops rows that differ from the clean data, requires ground truth |

## Usage

```bash
# drop_missing mode (default) — drop rows with missing values
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_deleteall \
    --mode drop_missing \
    --label_column style \
    --task_type classification

# drop_errors mode — drop all erroneous rows
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_deleteall_errors \
    --mode drop_errors \
    --label_column style

# Batch run on all datasets
bash MethodsRunScript/run_deleteall/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data | - |
| `--task_name` | Yes | Task name | - |
| `--mode` | No | Deletion mode | `drop_missing` |
| `--output_path` | No | Output path | `results/deleteall/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Mode Comparison

| Mode | Detection | Use case | Row-count impact |
|------|----------|----------|------------|
| `drop_missing` | Empty value / NaN detection | When missing values are few | Drops a small number of rows |
| `drop_errors` | Comparison against ground truth | When ground truth is available | May drop many rows |

## Output Files

```
results/deleteall/{task_name}/
├── {task_name}_cleaned.csv          # Data after deletion
├── {task_name}_total_evaluation.txt # Full evaluation report
└── {task_name}.log                  # Run log
```

## Notes

- `drop_errors` mode may drastically reduce the row count
- Dropping rows can affect data distribution and model training
- Best suited to cases with few erroneous rows
