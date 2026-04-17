# RepairAll Baseline

## Overview

RepairAll is a ground-truth-based repair baseline that directly replaces erroneous values in the dirty data with their clean counterparts. It represents the **theoretical upper bound** of cleaning effectiveness.

## Method Type

| Type | Description |
|------|------|
| **Type 2** | Requires full ground truth |
| Ground-truth usage | 100% (uses all ground truth) |

## Usage

```bash
# Single dataset
python MethodsRunScript/run_repairall/run_repairall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_repairall \
    --output_path results/repairall/ \
    --label_column style \
    --task_type classification

# Batch run on all datasets
bash MethodsRunScript/run_repairall/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (ground truth) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/repairall/` |
| `--index_attribute` | No | Index column name | `index` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Output Files

```
results/repairall/{task_name}/
├── {task_name}_cleaned.csv          # Repaired data (= clean data)
├── {task_name}_total_evaluation.txt # Full evaluation report
└── {task_name}.log                  # Run log
```

## Purpose

- Establishes the **theoretical upper bound** on cleaning effectiveness
- Measures the gap between other methods and "perfect repair"
- Validates the downstream-task performance achievable on fully clean data

## Comparison with DoNothing

| Method | Output | Purpose |
|------|------|------|
| DoNothing | Original dirty data | Lower bound on performance |
| RepairAll | Fully clean data | Upper bound on performance |
