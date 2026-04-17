# Raha & Baran Baseline

## Overview

Raha is a configuration-free ensemble error-detection system; Baran is the corresponding error-repair system. Together they provide an end-to-end data cleaning pipeline.

**Papers**:
- [Raha: A Configuration-Free Error Detection System](https://dl.acm.org/doi/10.1145/3299869.3324956) (SIGMOD 2019)
- [Baran: Effective Error Correction via a Unified Context Representation](https://www.vldb.org/pvldb/vol13/p1948-mahdavi.pdf) (VLDB 2020)

## Method Type

| Type | Description |
|------|------|
| **Type 3** | Requires user-labeled examples |
| Ground-truth usage | On-demand labeling (`labeling_budget`) |

## Core Ideas

### Raha (error detection)
1. Runs multiple error-detection strategies (outliers, pattern violations, FD violations, etc.)
2. Clusters the candidate error features
3. Asks the user to label a small sample
4. Trains a classifier to identify all errors

### Baran (error repair)
1. Builds a unified context representation
2. Generates candidate repair values
3. Ranks repair suggestions using contextual similarity
4. Applies the best repair

## Usage

```bash
# Single dataset
python MethodsRunScript/run_raha_baran/run_raha_baran_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_raha_baran \
    --output_path results/raha_baran/ \
    --label_column style \
    --task_type classification \
    --labeling_budget 20

# Batch run on all datasets
bash MethodsRunScript/run_raha_baran/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (simulates user labels) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/raha_baran/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--labeling_budget` | No | Labeling budget (number of tuples) | `20` |
| `--models` | No | Evaluation models | `rf lr` |

## Output Files

```
results/raha_baran/{task_name}/
├── {task_name}_cleaned.csv          # Cleaned data
├── {task_name}_detection.csv        # Detected errors
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Key Features

- **Configuration-free**: No predefined rules or thresholds required
- **Ensemble strategies**: Combines the strengths of multiple detection methods
- **Active learning**: Intelligently selects the most informative tuples to label
- **Context-aware**: Leverages data context to improve repair accuracy

## Comparison with Other Methods

| Method | Detection | Repair | Ground-truth usage |
|------|----------|----------|----------|
| **Raha/Baran** | Multi-strategy ensemble | Context similarity | Small set of labels |
| ActiveClean | - | Gradient-guided | On-demand cleaning |
| HoloClean | DC constraints | Probabilistic inference | None |

## Notes

- Labeling budget directly affects detection accuracy
- Works best on datasets with sufficient row counts
- Detection and repair are two separate stages
