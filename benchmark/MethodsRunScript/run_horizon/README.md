# Horizon Baseline

## Overview

Horizon is a scalable data cleaning method based on Functional Dependencies (FDs). It identifies and repairs FD violations to clean the data.

**Paper**: [Horizon: Scalable Dependency-Driven Data Cleaning](https://www.vldb.org/pvldb/vol14/p2546-yan.pdf) (VLDB 2021)

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Core Ideas

1. Parses functional-dependency rules (e.g., `A => B`)
2. Builds an FD pattern graph and scores the quality of each pattern
3. Uses SCC analysis and topological sort to decide the repair order
4. Selects optimal repair values based on pattern quality

## Usage

```bash
# Single dataset
python MethodsRunScript/run_horizon/run_horizon_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --rule_path Data/beers/rules.txt \
    --task_name beers_horizon \
    --output_path results/horizon/ \
    --label_column style \
    --task_type classification

# Batch run on all datasets
bash MethodsRunScript/run_horizon/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (for evaluation) | - |
| `--rule_path` | **Yes** | Path to the rules file | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/horizon/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--models` | No | Evaluation models | `rf lr` |

## Rule File Format

The rules file must contain a `[HORIZON_FD]` section:

```
[HORIZON_FD]
# Format: LHS => RHS
brewery_id => brewery_name
brewery_id => city
brewery_id => state
style => abv
```

**Rule syntax**:
- `LHS => RHS`: The left-hand attribute determines the right-hand attribute
- Supports both `=>` and `⇒` arrow syntax
- Lines starting with `#` are comments

## Output Files

```
results/horizon/{task_name}/
├── {task_name}_cleaned.csv          # Repaired data
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Comparison with HoloClean

| Method | Constraint Type | Repair Strategy | Characteristics |
|------|----------|----------|------|
| **Horizon** | FD (functional dependency) | Pattern-quality first | Fast, scalable |
| HoloClean | DC (denial constraint) | Probabilistic inference | More flexible, slower |

## Notes

- **A rules file is required**; otherwise Horizon cannot run
- Only repairs attributes that appear in the FDs; other attributes are untouched
- Rule quality directly affects cleaning effectiveness
- Works best on datasets with well-defined business rules
