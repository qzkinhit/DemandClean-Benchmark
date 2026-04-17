# HoloClean Baseline

## Overview

HoloClean is a probabilistic-graphical-model-based data cleaning system that combines multiple signals (constraints, statistics, knowledge bases) to automatically repair data errors.

**Paper**: [HoloClean: Holistic Data Repairs with Probabilistic Inference](https://www.vldb.org/pvldb/vol10/p1190-rekatsinas.pdf) (VLDB 2017)

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Core Ideas

1. Uses Denial Constraints (DC) to detect errors
2. Generates candidate repair values
3. Builds a factor graph fusing multiple signals
4. Uses probabilistic inference to pick the best repair

## Usage

```bash
# Single dataset
python MethodsRunScript/run_holoclean/run_holoclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --rule_path Data/beers/rules.txt \
    --task_name beers_holoclean \
    --output_path results/holoclean/ \
    --label_column style \
    --task_type classification

# Batch run on all datasets
bash MethodsRunScript/run_holoclean/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | No | Path to clean data (for evaluation) | - |
| `--rule_path` | No | Path to the rules file | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/holoclean/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--db_user` | No | PostgreSQL user | `holocleanuser` |
| `--db_name` | No | Database name | `holo` |
| `--epochs` | No | Training epochs | `10` |
| `--learning_rate` | No | Learning rate | `0.001` |
| `--threads` | No | Number of threads | `1` |
| `--weak_label_thresh` | No | Weak-label threshold | `0.99` |
| `--models` | No | Evaluation models | `rf lr` |

## Rule File Format

The rules file must contain a `[HOLOCLEAN_DC]` section:

```
[HOLOCLEAN_DC]
# Denial Constraint format
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.brewery_name,t2.brewery_name)
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.city,t2.city)
```

**DC syntax**:
- `t1&t2`: Declares two tuple variables
- `EQ(t1.attr, t2.attr)`: Equality predicate
- `IQ(t1.attr, t2.attr)`: Inequality predicate
- `LT/GT/LTE/GTE`: Comparison predicates

## Requirements

- **PostgreSQL**: A `holo` database must be created
- **Python 3.7**: HoloClean upstream only supports Python 3.7
- PyTorch, psycopg2, sqlalchemy

### PostgreSQL Setup

```sql
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
GRANT ALL ON SCHEMA public TO holocleanuser;
```

## Output Files

```
results/holoclean/{task_name}/
├── {task_name}_cleaned.csv          # Repaired data
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Comparison with Horizon

| Method | Constraint Type | Repair Strategy | Characteristics |
|------|----------|----------|------|
| Horizon | FD | Pattern quality | Fast, scalable |
| **HoloClean** | DC | Probabilistic inference | More flexible, signal fusion |

## Notes

- Python 3.10+ may have compatibility issues
- Requires PostgreSQL
- Slow and memory-intensive on large datasets
- Skips repair when no DC rules are provided
