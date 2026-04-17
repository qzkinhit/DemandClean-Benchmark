# UniClean Baseline

## Overview

UniClean is a multi-signal unified data cleaning framework built on PySpark for distributed processing. It applies a suite of predefined cleaners (range checks, pattern matching, outlier detection, etc.) to rule-based data cleaning.

**Paper**: UniClean: A Multi-Signal Unified Data Cleaning Framework (VLDB 2025)

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Core Ideas

1. Defines a set of cleaners (e.g., numeric range checks, pattern matches, outlier detection)
2. Applies the appropriate cleaner to each column
3. Cleaners repair the data automatically based on rules

## Usage

```bash
# Single dataset
python MethodsRunScript/run_uniclean/run_uniclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --dataset beers \
    --task_name beers_uniclean \
    --output_path results/uniclean/ \
    --label_column style \
    --task_type classification

# Batch run on all datasets
bash MethodsRunScript/run_uniclean/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dirty_path` | Yes | Path to dirty data | - |
| `--clean_path` | Yes | Path to clean data (for evaluation) | - |
| `--dataset` | Yes | Dataset name | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/uniclean/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--single_max` | No | Max records processed per batch | `10000` |
| `--executor_memory` | No | Spark executor memory | `8g` |
| `--driver_memory` | No | Spark driver memory | `8g` |

## Rule File Format

Cleaners are defined in the `[UNICLEAN]` section of `rules.txt`:

```
[UNICLEAN]
# Numeric-type checks
Number("ibu")
Number("abv")

# Pattern matching
Pattern("phone", r"^\d{3}-\d{4}$")

# Outlier detection
Outlier("price", [], "price_outlier")

# Attribute relationship
AttrRelation("brewery_id", ["brewery_name", "city", "state"])
```

## Supported Cleaners

| Cleaner | Description | Example |
|--------|------|------|
| `Number(col)` | Numeric-type check | `Number("price")` |
| `Pattern(col, regex)` | Regex pattern match | `Pattern("email", r".*@.*")` |
| `Outlier(col, bounds, name)` | Outlier detection | `Outlier("age", [0, 120], "age_check")` |
| `Date(col, format)` | Date-format check | `Date("date", "%Y-%m-%d")` |
| `AttrRelation(key, deps)` | Attribute-dependency relationship | `AttrRelation("id", ["name"])` |

## Output Files

```
results/uniclean/{task_name}/
├── {task_name}_cleaned.csv          # Cleaned data
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Requirements

- PySpark
- Java 8+

## Notes

- Requires a configured Spark environment
- Memory parameters may need tuning on large datasets
- Cleaning quality depends on the configuration of the cleaners
