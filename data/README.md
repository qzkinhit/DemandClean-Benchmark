# data/ — Benchmark Datasets

## Overview

This directory contains 9 benchmark datasets for data cleaning experiments. Each dataset includes clean data, dirty data (with injected/real errors), and FD/domain rules.

## Dataset Summary

| Dataset | Task | Model | Features | Records | Label Column | Error Types |
|---------|------|-------|----------|---------|-------------|-------------|
| **adult** | Classification | Random Forest | 15 | 45,222 | income | Rule violations, outliers, label noise |
| **beers** | Classification | XGBoost | 8 | 2,410 | style | Missing, outliers, label noise |
| **bike** | Regression | Random Forest | 12 | 17,379 | cnt | Missing, noise |
| **breast_cancer** | Classification | Random Forest | 31 | 569 | diagnosis | Missing |
| **har** | Classification | Random Forest | 562 | 10,299 | Activity | Missing, sensor noise |
| **mercedes** | Regression | Ridge | 377 | 4,209 | y | Missing |
| **nasa** | Regression | Ridge | 6 | 1,503 | Sound_pressure | Missing, noise |
| **smartfactory** | Classification | Random Forest | 6 | 7,936 | Machine_failure | Missing, outliers |
| **soilmoisture** | Regression | Ridge | 14 | 1,778 | soil_moisture | Missing |

## File Structure

Each dataset directory contains:

```
{dataset_name}/
├── README.md           # Dataset description
├── clean_index.csv     # Clean data with index column
├── dirty_index.csv     # Dirty data with index column (errors injected)
└── rules.txt           # FD/DOMAIN/CFD/REGEX rules for error detection & injection
```

## Rules File Format

Rules files support 6 section types used by DemandClean's `AutoDetector` and `ErrorInjector`:

```
[REGEX]
col_name: ^pattern$

[DOMAIN]
col_name: INT [min, max]
col_name: FLOAT [min, max]
col_name: ENUM {val1, val2, ...}

[FD]
lhs_col -> rhs_col

[CFD]
condition => col_name EXCESS >= threshold FROM_BASELINE baseline

[DC]
NOT(t1.col < 0)

[STATISTICAL]
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
```

## Index Column

- The `index` column (first column in CSV) is used for row tracking during cleaning
- Index remains consistent between clean and dirty versions
- Evaluation uses index-based alignment to handle row deletions

## Adding New Datasets

1. Create a directory named after the dataset under `data/`
2. Prepare `clean_index.csv` (ground truth) and `dirty_index.csv` (with errors)
3. Write `rules.txt` with DOMAIN/FD/CFD rules
4. Add `README.md` describing data source and error characteristics
5. Register the dataset in `run_demandclean/run_demandclean_base.py`
