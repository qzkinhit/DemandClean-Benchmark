# data/ — Benchmark Datasets

## Overview

This directory contains **12 benchmark datasets** from two sources, all used with clean/dirty pairs
adopted **unmodified** (we only append a row `index` column and normalize missing-value markers):

- **9 REIN datasets** (`adult`, `beers`, `bike`, `breast_cancer`, `har`, `mercedes`, `nasa`, `smartfactory`, `soilmoisture`) — from the public [REIN benchmark](https://github.com/mohamedyd/rein-benchmark) (Abdelaal et al., EDBT 2023).
- **3 UniClean real-world datasets** (`hospitals`, `flights`, `soccer`) — from the UniClean benchmark, carrying **native real-world errors** (not synthetic). Added to evaluate generalization to naturalistic error distributions.

## Data Provenance

**REIN datasets (9):**
- **`beers`** carries **real-world errors** in its native dirty form (from the Raha/Baran lineage) and is used unmodified.
- The **remaining eight** are real-world tables whose **errors were synthesized by the REIN authors** from the clean reference versions (using REIN's BART-based error generator), covering missing values, semantic, and syntactic errors.

**UniClean real-world datasets (3):**
- **`hospitals`** (US hospital quality records): native **typos** (systematic l→x) + FD violations (~3% error rate).
- **`flights`** (flight arrival/departure times): native **missing values + format inconsistency** in time columns (38–66% per-column corruption). Downstream label `arrival_delay_bucket` is derived from scheduled vs. actual arrival times (kept as features).
- **`soccer`** (10k subset of 200k player records): native missing/format errors. Downstream label = `manager`.

**We do not inject any errors into these benchmark datasets ourselves.** The adversarial error injection in the DemandClean training pipeline (`error_injector.py`) is a **separate, self-supervised mechanism** applied only to the agent's training base `D_base`, and is independent of these benchmark datasets.

> REIN: Mohamed Abdelaal, Christian Hammacher, Harald Schöning. *REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines.* EDBT 2023, pp. 499–511. doi:10.48786/EDBT.2023.43

## Dataset Summary

| Dataset | Task | Model | Features | Records | Label Column | Error Types |
|---------|------|-------|----------|---------|-------------|-------------|
| **adult** | Classification | Random Forest | 15 | 45,222 | income | Rule violations, outliers, label noise |
| **beers** | Classification | Random Forest | 9 | 2,410 | style | Missing, outliers, label noise |
| **bike** | Regression | Random Forest | 16 | 17,379 | cnt | Missing, noise |
| **breast_cancer** | Classification | Random Forest | 10 | 699 | class | Missing |
| **har** | Clustering | KMeans | 4 | 70,000 | gt | Missing, sensor noise |
| **mercedes** | Regression | Ridge | 377 | 4,209 | y | Missing |
| **nasa** | Regression | Ridge | 6 | 1,503 | sound_pressure_level | Missing, noise |
| **smartfactory** | Classification | Random Forest | 19 | 23,645 | labels | Missing, outliers |
| **soilmoisture** | Regression | Ridge | 128 | 679 | soil_moisture | Missing |
| **hospitals** ⁺ | Classification | Random Forest | 18 | 1,000 | Condition | Typos, FD violations |
| **flights** ⁺ | Classification | Random Forest | 6 | 2,376 | arrival_delay_bucket | Missing, format inconsistency |
| **soccer** ⁺ | Classification | Random Forest | 9 | 10,610 | manager | Missing, format errors |

> ⁺ = UniClean real-world datasets (native errors). **Records/Features** are counted from the actual CSVs (features exclude `index`). For the per-dataset breakdown in the paper, see Table 2 of the main paper.

## File Structure

Each dataset directory contains:

```
{dataset_name}/
├── README.md           # Dataset description
├── clean_index.csv     # Clean reference version (REIN clean), with index column
├── dirty_index.csv     # Dirty version (REIN dirty), with index column
└── rules.txt           # FD/DOMAIN/CFD/REGEX rules used for error detection
```

## Rules File Format

Rules files support 6 section types. They are consumed by DemandClean's `AutoDetector` for error detection on these benchmark datasets, and the same rule schema is reused by the training-time `ErrorInjector` when generating adversarial samples on the agent's own training base `D_base` (not on the benchmark data):

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
2. Prepare `clean_index.csv` (ground truth) and `dirty_index.csv` (dirty version)
3. Write `rules.txt` with DOMAIN/FD/CFD rules
4. Add `README.md` describing data source and error characteristics
5. Register the dataset in `run_demandclean/run_demandclean_base.py`
