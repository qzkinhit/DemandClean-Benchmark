# tools/ — Evaluation and Auxiliary Utilities

## Overview

This directory contains evaluation tools, data processing utilities, and analysis scripts used throughout the DemandClean pipeline.

## File List

### Core Evaluation

| File | Description |
|------|-------------|
| `getScore.py` | Traditional data cleaning metrics (accuracy, recall, F1, EDR, hybrid distance, R-EDR) |
| `getScoreML.py` | Unified Clean4ML evaluation (downstream task + tolerance + Snoopy + cost) |

### Data Processing

| File | Description |
|------|-------------|
| `readData.py` | Data loading utilities |
| `saveData.py` | Data saving utilities |
| `inject_errors.py` | Error injection tool (random and systematic errors) |
| `insert_null.py` | Null value insertion tool |
| `get_error_num.py` | Error statistics counter |
| `get_subset.py` | Data subset extraction |
| `rules_parser.py` | FD rule parser (legacy, see `demandclean/detectors/rule_parser.py` for the current version) |

### Vectorization

| File | Description |
|------|-------------|
| `adult_vectorize.py` | Adult dataset vectorization (TF-IDF + StandardScaler) |
| `eeg_vectorize.py` | EEG Eye State dataset vectorization |

### Visualization

| File | Description |
|------|-------------|
| `get_plt.py` | Result visualization plots |
| `resultPLT.py` | Result plotting utilities |

### Analysis

| File | Description |
|------|-------------|
| `shapley_analysis.py` | Shapley value analysis for 3 dimensions (action, feature, error type importance) |
| `tolerance_analysis.py` | Model tolerance threshold analysis |
| `get_T_table.py` | T-table generation |

### Sub-directories

| Directory | Description |
|-----------|-------------|
| `generate_index/` | Index generation tools for datasets (`clean_index.py`, `dirty_index.py`, `description.py`) |
| `snoopy/` | Snoopy data quality upper bound evaluation tool (external library) |

## Key APIs

### getScore.py — Traditional Cleaning Metrics

```python
from tools.getScore import calculate_all_metrics

results = calculate_all_metrics(
    clean, dirty, cleaned, attributes,
    output_path, task_name, index_attribute
)
# Returns: accuracy, recall, f1_score, edr, hybrid_distance, r_edr
```

**Metrics:**
- **Accuracy**: correctly repaired cells / total repaired cells
- **Recall**: correctly repaired cells / total cells that need repair
- **F1**: harmonic mean of accuracy and recall
- **EDR**: Error Distance Reduction = (D_dirty - D_cleaned) / D_dirty
- **Hybrid Distance**: MSE (numeric) + Jaccard (categorical)
- **R-EDR**: Record-based EDR (per-row error reduction)

**Data alignment**: Uses three-way index intersection (`clean ∩ dirty ∩ cleaned`) to handle row deletions.

### getScoreML.py — Unified Evaluation

```python
from tools.getScoreML import run_all_evaluation

results = run_all_evaluation(
    dirty_path, cleaned_path, clean_path,
    label_col, task_type, task_name, output_dir
)
```

**5-module evaluation pipeline:**
1. Traditional cleaning metrics (getScore)
2. Downstream task performance (RF, LR, SVM, KNN, DT, GB)
3. Model tolerance (prior and posterior)
4. Snoopy upper bound
5. Ground truth cost analysis
