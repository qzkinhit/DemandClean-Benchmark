# Utils Directory

This directory contains utility functions commonly used in the data cleaning experiments.

## Files

### Evaluation

| File | Description |
|------|-------------|
| `getScore.py` | Traditional data quality metrics (accuracy, recall, EDR, etc.) |
| `getScoreML.py` | Downstream task evaluation and model tolerance |

### Data processing

| File | Description |
|------|-------------|
| `inject_errors.py` | Error injection utility (random/system errors) |
| `insert_null.py` | Null value insertion utility |
| `get_error_num.py` | Error counting utility |

### Data transformation

| File | Description |
|------|-------------|
| `adult_vectorize.py` | Adult dataset vectorization |
| `eeg_vectorize.py` | EEG dataset vectorization |

### Visualization

| File | Description |
|------|-------------|
| `get_plt.py` | Result visualization |
| `resultPLT.py` | Result plotting |

### Helpers

| File | Description |
|------|-------------|
| `readData.py` | Data loading |
| `saveData.py` | Data saving |
| `get_subset.py` | Subset extraction |
| `generate_index/` | Index-generation tools |

---

## Key Functions

### getScore.py — Traditional Metrics

```python
from utils.getScore import calculate_all_metrics

results = calculate_all_metrics(
    clean, dirty, cleaned, attributes,
    output_path, task_name, index_attribute
)
# Returns: accuracy, recall, f1_score, edr, hybrid_distance, r_edr
```

### getScoreML.py — Downstream Task Evaluation

```python
from utils.getScoreML import comprehensive_evaluation

results = comprehensive_evaluation(
    dirty_data, cleaned_data, clean_data,
    label_column='label',
    task_type='classification',
    models=['rf', 'lr'],
    method_type=1,
    ground_truth_used=0
)
# Returns: downstream task performance, tolerance, ground-truth usage cost
```

### inject_errors.py — Error Injection

```python
from utils.inject_errors import inject_random_error, inject_system_error

# Random error injection
dirty_df = inject_random_error(clean_df, percent=0.1)

# System error injection (based on model importance)
dirty_df = inject_system_error(clean_df, percent=0.1, target_column='label')
```

---

## Details

### `inject_error.py`

Performs **error injection** on the vectorized adult and eeg feature datasets. Two injection types are supported: **random errors** and **system errors**.

#### Key functions

1. **`inject_random_error(df, percent)`**:
   - Randomly selects a fraction of rows and replaces every numeric feature in those rows with 3x the column maximum.

2. **`inject_system_error(df, percent, target_column)`**:
   - Uses SGDClassifier feature weights to pick the top `x%` of rows and replaces their three most important features with the column mean.

#### Command-line examples

```bash
# Random error injection
python inject_errors.py --input adult_data_vectorized.csv --output adult_with_random_errors.csv --error_type random --percent 5

# System error injection
python inject_error.py --input adult_vectorized.csv --output adult_with_system_errors.csv --error_type system --percent 10
```

---

### `eeg_vectorize.py`

Vectorizes the **EEG Eye State dataset** by extracting per-timestep statistical features.

#### Command-line example

```bash
python vectorize_eeg.py --input eeg_eye_state.arff --output eeg_vectorized.csv
```

---

### `adult_vectorize.py`

Vectorizes the **Adult dataset**.

#### Feature processing

1. **Numeric features** (`age`, `fnlwgt`, `education-num`, `hours-per-week`): standardized.
2. **Categorical features** (`workclass`, `education`, etc.): TF-IDF bag-of-words encoded.
3. **Income label**: `<=50K` -> 0, `>50K` -> 1.

#### Command-line example

```bash
python adult_vectorize.py --input adult.csv --output adult_vectorized.csv
```

---

## Metric Reference

### Traditional metrics

- **Accuracy**: correctly repaired / total repaired
- **Recall**: correctly repaired / total actual errors
- **F1**: 2 * Accuracy * Recall / (Accuracy + Recall)
- **EDR**: (D_dirty_to_clean - D_repaired_to_clean) / D_dirty_to_clean
- **R-EDR**: record-based error reduction rate

### Downstream task metrics

- **Classification**: Accuracy, F1, Precision, Recall
- **Regression**: MSE, MAE, R^2
- **Clustering**: Silhouette Score, ARI

### Model tolerance

- **Prior tolerance**: P_demand_clean / P_do_nothing
- **Posterior tolerance**: (P_demand_clean - P_do_nothing) / (P_repair_all - P_do_nothing)
