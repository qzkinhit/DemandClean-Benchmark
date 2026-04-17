# Dataset: Mercedes

## Basic Information

| Item | Value |
|------|-------|
| Task type | Regression |
| Target column | `y` |
| Size | 4,209 records x 378 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (376 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| X0-X8 | Categorical | Categorical features |
| X10-X385 | Binary | Anonymized vehicle configuration flags (0/1) |

**Note**: feature columns are named X0, X1, X2, ... X385 (376 columns; some indices are missing: X7, X9, X72, X121, X149, X188, X193, X303, X381).

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| y | Numeric | Vehicle test time in seconds |

## Error Information
- **Error types**: Missing values, outliers, implicit missing values
- **Error entry count**: 4,209
- **Error cell count**: 301,972

## Source
Daimler. Mercedes-Benz Greener Manufacturing. https://www.kaggle.com/c/mercedes-benz-greener-manufacturing. 2017.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/mercedes/dirty_with_index.csv \
  --clean_path Data/mercedes/clean_with_index.csv \
  --task_name mercedes_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column y \
  --task_type regression \
  --index_attribute index \
  --verbose
```
