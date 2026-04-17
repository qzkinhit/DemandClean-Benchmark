# Dataset: HAR (Human Activity Recognition)

## Basic Information

| Item | Value |
|------|-------|
| Task type | Clustering |
| Target column | `gt` |
| Size | 70,000 records x 5 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (3 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| x | Numeric | Accelerometer X-axis signal |
| y | Numeric | Accelerometer Y-axis signal |
| z | Numeric | Accelerometer Z-axis signal |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| gt | Multiclass | Activity ground-truth label |

## Error Information
- **Error types**: Missing values, outliers
- **Error entry count**: 38,891
- **Error cell count**: 51,180

## Source
Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. 2013.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/har/dirty_with_index.csv \
  --clean_path Data/har/clean_with_index.csv \
  --task_name har_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column gt \
  --task_type clustering \
  --index_attribute index \
  --verbose
```
