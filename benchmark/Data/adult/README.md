# Dataset: Adult

## Basic Information

| Item | Value |
|------|-------|
| Task type | Classification |
| Target column | `income` |
| Size | 45,222 records x 15 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (14 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| age | Numeric | Age |
| workclass | Categorical | Class of employer |
| fnlwgt | Numeric | Final weight |
| education | Categorical | Education level |
| educational_num | Numeric | Years of education |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Occupation |
| relationship | Categorical | Family relationship |
| race | Categorical | Race |
| gender | Categorical | Gender |
| capital_gain | Numeric | Capital gain |
| capital_loss | Numeric | Capital loss |
| hours_per_week | Numeric | Hours worked per week |
| native_country | Categorical | Country of origin |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| income | Binary | Whether income exceeds 50K (0/1) |

## Error Information
- **Error types**: Rule violations, outliers
- **Error entry count**: 1,701
- **Error cell count**: 1,701

## Source
Becker, B. & Kohavi, R. Adult. https://archive.ics.uci.edu/dataset/2/adult. 1996.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/adult/dirty_with_index.csv \
  --clean_path Data/adult/clean_with_index.csv \
  --task_name adult_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column income \
  --task_type classification \
  --index_attribute index \
  --verbose
```
