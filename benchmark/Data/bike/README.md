# Dataset: Bike

## Basic Information

| Item | Value |
|------|-------|
| Task type | Regression |
| Target column | `cnt` |
| Size | 17,379 records x 17 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Excluded columns (not used for model training)
| Column | Type | Description |
|--------|------|-------------|
| dteday | Date | Date string, requires special handling |

### Feature columns (14 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| season | Categorical | Season (1: spring, 2: summer, 3: fall, 4: winter) |
| yr | Numeric | Year (0: 2011, 1: 2012) |
| mnth | Numeric | Month (1-12) |
| hr | Numeric | Hour (0-23) |
| holiday | Binary | Holiday flag |
| weekday | Numeric | Day of week (0-6) |
| workingday | Binary | Working-day flag |
| weathersit | Categorical | Weather (1: clear, 2: cloudy, 3: light rain/snow, 4: severe) |
| temp | Numeric | Normalized temperature |
| atemp | Numeric | Normalized feels-like temperature |
| hum | Numeric | Normalized humidity |
| windspeed | Numeric | Normalized wind speed |
| casual | Numeric | Casual rental count |
| registered | Numeric | Registered rental count |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| cnt | Numeric | Total rental count (casual + registered) |

## Error Information
- **Error types**: Rule violations, outliers
- **Error entry count**: 16,926
- **Error cell count**: 45,205

## Source
Fanaee-T, H. Bike Sharing. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. 2013.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/bike/dirty_with_index.csv \
  --clean_path Data/bike/clean_with_index.csv \
  --task_name bike_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column cnt \
  --task_type regression \
  --index_attribute index \
  --exclude_columns dteday \
  --verbose
```
