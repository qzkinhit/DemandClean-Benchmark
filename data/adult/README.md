# Dataset: Adult

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Classification |
| Target Column | `income` |
| Data Scale | 45,222 records × 15 columns (14 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (14 total)
| Attribute | Type | Description |
|--------|------|------|
| age | Numeric | Age |
| workclass | Categorical | Work class |
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
| hours_per_week | Numeric | Working hours per week |
| native_country | Categorical | Country of origin |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| income | Binary | Whether income exceeds 50K (0/1) |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 0 |
| Total cells | 633,108 |
| Cell error rate | 0.00% |
| Error rows | 0 / 45,222 |
| Row error rate | 0.0% |
| Label errors | 1,701 |
| Label error rate | 3.76% |

> Feature columns contain no errors. All 1,701 errors are in the label column `income` (3.76% of labels).

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 0 | - |
| Syntactic | 0 | - |
| Missing | 0 | - |
| Label errors | 1,701 | 100% |

### Per-Column Error Distribution

Feature columns contain no errors. All 1,701 errors are located in the label column `income`.

## Data Source
Becker, B. & Kohavi, R. Adult. https://archive.ics.uci.edu/dataset/2/adult. 1996.
