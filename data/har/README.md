# Dataset: HAR (Human Activity Recognition)

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Clustering |
| Target Column | `gt` |
| Data Scale | 70,000 records × 4 columns (3 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (3 total)
| Attribute | Type | Description |
|--------|------|------|
| x | Numeric | Accelerometer X-axis reading |
| y | Numeric | Accelerometer Y-axis reading |
| z | Numeric | Accelerometer Z-axis reading |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| gt | Multi-class | Activity ground-truth label, used to determine the number of clusters k and to compute ARI |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 37,035 |
| Total cells | 210,000 |
| Cell error rate | 17.64% |
| Error rows | 31,012 / 70,000 |
| Row error rate | 44.3% |
| Label errors | 2,087 |
| Label error rate | 2.98% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 23,061 | 62.27% |
| Syntactic | 13,974 | 37.73% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| y | 12,367 | 17.67% | 7,654 | 4,713 |
| x | 12,334 | 17.62% | 7,716 | 4,618 |
| z | 12,334 | 17.62% | 7,691 | 4,643 |

## Data Source
Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. 2013.
