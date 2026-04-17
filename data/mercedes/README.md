# Dataset: Mercedes

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Regression |
| Target Column | `y` |
| Data Scale | 4,209 records × 377 columns (376 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (376 total)
| Attribute | Type | Description |
|--------|------|------|
| X0-X8 | Categorical | Categorical features |
| X10-X385 | Binary | Anonymized vehicle configuration features (0/1) |

**Note**: Feature columns are named X0, X1, X2, ... X385, with 376 feature columns in total (some indices missing: X7, X9, X72, X121, X149, X188, X193, X303, X381).

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| y | Numeric | Vehicle testing time (seconds) |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 301,127 |
| Total cells | 1,582,584 |
| Cell error rate | 19.03% |
| Error rows | 4,209 / 4,209 |
| Row error rate | 100.0% |
| Label errors | 0 |
| Label error rate | 0.0% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 165,155 | 54.85% |
| Syntactic | 135,972 | 45.15% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution (Top-10)
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| X304 | 910 | 21.62% | 494 | 416 |
| X39 | 905 | 21.50% | 484 | 421 |
| X322 | 902 | 21.43% | 512 | 390 |
| X76 | 901 | 21.41% | 475 | 426 |
| X261 | 900 | 21.38% | 496 | 404 |
| X113 | 898 | 21.34% | 498 | 400 |
| X280 | 897 | 21.31% | 500 | 397 |
| X311 | 895 | 21.26% | 481 | 414 |
| X202 | 892 | 21.19% | 492 | 400 |
| X178 | 891 | 21.17% | 489 | 402 |

> The dataset has 376 feature columns, with per-column error rates evenly distributed between ~19-22%. The table above shows only the top 10 columns by error count.

## Data Source
Daimler. Mercedes-Benz Greener Manufacturing. https://www.kaggle.com/c/mercedes-benz-greener-manufacturing. 2017.
