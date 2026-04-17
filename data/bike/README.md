# Dataset: Bike

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Regression |
| Target Column | `cnt` |
| Data Scale | 17,379 records × 16 columns (15 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (15 total)
| Attribute | Type | Description |
|--------|------|------|
| dteday | Date | Date string |
| season | Categorical | Season (1: Spring, 2: Summer, 3: Fall, 4: Winter) |
| yr | Numeric | Year (0: 2011, 1: 2012) |
| mnth | Numeric | Month (1-12) |
| hr | Numeric | Hour (0-23) |
| holiday | Binary | Is holiday |
| weekday | Numeric | Day of week (0-6) |
| workingday | Binary | Is working day |
| weathersit | Categorical | Weather (1: Clear, 2: Cloudy, 3: Light rain/snow, 4: Severe) |
| temp | Numeric | Normalized temperature |
| atemp | Numeric | Normalized apparent temperature |
| hum | Numeric | Normalized humidity |
| windspeed | Numeric | Normalized wind speed |
| casual | Numeric | Count of casual users |
| registered | Numeric | Count of registered users |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| cnt | Numeric | Total rental count (casual + registered) |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 44,324 |
| Total cells | 260,685 |
| Cell error rate | 17.00% |
| Error rows | 16,900 / 17,379 |
| Row error rate | 97.2% |
| Label errors | 0 |
| Label error rate | 0.0% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 23,935 | 54.00% |
| Syntactic | 20,389 | 46.00% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| dteday | 11,820 | 68.01% | 11,418 | 402 |
| mnth | 7,482 | 43.05% | 6,583 | 899 |
| holiday | 7,482 | 43.05% | 1 | 7,481 |
| season | 7,470 | 42.98% | 217 | 7,253 |
| workingday | 1,367 | 7.87% | 948 | 419 |
| temp | 886 | 5.10% | 475 | 411 |
| windspeed | 886 | 5.10% | 492 | 394 |
| casual | 885 | 5.09% | 480 | 405 |
| atemp | 884 | 5.09% | 523 | 361 |
| registered | 882 | 5.08% | 478 | 404 |
| weathersit | 873 | 5.02% | 464 | 409 |
| yr | 861 | 4.95% | 485 | 376 |
| weekday | 859 | 4.94% | 462 | 397 |
| hr | 856 | 4.93% | 471 | 385 |
| hum | 831 | 4.78% | 438 | 393 |

## Data Source
Fanaee-T, H. Bike Sharing. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. 2013.
