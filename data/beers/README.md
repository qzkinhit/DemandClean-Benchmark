# Dataset: Beers

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Classification |
| Target Column | `state` |
| Data Scale | 2,410 records × 10 columns (9 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |
| Missing Marker | `empty` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (9 total)
| Attribute | Type | Description |
|--------|------|------|
| id | Numeric | Beer ID |
| beer_name | Text | Beer name |
| style | Multi-class | Beer style |
| ounces | Numeric | Volume (ounces) |
| abv | Numeric | Alcohol by volume |
| ibu | Numeric | International Bitterness Units |
| brewery_id | Numeric | Brewery ID |
| brewery_name | Text | Brewery name |
| city | Text | City |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| state | Text categorical | State |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 3,232 |
| Total cells | 21,690 |
| Cell error rate | 14.90% |
| Error rows | 2,410 / 2,410 |
| Row error rate | 100.0% |
| Label errors | 127 |
| Label error rate | 5.27% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 3,231 | 99.97% |
| Syntactic | 0 | 0.00% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| ounces | 2,409 | 99.96% | 2,409 | 0 |
| abv | 692 | 28.71% | 692 | 0 |
| city | 127 | 5.27% | 127 | 0 |
| style | 2 | 0.08% | 2 | 0 |
| brewery_id | 2 | 0.08% | 1 | 0 |

## Data Source
J.-N. Hould. Craft beers dataset. https://www.kaggle.com/nickhould/craft-cans. 2017.
