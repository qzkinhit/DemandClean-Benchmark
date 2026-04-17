# Dataset: NASA (Airfoil Self-Noise)

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Regression |
| Target Column | `sound_pressure_level` |
| Data Scale | 1,503 records × 6 columns (5 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (5 total)
| Attribute | Type | Description |
|--------|------|------|
| frequency | Numeric | Frequency (Hz) |
| angle | Numeric | Angle of attack (degrees) |
| chord_length | Numeric | Chord length (meters) |
| velocity | Numeric | Free-stream velocity (m/s) |
| thickness | Numeric | Suction-side displacement thickness (meters) |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| sound_pressure_level | Numeric | Scaled sound pressure level (decibels) |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 731 |
| Total cells | 7,515 |
| Cell error rate | 9.73% |
| Error rows | 731 / 1,503 |
| Row error rate | 48.6% |
| Label errors | 0 |
| Label error rate | 0.0% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 475 | 64.98% |
| Syntactic | 256 | 35.02% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| velocity | 157 | 10.45% | 115 | 42 |
| angle | 150 | 9.98% | 108 | 42 |
| chord_length | 148 | 9.85% | 80 | 68 |
| frequency | 138 | 9.18% | 107 | 31 |
| thickness | 138 | 9.18% | 65 | 73 |

## Data Source
Brooks, T., Pope, D., & Marcolini, M. Airfoil Self-Noise. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise. 1989.
