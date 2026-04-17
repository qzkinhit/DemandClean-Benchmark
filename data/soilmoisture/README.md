# Dataset: SoilMoisture

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Regression |
| Target Column | `soil_moisture` |
| Data Scale | 679 records × 128 columns (127 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (127 total)
| Attribute | Type | Description |
|--------|------|------|
| datetime | Timestamp | Acquisition timestamp |
| soil_temperature | Numeric | Soil temperature |
| 454-950 | Numeric | Hyperspectral band reflectance (125 bands covering 454nm-950nm) |

**Note**: Feature columns 454, 458, 462, ... 950 represent reflectance values at their corresponding wavelengths (nm).

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| soil_moisture | Numeric | Soil moisture content |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 26,014 |
| Total cells | 86,233 |
| Cell error rate | 30.17% |
| Error rows | 679 / 679 |
| Row error rate | 100.0% |
| Label errors | 0 |
| Label error rate | 0.0% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 14,367 | 55.23% |
| Syntactic | 11,647 | 44.77% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution (Top-10)
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| 766 | 238 | 35.05% | 142 | 96 |
| 602 | 237 | 34.90% | 137 | 100 |
| 506 | 228 | 33.58% | 115 | 113 |
| 782 | 228 | 33.58% | 126 | 102 |
| 466 | 227 | 33.43% | 100 | 127 |
| 554 | 225 | 33.14% | 118 | 107 |
| 722 | 223 | 32.84% | 131 | 92 |
| 930 | 223 | 32.84% | 132 | 91 |
| 794 | 222 | 32.70% | 106 | 116 |
| 454 | 221 | 32.55% | 111 | 110 |

> The dataset has 127 feature columns, with per-column error rates distributed between ~28-35%. The table above shows only the top 10 columns by error count.

## Data Source
Riese, F. M., & Keller, S. Hyperspectral benchmark dataset on soil moisture. https://github.com/felixriese/hyperspectral-soilmoisture-dataset. 2018.
