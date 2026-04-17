# Dataset: SmartFactory

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Classification |
| Target Column | `labels` |
| Data Scale | 23,645 records × 19 columns (18 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (18 total)
| Attribute | Type | Description |
|--------|------|------|
| i_w_blo_weg | Numeric | Bottom-left sensor input displacement |
| o_w_blo_power | Numeric | Bottom-left sensor output power |
| o_w_blo_voltage | Numeric | Bottom-left sensor output voltage |
| i_w_bhl_weg | Numeric | Back-left sensor input displacement |
| o_w_bhl_power | Numeric | Back-left sensor output power |
| o_w_bhl_voltage | Numeric | Back-left sensor output voltage |
| i_w_bhr_weg | Numeric | Back-right sensor input displacement |
| o_w_bhr_power | Numeric | Back-right sensor output power |
| o_w_bhr_voltage | Numeric | Back-right sensor output voltage |
| i_w_bru_weg | Numeric | Bottom-right sensor input displacement |
| o_w_bru_power | Numeric | Bottom-right sensor output power |
| o_w_bru_voltage | Numeric | Bottom-right sensor output voltage |
| i_w_hr_weg | Numeric | Right sensor input displacement |
| o_w_hr_power | Numeric | Right sensor output power |
| o_w_hr_voltage | Numeric | Right sensor output voltage |
| i_w_hl_weg | Numeric | Left sensor input displacement |
| o_w_hl_power | Numeric | Left sensor output power |
| o_w_hl_voltage | Numeric | Left sensor output voltage |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| labels | Multi-class | Equipment status label |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 7,093 |
| Total cells | 425,610 |
| Cell error rate | 1.67% |
| Error rows | 7,093 / 23,645 |
| Row error rate | 30.0% |
| Label errors | 0 |
| Label error rate | 0.0% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 7,093 | 100.00% |
| Syntactic | 0 | 0.00% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| o_w_hl_power | 429 | 1.81% | 429 | 0 |
| o_w_bhl_power | 413 | 1.75% | 413 | 0 |
| o_w_bru_voltage | 413 | 1.75% | 413 | 0 |
| o_w_bru_power | 409 | 1.73% | 409 | 0 |
| o_w_bhl_voltage | 404 | 1.71% | 404 | 0 |
| i_w_hl_weg | 402 | 1.70% | 402 | 0 |
| o_w_blo_power | 398 | 1.68% | 398 | 0 |
| o_w_hr_power | 398 | 1.68% | 398 | 0 |
| o_w_blo_voltage | 396 | 1.67% | 396 | 0 |
| i_w_bru_weg | 391 | 1.65% | 391 | 0 |
| i_w_hr_weg | 390 | 1.65% | 390 | 0 |
| i_w_bhl_weg | 385 | 1.63% | 385 | 0 |
| o_w_hr_voltage | 382 | 1.62% | 382 | 0 |
| o_w_hl_voltage | 382 | 1.62% | 382 | 0 |
| o_w_bhr_voltage | 381 | 1.61% | 381 | 0 |
| o_w_bhr_power | 378 | 1.60% | 378 | 0 |
| i_w_blo_weg | 374 | 1.58% | 374 | 0 |
| i_w_bhr_weg | 368 | 1.56% | 368 | 0 |

## Data Source
Oliver Birgelen, Alexander; Niggemann. Smart Factory: High Storage System Data for Energy Optimization. https://www.kaggle.com/inIT-OWL/high-storage-system-data-for-energy-optimization. 2018.
