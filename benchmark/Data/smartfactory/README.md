# Dataset: SmartFactory

## Basic Information

| Item | Value |
|------|-------|
| Task type | Classification |
| Target column | `labels` |
| Size | 23,645 records x 19 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (18 total)
| Attribute | Type | Description |
|-----------|------|-------------|
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

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| labels | Multiclass | Equipment status label |

## Error Information
- **Error types**: Missing values, outliers
- **Error entry count**: 7,093
- **Error cell count**: 7,093

## Source
Oliver Birgelen, Alexander; Niggemann. Smart Factory: High Storage System Data for Energy Optimization. https://www.kaggle.com/inIT-OWL/high-storage-system-data-for-energy-optimization. 2018.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/smartfactory/dirty_with_index.csv \
  --clean_path Data/smartfactory/clean_with_index.csv \
  --task_name smartfactory_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column labels \
  --task_type classification \
  --index_attribute index \
  --verbose
```
