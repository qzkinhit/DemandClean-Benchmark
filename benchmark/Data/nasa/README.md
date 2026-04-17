# Dataset: NASA (Airfoil Self-Noise)

## Basic Information

| Item | Value |
|------|-------|
| Task type | Regression |
| Target column | `sound_pressure_level` |
| Size | 1,503 records x 7 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (5 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| frequency | Numeric | Frequency (Hz) |
| angle | Numeric | Angle of attack (degrees) |
| chord_length | Numeric | Chord length (meters) |
| velocity | Numeric | Free-stream velocity (m/s) |
| thickness | Numeric | Suction-side displacement thickness (meters) |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| sound_pressure_level | Numeric | Scaled sound pressure level (decibels) |

## Error Information
- **Error types**: Missing values, outliers
- **Error entry count**: 731
- **Error cell count**: 731

## Source
Brooks, T., Pope, D., & Marcolini, M. Airfoil Self-Noise. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise. 1989.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/nasa/dirty_with_index.csv \
  --clean_path Data/nasa/clean_with_index.csv \
  --task_name nasa_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column sound_pressure_level \
  --task_type regression \
  --index_attribute index \
  --verbose
```
