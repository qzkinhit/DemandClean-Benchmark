# Dataset: SoilMoisture

## Basic Information

| Item | Value |
|------|-------|
| Task type | Regression |
| Target column | `soil_moisture` |
| Size | 679 records x 131 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Excluded columns (not used for model training)
| Column | Type | Description |
|--------|------|-------------|
| datetime | Timestamp | Acquisition timestamp, requires special handling |

### Feature columns (128 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| soil_temperature | Numeric | Soil temperature |
| 454-950 | Numeric | Hyperspectral band reflectance (125 bands, wavelengths 454 nm-950 nm) |

**Note**: feature columns 454, 458, 462, ... 950 are reflectance values at the corresponding wavelengths (nm).

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| soil_moisture | Numeric | Soil moisture content |

## Error Information
- **Error types**: Missing values, outliers
- **Error entry count**: 679
- **Error cell count**: 26,014

## Source
Riese, F. M., & Keller, S. Hyperspectral benchmark dataset on soil moisture. https://github.com/felixriese/hyperspectral-soilmoisture-dataset. 2018.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/soilmoisture/dirty_with_index.csv \
  --clean_path Data/soilmoisture/clean_with_index.csv \
  --task_name soilmoisture_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column soil_moisture \
  --task_type regression \
  --index_attribute index \
  --exclude_columns datetime \
  --verbose
```
