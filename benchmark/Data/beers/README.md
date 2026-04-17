# Dataset: Beers

## Basic Information

| Item | Value |
|------|-------|
| Task type | Classification |
| Target column | `style` |
| Size | 2,410 records x 11 attributes |
| Indexed files | `clean_index.csv`, `dirty_index.csv` |
| Missing value marker | `empty` |

## File Layout

### Main data files
| File | Description | Usage |
|------|-------------|-------|
| `clean.csv` | Clean data (no index) | Raw clean data |
| `dirty.csv` | Dirty data (no index) | Raw dirty data |
| `clean_index.csv` | Clean data with `index` column | **Recommended** primary evaluation file |
| `dirty_index.csv` | Dirty data with `index` column | **Recommended** primary evaluation file |
| `clean_with_index.csv` | Same as `clean_index.csv` | Legacy compatibility |
| `dirty_with_index.csv` | Same as `dirty_index.csv` | Legacy compatibility |

### HoloClean-specific files
| File | Description |
|------|-------------|
| `clean_holoclean.csv` | Transposed format (tid, attribute, correct_val) |
| `dirty_index_holoclean.csv` | Dirty data without the `index` column |
| `*_ori_empty.csv` | Variant with missing values normalized to empty strings |

### Rule files
| File | Description |
|------|-------------|
| `dc_rules_holoclean.txt` | Denial Constraints in HoloClean format |
| `dc_rules-validate-fd-horizon.txt` | Functional Dependencies in Horizon format |
| `fd_rule.txt` | Functional dependency rules |
| `rules.txt` | Generic rule file |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Excluded columns (not used for model training)
| Column | Type | Description |
|--------|------|-------------|
| id | Numeric | Beer ID (identifier) |
| beer_name | Text | Beer name (text identifier) |
| brewery_id | Numeric | Brewery ID (identifier) |
| brewery_name | Text | Brewery name (text identifier) |
| city | Text | City (high-cardinality categorical) |
| state | Text | State (categorical) |

### Feature columns (3 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| ounces | Numeric | Volume in ounces |
| abv | Numeric | Alcohol by volume |
| ibu | Numeric | International Bitterness Units |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| style | Multiclass | Beer style |

## Error Information
- **Error types**: Missing values, rule violations, typos
- **Error entry count**: 2,410
- **Error cell count**: 3,357
- **Missing value marker**: `empty`

## Source
J.-N. Hould. Craft beers dataset. https://www.kaggle.com/nickhould/craft-cans. 2017.

## Example Command

```bash
# DeleteAll baseline
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/beers/dirty_index.csv \
  --clean_path Data/beers/clean_index.csv \
  --task_name beers_deleteall \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column style \
  --task_type classification \
  --index_attribute index \
  --exclude_columns id beer_name brewery_id brewery_name city state
```
