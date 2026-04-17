# Dataset: Breast_Cancer

## Basic Information

| Item | Value |
|------|-------|
| Task type | Classification |
| Target column | `class` |
| Size | 699 records x 11 attributes |
| Indexed files | `clean_with_index.csv`, `dirty_with_index.csv` |

## Column Definitions

### Index column
| Column | Description |
|--------|-------------|
| `index` | row index, not used for model training |

### Feature columns (9 total)
| Attribute | Type | Description |
|-----------|------|-------------|
| Clump Thickness | Numeric | Clump thickness (1-10) |
| Uniformity of Cell Size | Numeric | Cell size uniformity (1-10) |
| Uniformity of Cell Shape | Numeric | Cell shape uniformity (1-10) |
| Marginal Adhesion | Numeric | Marginal adhesion (1-10) |
| Single Epithelial Cell Size | Numeric | Single epithelial cell size (1-10) |
| Bare Nuclei | Numeric | Bare nuclei (1-10) |
| Bland Chromatin | Numeric | Bland chromatin (1-10) |
| Normal Nucleoli | Numeric | Normal nucleoli (1-10) |
| Mitoses | Numeric | Mitoses (1-10) |

### Label column
| Attribute | Type | Description |
|-----------|------|-------------|
| class | Binary | Tumor class (2: benign, 4: malignant) |

## Error Information
- **Error types**: Missing values, outliers, typos
- **Error entry count**: 453
- **Error cell count**: 631

## Source
Wolberg, W. Breast Cancer Wisconsin (Original). https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original. 1990.

## Example Command

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/breast_cancer/dirty_with_index.csv \
  --clean_path Data/breast_cancer/clean_with_index.csv \
  --task_name breast_cancer_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column class \
  --task_type classification \
  --index_attribute index \
  --verbose
```
