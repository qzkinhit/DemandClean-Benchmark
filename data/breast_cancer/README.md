# Dataset: Breast_Cancer

## Basic Information

| Item | Value |
|------|-----|
| Task Type | Classification |
| Target Column | `class` |
| Data Scale | 699 records × 10 columns (9 features + 1 label) |
| Index Files | `clean_index.csv`, `dirty_index.csv` |

## Column Definitions

### Index Column
| Column | Description |
|------|------|
| `index` | Row index, not used in model training |

### Feature Columns (9 total)
| Attribute | Type | Description |
|--------|------|------|
| Clump Thickness | Numeric | Clump thickness (1-10) |
| Uniformity of Cell Size | Numeric | Uniformity of cell size (1-10) |
| Uniformity of Cell Shape | Numeric | Uniformity of cell shape (1-10) |
| Marginal Adhesion | Numeric | Marginal adhesion (1-10) |
| Single Epithelial Cell Size | Numeric | Single epithelial cell size (1-10) |
| Bare Nuclei | Numeric | Bare nuclei (1-10) |
| Bland Chromatin | Numeric | Bland chromatin (1-10) |
| Normal Nucleoli | Numeric | Normal nucleoli (1-10) |
| Mitoses | Numeric | Mitoses (1-10) |

### Label Column
| Attribute | Type | Description |
|--------|------|------|
| class | Binary | Tumor class (2: Benign, 4: Malignant) |

## Error Statistics

### Overview
| Metric | Value |
|------|-----|
| Error cells | 531 |
| Total cells | 6,291 |
| Cell error rate | 8.44% |
| Error rows | 387 / 699 |
| Row error rate | 55.4% |
| Label errors | 15 |
| Label error rate | 2.15% |

### Error Type Distribution
| Type | Count | Ratio |
|------|------|------|
| Semantic | 371 | 69.87% |
| Syntactic | 138 | 25.99% |
| Missing | 0 | 0.00% |

### Per-Column Error Distribution
| Column | Errors | Error Rate | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| Clump Thickness | 73 | 10.44% | 49 | 24 |
| Bare Nuclei | 72 | 10.30% | 50 | 0 |
| Normal Nucleoli | 63 | 9.01% | 48 | 15 |
| Single Epithelial Cell Size | 61 | 8.73% | 44 | 17 |
| Bland Chromatin | 61 | 8.73% | 39 | 22 |
| Uniformity of Cell Shape | 55 | 7.87% | 44 | 11 |
| Uniformity of Cell Size | 53 | 7.58% | 38 | 15 |
| Marginal Adhesion | 51 | 7.30% | 33 | 18 |
| Mitoses | 42 | 6.01% | 26 | 16 |

## Data Source
Wolberg, W. Breast Cancer Wisconsin (Original). https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original. 1990.
