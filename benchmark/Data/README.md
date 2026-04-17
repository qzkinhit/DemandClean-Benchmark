# Data Directory

This directory contains the datasets used in the baseline data cleaning experiments. Each dataset ships with both a clean and a dirty version, along with index columns for row tracking.

## Dataset Overview

| Dataset | Task Type | Attributes | Records | Error Types | Source |
|---------|-----------|------------|---------|-------------|--------|
| adult | Classification (C) | 15 | 45,222 | Rule violations, outliers | UCI ML Repository |
| beers | Regression (R) | - | - | Missing values, outliers | Kaggle |
| bike | Regression (R) | - | - | Missing values, noise | UCI ML Repository |
| breast_cancer | Classification (C) | - | - | Missing values | UCI ML Repository |
| har | Classification (C) | - | - | Missing values, noise | UCI ML Repository |
| mercedes | Regression (R) | - | - | Missing values | Kaggle |
| nasa | - | - | - | Missing values | NASA |
| smartfactory | - | - | - | Missing values, outliers | Industrial data |
| soilmoisture | - | - | - | Missing values | Sensor data |

## File Naming Convention

Each dataset directory should contain the following files:

```
{dataset_name}/
├── clean.csv              # Clean data (ground truth)
├── dirty.csv              # Dirty data (with injected errors)
├── clean_with_index.csv   # Clean data with index column
├── dirty_with_index.csv   # Dirty data with index column
├── constraints.txt        # Constraint file (if applicable)
├── README.md              # Dataset description
└── *.py                   # ML task scripts
```

## Index Convention

Index columns are used to track modifications during the cleaning process:

- `index`: row index starting from 0
- The index is consistent between the clean and dirty versions
- Evaluation uses the index to align rows

## Task Types

- **C (Classification)**: classification tasks
- **R (Regression)**: regression tasks
- **Clustering**: clustering tasks

## Error Types

1. **Missing Values**: NULL, NaN, empty strings
2. **Outliers**: extreme values far from the normal distribution
3. **Rule Violations**: violations of business rules or constraints
4. **Noise**: random errors or measurement noise
5. **Duplicates**: duplicate records
6. **Inconsistency**: different representations of the same entity

## Dataset Details

### Adult

**Source**: UCI Machine Learning Repository

**Task**: predict whether income exceeds 50K.

**Attributes**:
- age, workclass, fnlwgt, education, educational_num
- marital_status, occupation, relationship, race, gender
- capital_gain, capital_loss, hours_per_week
- native_country, income (label)

**Error types**: rule violations, outliers

**Native error count**: 1,701

### Beers

**Source**: Kaggle

**Task**: predict beer rating.

**Key attributes**: abv, ibu, rating, etc.

**Error types**: missing values, outliers

### HAR (Human Activity Recognition)

**Source**: UCI Machine Learning Repository

**Task**: human activity recognition.

**Error types**: missing values, sensor noise

## Adding a New Dataset

1. Create a folder named after the dataset under `Data/`.
2. Prepare `clean.csv` and `dirty.csv`.
3. Generate indexed versions using `utils/generate_index`.
4. Write a `README.md` describing the data source and error types.
5. If constraint rules apply, create `constraints.txt`.

### Constraint File Format

Use the Denial Constraint format:

```
# FD: A -> B
t1&t2&EQ(t1.A,t2.A)&IQ(t1.B,t2.B)

# Range constraint
t1&LT(t1.age,0)

# Pattern constraint
t1&NOT(MATCH(t1.email,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"))
```

## Usage Example

```python
import pandas as pd

# Load data
clean = pd.read_csv('Data/adult/clean.csv')
dirty = pd.read_csv('Data/adult/dirty.csv')

# Count errors
diff = (clean != dirty).sum().sum()
print(f"Number of error cells: {diff}")
```

## References

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml
- Kaggle Datasets: https://www.kaggle.com/datasets
