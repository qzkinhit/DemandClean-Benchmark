# MLImputer

## Overview

MLImputer is an ML-based missing-value imputation method that trains a model to predict missing values.

## Key Features

- **MICE imputation**: Multiple Imputation by Chained Equations
- **KNN imputation**: Based on K-nearest neighbors
- **Random Forest imputation**: Uses Random Forest as the base estimator

## Ground-Truth Usage

**Type**: Fully automatic, no human effort required (Type 1)

MLImputer trains its model on the available data to predict missing values and requires no additional annotation.

## File Structure

```
MLImputer/
├── __init__.py             # Package init
├── mlimputer_wrapper.py    # MLImputer wrapper class
├── readme.md               # Documentation
└── requirements.txt        # Dependencies
```

## Supported Methods

| Method | Description |
|------|------|
| `mice` | Multiple Imputation by Chained Equations (iterative) |
| `knn` | K-nearest-neighbor imputation |
| `rf` | Random Forest imputation |

## Example Usage

```python
from Methods.MLImputer import MLImputerWrapper

# Create the imputer
imputer = MLImputerWrapper(
    method='mice',
    max_iter=10,
    verbose=True
)

# Run imputation
repaired_df, info = imputer.clean(
    dirty_path='data/dirty.csv',
    output_path='results/imputed.csv'
)

print(f"Imputed cells: {info['imputed_cells']}")
```

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| method | 'mice' | Imputation method |
| max_iter | 10 | Maximum iterations for MICE |
| n_neighbors | 5 | Number of neighbors for KNN |
| random_state | 42 | Random seed |
