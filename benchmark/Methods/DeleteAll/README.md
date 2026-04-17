# DeleteAll Baseline

## Overview

DeleteAll is a simple baseline that "cleans" data by **dropping rows that contain problems**.

## Supported Modes

### 1. drop_missing (default)
- Drops all rows containing missing values (NaN, empty string, "N/A", etc.)
- **Type 1**: Fully automatic, no human effort required
- **Ground-truth cost**: 0

### 2. drop_errors
- Drops all rows that differ from the clean data
- **Type 2**: Requires comparison against clean data
- **Ground-truth cost**: Proportional to the number of dropped rows

## Purpose

- Provides a baseline for an aggressive deletion-based cleaning strategy
- Highlights the trade-off between preserving row count and data quality
- Validates a model's sensitivity to reduced row count

## Usage

```python
from Methods.DeleteAll.deleteall_wrapper import DeleteAllWrapper

# Mode 1: drop rows containing missing values only
cleaner = DeleteAllWrapper(mode='drop_missing', verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv'
)

# Mode 2: drop all erroneous rows
cleaner = DeleteAllWrapper(mode='drop_errors', verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv',
    clean_path='path/to/clean.csv'
)
```

## Implementation Notes

This method is newly implemented in this repository and is not based on any upstream code.
