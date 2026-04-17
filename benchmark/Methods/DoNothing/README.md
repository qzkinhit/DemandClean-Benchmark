# DoNothing Baseline

## Overview

DoNothing is the simplest baseline: it **performs no cleaning** and returns the original dirty data as-is.

## Purpose

- Establishes a lower bound on performance
- Serves as a reference point for measuring the improvement brought by other cleaning methods
- Validates the correctness of the evaluation pipeline

## Ground-Truth Usage

- **Type 1**: Fully automatic, no human effort required
- **Ground-truth cost**: 0

## Usage

```python
from Methods.DoNothing.donothing_wrapper import DoNothingWrapper

cleaner = DoNothingWrapper(verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv'
)
```

## Implementation Notes

This method is newly implemented in this repository and is not based on any upstream code.
