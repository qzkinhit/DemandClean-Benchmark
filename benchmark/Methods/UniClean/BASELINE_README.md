# UniClean — Multi-Signal Unified Data Cleaning

## Upstream Information
- **Paper**: UniClean: A Unified Framework for Data Cleaning with Multi-Signal Fusion (VLDB 2025)
- **Type**: Type 1 — Fully automatic

## Method Description
UniClean achieves efficient data cleaning by fusing multiple cleaning signals (constraints, statistics, patterns, etc.) and optimizing the cleaning workflow.

## Installation

```bash
pip install pyspark==3.1.1
```

Or use the local `requirements.txt`:
```bash
pip install -r Methods/UniClean/requirements.txt
```

## Core Files

- `Clean.py` — Core cleaning logic (e.g., `CleanonLocalWithnoSmple`)
- `SampleScrubber/cleaner/single.py` — Single-attribute cleaners (`Number`, `Pattern`, `Outlier`)
- `SampleScrubber/cleaner/multiple.py` — Multi-attribute cleaner (`AttrRelation`)
- `AnalyticsCache/` — Analytics and cache module

## Data Format Requirements

Data must contain an `index` column:
```
Data/{dataset}/
├── dirty_with_index.csv
└── clean_with_index.csv
```

## Usage

### Option 1: via the wrapper
```python
from Methods.UniClean.uniclean_wrapper import UniCleanWrapper, get_beers_cleaners

cleaners = get_beers_cleaners()
wrapper = UniCleanWrapper(cleaners=cleaners)
df, info = wrapper.clean('Data/beers/dirty_with_index.csv')
```

### Option 2: run the upstream script directly
```bash
python Methods/UniClean/main_beers.py \
    --file_load Data/beers/dirty_with_index.csv \
    --clean_path Data/beers/clean_with_index.csv \
    --save_path results/uniclean/
```

## Example Cleaner Configuration

```python
from SampleScrubber.cleaner.single import Number, Pattern, Outlier
from SampleScrubber.cleaner.multiple import AttrRelation

# Cleaners for the beers dataset
cleaners = [
    Number("ounces", name="Number_ounces"),
    Number("abv", name="Number_abv"),
    AttrRelation(["brewery_id"], ["brewery_name"], '0'),
    AttrRelation(["brewery_id"], ["city"], '1'),
    AttrRelation(["brewery_id"], ["state"], '2')
]
```

## Differences from the Upstream Implementation

**None** — the wrapper only packages the upstream implementation; no simplified variant is included.

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| single_max | 10000 | Max records processed per batch |
| batch_size | 500 | Batch size |
| executor_memory | 8g | Spark executor memory |
| driver_memory | 8g | Spark driver memory |
