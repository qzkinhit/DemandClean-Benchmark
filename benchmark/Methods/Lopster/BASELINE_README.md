# Lopster — Latent-Space Data Cleaning

## Upstream Information
- **Paper**: Generalizable Data Cleaning of Tabular Data in Latent Space (VLDB 2024)
- **GitHub**: https://github.com/DataManagementLab/data_cleaning_with_latent_operators
- **Authors**: Eduardo dos Reis, Mohamed Abdelaal, Carsten Binnig

## Method Description
Lopster is a VAE-based general-purpose data cleaning method. It learns a latent-space representation of the data and uses it to detect and repair errors.

## Ground-Truth Usage

**Type 2 — Requires training data**

- Lopster trains its VAE on `clean.csv` to learn the latent-space representation
- **Ground-truth cost = number of rows in `clean.csv`** (all clean data is used for training)
- Lopster therefore needs a reasonable amount of clean data to learn the distribution of "normal" data

## Installation

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib
```

Or use the local `requirements.txt`:
```bash
pip install -r Methods/Lopster/requirements.txt
```

## Data Format Requirements

**Important**: The upstream implementation expects a specific layout:

```
Data/{dataset_name}/
├── clean.csv      # Clean data (for training)
└── dirty01.csv    # Dirty data (note: named dirty01.csv, not dirty.csv)
```

## Configuration File

Dataset information must be registered in `dataset_configuration.json`. Follow the existing entries as reference.

## Usage

### Option 1: via the wrapper
```python
from Methods.Lopster.lopster_wrapper import LopsterWrapper, prepare_data_for_lopster

# Prepare the required data layout
prepare_data_for_lopster(
    'Data/beers/dirty.csv',
    'Data/beers/clean.csv',
    'beers',
    'Data/'
)

# Run cleaning
wrapper = LopsterWrapper(epochs=100, latent_dim=120)
df, info = wrapper.clean('beers', 'Data/')
```

### Option 2: run the upstream script directly
```bash
python Methods/Lopster/lopster.py --dataset beers --path Data/ --epochs 100 --latent 120
```

## Differences from the Upstream Implementation

**None** — the wrapper only packages the upstream implementation; no simplified variant is included.

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| epochs | 100 | Training epochs |
| learning_rate | 0.001 | Learning rate |
| latent | 120 | Latent-space dimension |
| batch_size | 256 | Batch size |
| K | 12 | Translation-operator parameter |

## Output

The cleaned data is written to `{path}/{dataset}/lopster.csv`.
