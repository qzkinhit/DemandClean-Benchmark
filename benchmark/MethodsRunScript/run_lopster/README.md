# Lopster Baseline

## Overview

Lopster is a VAE-based (variational autoencoder) data cleaning method. It learns a latent representation of the data and uses it to detect and repair errors.

**Paper**: [Lopster: Learning to Repair Tables](https://dl.acm.org/doi/10.14778/3632093.3632099) (VLDB 2024)

## Method Type

| Type | Description |
|------|------|
| **Type 1** | Fully automatic, no ground truth required |
| Ground-truth usage | 0 (used only for evaluation) |

## Core Ideas

1. A VAE learns the latent distribution of the data
2. Reconstruction error detects anomalies / errors
3. Nearest-neighbor lookup in the latent space yields repair values
4. The process is iteratively refined

## Usage

```bash
# Single dataset
python MethodsRunScript/run_lopster/run_lopster_base.py \
    --dataset beers \
    --data_path Data \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_lopster \
    --output_path results/lopster/ \
    --label_column style \
    --task_type classification \
    --epochs 100 \
    --latent_dim 120

# Batch run on all datasets
bash MethodsRunScript/run_lopster/run.sh
```

## Parameters

| Parameter | Required | Description | Default |
|------|------|------|--------|
| `--dataset` | Yes | Dataset name | - |
| `--data_path` | No | Data root directory | `Data` |
| `--clean_path` | No | Path to clean data (for evaluation) | - |
| `--task_name` | Yes | Task name | - |
| `--output_path` | No | Output path | `results/lopster/` |
| `--label_column` | No | Label column name | - |
| `--task_type` | No | Task type | `classification` |
| `--latent_dim` | No | Latent-space dimension | `120` |
| `--epochs` | No | Training epochs | `100` |
| `--learning_rate` | No | Learning rate | `0.001` |
| `--batch_size` | No | Batch size | `256` |
| `--K` | No | K for nearest neighbors | `12` |
| `--clean_ratio` | No | Fraction of clean data to use | `1.0` |
| `--models` | No | Evaluation models | `rf lr` |

## Output Files

```
results/lopster/{task_name}/
├── {task_name}_cleaned.csv          # Cleaned data
├── {task_name}_total_evaluation.txt # Evaluation report
├── {task_name}.log                  # Run log
└── model/                           # Trained VAE model
```

## Evaluation Metrics

Metrics defined in the Lopster paper:
- **col_avg_rmse**: Mean of per-column normalized RMSE on numeric columns (computed after `StandardScaler` normalization)
- **col_avg_f1**: Weighted F1 averaged across categorical columns

## Key Features

- **Deep learning**: Uses a neural network to learn data representations
- **Rule-free**: Requires no predefined rules or constraints
- **General purpose**: Works on many dataset types
- **GPU acceleration**: Supports CUDA training

## Comparison with Other Methods

| Method | Approach | Requires Rules | Requires Training |
|------|------|----------|----------|
| **Lopster** | VAE latent representation | No | Yes |
| HoloClean | Probabilistic graphical model | Yes (DC) | Yes |
| Horizon | FD pattern graph | Yes (FD) | No |

## Notes

- Training time scales with dataset size and epochs
- A too-small `latent_dim` can lead to underfitting
- A GPU environment is recommended to speed up training
