# CtxPipe Baseline

## Overview

CtxPipe (Context-aware Data Preparation Pipeline) is a reinforcement-learning-based tool for automatically generating data preparation pipelines, selecting the optimal processing flow for the context of the data.

**Paper**: [CtxPipe: Context-aware Data Preparation Pipeline Construction for Machine Learning](https://dl.acm.org/doi/10.1145/3626246.3653389) (SIGMOD 2025)

**Notes**:
- `Methods/ctxpipe` mirrors the official repository exactly; no source modifications.
- Inference uses the pretrained model `ctx_50000`; **no training required**.

## Method Type

| Type | Description |
|------|-------------|
| **Type 1** | Fully automatic; no ground truth required |
| Ground truth | 0 (used only for evaluation) |

## Core Idea

1. Extract table context vectors with the GTE-large embedding model.
2. Use a DQN agent to pick the best data-processing component at each step.
3. Automatically assemble the end-to-end data preparation pipeline.

## How to Run

```bash
# Single dataset
python MethodsRunScript/run_ctxpipe/run_ctxpipe_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_ctxpipe \
    --output_path results/ctxpipe/ \
    --label_index 4 \
    --task_type classification \
    --model_tag ctx_50000

# Batch run across all datasets
bash MethodsRunScript/run_ctxpipe/run.sh
```

## Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--dirty_path` | yes | Path to the dirty data | - |
| `--clean_path` | yes | Path to the clean data (for evaluation) | - |
| `--task_name` | yes | Task name | - |
| `--output_path` | no | Output directory | `results/ctxpipe/` |
| `--label_index` | no | Label column index (0-based) | auto-detect |
| `--task_type` | no | Task type | `classification` |
| `--model_tag` | no | Pretrained model tag | `ctx_50000` |
| `--skip_evaluation` | no | Skip evaluation | `False` |

## Dataset Configuration

| Dataset | Task Type | label_index | Label Column |
|---------|-----------|-------------|--------------|
| adult | classification | 14 | income |
| beers | classification | 4 | style |
| breast_cancer | classification | 9 | class |
| smartfactory | classification | 18 | labels |
| bike | regression | 15 | cnt |
| mercedes | regression | 1 | y |
| nasa | regression | 5 | sound_pressure_level |
| soilmoisture | regression | 2 | soil_moisture |
| har | clustering | 3 | gt |

## Pipeline Components

CtxPipe picks the best combination over these components:

| Component | Options |
|-----------|---------|
| Numeric imputation | mean, median, mode |
| Categorical imputation | mode |
| Encoder | label encoding, one-hot encoding |
| Feature preprocessing | MinMaxScaler, StandardScaler, RobustScaler |
| Feature engineering | polynomial features, PCA, kernel PCA |
| Feature selection | variance threshold |

## Output Files

```
results/ctxpipe/{task_name}/
├── {task_name}_ctxpipe_output.csv   # Processed data
├── {task_name}_pipeline_info.txt    # Selected pipeline info
├── {task_name}_total_evaluation.txt # Evaluation report
└── {task_name}.log                  # Run log
```

## Dependencies

- PyTorch
- sentence-transformers (GTE-large)
- Download the embedding model to `Methods/ctxpipe/embed/gte-large/`

## Device Notes

- Auto-detected: CUDA when a GPU is available, CPU otherwise.
- The adapter handles device compatibility automatically.

## Notes

- Uses the dedicated conda environment `ctxpipe-pt112`.
- Per-dataset runtime is a few to a few tens of seconds.
- No training needed; inference uses the pretrained model directly.
