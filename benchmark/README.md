# Clean4MLBaseline

A benchmark repository for evaluating data cleaning methods with a focus on downstream machine learning task performance.

## Overview

This repository contains reproducible implementations of 14 data cleaning methods, evaluated not only on traditional data quality metrics but also on:

- **Downstream task performance**: classification, regression, and clustering
- **Model tolerance**: prior and posterior tolerance metrics
- **Ground truth cost**: Type 1/2/3 method categorization
- **Snoopy upper bound**: data quality ceiling analysis

## Directory Structure

```
benchmark/
├── Data/                          # 9 benchmark datasets
│   ├── adult/                     # Income prediction (classification)
│   ├── beers/                     # Beer style classification
│   ├── bike/                      # Bike rental regression
│   ├── breast_cancer/             # Cancer diagnosis (classification)
│   ├── har/                       # Human activity recognition (clustering)
│   ├── mercedes/                  # Car test time (regression)
│   ├── nasa/                      # Sound pressure prediction (regression)
│   ├── smartfactory/              # Smart factory (classification)
│   └── soilmoisture/              # Soil moisture (regression)
│
├── Methods/                       # 14 cleaning method implementations
│   ├── DoNothing/                 # No-op baseline
│   ├── DeleteAll/                 # Delete rows with missing values
│   ├── RepairAll/                 # Perfect repair baseline
│   ├── SimpleImputer/             # Statistical imputation (mean/median/mode)
│   ├── MLImputer/                 # ML-based imputation (MICE/KNN/RF)
│   ├── Horizon/                   # Dependency-driven cleaning
│   ├── Baran_Raha/                # Error detection and repair
│   ├── HoloClean/                 # Probabilistic graphical model cleaning
│   ├── UniClean/                  # Multi-signal fusion cleaning
│   ├── Lopster/                   # Latent space representation learning
│   ├── ActiveClean/               # Model-guided iterative cleaning
│   ├── BoostClean/                # Detector ensemble cleaning
│   └── ctxpipe/                   # Context-aware data preparation
│
├── MethodsRunScript/              # Per-method run scripts
├── tools/                         # Evaluation tools
├── results/                       # Pre-computed results
├── logs/                          # Execution logs
├── run_all.sh                     # One-click runner for all baselines
└── requirements.txt               # Python dependencies
```

## Environment Setup

### Quick Start (Conda)

This project uses Conda virtual environments:

| Environment | Python | Purpose |
|-------------|--------|---------|
| `multibaseline` | 3.7 | Most baselines |
| `ctxpipe-pt112` | 3.8 | CTXPipe (requires PyTorch 1.12) |

```bash
# 1. Create main environment
conda create -n multibaseline python=3.7 -y
conda activate multibaseline
pip install -r requirements.txt
pip install tensorflow==2.11.0  # Required by Lopster

# 2. Create CTXPipe environment (optional)
conda create -n ctxpipe-pt112 python=3.8 -y
conda activate ctxpipe-pt112
pip install torch==1.12.1 torchvision==0.13.1
pip install pandas scikit-learn transformers sentence-transformers loguru peewee tqdm safetensors
```

### External Dependencies

| Component | Version | Used By | Setup |
|-----------|---------|---------|-------|
| Java | OpenJDK 11 | UniClean (PySpark) | Set `JAVA_HOME` |
| Spark | 3.1.2 | UniClean | Set `SPARK_HOME` |
| PostgreSQL | 14.x | HoloClean | Create `holo` database |

### PostgreSQL Setup (HoloClean)

```bash
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql

sudo -u postgres psql
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
ALTER USER holocleanuser CREATEDB;
\q
```

### Spark Setup (UniClean)

```bash
wget https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
tar -xzf spark-3.1.2-bin-hadoop3.2.tgz && mv spark-3.1.2-bin-hadoop3.2 spark

# Add to ~/.bashrc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_HOME=$(pwd)/spark
export PATH=$SPARK_HOME/bin:$PATH
export PYSPARK_PYTHON=python3
```

## Baseline Methods

### Control Baselines

| Method | Type | Description | Performance |
|--------|------|-------------|-------------|
| **DoNothing** | Type 1 | Returns dirty data as-is | Lower bound |
| **DeleteAll** | Type 1 | Removes all rows with missing values | — |
| **RepairAll** | Type 2 | Replaces all errors with ground truth | Upper bound |

### Data-Oriented Methods

| Method | Venue | Signal | Type |
|--------|-------|--------|------|
| **HoloClean** | VLDB 2017 | Constraints + statistics + knowledge fusion | Type 1 |
| **Horizon** | VLDB 2021 | Functional dependency pattern selection | Type 1 |
| **Baran/Raha** | SIGMOD'19 / VLDB'20 | Context features + semi-supervised learning | Type 3 |
| **UniClean** | VLDB 2025 | Multi-signal fusion + workflow optimization | Type 1 |
| **Lopster** | VLDB 2024 | Latent space representation learning | Type 1 |

### Model-Oriented Methods

| Method | Venue | Signal | Type |
|--------|-------|--------|------|
| **ActiveClean** | SIGMOD 2016 | Gradient-based prioritization + human interaction | Type 3 |
| **BoostClean** | SIGMOD 2017 | Detector-repair ensemble + auto optimization | Type 2 |

### Data Preparation Methods

| Method | Description | Type |
|--------|-------------|------|
| **CTXPipe** | Context embedding + RL pipeline generation | Type 1 |
| **SimpleImputer** | Mean/median/mode statistical imputation | Type 1 |
| **MLImputer** | MICE/KNN/RF machine learning imputation | Type 1 |

### Ground Truth Usage Types

- **Type 1**: Fully automatic, no human involvement
- **Type 2**: Requires a small validation set for quality assessment
- **Type 3**: Iterative human-in-the-loop cleaning

## Datasets

| Dataset | Task | Target | Rows | Error Types |
|---------|------|--------|------|-------------|
| adult | Classification | income | 45,222 | Rule violations, outliers |
| beers | Classification | style | 2,410 | Missing values, outliers |
| bike | Regression | cnt | 17,379 | Missing values, noise |
| breast_cancer | Classification | class | 699 | Missing values |
| har | Clustering | gt | 10,299 | Missing values, noise |
| mercedes | Regression | y | 4,209 | Missing values |
| nasa | Regression | sound_pressure_level | 1,503 | Missing values |
| smartfactory | Classification | labels | 7,484 | Missing values, outliers |
| soilmoisture | Regression | soil_moisture | 1,440 | Missing values |

### Data Format

Each dataset directory contains:
```
Data/<dataset>/
├── clean_index.csv    # Clean data (with index column)
├── dirty_index.csv    # Dirty data (with index column)
├── clean.csv          # Clean data (without index)
├── dirty.csv          # Dirty data (without index)
├── rules.txt          # Cleaning rules (FD, DC, etc.)
└── README.md          # Dataset description
```

Missing values are uniformly marked as `empty`.

## Quick Start

### Run a Single Baseline

```bash
conda activate multibaseline

python MethodsRunScript/run_simpleimputer/run_simpleimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_simpleimputer \
    --output_path results/simpleimputer/ \
    --label_column style \
    --task_type classification \
    --models rf lr
```

### Using run.sh Scripts

Each method has a `run.sh` script:

```bash
bash MethodsRunScript/run_donothing/run.sh
bash MethodsRunScript/run_simpleimputer/run.sh
bash MethodsRunScript/run_horizon/run.sh
```

### Run All Baselines

```bash
# Run all sequentially
bash run_all.sh --version v1

# Run all in parallel (uses GNU screen)
bash run_all.sh --version v1 --parallel

# Run specific baseline on specific dataset
bash run_all.sh --version v1 --baseline raha_baran --dataset beers

# Check status / stop
bash run_all.sh --version v1 --status
bash run_all.sh --version v1 --stop
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dirty_path` | Path to dirty data | Required |
| `--clean_path` | Path to clean data (for evaluation) | Required |
| `--task_name` | Task identifier | Required |
| `--output_path` | Output directory | `results/<method>/` |
| `--index_attribute` | Index column name | `index` |
| `--label_column` | Target column name | None |
| `--task_type` | Task type | `classification` |
| `--models` | Evaluation models | `['rf', 'lr']` |

## Evaluation Metrics

### Data Quality Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Correctly repaired / total repaired |
| Recall | Correctly repaired / total errors |
| F1 Score | Harmonic mean of accuracy and recall |
| EDR | Error detection rate |
| R-EDR | Record-based error detection rate |
| Hybrid Distance | Weighted MSE + Jaccard distance |

### Downstream Task Metrics

| Task Type | Metrics |
|-----------|---------|
| Classification | Accuracy, F1 (weighted), Precision, Recall |
| Regression | MSE, MAE, R-squared |
| Clustering | Silhouette Score, ARI |

### Model Tolerance

```
Prior tolerance  = P_cleaned / P_dirty
Posterior tolerance = (P_cleaned - P_dirty) / (P_clean - P_dirty)
```

- `P_dirty`: model performance on dirty data
- `P_cleaned`: model performance on cleaned data
- `P_clean`: model performance on perfectly clean data

## Output Format

Results are saved to `results/<task_name>/`:

```
results/<task_name>/
├── <task_name>_cleaned.csv              # Cleaned data
├── <task_name>_report.txt               # Unified evaluation report
├── <task_name>.log                      # Full execution log
├── <task_name>_pipeline_info.txt        # Pipeline info (if applicable)
└── <task_name>_evaluation_results.txt   # Detailed evaluation results
```

## Adding a New Method

1. Create directories: `Methods/NewMethod/` and `MethodsRunScript/run_newmethod/`

2. Implement a wrapper class:

```python
# Methods/NewMethod/newmethod_wrapper.py
class NewMethodWrapper:
    def __init__(self, **kwargs):
        self.ground_truth_used = 0

    def setup(self):
        pass

    def clean(self, dirty_path, output_path=None):
        # Return cleaned DataFrame
        pass

    def get_ground_truth_cost(self):
        return self.ground_truth_used
```

3. Create a run script using the shared evaluation API:

```python
from tools.getScoreML import run_all_evaluation

cleaned_data = wrapper.clean(dirty_path)
cleaned_data.to_csv(res_path, index=False)

results = run_all_evaluation(
    dirty_path=dirty_path,
    cleaned_path=res_path,
    clean_path=clean_path,
    output_path=output_path,
    task_name=task_name,
    label_column=label_column,
    task_type=task_type,
    models=models,
    method_type=1,
    ground_truth_used=wrapper.get_ground_truth_cost()
)
```

## References

```bibtex
@article{rekatsinas2017holoclean,
  title={HoloClean: Holistic Data Repairs with Probabilistic Inference},
  author={Rekatsinas, Theodoros and others},
  journal={PVLDB},
  year={2017}
}

@inproceedings{krishnan2016activeclean,
  title={ActiveClean: Interactive Data Cleaning For Statistical Modeling},
  author={Krishnan, Sanjay and others},
  booktitle={PVLDB},
  year={2016}
}

@article{mahdavi2019raha,
  title={Raha: A Configuration-Free Error Detection System},
  author={Mahdavi, Mohammad and others},
  journal={SIGMOD},
  year={2019}
}

@article{mahdavi2020baran,
  title={Baran: Effective Error Correction via a Unified Context Representation},
  author={Mahdavi, Mohammad and others},
  journal={PVLDB},
  year={2020}
}

@article{li2021horizon,
  title={Horizon: Scalable Dependency-Driven Data Cleaning},
  author={Li, Peng and others},
  journal={PVLDB},
  year={2021}
}

@article{neutatz2024lopster,
  title={Lopster: Data Cleaning with Latent Operators},
  author={Neutatz, Felix and others},
  journal={PVLDB},
  year={2024}
}

@article{zhao2025uniclean,
  title={UniClean: A Unified Data Cleaning Framework},
  author={Zhao, Xiang and others},
  journal={PVLDB},
  year={2025}
}

@article{narayan2024ctxpipe,
  title={CtxPipe: Context-Aware Data Preparation Pipeline Construction},
  author={Narayan, Kumar and others},
  journal={SIGMOD},
  year={2024}
}
```

## License

MIT License
