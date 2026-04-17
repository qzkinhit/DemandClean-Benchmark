# DemandClean: A DQN-Based Demand-Driven Data Cleaning Framework

> **Core idea**: Data cleaning is modeled as a sequential decision-making MDP. The agent learns to identify which errors significantly impact downstream tasks (and thus warrant ground-truth repair using the budget), which can be handled with free value estimation, and which can be safely skipped.

---

## End-to-End Pipeline

```
 dirty_index.csv (2410 rows, CSV strings)
       |
  +----+----+
  | Data Preprocessing | LabelEncoder + StandardScaler
  +----+----+
       | (2410 rows, LE+SS numpy with NaN)
  +----+----+
  | Three-way Split | train=1446(60%) / val=482(20%) / test=482(20%)
  +----+----+
       |
  +----+----+
  | Error Detection | AutoDetector: missing -> syntactic (RAHA+DOMAIN) -> semantic (FD) -> labels
  +----+----+
       | ~3256 erroneous cells
  +----+----+
  | RAHA Pre-repair | 20 annotated rows repaired directly
  +----+----+
       |
  +----+--------+
  | Clean Base  | DeleteFix vs VE-Fill (5-fold CV + sqrt(n/N) weighting)
  +----+--------+
       |
  +----+----+
  | DQN Training | 300 episodes: injection -> agent decision -> reward -> experience replay
  +----+----+
       |
  +----+----+
  | Two-phase Inference | Phase1: Plan (user only provides ~30 ground-truth values) -> Phase2: Execute
  +----+----+
       |
  +----+----+
  | Evaluation | 6 baselines + EDR + Shapley + cost accounting
  +---------+
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn torch

# 2. Run beers v6 (main version)
python run_demandclean/run_demandclean_base.py --dataset beers --versions v6 --n_episodes 300

# 3. Inspect results
ls results/demandclean/beers/v6_auto_dueling_two/
```

### Python API

```python
from demandclean import DemandClean

# Create v6 configuration
dc = DemandClean(
    task_type='classification',
    model_type='random_forest',
    agent_type='dueling_two_stage',
    detector_mode='auto',
    inference_mode='two_phase',
    n_episodes=300,
    count_raha_cost=True,
)

# Train
dc.fit(X_dirty, y, X_clean_val=X_clean_val, y_clean_val=y_clean_val)

# Two-phase inference
plan = dc.plan(X_dirty, y)                # Phase1: generate repair plan
X_cleaned, y_cleaned, mask = dc.execute(  # Phase2: execute repair
    X_dirty, true_values, y_dirty=y
)
```

---

## Version Matrix

| Version | Detector | Agent Type | Inference Mode | Ablation Dimension |
|---------|----------|-----------|----------------|--------------------|
| v3 | oracle | plain_single | single | Baseline: oracle + simplest agent |
| v4 | oracle | plain_two | two_phase | Two-phase vs single-phase |
| v5 | auto | dueling_single | single | Dueling single-phase |
| **v6** | **auto** | **dueling_two_stage** | **two_phase** | **Main version** |
| v7 | auto | plain_single | single | Ablation: no Dueling |
| v8 | auto | plain_two | two_phase | Ablation: no Dueling + two-phase |

Ablation mapping: **v6 vs v5** = two-phase gain | **v6 vs v7** = Dueling gain | **v6 vs v3** = auto detection vs Oracle

---

## Datasets

| Dataset | Task | Model | Rows | Features | Label Column | Characteristics |
|---------|------|-------|------|----------|--------------|-----------------|
| beers | classification | RF | 2410 | 8 | style | FD rules, many missing ibu |
| adult | classification | RF | 32561 | 14 | income | Large dataset, multiple categorical columns |
| bike | regression | RF | 8645 | 11 | cnt | All numerical, DC rules |
| breast_cancer | classification | RF | 699 | 9 | class | Small dataset |
| har | clustering | KMeans | 10299 | 3 | gt | Acceleration data |
| mercedes | regression | Ridge | 4209 | 16 | y | Mixed categorical + numerical |
| nasa | regression | Ridge | 1503 | 5 | sound_pressure_level | All FLOAT |
| smartfactory | classification | RF | 2000 | 10 | machine_status | Industrial scenario |
| soilmoisture | regression | Ridge | 1500 | 8 | soil_moisture | Sensor data |

---

## Technical Documentation Index

> Full documentation is under [`docs/`](docs/README.md), comprising 15 chapters.

| Chapter | Title | One-line Summary |
|---------|-------|------------------|
| 01 | System Overview | Core idea, 10-step pipeline, v6 configuration, module dependencies |
| 02 | Data Preprocessing and Encoding | CSV->LE->LE+SS encoding pipeline, NaN preservation, edit-distance tolerance |
| 03 | Data Splitting and Clean Base | Oracle three-way split, DeleteFix vs VE-Fill + sqrt(n/N) weighting |
| 04 | Rule System | Format, semantics, and triple usage of 6 rule types in rules.txt |
| 05 | Error Detection | 4-stage pipeline, RAHA chunking (20 columns / 10000 rows), FD majority voting |
| 06 | Error Injection | Injection = inverse of detection, FD strict <half, RAHA-aware syntactic injection |
| 07 | Configuration System | 5 enums, all DemandCleanConfig fields, version mapping |
| 08 | State Space | Per-dimension definitions of the 8D vector: error_type/importance/distance/... |
| 09 | DQN Agent | 4 variants, Dueling V+A decomposition, two-stage (3+2) decision |
| 10 | Cleaning Environment and Reward | 4 actions + 3 fallbacks, tanh dynamic repair modulator, three-layer repair-rate control |
| 11 | Value Estimation Chain | 7 priority levels: FD->DC->CFD->numeric extraction->edit distance->KNN->Fallback |
| 12 | Training Pipeline | clean_base/self_supervised, adaptive injection ratio, shared VE |
| 13 | Inference Pipeline | Two-phase Plan+Execute, user only provides ground-truth values from the plan |
| 14 | Evaluation System | 6 baselines, EDR cost-effectiveness, 3-dimensional Shapley, cost accounting |
| 15 | Model Adapters | Factory pattern, unified interface for classification/regression/clustering, score normalization |

---

## CLI Reference

```bash
python run_demandclean/run_demandclean_base.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | Dataset name (beers/adult/bike/...) |
| `--versions` | v6 | Version list (v3,v5,v6,v7,v8) |
| `--n_episodes` | 300 | Number of training episodes |
| `--oracle` | False | Use Oracle three-way split |
| `--verbose` | True | Verbose logging |
| `--resume` | auto | Resume mode (auto/force_new) |
| `--all_datasets` | False | Run all 9 datasets |
| `--visualize_only` | False | Skip training, only regenerate visualizations |
| `--apply_raha_truth` | True | Pre-repair RAHA-annotated rows |
| `--count_raha_cost` | True | Include RAHA annotation cost in accounting |

---

## Ablation Studies

v6 is the main version; other versions are ablation controls:

```
v6 (full)
 |-- Remove two-phase inference -> v5 (single-phase Dueling)
 |-- Remove Dueling             -> v8 (two-phase Plain)
 |-- Remove both                -> v7 (single-phase Plain)
 +-- Swap to Oracle detection   -> v3 (Oracle upper bound)
```

| Comparison | Ablation Dimension | Hypothesis Tested |
|------------|--------------------|-------------------|
| v6 vs v5 | Two-phase inference | Plan+Execute saves ground-truth cost over direct cleaning |
| v6 vs v7 | Dueling network | V+A decomposition accelerates learning convergence |
| v6 vs v8 | Dueling network (two-phase preserved) | Effect of network structure on the two-phase variant |
| v6 vs v3 | Automatic detection | Gap between auto detection and Oracle cheating |

---

## License

MIT License
