# Rein-Baseline Reproduction Results

This directory collects the run artifacts of **Rein-Baseline** on each dataset. It contains three result categories: **error detection**, **error cleaning**, and **model training**.

---

## 1. Error Detection
**Command:**
```bash
python3 scripts/detect_errors.py \
  --dataset_name <dataset_name> \
  --detect_method <detect_method>
  --n_iterations 1 
```
**Arguments:**
- `<dataset_name>`: Selected dataset
- `<detect_method>`: Detector to run

**Result file:** `results/detection_results.csv`

### Included detectors
- `mvdetector`
- `outlierdetector`
- `min_k`
- `max_entropy`

---

## 2. Error Cleaning
**Command:**
```bash
python3 scripts/repair_errors.py \
  --dataset_name <dataset_name> \
  --repair_method <repair_method> \
  --n_iterations 1 \
  --store_postgres
```
**Arguments:**
- `<dataset_name>`: Selected dataset
- `<repair_method>`: Cleaner and repair method

**Result file:** `results/cleaning_results.csv`

### Included cleaners
- `standardImputer`
- `cleanWithGroundTruth`
- `mlImputer`
- `duplicatesCleaner`

---

## 3. Model Training
**Command:**
```bash
python3 scripts/train_model.py \
  --dataset_name <dataset_name> \
  --ml_models <ml_models> \
  --n_iterations 1 \
  --hyperopt False \
  --early_termination False \
  --store_postgres
```
**Arguments:**
- `<dataset_name>`: Selected dataset
- `<ml_models>`: Model(s) to train

**Result file:** `results/model_results.csv`

### 3.1 Classification
- `forest_clf`
- `logit_clf`
- `tree_clf`
- `cleanlab`

### 3.2 Regression
- `forest_reg`
- `lin_reg`
- `tree_reg`

### 3.3 Clustering
- 
