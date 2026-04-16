# Rein-Baseline 复现结果

本目录用于汇总 **Rein-Baseline** 在各数据集上的运行产物，包含三类结果文件：**错误检测**、**错误清洗**、**模型训练**。

---

## 1. 错误检测
**错误检测命令：**
```bash
python3 scripts/detect_errors.py \
  --dataset_name <dataset_name> \
  --detect_method <detect_method>
  --n_iterations 1 
```
**命令参数：**
<dataset_name>：指定数据集
<detect_method>：指定要运行的检测器

**结果文件：** `results/detection_results.csv`

### 包含检测器
- `mvdetector`
- `outlierdetector`
- `min_k`
- `max_entropy`
---

## 2. 错误清洗
**错误清洗命令：**
```bash
python3 scripts/repair_errors.py \
  --dataset_name <dataset_name> \
  --repair_method <repair_method> \
  --n_iterations 1 \
  --store_postgres
```
**命令参数：**
<dataset_name>：指定数据集
<repair_method>：指定清洗器和修复方法

**结果文件：** `results/cleaning_results.csv`

### 已包含清洗器
- `standardImputer`
- `cleanWithGroundTruth`
- `mlImputer`
- `duplicatesCleaner`
---

## 3. 模型训练
**模型训练命令：**
```bash
python3 scripts/train_model.py \
  --dataset_name <dataset_name> \
  --ml_models <ml_models> \
  --n_iterations 1 \
  --hyperopt False \
  --early_termination False \
  --store_postgres
```
**命令参数：**
<dataset_name>：指定数据集
<ml_models>：指定要训练的模型

**结果文件：** `results/model_results.csv`

### 3.1 分类任务（Classification）
- `forest_clf`
- `logit_clf`
- `tree_clf`
- `cleanlab`

### 3.2 回归任务（Regression）
- `forest_reg`
- `lin_reg`
- `tree_reg`

### 3.3 聚类任务（Clustering）
- 

