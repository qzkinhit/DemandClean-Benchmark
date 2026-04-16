# tools/ — 评估与辅助工具

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

This directory contains evaluation tools, data processing utilities, and analysis scripts used throughout the DemandClean pipeline.

### File List

#### Core Evaluation

| File | Description |
|------|-------------|
| `getScore.py` | Traditional data cleaning metrics (accuracy, recall, F1, EDR, hybrid distance, R-EDR) |
| `getScoreML.py` | Unified Clean4ML evaluation (downstream task + tolerance + Snoopy + cost) |

#### Data Processing

| File | Description |
|------|-------------|
| `readData.py` | Data loading utilities |
| `saveData.py` | Data saving utilities |
| `inject_errors.py` | Error injection tool (random and systematic errors) |
| `insert_null.py` | Null value insertion tool |
| `get_error_num.py` | Error statistics counter |
| `get_subset.py` | Data subset extraction |
| `rules_parser.py` | FD rule parser (legacy, see `demandclean/detectors/rule_parser.py` for the current version) |

#### Vectorization

| File | Description |
|------|-------------|
| `adult_vectorize.py` | Adult dataset vectorization (TF-IDF + StandardScaler) |
| `eeg_vectorize.py` | EEG Eye State dataset vectorization |

#### Visualization

| File | Description |
|------|-------------|
| `get_plt.py` | Result visualization plots |
| `resultPLT.py` | Result plotting utilities |

#### Analysis

| File | Description |
|------|-------------|
| `shapley_analysis.py` | Shapley value analysis for 3 dimensions (action, feature, error type importance) |
| `tolerance_analysis.py` | Model tolerance threshold analysis |
| `get_T_table.py` | T-table generation |

#### Sub-directories

| Directory | Description |
|-----------|-------------|
| `generate_index/` | Index generation tools for datasets (`clean_index.py`, `dirty_index.py`, `description.py`) |
| `snoopy/` | Snoopy data quality upper bound evaluation tool (external library) |

### Key APIs

#### getScore.py — Traditional Cleaning Metrics

```python
from tools.getScore import calculate_all_metrics

results = calculate_all_metrics(
    clean, dirty, cleaned, attributes,
    output_path, task_name, index_attribute
)
# Returns: accuracy, recall, f1_score, edr, hybrid_distance, r_edr
```

**Metrics:**
- **Accuracy**: correctly repaired cells / total repaired cells
- **Recall**: correctly repaired cells / total cells that need repair
- **F1**: harmonic mean of accuracy and recall
- **EDR**: Error Distance Reduction = (D_dirty - D_cleaned) / D_dirty
- **Hybrid Distance**: MSE (numeric) + Jaccard (categorical)
- **R-EDR**: Record-based EDR (per-row error reduction)

**Data alignment**: Uses three-way index intersection (`clean ∩ dirty ∩ cleaned`) to handle row deletions.

#### getScoreML.py — Unified Evaluation

```python
from tools.getScoreML import run_all_evaluation

results = run_all_evaluation(
    dirty_path, cleaned_path, clean_path,
    label_col, task_type, task_name, output_dir
)
```

**5-module evaluation pipeline:**
1. Traditional cleaning metrics (getScore)
2. Downstream task performance (RF, LR, SVM, KNN, DT, GB)
3. Model tolerance (prior and posterior)
4. Snoopy upper bound
5. Ground truth cost analysis

---

<a name="中文"></a>
## 中文

### 概述

本目录包含 DemandClean 管道中使用的评估工具、数据处理工具和分析脚本。

### 文件列表

#### 核心评估

| 文件 | 说明 |
|------|------|
| `getScore.py` | 传统数据清洗指标（准确率、召回率、F1、EDR、混合距离、R-EDR） |
| `getScoreML.py` | 统一 Clean4ML 测评（下游任务 + 容忍度 + Snoopy + 成本） |

#### 数据处理

| 文件 | 说明 |
|------|------|
| `readData.py` | 数据读取工具 |
| `saveData.py` | 数据保存工具 |
| `inject_errors.py` | 错误注入工具（随机/系统性错误） |
| `insert_null.py` | 空值插入工具 |
| `get_error_num.py` | 错误数量统计 |
| `get_subset.py` | 数据子集提取 |
| `rules_parser.py` | FD 规则解析器（旧版，当前版本见 `demandclean/detectors/rule_parser.py`） |

#### 分析

| 文件 | 说明 |
|------|------|
| `shapley_analysis.py` | Shapley 值三维度分析（动作、特征、错误类型重要性） |
| `tolerance_analysis.py` | 模型容忍阈值分析 |

#### 子目录

| 目录 | 说明 |
|------|------|
| `generate_index/` | 数据集索引生成工具 |
| `snoopy/` | Snoopy 数据质量上界评估工具（外部库） |

### 核心 API

#### getScore.py — 传统清洗指标

```python
from tools.getScore import calculate_all_metrics

results = calculate_all_metrics(
    clean, dirty, cleaned, attributes,
    output_path, task_name, index_attribute
)
# 返回: accuracy, recall, f1_score, edr, hybrid_distance, r_edr
```

**数据对齐**：使用三方索引交集（`clean ∩ dirty ∩ cleaned`）处理行删除。

#### getScoreML.py — 统一测评

```python
from tools.getScoreML import run_all_evaluation

results = run_all_evaluation(
    dirty_path, cleaned_path, clean_path,
    label_col, task_type, task_name, output_dir
)
```

**5模块测评管道：**
1. 传统清洗指标 (getScore)
2. 下游任务性能 (RF, LR, SVM, KNN, DT, GB)
3. 模型容忍度（先验和后验）
4. Snoopy 上界
5. 真值使用成本分析

### 评估指标详解

#### 传统指标
- **准确率**: 正确修复数 / 总修复数
- **召回率**: 正确修复数 / 应修复数
- **F1**: 2 × 准确率 × 召回率 / (准确率 + 召回率)
- **EDR**: (D_dirty_to_clean - D_repaired_to_clean) / D_dirty_to_clean
- **混合距离**: MSE (数值列) + Jaccard (分类列)
- **R-EDR**: 基于记录的错误减少率

#### 模型容忍度
- **先验容忍度**: P_demand_clean / P_do_nothing
- **后验容忍度**: (P_demand_clean - P_do_nothing) / (P_repair_all - P_do_nothing)
