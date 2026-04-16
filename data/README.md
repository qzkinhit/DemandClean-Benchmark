# data/ — 基准数据集 Benchmark Datasets

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

This directory contains 9 benchmark datasets for data cleaning experiments. Each dataset includes clean data, dirty data (with injected/real errors), and FD/domain rules.

### Dataset Summary

| Dataset | Task | Model | Features | Records | Label Column | Error Types |
|---------|------|-------|----------|---------|-------------|-------------|
| **adult** | Classification | Random Forest | 15 | 45,222 | income | Rule violations, outliers, label noise |
| **beers** | Classification | XGBoost | 8 | 2,410 | style | Missing, outliers, label noise |
| **bike** | Regression | Random Forest | 12 | 17,379 | cnt | Missing, noise |
| **breast_cancer** | Classification | Random Forest | 31 | 569 | diagnosis | Missing |
| **har** | Classification | Random Forest | 562 | 10,299 | Activity | Missing, sensor noise |
| **mercedes** | Regression | Ridge | 377 | 4,209 | y | Missing |
| **nasa** | Regression | Ridge | 6 | 1,503 | Sound_pressure | Missing, noise |
| **smartfactory** | Classification | Random Forest | 6 | 7,936 | Machine_failure | Missing, outliers |
| **soilmoisture** | Regression | Ridge | 14 | 1,778 | soil_moisture | Missing |

### File Structure

Each dataset directory contains:

```
{dataset_name}/
├── README.md           # Dataset description
├── clean_index.csv     # Clean data with index column
├── dirty_index.csv     # Dirty data with index column (errors injected)
└── rules.txt           # FD/DOMAIN/CFD/REGEX rules for error detection & injection
```

### Rules File Format

Rules files support 6 section types used by DemandClean's `AutoDetector` and `ErrorInjector`:

```
[REGEX]
col_name: ^pattern$

[DOMAIN]
col_name: INT [min, max]
col_name: FLOAT [min, max]
col_name: ENUM {val1, val2, ...}

[FD]
lhs_col -> rhs_col

[CFD]
condition => col_name EXCESS >= threshold FROM_BASELINE baseline

[DC]
NOT(t1.col < 0)

[STATISTICAL]
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
```

### Index Column

- The `index` column (first column in CSV) is used for row tracking during cleaning
- Index remains consistent between clean and dirty versions
- Evaluation uses index-based alignment to handle row deletions

### Adding New Datasets

1. Create a directory named after the dataset under `data/`
2. Prepare `clean_index.csv` (ground truth) and `dirty_index.csv` (with errors)
3. Write `rules.txt` with DOMAIN/FD/CFD rules
4. Add `README.md` describing data source and error characteristics
5. Register the dataset in `run_demandclean/run_demandclean_base.py`

---

<a name="中文"></a>
## 中文

### 概述

本目录包含 9 个用于数据清洗实验的基准数据集。每个数据集包含干净数据、脏数据（含注入/真实错误）和 FD/Domain 规则。

### 数据集概览

| 数据集 | 任务类型 | 模型 | 属性数 | 记录数 | 标签列 | 错误类型 |
|--------|----------|------|--------|--------|--------|---------|
| **adult** | 分类 | Random Forest | 15 | 45,222 | income | 规则违例、异常值、标签噪声 |
| **beers** | 分类 | XGBoost | 8 | 2,410 | style | 缺失值、异常值、标签噪声 |
| **bike** | 回归 | Random Forest | 12 | 17,379 | cnt | 缺失值、噪声 |
| **breast_cancer** | 分类 | Random Forest | 31 | 569 | diagnosis | 缺失值 |
| **har** | 分类 | Random Forest | 562 | 10,299 | Activity | 缺失值、传感器噪声 |
| **mercedes** | 回归 | Ridge | 377 | 4,209 | y | 缺失值 |
| **nasa** | 回归 | Ridge | 6 | 1,503 | Sound_pressure | 缺失值、噪声 |
| **smartfactory** | 分类 | Random Forest | 6 | 7,936 | Machine_failure | 缺失值、异常值 |
| **soilmoisture** | 回归 | Ridge | 14 | 1,778 | soil_moisture | 缺失值 |

### 文件结构

每个数据集目录包含：

```
{数据集名}/
├── README.md           # 数据集说明
├── clean_index.csv     # 带索引的干净数据
├── dirty_index.csv     # 带索引的脏数据（含错误）
└── rules.txt           # FD/DOMAIN/CFD/REGEX 规则
```

### 规则文件格式

规则文件支持 6 种类型，用于 DemandClean 的 `AutoDetector` 和 `ErrorInjector`：

```
[REGEX]
列名: ^正则表达式$

[DOMAIN]
列名: INT [最小值, 最大值]
列名: FLOAT [最小值, 最大值]
列名: ENUM {值1, 值2, ...}

[FD]
左侧列 -> 右侧列

[CFD]
条件 => 列名 EXCESS >= 阈值 FROM_BASELINE 基线值

[DC]
NOT(t1.列名 < 0)

[STATISTICAL]
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
```

### 索引列

- `index` 列（CSV 第一列）用于清洗过程中的行追踪
- 索引在干净版本和脏版本之间保持一致
- 评估时使用基于索引的对齐来处理行删除

### 添加新数据集

1. 在 `data/` 下创建以数据集命名的目录
2. 准备 `clean_index.csv`（真值）和 `dirty_index.csv`（含错误）
3. 编写 `rules.txt`，包含 DOMAIN/FD/CFD 规则
4. 添加 `README.md` 描述数据来源和错误特征
5. 在 `run_demandclean/run_demandclean_base.py` 中注册数据集
