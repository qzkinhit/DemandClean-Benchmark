# Utils 工具函数目录

本目录包含数据清洗实验中常用的工具函数。

## 文件说明

### 评估相关

| 文件 | 说明 |
|------|------|
| `getScore.py` | 传统数据质量评估指标（准确率、召回率、EDR等） |
| `getScoreML.py` | 下游任务评估与模型容忍度计算 |

### 数据处理

| 文件 | 说明 |
|------|------|
| `inject_errors.py` | 错误注入工具（随机/系统错误） |
| `insert_null.py` | 空值插入工具 |
| `get_error_num.py` | 错误统计工具 |

### 数据转换

| 文件 | 说明 |
|------|------|
| `adult_vectorize.py` | Adult数据集向量化 |
| `eeg_vectorize.py` | EEG数据集向量化 |

### 可视化

| 文件 | 说明 |
|------|------|
| `get_plt.py` | 结果可视化 |
| `resultPLT.py` | 结果绘图 |

### 辅助工具

| 文件 | 说明 |
|------|------|
| `readData.py` | 数据读取 |
| `saveData.py` | 数据保存 |
| `get_subset.py` | 数据子集提取 |
| `generate_index/` | 索引生成工具 |

---

## 主要功能

### getScore.py - 传统评估指标

```python
from utils.getScore import calculate_all_metrics

results = calculate_all_metrics(
    clean, dirty, cleaned, attributes,
    output_path, task_name, index_attribute
)
# 返回: accuracy, recall, f1_score, edr, hybrid_distance, r_edr
```

### getScoreML.py - 下游任务评估

```python
from utils.getScoreML import comprehensive_evaluation

results = comprehensive_evaluation(
    dirty_data, cleaned_data, clean_data,
    label_column='label',
    task_type='classification',
    models=['rf', 'lr'],
    method_type=1,
    ground_truth_used=0
)
# 返回: 下游任务性能、容忍度、真值使用成本
```

### inject_errors.py - 错误注入

```python
from utils.inject_errors import inject_random_error, inject_system_error

# 随机错误注入
dirty_df = inject_random_error(clean_df, percent=0.1)

# 系统错误注入（基于模型重要性）
dirty_df = inject_system_error(clean_df, percent=0.1, target_column='label')
```

---

## 详细说明

### `inject_error.py`

用于在adult和eeg的特征向量数据集上进行**错误注入**操作。错误注入的类型有两种：**随机错误（random errors）** 和 **系统错误（system errors）**。

#### 主要函数

1. **`inject_random_error(df, percent)`**:
   - 随机选择一定比例的**行**，将这些行的所有数值型特征替换为该列最大值的 3 倍

2. **`inject_system_error(df, percent, target_column)`**:
   - 基于 **SGDClassifier** 模型权重选择前 `x%` 的数据行，将最重要的 3 个特征替换为均值

#### 命令行示例

```bash
# 随机错误注入
python inject_errors.py --input adult_data_vectorized.csv --output adult_with_random_errors.csv --error_type random --percent 5

# 系统错误注入
python inject_error.py --input adult_vectorized.csv --output adult_with_system_errors.csv --error_type system --percent 10
```

---

### `eeg_vectorize.py`

对 **EEG Eye State 数据集** 进行向量化处理，提取每个时间步的统计特征。

#### 命令行示例

```bash
python vectorize_eeg.py --input eeg_eye_state.arff --output eeg_vectorized.csv
```

---

### `adult_vectorize.py`

对 **Adult 数据集** 进行向量化处理。

#### 特征处理说明

1. **数值型特征** (`age`, `fnlwgt`, `education-num`, `hours-per-week`)：使用**标准化**
2. **类别型特征** (`workclass`, `education`等)：使用 **TF-IDF词袋编码**
3. **收入标签**：`<=50K` → 0, `>50K` → 1

#### 命令行示例

```bash
python adult_vectorize.py --input adult.csv --output adult_vectorized.csv
```

---

## 评估指标详解

### 传统指标

- **准确率**: 正确修复数 / 总修复数
- **召回率**: 正确修复数 / 应修复数
- **F1值**: 2 * 准确率 * 召回率 / (准确率 + 召回率)
- **EDR**: (D_dirty_to_clean - D_repaired_to_clean) / D_dirty_to_clean
- **R-EDR**: 基于记录的错误减少率

### 下游任务指标

- **分类**: Accuracy, F1, Precision, Recall
- **回归**: MSE, MAE, R²
- **聚类**: Silhouette Score, ARI

### 模型容忍度

- **先验容忍度**: P_demand_clean / P_do_nothing
- **后验容忍度**: (P_demand_clean - P_do_nothing) / (P_repair_all - P_do_nothing)
