# Data 数据集目录

本目录包含用于数据清洗Baseline实验的数据集。每个数据集包含干净版本和脏版本，并附带索引用于追踪。

## 数据集概览

| 数据集 | 任务类型 | 属性数 | 记录数 | 错误类型 | 来源 |
|--------|----------|--------|--------|----------|------|
| adult | 分类 (C) | 15 | 45,222 | 规则违例, 异常值 | UCI ML Repository |
| beers | 回归 (R) | - | - | 缺失值, 异常值 | Kaggle |
| bike | 回归 (R) | - | - | 缺失值, 噪声 | UCI ML Repository |
| breast_cancer | 分类 (C) | - | - | 缺失值 | UCI ML Repository |
| har | 分类 (C) | - | - | 缺失值, 噪声 | UCI ML Repository |
| mercedes | 回归 (R) | - | - | 缺失值 | Kaggle |
| nasa | - | - | - | 缺失值 | NASA |
| smartfactory | - | - | - | 缺失值, 异常值 | 工业数据 |
| soilmoisture | - | - | - | 缺失值 | 传感器数据 |

## 文件命名规范

每个数据集目录应包含以下文件：

```
{dataset_name}/
├── clean.csv              # 干净数据（真值）
├── dirty.csv              # 脏数据（含错误）
├── clean_with_index.csv   # 带索引的干净数据
├── dirty_with_index.csv   # 带索引的脏数据
├── constraints.txt        # 约束文件（如适用）
├── README.md              # 数据集说明
└── *.py                   # ML任务脚本
```

## 索引说明

索引列用于追踪数据清洗过程中的修改：

- `index`: 行索引，从0开始
- 索引在clean和dirty版本之间保持一致
- 评估时使用索引对齐数据

## 任务类型

- **C (Classification)**: 分类任务
- **R (Regression)**: 回归任务
- **Clustering**: 聚类任务

## 错误类型

1. **缺失值 (Missing Values)**: NULL, NaN, 空字符串
2. **异常值 (Outliers)**: 偏离正常分布的极端值
3. **规则违例 (Rule Violations)**: 违反业务规则或约束
4. **噪声 (Noise)**: 随机错误或测量误差
5. **重复 (Duplicates)**: 重复记录
6. **不一致 (Inconsistency)**: 同一实体的不同表示

## 数据集详细说明

### Adult

**来源**: UCI Machine Learning Repository

**任务**: 预测收入是否超过50K

**属性**:
- age, workclass, fnlwgt, education, educational_num
- marital_status, occupation, relationship, race, gender
- capital_gain, capital_loss, hours_per_week
- native_country, income (标签)

**错误类型**: 规则违例, 异常值

**原生错误条目数**: 1,701

### Beers

**来源**: Kaggle

**任务**: 预测啤酒评分

**主要属性**: abv, ibu, rating等

**错误类型**: 缺失值, 异常值

### HAR (Human Activity Recognition)

**来源**: UCI Machine Learning Repository

**任务**: 人体活动识别

**错误类型**: 缺失值, 传感器噪声

## 添加新数据集

1. 在Data目录下创建以数据集名称命名的文件夹
2. 准备clean.csv和dirty.csv文件
3. 使用utils/generate_index生成带索引版本
4. 编写README.md说明数据来源和错误类型
5. 如有约束规则，创建constraints.txt

### 约束文件格式

使用否定约束(Denial Constraints)格式：

```
# FD: A -> B
t1&t2&EQ(t1.A,t2.A)&IQ(t1.B,t2.B)

# 范围约束
t1&LT(t1.age,0)

# 模式约束
t1&NOT(MATCH(t1.email,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"))
```

## 使用示例

```python
import pandas as pd

# 加载数据
clean = pd.read_csv('Data/adult/clean.csv')
dirty = pd.read_csv('Data/adult/dirty.csv')

# 检查错误
diff = (clean != dirty).sum().sum()
print(f"错误单元格数: {diff}")
```

## 参考文献

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml
- Kaggle Datasets: https://www.kaggle.com/datasets
