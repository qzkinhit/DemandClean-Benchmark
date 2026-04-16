# 数据集：Adult

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `income` |
| 数据规模 | 45,222 条记录 × 15 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共14列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| age | 数值 | 年龄 |
| workclass | 分类 | 工作类型 |
| fnlwgt | 数值 | 最终权重 |
| education | 分类 | 教育程度 |
| educational_num | 数值 | 教育年数 |
| marital_status | 分类 | 婚姻状态 |
| occupation | 分类 | 职业 |
| relationship | 分类 | 家庭关系 |
| race | 分类 | 种族 |
| gender | 分类 | 性别 |
| capital_gain | 数值 | 资本收益 |
| capital_loss | 数值 | 资本损失 |
| hours_per_week | 数值 | 每周工作小时数 |
| native_country | 分类 | 原籍国 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| income | 二分类 | 收入是否超过50K (0/1) |

## 错误信息
- **错误类型**: 规则违例(Rule Violation), 异常值(Outlier)
- **错误条目数**: 1,701
- **错误单元格数**: 1,701

## 数据来源
Becker, B. & Kohavi, R. Adult. https://archive.ics.uci.edu/dataset/2/adult. 1996.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/adult/dirty_with_index.csv \
  --clean_path Data/adult/clean_with_index.csv \
  --task_name adult_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column income \
  --task_type classification \
  --index_attribute index \
  --verbose
```
