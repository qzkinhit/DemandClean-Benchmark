# 数据集：HAR (Human Activity Recognition)

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 聚类 (Clustering) |
| 目标列 | `gt` |
| 数据规模 | 70,000 条记录 × 5 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共3列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| x | 数值 | 加速度计X轴数据 |
| y | 数值 | 加速度计Y轴数据 |
| z | 数值 | 加速度计Z轴数据 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| gt | 多分类 | 活动类型真值标签 (ground truth) |

## 错误信息
- **错误类型**: 缺失值(Missing Value), 异常值(Outlier)
- **错误条目数**: 38,891
- **错误单元格数**: 51,180

## 数据来源
Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. 2013.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/har/dirty_with_index.csv \
  --clean_path Data/har/clean_with_index.csv \
  --task_name har_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column gt \
  --task_type clustering \
  --index_attribute index \
  --verbose
```
