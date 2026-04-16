# 数据集：Bike

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 回归 (Regression) |
| 目标列 | `cnt` |
| 数据规模 | 17,379 条记录 × 17 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 排除列 (Excluded) - 不参与模型训练
| 列名 | 类型 | 说明 |
|------|------|------|
| dteday | 日期 | 日期字符串，需特殊处理 |

### 特征列 (Features) - 共14列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| season | 分类 | 季节 (1:春, 2:夏, 3:秋, 4:冬) |
| yr | 数值 | 年份 (0:2011, 1:2012) |
| mnth | 数值 | 月份 (1-12) |
| hr | 数值 | 小时 (0-23) |
| holiday | 二值 | 是否假日 |
| weekday | 数值 | 星期几 (0-6) |
| workingday | 二值 | 是否工作日 |
| weathersit | 分类 | 天气情况 (1:晴, 2:多云, 3:小雨/雪, 4:恶劣天气) |
| temp | 数值 | 归一化温度 |
| atemp | 数值 | 归一化体感温度 |
| hum | 数值 | 归一化湿度 |
| windspeed | 数值 | 归一化风速 |
| casual | 数值 | 临时用户租借数 |
| registered | 数值 | 注册用户租借数 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| cnt | 数值 | 总租借数量 (casual + registered) |

## 错误信息
- **错误类型**: 规则违例(Rule Violation), 异常值(Outlier)
- **错误条目数**: 16,926
- **错误单元格数**: 45,205

## 数据来源
Fanaee-T, H. Bike Sharing. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset. 2013.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/bike/dirty_with_index.csv \
  --clean_path Data/bike/clean_with_index.csv \
  --task_name bike_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column cnt \
  --task_type regression \
  --index_attribute index \
  --exclude_columns dteday \
  --verbose
```
