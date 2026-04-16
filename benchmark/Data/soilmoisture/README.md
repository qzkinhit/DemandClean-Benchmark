# 数据集：SoilMoisture

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 回归 (Regression) |
| 目标列 | `soil_moisture` |
| 数据规模 | 679 条记录 × 131 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 排除列 (Excluded) - 不参与模型训练
| 列名 | 类型 | 说明 |
|------|------|------|
| datetime | 时间 | 采集时间戳，需特殊处理 |

### 特征列 (Features) - 共128列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| soil_temperature | 数值 | 土壤温度 |
| 454-950 | 数值 | 高光谱波段反射率 (共125个波段，波长范围454nm-950nm) |

**注意**: 特征列454, 458, 462, ... 950 表示对应波长(nm)的光谱反射率值

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| soil_moisture | 数值 | 土壤含水量 |

## 错误信息
- **错误类型**: 缺失值(Missing Values), 异常值(Outliers)
- **错误条目数**: 679
- **错误单元格数**: 26,014

## 数据来源
Riese, F. M., & Keller, S. Hyperspectral benchmark dataset on soil moisture. https://github.com/felixriese/hyperspectral-soilmoisture-dataset. 2018.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/soilmoisture/dirty_with_index.csv \
  --clean_path Data/soilmoisture/clean_with_index.csv \
  --task_name soilmoisture_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column soil_moisture \
  --task_type regression \
  --index_attribute index \
  --exclude_columns datetime \
  --verbose
```
