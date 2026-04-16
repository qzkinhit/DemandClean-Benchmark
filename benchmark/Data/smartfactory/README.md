# 数据集：SmartFactory

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `labels` |
| 数据规模 | 23,645 条记录 × 19 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共18列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| i_w_blo_weg | 数值 | 左下传感器输入位移 |
| o_w_blo_power | 数值 | 左下传感器输出功率 |
| o_w_blo_voltage | 数值 | 左下传感器输出电压 |
| i_w_bhl_weg | 数值 | 左后传感器输入位移 |
| o_w_bhl_power | 数值 | 左后传感器输出功率 |
| o_w_bhl_voltage | 数值 | 左后传感器输出电压 |
| i_w_bhr_weg | 数值 | 右后传感器输入位移 |
| o_w_bhr_power | 数值 | 右后传感器输出功率 |
| o_w_bhr_voltage | 数值 | 右后传感器输出电压 |
| i_w_bru_weg | 数值 | 右下传感器输入位移 |
| o_w_bru_power | 数值 | 右下传感器输出功率 |
| o_w_bru_voltage | 数值 | 右下传感器输出电压 |
| i_w_hr_weg | 数值 | 右传感器输入位移 |
| o_w_hr_power | 数值 | 右传感器输出功率 |
| o_w_hr_voltage | 数值 | 右传感器输出电压 |
| i_w_hl_weg | 数值 | 左传感器输入位移 |
| o_w_hl_power | 数值 | 左传感器输出功率 |
| o_w_hl_voltage | 数值 | 左传感器输出电压 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| labels | 多分类 | 设备状态标签 |

## 错误信息
- **错误类型**: 缺失值(Missing Value), 异常值(Outlier)
- **错误条目数**: 7,093
- **错误单元格数**: 7,093

## 数据来源
Oliver Birgelen, Alexander; Niggemann. Smart Factory: High Storage System Data for Energy Optimization. https://www.kaggle.com/inIT-OWL/high-storage-system-data-for-energy-optimization. 2018.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/smartfactory/dirty_with_index.csv \
  --clean_path Data/smartfactory/clean_with_index.csv \
  --task_name smartfactory_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column labels \
  --task_type classification \
  --index_attribute index \
  --verbose
```
