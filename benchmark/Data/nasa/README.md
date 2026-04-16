# 数据集：NASA (Airfoil Self-Noise)

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 回归 (Regression) |
| 目标列 | `sound_pressure_level` |
| 数据规模 | 1,503 条记录 × 7 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共5列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| frequency | 数值 | 频率 (Hz) |
| angle | 数值 | 攻角 (度) |
| chord_length | 数值 | 弦长 (米) |
| velocity | 数值 | 自由流速度 (米/秒) |
| thickness | 数值 | 吸力侧位移厚度 (米) |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| sound_pressure_level | 数值 | 缩放声压级 (分贝) |

## 错误信息
- **错误类型**: 缺失值(Missing Value), 异常值(Outlier)
- **错误条目数**: 731
- **错误单元格数**: 731

## 数据来源
Brooks, T., Pope, D., & Marcolini, M. Airfoil Self-Noise. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise. 1989.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/nasa/dirty_with_index.csv \
  --clean_path Data/nasa/clean_with_index.csv \
  --task_name nasa_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column sound_pressure_level \
  --task_type regression \
  --index_attribute index \
  --verbose
```
