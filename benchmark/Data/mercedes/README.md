# 数据集：Mercedes

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 回归 (Regression) |
| 目标列 | `y` |
| 数据规模 | 4,209 条记录 × 378 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共376列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| X0-X8 | 分类 | 类别特征 |
| X10-X385 | 二值 | 匿名化的车辆配置特征 (0/1) |

**注意**: 特征列命名为 X0, X1, X2, ... X385，共376个特征列（部分编号缺失：X7, X9, X72, X121, X149, X188, X193, X303, X381）

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| y | 数值 | 车辆测试时间（秒） |

## 错误信息
- **错误类型**: 缺失值(Missing Value), 异常值(Outlier), 隐性缺失值(Implicit Missing Value)
- **错误条目数**: 4,209
- **错误单元格数**: 301,972

## 数据来源
Daimler. Mercedes-Benz Greener Manufacturing. https://www.kaggle.com/c/mercedes-benz-greener-manufacturing. 2017.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/mercedes/dirty_with_index.csv \
  --clean_path Data/mercedes/clean_with_index.csv \
  --task_name mercedes_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column y \
  --task_type regression \
  --index_attribute index \
  --verbose
```
