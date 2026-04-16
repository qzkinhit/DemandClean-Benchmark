# SimpleImputer Baseline

## 简介

SimpleImputer 是基于 scikit-learn 的简单缺失值填充方法，使用统计量（均值、中位数、众数等）填充缺失值。

**论文**: scikit-learn 内置方法

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 运行方式

```bash
# 使用默认策略（均值填充）
python MethodsRunScript/run_simpleimputer/run_simpleimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_simpleimputer \
    --output_path results/simpleimputer/ \
    --label_column style \
    --task_type classification

# 指定填充策略
python MethodsRunScript/run_simpleimputer/run_simpleimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_simpleimputer \
    --strategy median

# 批量运行所有数据集
bash MethodsRunScript/run_simpleimputer/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/simpleimputer/` |
| `--strategy` | 否 | 填充策略 | `mean` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 支持的填充策略

| 策略 | 说明 | 适用类型 |
|------|------|----------|
| `mean` | 均值填充 | 数值列 |
| `median` | 中位数填充 | 数值列 |
| `most_frequent` | 众数填充 | 所有类型 |
| `constant` | 常数填充 | 所有类型 |

## 输出文件

```
results/simpleimputer/{task_name}/
├── {task_name}_cleaned.csv          # 填充后的数据
├── {task_name}_total_evaluation.txt # 完整评估报告
└── {task_name}.log                  # 运行日志
```

## 特点

- 简单高效，适合快速基准测试
- 仅处理缺失值，不修复错误值
- 数值列和类别列分别处理
