# BoostClean Baseline

## 简介

BoostClean 是一种面向模型的数据清洗方法，使用 Boosting 策略集成多种错误检测器和修复器，自动选择最优组合来提升下游ML模型性能。

**论文**: [BoostClean: Automatic Error Detection and Repair for Machine Learning](https://arxiv.org/abs/1711.01299)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 2** | 需要验证集真值 |
| 真值使用 | 验证集（validation_ratio） |

## 核心思想

1. 定义一组错误检测器（缺失值、异常值、类型错误等）
2. 定义一组修复器（删除、填充、规则修复等）
3. 使用 Boosting 策略迭代选择最优的检测器-修复器组合
4. 以验证集上的模型性能为优化目标

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_boostclean/run_boostclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_boostclean \
    --output_path results/boostclean/ \
    --label_column style \
    --task_type classification \
    --boosting_rounds 5

# 批量运行所有数据集
bash MethodsRunScript/run_boostclean/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径 | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/boostclean/` |
| `--label_column` | 是 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--boosting_rounds` | 否 | Boosting 轮数 | `5` |
| `--quantitative_thresh` | 否 | 数值异常检测阈值 | `10` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 输出文件

```
results/boostclean/{task_name}/
├── {task_name}_cleaned.csv          # 清洗后的数据
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 与 ActiveClean 对比

| 方法 | 选择策略 | 真值使用 | 优化目标 |
|------|----------|----------|----------|
| ActiveClean | 梯度引导 | 按需清洗 | 模型损失 |
| **BoostClean** | Boosting | 验证集 | 验证集性能 |

## 注意事项

- 需要划分验证集，数据量较小时可能影响效果
- 迭代次数越多效果越好，但计算成本越高
- 适合有明确下游任务的场景
