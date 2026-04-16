# DoNothing Baseline

## 简介

DoNothing 是最简单的 baseline 方法，**不对数据做任何清洗**，直接返回原始脏数据。用于建立性能下界，验证其他清洗方法的改进效果。

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 运行方式

```bash
# 单数据集运行
python MethodsRunScript/run_donothing/run_donothing_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_donothing \
    --output_path results/donothing/ \
    --label_column style \
    --task_type classification \
    --models rf lr

# 批量运行所有数据集
bash MethodsRunScript/run_donothing/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/donothing/` |
| `--index_attribute` | 否 | 索引列名 | `index` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 支持的任务类型

| 任务类型 | 模型 |
|----------|------|
| classification | rf, lr, svm, knn, dt, gb |
| regression | rf, lr, ridge, lasso, knn, gb |
| clustering | kmeans, agglomerative |

## 输出文件

```
results/donothing/{task_name}/
├── {task_name}_cleaned.csv          # 输出数据（与输入相同）
├── {task_name}_total_evaluation.txt # 完整评估报告
└── {task_name}.log                  # 运行日志
```

## 用途

- 建立清洗效果的性能下界
- 验证其他清洗方法是否有实际改进
- 对比脏数据直接用于下游任务的效果
