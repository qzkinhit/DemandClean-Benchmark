# MLImputer Baseline

## 简介

MLImputer 是基于机器学习的缺失值填充方法，使用 MICE、KNN、随机森林等算法预测并填充缺失值。

**实现**: 基于 scikit-learn 的 IterativeImputer 和 KNNImputer

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 运行方式

```bash
# 使用 MICE 方法（默认）
python MethodsRunScript/run_mlimputer/run_mlimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_mlimputer \
    --output_path results/mlimputer/ \
    --method mice \
    --label_column style \
    --task_type classification

# 使用 KNN 方法
python MethodsRunScript/run_mlimputer/run_mlimputer_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_mlimputer_knn \
    --method knn

# 批量运行所有数据集
bash MethodsRunScript/run_mlimputer/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/mlimputer/` |
| `--method` | 否 | 插补方法 | `mice` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 支持的插补方法

| 方法 | 说明 | 特点 |
|------|------|------|
| `mice` | Multiple Imputation by Chained Equations | 迭代式，效果好 |
| `knn` | K-Nearest Neighbors | 基于相似样本 |
| `rf` | Random Forest | 使用随机森林预测 |

## 输出文件

```
results/mlimputer/{task_name}/
├── {task_name}_cleaned.csv          # 填充后的数据
├── {task_name}_total_evaluation.txt # 完整评估报告
└── {task_name}.log                  # 运行日志
```

## 与 SimpleImputer 的对比

| 方法 | 原理 | 效果 | 速度 |
|------|------|------|------|
| SimpleImputer | 统计量填充 | 一般 | 快 |
| MLImputer | ML模型预测 | 较好 | 较慢 |

## 注意事项

- MICE 方法在大数据集上可能较慢
- 仅处理缺失值，不修复错误值
- 需要数值编码才能使用
