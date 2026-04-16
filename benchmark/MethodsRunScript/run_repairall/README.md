# RepairAll Baseline

## 简介

RepairAll 是基于真值修复的 baseline 方法，直接使用干净数据替换脏数据中的错误值。代表清洗效果的**理论上界**。

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 2** | 需要完整真值 |
| 真值使用 | 100%（使用全部真值） |

## 运行方式

```bash
# 单数据集运行
python MethodsRunScript/run_repairall/run_repairall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_repairall \
    --output_path results/repairall/ \
    --label_column style \
    --task_type classification

# 批量运行所有数据集
bash MethodsRunScript/run_repairall/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（真值） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/repairall/` |
| `--index_attribute` | 否 | 索引列名 | `index` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 输出文件

```
results/repairall/{task_name}/
├── {task_name}_cleaned.csv          # 修复后的数据（=干净数据）
├── {task_name}_total_evaluation.txt # 完整评估报告
└── {task_name}.log                  # 运行日志
```

## 用途

- 建立清洗效果的**理论上界**
- 评估其他方法与"完美修复"的差距
- 验证下游任务在干净数据上的最佳性能

## 与 DoNothing 的对比

| 方法 | 输出 | 用途 |
|------|------|------|
| DoNothing | 原始脏数据 | 性能下界 |
| RepairAll | 完全干净数据 | 性能上界 |
