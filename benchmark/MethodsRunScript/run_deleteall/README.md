# DeleteAll Baseline

## 简介

DeleteAll 是基于删除策略的 baseline 方法，通过删除含有问题的行来"清洗"数据。支持两种删除模式。

## 方法类型

| 模式 | 类型 | 说明 |
|------|------|------|
| drop_missing | **Type 1** | 删除含缺失值的行，全自动 |
| drop_errors | **Type 2** | 删除与干净数据不一致的行，需要真值 |

## 运行方式

```bash
# drop_missing 模式（默认）- 删除含缺失值的行
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_deleteall \
    --mode drop_missing \
    --label_column style \
    --task_type classification

# drop_errors 模式 - 删除所有错误行
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_deleteall_errors \
    --mode drop_errors \
    --label_column style

# 批量运行所有数据集
bash MethodsRunScript/run_deleteall/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径 | - |
| `--task_name` | 是 | 任务名称 | - |
| `--mode` | 否 | 删除模式 | `drop_missing` |
| `--output_path` | 否 | 结果输出路径 | `results/deleteall/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 删除模式对比

| 模式 | 检测方式 | 适用场景 | 数据量影响 |
|------|----------|----------|------------|
| `drop_missing` | 检测空值/NaN | 缺失值较少时 | 可能删除少量行 |
| `drop_errors` | 与真值对比 | 需要真值时 | 可能删除大量行 |

## 输出文件

```
results/deleteall/{task_name}/
├── {task_name}_cleaned.csv          # 删除后的数据
├── {task_name}_total_evaluation.txt # 完整评估报告
└── {task_name}.log                  # 运行日志
```

## 注意事项

- `drop_errors` 模式会导致数据量大幅减少
- 删除行可能影响数据分布和模型训练
- 适用于错误行较少的场景
