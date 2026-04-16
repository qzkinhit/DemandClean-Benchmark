# ActiveClean

## 简介

ActiveClean 是一个面向模型的迭代式数据清洗方法，通过主动学习选择对模型影响最大的脏数据进行清洗。

**论文**: [ActiveClean: Interactive Data Cleaning For Statistical Modeling](https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf) (VLDB 2016)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 3** | 迭代式主动学习，需要人工标注 |
| 真值使用 | 按需使用（每轮选择 batch_size 个样本） |

## 核心思想

1. 训练初始模型
2. 使用模型梯度识别对模型影响最大的脏数据
3. 请求用户清洗选中的样本
4. 更新模型，重复步骤2-4直到收敛

## 运行方式

```bash
# 单数据集运行
python MethodsRunScript/run_activeclean/run_activeclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_activeclean \
    --output_path results/activeclean/ \
    --label_column style \
    --task_type classification \
    --batch_size 50 \
    --total_budget 500

# 批量运行所有数据集
bash MethodsRunScript/run_activeclean/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（模拟人工标注） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/activeclean/` |
| `--label_column` | 是 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--batch_size` | 否 | 每轮清洗的样本数 | `50` |
| `--total_budget` | 否 | 总清洗预算 | `10000` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 输出文件

```
results/activeclean/{task_name}/
├── {task_name}_cleaned.csv          # 清洗后的数据
├── {task_name}_total_evaluation.txt # 完整评估报告（含真值使用成本）
└── {task_name}.log                  # 运行日志
```

## 特点

- **面向模型优化**: 优先清洗对模型影响最大的样本
- **预算可控**: 可设置清洗预算，控制人工成本
- **迭代收敛**: 持续优化直到模型性能收敛

## 与其他方法对比

| 方法 | 清洗策略 | 真值使用 |
|------|----------|----------|
| ActiveClean | 梯度引导选择 | 按需（Type 3） |
| BoostClean | Boosting 选择 | 验证集（Type 2） |
| Raha/Baran | 主动学习 | 按需（Type 3） |
