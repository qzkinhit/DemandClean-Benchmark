# UniClean Baseline

## 简介

UniClean 是一种多信号融合的数据清洗框架，基于 PySpark 实现分布式处理，使用多种预定义的清洗器（Cleaner）对数据进行规则化清洗。

**论文**: UniClean: A Multi-Signal Unified Data Cleaning Framework (VLDB 2025)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 核心思想

1. 定义一组清洗器（Cleaner），如数值范围检查、模式匹配、异常值检测等
2. 对每一列应用对应的清洗器
3. 清洗器基于规则自动修复数据

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_uniclean/run_uniclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --dataset beers \
    --task_name beers_uniclean \
    --output_path results/uniclean/ \
    --label_column style \
    --task_type classification

# 批量运行所有数据集
bash MethodsRunScript/run_uniclean/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--dataset` | 是 | 数据集名称 | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/uniclean/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--single_max` | 否 | 单次处理最大记录数 | `10000` |
| `--executor_memory` | 否 | Spark executor 内存 | `8g` |
| `--driver_memory` | 否 | Spark driver 内存 | `8g` |

## 规则文件格式

清洗器定义在 `rules.txt` 的 `[UNICLEAN]` 部分：

```
[UNICLEAN]
# 数值类型检查
Number("ibu")
Number("abv")

# 模式匹配
Pattern("phone", r"^\d{3}-\d{4}$")

# 异常值检测
Outlier("price", [], "price_outlier")

# 属性关系
AttrRelation("brewery_id", ["brewery_name", "city", "state"])
```

## 支持的清洗器

| 清洗器 | 说明 | 示例 |
|--------|------|------|
| `Number(col)` | 数值类型检查 | `Number("price")` |
| `Pattern(col, regex)` | 正则模式匹配 | `Pattern("email", r".*@.*")` |
| `Outlier(col, bounds, name)` | 异常值检测 | `Outlier("age", [0, 120], "age_check")` |
| `Date(col, format)` | 日期格式检查 | `Date("date", "%Y-%m-%d")` |
| `AttrRelation(key, deps)` | 属性依赖关系 | `AttrRelation("id", ["name"])` |

## 输出文件

```
results/uniclean/{task_name}/
├── {task_name}_cleaned.csv          # 清洗后的数据
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 依赖要求

- PySpark
- Java 8+

## 注意事项

- 需要配置 Spark 环境
- 大数据集可能需要调整内存参数
- 清洗效果依赖于清洗器的配置质量
