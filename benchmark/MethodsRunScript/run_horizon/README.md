# Horizon Baseline

## 简介

Horizon 是一种基于函数依赖（Functional Dependency）的可扩展数据清洗方法，通过识别和修复 FD 违规来清洗数据。

**论文**: [Horizon: Scalable Dependency-Driven Data Cleaning](https://www.vldb.org/pvldb/vol14/p2546-yan.pdf) (VLDB 2021)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 核心思想

1. 解析函数依赖规则（如 `A => B`）
2. 构建 FD 模式图，计算每个模式的质量
3. 使用强连通分量和拓扑排序确定修复顺序
4. 根据模式质量选择最优修复值

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_horizon/run_horizon_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --rule_path Data/beers/rules.txt \
    --task_name beers_horizon \
    --output_path results/horizon/ \
    --label_column style \
    --task_type classification

# 批量运行所有数据集
bash MethodsRunScript/run_horizon/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--rule_path` | **是** | 规则文件路径 | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/horizon/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 规则文件格式

规则文件需包含 `[HORIZON_FD]` 部分：

```
[HORIZON_FD]
# 格式: LHS => RHS
brewery_id => brewery_name
brewery_id => city
brewery_id => state
style => abv
```

**规则说明**:
- `LHS => RHS`: 左侧属性决定右侧属性的值
- 支持 `=>` 和 `⇒` 两种箭头格式
- 以 `#` 开头的行为注释

## 输出文件

```
results/horizon/{task_name}/
├── {task_name}_cleaned.csv          # 修复后的数据
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 与 HoloClean 对比

| 方法 | 约束类型 | 修复策略 | 特点 |
|------|----------|----------|------|
| **Horizon** | FD（函数依赖） | 模式质量优先 | 快速、可扩展 |
| HoloClean | DC（否定约束） | 概率推理 | 更灵活、更慢 |

## 注意事项

- **必须提供规则文件**，否则无法运行
- 仅修复 FD 涉及的列，不处理其他列
- 规则质量直接影响清洗效果
- 适合具有明确业务规则的数据集
