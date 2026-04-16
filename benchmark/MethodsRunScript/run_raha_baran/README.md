# Raha & Baran Baseline

## 简介

Raha 是一种基于配置无关方法集成的错误检测系统，Baran 是其对应的错误修复系统。两者结合实现端到端的数据清洗。

**论文**:
- [Raha: A Configuration-Free Error Detection System](https://dl.acm.org/doi/10.1145/3299869.3324956) (SIGMOD 2019)
- [Baran: Effective Error Correction via a Unified Context Representation](https://www.vldb.org/pvldb/vol13/p1948-mahdavi.pdf) (VLDB 2020)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 3** | 需要用户标注样本 |
| 真值使用 | 按需标注（labeling_budget） |

## 核心思想

### Raha（错误检测）
1. 运行多种错误检测策略（异常值、模式违规、FD违规等）
2. 使用聚类生成候选错误特征
3. 用户标注少量样本
4. 训练分类器识别所有错误

### Baran（错误修复）
1. 构建统一的上下文表示
2. 生成候选修复值
3. 使用上下文相似度排序修复建议
4. 应用最优修复

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_raha_baran/run_raha_baran_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_raha_baran \
    --output_path results/raha_baran/ \
    --label_column style \
    --task_type classification \
    --labeling_budget 20

# 批量运行所有数据集
bash MethodsRunScript/run_raha_baran/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（模拟用户标注） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/raha_baran/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--labeling_budget` | 否 | 标注预算（元组数） | `20` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 输出文件

```
results/raha_baran/{task_name}/
├── {task_name}_cleaned.csv          # 清洗后的数据
├── {task_name}_detection.csv        # 检测到的错误
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 特点

- **配置无关**: 无需预定义规则或阈值
- **策略集成**: 融合多种检测方法的优势
- **主动学习**: 智能选择最有价值的标注样本
- **上下文感知**: 利用数据上下文提升修复准确性

## 与其他方法对比

| 方法 | 检测策略 | 修复策略 | 真值使用 |
|------|----------|----------|----------|
| **Raha/Baran** | 多策略集成 | 上下文相似度 | 少量标注 |
| ActiveClean | - | 梯度引导 | 按需清洗 |
| HoloClean | DC约束 | 概率推理 | 无需 |

## 注意事项

- 标注预算影响检测准确率
- 需要足够的数据量才能发挥优势
- 检测和修复是分开的两个阶段
