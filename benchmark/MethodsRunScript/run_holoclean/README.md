# HoloClean Baseline

## 简介

HoloClean 是一个基于概率图模型的数据清洗系统，通过融合多种信号（约束、统计、知识库）来自动修复数据错误。

**论文**: [HoloClean: Holistic Data Repairs with Probabilistic Inference](https://www.vldb.org/pvldb/vol10/p1190-rekatsinas.pdf) (VLDB 2017)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 核心思想

1. 使用 Denial Constraints (DC) 检测错误
2. 生成候选修复值
3. 构建因子图，融合多种信号
4. 使用概率推理选择最优修复

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_holoclean/run_holoclean_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --rule_path Data/beers/rules.txt \
    --task_name beers_holoclean \
    --output_path results/holoclean/ \
    --label_column style \
    --task_type classification

# 批量运行所有数据集
bash MethodsRunScript/run_holoclean/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 否 | 干净数据路径（用于评估） | - |
| `--rule_path` | 否 | 规则文件路径 | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/holoclean/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--db_user` | 否 | PostgreSQL 用户名 | `holocleanuser` |
| `--db_name` | 否 | 数据库名称 | `holo` |
| `--epochs` | 否 | 训练轮数 | `10` |
| `--learning_rate` | 否 | 学习率 | `0.001` |
| `--threads` | 否 | 线程数 | `1` |
| `--weak_label_thresh` | 否 | 弱标签阈值 | `0.99` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 规则文件格式

规则文件需包含 `[HOLOCLEAN_DC]` 部分：

```
[HOLOCLEAN_DC]
# Denial Constraint 格式
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.brewery_name,t2.brewery_name)
t1&t2&EQ(t1.brewery_id,t2.brewery_id)&IQ(t1.city,t2.city)
```

**DC 语法说明**:
- `t1&t2`: 定义两个元组变量
- `EQ(t1.attr, t2.attr)`: 相等谓词
- `IQ(t1.attr, t2.attr)`: 不等谓词
- `LT/GT/LTE/GTE`: 比较谓词

## 依赖要求

- **PostgreSQL**: 需要创建 `holo` 数据库
- **Python 3.7**: HoloClean 官方仅支持 Python 3.7
- PyTorch, psycopg2, sqlalchemy

### PostgreSQL 配置

```sql
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
GRANT ALL ON SCHEMA public TO holocleanuser;
```

## 输出文件

```
results/holoclean/{task_name}/
├── {task_name}_cleaned.csv          # 修复后的数据
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 与 Horizon 对比

| 方法 | 约束类型 | 修复策略 | 特点 |
|------|----------|----------|------|
| Horizon | FD | 模式质量 | 快速、可扩展 |
| **HoloClean** | DC | 概率推理 | 更灵活、信号融合 |

## 注意事项

- Python 3.10+ 可能存在兼容性问题
- 需要 PostgreSQL 数据库支持
- 大数据集处理较慢、内存消耗大
- 无 DC 规则时会跳过修复
