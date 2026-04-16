# Lopster Baseline

## 简介

Lopster 是一种基于变分自编码器（VAE）的数据清洗方法，通过学习数据的潜在表示来检测和修复错误。

**论文**: [Lopster: Learning to Repair Tables](https://dl.acm.org/doi/10.14778/3632093.3632099) (VLDB 2024)

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 核心思想

1. 使用 VAE 学习数据的潜在分布
2. 通过重构误差检测异常/错误
3. 使用潜在空间中的最近邻进行值修复
4. 迭代优化修复结果

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_lopster/run_lopster_base.py \
    --dataset beers \
    --data_path Data \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_lopster \
    --output_path results/lopster/ \
    --label_column style \
    --task_type classification \
    --epochs 100 \
    --latent_dim 120

# 批量运行所有数据集
bash MethodsRunScript/run_lopster/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dataset` | 是 | 数据集名称 | - |
| `--data_path` | 否 | 数据根目录 | `Data` |
| `--clean_path` | 否 | 干净数据路径（用于评估） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/lopster/` |
| `--label_column` | 否 | 标签列名 | - |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--latent_dim` | 否 | 潜在空间维度 | `120` |
| `--epochs` | 否 | 训练轮数 | `100` |
| `--learning_rate` | 否 | 学习率 | `0.001` |
| `--batch_size` | 否 | 批大小 | `256` |
| `--K` | 否 | K近邻参数 | `12` |
| `--clean_ratio` | 否 | clean数据使用比例 | `1.0` |
| `--models` | 否 | 评估模型列表 | `rf lr` |

## 输出文件

```
results/lopster/{task_name}/
├── {task_name}_cleaned.csv          # 清洗后的数据
├── {task_name}_total_evaluation.txt # 评估报告
├── {task_name}.log                  # 运行日志
└── model/                           # 训练的VAE模型
```

## 评估指标

Lopster 论文定义的评估指标：
- **col_avg_rmse**: 数值列归一化RMSE均值（StandardScaler归一化后计算）
- **col_avg_f1**: 类别列weighted F1均值

## 特点

- **深度学习**: 使用神经网络学习数据表示
- **无规则**: 不需要预定义规则或约束
- **通用性**: 适用于各种类型的数据集
- **GPU加速**: 支持 CUDA 加速训练

## 与其他方法对比

| 方法 | 原理 | 需要规则 | 需要训练 |
|------|------|----------|----------|
| **Lopster** | VAE潜在表示 | 否 | 是 |
| HoloClean | 概率图模型 | 是（DC） | 是 |
| Horizon | FD模式图 | 是（FD） | 否 |

## 注意事项

- 训练时间与数据集大小和 epochs 相关
- latent_dim 过小可能导致欠拟合
- 建议在 GPU 环境下运行以加速训练
