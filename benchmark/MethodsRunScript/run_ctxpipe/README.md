# CtxPipe Baseline

## 简介

CtxPipe (Context-aware Data Preparation Pipeline) 是一个基于强化学习的数据准备管道自动生成工具，能够根据数据上下文自动选择最优的数据处理流程。

**论文**: [CtxPipe: Context-aware Data Preparation Pipeline Construction for Machine Learning](https://dl.acm.org/doi/10.1145/3626246.3653389) (SIGMOD 2025)

**重要说明**:
- `Methods/ctxpipe` 目录与官方仓库完全一致，未做代码修改
- 使用预训练模型 `ctx_50000` 进行推理，**无需训练**

## 方法类型

| 类型 | 说明 |
|------|------|
| **Type 1** | 全自动，无需真值 |
| 真值使用 | 0（仅用于评估） |

## 核心思想

1. 使用 GTE-large 嵌入模型提取数据表的上下文向量
2. 通过 DQN 强化学习选择最优数据处理组件
3. 自动构建端到端的数据准备管道

## 运行方式

```bash
# 单个数据集
python MethodsRunScript/run_ctxpipe/run_ctxpipe_base.py \
    --dirty_path Data/beers/dirty_index.csv \
    --clean_path Data/beers/clean_index.csv \
    --task_name beers_ctxpipe \
    --output_path results/ctxpipe/ \
    --label_index 4 \
    --task_type classification \
    --model_tag ctx_50000

# 批量运行所有数据集
bash MethodsRunScript/run_ctxpipe/run.sh
```

## 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--dirty_path` | 是 | 脏数据路径 | - |
| `--clean_path` | 是 | 干净数据路径（用于评估） | - |
| `--task_name` | 是 | 任务名称 | - |
| `--output_path` | 否 | 结果输出路径 | `results/ctxpipe/` |
| `--label_index` | 否 | 标签列索引（从0开始） | 自动检测 |
| `--task_type` | 否 | 任务类型 | `classification` |
| `--model_tag` | 否 | 预训练模型标签 | `ctx_50000` |
| `--skip_evaluation` | 否 | 跳过评估 | `False` |

## 数据集配置

| 数据集 | 任务类型 | label_index | 标签列 |
|--------|----------|-------------|--------|
| adult | classification | 14 | income |
| beers | classification | 4 | style |
| breast_cancer | classification | 9 | class |
| smartfactory | classification | 18 | labels |
| bike | regression | 15 | cnt |
| mercedes | regression | 1 | y |
| nasa | regression | 5 | sound_pressure_level |
| soilmoisture | regression | 2 | soil_moisture |
| har | clustering | 3 | gt |

## 管道组件

CtxPipe 自动选择以下组件的最优组合：

| 组件 | 选项 |
|------|------|
| 数值填充 | 均值、中位数、众数 |
| 类别填充 | 众数 |
| 编码器 | 标签编码、独热编码 |
| 特征预处理 | MinMaxScaler, StandardScaler, RobustScaler |
| 特征工程 | 多项式特征、PCA、核PCA |
| 特征选择 | 方差阈值 |

## 输出文件

```
results/ctxpipe/{task_name}/
├── {task_name}_ctxpipe_output.csv   # 处理后的数据
├── {task_name}_pipeline_info.txt    # 选择的管道信息
├── {task_name}_total_evaluation.txt # 评估报告
└── {task_name}.log                  # 运行日志
```

## 依赖要求

- PyTorch
- sentence-transformers（GTE-large 模型）
- 需要下载嵌入模型到 `Methods/ctxpipe/embed/gte-large/`

## 设备说明

- 自动检测：有 GPU 时使用 CUDA，无 GPU 时使用 CPU
- 适配器会自动处理设备兼容性

## 注意事项

- 使用独立的 conda 环境 `ctxpipe-pt112`
- 每个数据集运行时间几秒到几十秒
- 无需训练，直接使用预训练模型推理
