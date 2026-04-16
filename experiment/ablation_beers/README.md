# Beers (IPA) 消融实验

> 基于 Beer 数据集 (ABV, IBU) 的小规模消融实验，对比 5 种基线策略 + 12 种 DQN 策略变体

---

## 实验目的

在低维 (2 特征) Beer 数据集上，系统对比 DemandClean 各版本策略的清洗效果：
1. **基线对比**: NoFix / OverFix / RelaxFix / FullFix / DemandFix
2. **DQN 架构消融**: Plain DQN vs Dueling Double DQN
3. **决策阶段消融**: Single Stage vs Two Stage
4. **检测器消融**: Oracle (已知错误) vs Auto (自动检测)
5. **推理模式消融**: Single Phase vs Two Phase

## 目录结构

```
ablation_beers/
├── run_ablation.py          # 主实验脚本（使用 DemandClean 发行版 API）
├── README.md                # 本文件
├── datasets/beers/          # Beer 数据集
│   ├── clean.csv            # 干净数据
│   ├── dirty.csv            # 脏数据
│   └── README.md
├── result/                  # 实验结果（boundary plot + 训练曲线）
│   ├── {策略名}.png         # 每种策略的决策边界图
│   ├── dqn_*_training.png   # DQN 训练历程图
│   └── detector.pkl         # 检测器缓存
└── model/                   # 已训练的 PyTorch 模型 (.pt)
```

## 使用方法

```bash
# 从项目根目录运行
cd /path/to/TolerDM

# 默认运行所有策略（400轮训练）
python experiment/ablation_beers/run_ablation.py

# 加载已保存模型（跳过训练，直接推理）
python experiment/ablation_beers/run_ablation.py --load_model

# 自定义训练轮数
python experiment/ablation_beers/run_ablation.py --n_episodes 100

# 只运行指定策略
python experiment/ablation_beers/run_ablation.py --strategies NoFix,FullFix,FullUnsup_Dueling_Single
```

## 策略清单 (17 种)

### 基线策略 (5 种)
| 策略名 | 说明 |
|--------|------|
| NoFix | 仅删除缺失值行，保留其他错误 |
| OverFix | 删除所有检测到的错误行 |
| RelaxFix | 所有错误用 KNN 近邻值填充 |
| FullFix | 所有错误用真值修复（成本最高） |
| DemandFix | 按需修复：边界区域用真值，非边界用 KNN/删除 |

### DQN 策略 (12 种)
| 策略名 | 检测器 | Agent架构 | 推理模式 |
|--------|--------|-----------|---------|
| DQN_Single | oracle | single (Plain) | single_phase |
| DQN_TwoStage | oracle | two_stage (Plain) | single_phase |
| SemiSup_Single | oracle | single (Plain) | single_phase |
| SemiSup_TwoStage | oracle | two_stage (Plain) | single_phase |
| FullUnsup_Single | auto | single (Plain) | single_phase |
| FullUnsup_TwoStage | auto | two_stage (Plain) | single_phase |
| FullUnsup_Single_2P | auto | single (Plain) | two_phase |
| FullUnsup_TwoStage_2P | auto | two_stage (Plain) | two_phase |
| SemiSup_Dueling_Single | oracle | dueling_single | single_phase |
| SemiSup_Dueling_TwoStage | oracle | dueling_two_stage | single_phase |
| FullUnsup_Dueling_Single | auto | dueling_single | single_phase |
| FullUnsup_Dueling_TwoStage | auto | dueling_two_stage | single_phase |

## 与发行版 API 的对应关系

消融实验中的 12 种 DQN 策略现已统一通过 `DemandClean` API 实现：

```python
from demandclean.api.demand_clean import DemandClean

dc = DemandClean(
    task_type='classification',
    model_type='svm',
    agent_type='...',          # single / two_stage / dueling_single / dueling_two_stage
    detector_mode='...',       # oracle / auto
    inference_mode='...',      # single_phase / two_phase
    n_episodes=400,
    column_names=['abv', 'ibu'],
    label_col='is_ipa',
)
```

## 数据集说明

- **来源**: Beer Reviews Dataset, 筛选 ABV 和 IBU 均非空的行
- **特征**: ABV (酒精度), IBU (苦度)
- **标签**: is_ipa (是否为 IPA 风格)
- **错误注入**: 语义错误 (15%) + 句法错误 (25%) + 缺失值 (5%)
- **边界区域**: IBU ∈ [35, 65]，错误集中在此区域

## 可视化输出

### Decision Boundary Plot
每种策略生成一张决策边界图，展示：
- SVM 线性决策边界（实际 vs 理想）
- 训练数据点分布 (IPA vs Non-IPA)
- 准确率、真值成本、真实性 (Auth)、多样性 (Div)

### DQN Training History
每种 DQN 策略生成训练历程图，展示 reward 和 epsilon 变化曲线。

## 历史说明

本实验从 `history/experiments/pre_exp_ablation/beers_ipa_experiment/real_beers_experiment_with_detector.py` 迁移而来。主要改动：
1. 消除 TensorFlow 依赖，统一使用 PyTorch (发行版 API)
2. 12 个独立策略函数 → 统一 DemandClean API 参数化调用
3. 保留原始错误注入逻辑（beers 数据集特定）
4. 保留所有可视化函数（boundary plot、训练曲线）
5. 更新文件保存路径至当前目录结构
