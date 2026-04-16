# 搜索空间可视化实验

> 验证 DemandClean 清洗策略的搜索空间理论：在 Authenticity-Diversity 空间中映射所有可能策略的性能分布

---

## 实验目的

在 Beer (IPA) 数据集上验证数据清洗策略搜索空间的理论模型：
1. **四个极端点**: NoFix, FullFix, DeleteFix, RelaxFix
2. **随机采样**: 遍历所有动作概率组合 (no_action, repair, delete, replace_nearby)
3. **DQN 推理**: 标注 DemandFix 在搜索空间中的位置
4. **热力图**: 展示性能 (Accuracy) 在搜索空间中的分布

### 理论搜索空间模型

```
多样性 ↑
    │
FullFix  ●─────────────────● DeleteFix (多样性下界)
    │╲   搜索空间      │
    │ ╲  (月牙形)     │
    │  ╲             ╱
    │   ● DemandFix ╱
    │    ╲         ╱
    │     ╲       ╱
NoFix ●──┼──────●─────┼────→ 真实性
    │    RelaxFix
```

## 目录结构

```
search_space_beers/
├── run_search_space.py      # 主实验脚本（使用 DemandClean 发行版 API）
├── README.md                # 本文件
├── datasets/                # 符号链接 → ../ablation_beers/datasets/beers/
├── result/                  # 实验结果
│   ├── search_space_scatter.png      # 散点图（颜色=性能）
│   ├── search_space_heatmap.png      # 网格热力图
│   ├── search_space_combined.png     # 组合图
│   ├── search_space_cost_heatmap.png # 成本热力图
│   ├── search_space_cost_scatter.png # 成本散点图
│   ├── repair_vs_nonrepair.png       # repair vs delete+nearby 热力图
│   ├── action_pairs/                 # 动作组合热力图
│   ├── search_space_results.csv      # 采样结果数据
│   ├── dqn_distribution.csv          # DQN 推理结果
│   ├── extreme_points.json           # 极端点坐标
│   └── dirty_data.csv               # 注入错误后的数据
└── model/                   # DQN 模型
```

## 使用方法

```bash
# 从项目根目录运行
cd /path/to/TolerDM

# 只加载已有结果重新绘图（推荐）
python experiment/search_space_beers/run_search_space.py

# 重新运行完整实验（耗时较长）
# 需要在脚本中设置 RUN_EXPERIMENT = True
python experiment/search_space_beers/run_search_space.py
```

## 实验参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_RANDOM_SAMPLES` | 1000 | 随机采样数 |
| `N_DQN_RUNS` | 1 | DQN 推理轮数 |
| `SEMANTIC_RATE` | 0.15 | 语义错误注入率 |
| `SYNTACTIC_RATE` | 0.20 | 句法错误注入率 |
| `RUN_EXPERIMENT` | False | True=运行实验, False=只读取已有结果重新画图 |
| `LOAD_MODEL` | True | True=加载已训练模型, False=重新训练DQN |

## 可视化输出

### 1. 散点图 (`search_space_scatter.png`)
- X轴: Authenticity (真实性), Y轴: Diversity (多样性)
- 颜色: 分类准确率 (暖色=高, 冷色=低)
- ★: DQN DemandFix 位置

### 2. 网格热力图 (`search_space_heatmap.png`)
- 将搜索空间划分为网格
- 每个网格颜色表示平均准确率

### 3. 组合图 (`search_space_combined.png`)
- 散点图 + 热力图 + 极端点标注

### 4. 成本热力图 (`search_space_cost_heatmap.png`)
- 在 Auth-Div 空间中展示真值使用成本分布

### 5. 成本散点图 (`search_space_cost_scatter.png`)
- 无插值版本的成本分布

### 6. Repair vs Non-repair (`repair_vs_nonrepair.png`)
- repair_value 比例 vs (delete + replace_nearby) 比例的热力图

### 7. 动作组合热力图 (`action_pairs/`)
- 所有动作两两组合的性能热力图

## 额外依赖

```bash
pip install alphashape shapely
```

## 关键指标

### Authenticity (真实性)
```
Auth = 正确值数量 / 当前总值数量
```
衡量修复后数据与干净数据的一致程度。

### Diversity (多样性)
```
Div = 样本保留率 × 方差保留率
```
衡量清洗后数据保留了多少信息量。

## 历史说明

本实验从 `history/experiments/pre_exp_ablation/beers_ipa_experiment/history_logs/history/search_space_experiment.py` 迁移而来。主要改动：
1. DQN 训练/推理替换为 DemandClean 发行版 API (消除 TensorFlow 依赖)
2. 所有可视化函数完整保留，逻辑和风格不做修改
3. 随机采样逻辑完整保留
4. 更新文件保存路径至当前目录结构
