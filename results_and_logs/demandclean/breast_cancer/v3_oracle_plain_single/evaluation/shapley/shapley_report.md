# Shapley Value Analysis Report

- 数据规模: 419 行 × 9 列
- 错误总数: 328
- 任务类型: classification (random_forest)
- MC 采样次数: 200

## 维度 1: 动作重要性 (Action Importance)

| 排名 | 动作 | Shapley 值 | 占比 | 方向 |
|------|------|-----------|------|------|
| 1 | repair_value | +0.011044 | 78.6% | ↑ 正贡献 |
| 2 | no_action | +0.001004 | 7.1% | ↑ 正贡献 |
| 3 | delete | +0.001004 | 7.1% | ↑ 正贡献 |
| 4 | replace_nearby | -0.001004 | 7.1% | ↓ 负贡献 |

**结论**: 最有效的清洗动作是 **repair_value**
（replace_nearby 对性能有负面影响，建议减少使用）

## 维度 2: 特征重要性 (Feature Importance)

| 排名 | 特征 | Shapley 值 | 占比 | 清洗价值 |
|------|------|-----------|------|---------|
| 1 | Clump Thickness | +0.000000 | 0.0% | — 可忽略 |
| 2 | Uniformity of Cell Size | +0.000000 | 0.0% | — 可忽略 |
| 3 | Uniformity of Cell Shape | +0.000000 | 0.0% | — 可忽略 |
| 4 | Marginal Adhesion | +0.000000 | 0.0% | — 可忽略 |
| 5 | Single Epithelial Cell Size | +0.000000 | 0.0% | — 可忽略 |
| 6 | Bare Nuclei | +0.000000 | 0.0% | — 可忽略 |
| 7 | Bland Chromatin | +0.000000 | 0.0% | — 可忽略 |
| 8 | Normal Nucleoli | +0.000000 | 0.0% | — 可忽略 |
| 9 | Mitoses | +0.000000 | 0.0% | — 可忽略 |

**结论**: 
- 清洗价值极低的特征: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape（清洗它们几乎不影响性能）

## 维度 3: 错误类型重要性 (Error Type Importance)

| 排名 | 错误类型 | Shapley 值 | 占比 | 优先级 |
|------|---------|-----------|------|--------|
| 1 | label_noise | +0.012048 | 100.0% | 🔴 高 |
| 2 | semantic | +0.000000 | 0.0% | 🟢 低 |
| 3 | missing | +0.000000 | 0.0% | 🟢 低 |
| 4 | syntactic | +0.000000 | 0.0% | 🟢 低 |

**结论**: 对下游性能影响最大的错误类型是 **label_noise**，检测器应优先识别此类错误。

## 综合清洗建议

