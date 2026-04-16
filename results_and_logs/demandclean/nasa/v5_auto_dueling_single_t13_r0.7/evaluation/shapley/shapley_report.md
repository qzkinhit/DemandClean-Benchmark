# Shapley Value Analysis Report

- 数据规模: 901 行 × 5 列
- 错误总数: 482
- 任务类型: regression (random_forest)
- MC 采样次数: 200

## 维度 1: 动作重要性 (Action Importance)

| 排名 | 动作 | Shapley 值 | 占比 | 方向 |
|------|------|-----------|------|------|
| 1 | no_action | -0.220763 | 3.8% | ↓ 负贡献 |
| 2 | repair_value | -0.655132 | 11.4% | ↓ 负贡献 |
| 3 | delete | -1.418697 | 24.6% | ↓ 负贡献 |
| 4 | replace_nearby | -3.474695 | 60.2% | ↓ 负贡献 |

**结论**: 最有效的清洗动作是 **no_action**
（no_action, repair_value, delete, replace_nearby 对性能有负面影响，建议减少使用）

## 维度 2: 特征重要性 (Feature Importance)

| 排名 | 特征 | Shapley 值 | 占比 | 清洗价值 |
|------|------|-----------|------|---------|
| 1 | frequency | +0.000000 | 0.0% | — 可忽略 |
| 2 | angle | +0.000000 | 0.0% | — 可忽略 |
| 3 | chord_length | +0.000000 | 0.0% | — 可忽略 |
| 4 | velocity | +0.000000 | 0.0% | — 可忽略 |
| 5 | thickness | +0.000000 | 0.0% | — 可忽略 |

**结论**: 
- 清洗价值极低的特征: frequency, angle, chord_length（清洗它们几乎不影响性能）

## 维度 3: 错误类型重要性 (Error Type Importance)

| 排名 | 错误类型 | Shapley 值 | 占比 | 优先级 |
|------|---------|-----------|------|--------|
| 1 | semantic | -1.018919 | 16.3% | 🟢 低 |
| 2 | missing | -1.560714 | 24.9% | 🟡 中 |
| 3 | syntactic | -3.685446 | 58.8% | 🔴 高 |

**结论**: 对下游性能影响最大的错误类型是 **semantic**，检测器应优先识别此类错误。

## 综合清洗建议

