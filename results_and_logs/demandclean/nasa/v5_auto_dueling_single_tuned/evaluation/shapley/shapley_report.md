# Shapley Value Analysis Report

- 数据规模: 901 行 × 5 列
- 错误总数: 482
- 任务类型: regression (random_forest)
- MC 采样次数: 200

## 维度 1: 动作重要性 (Action Importance)

| 排名 | 动作 | Shapley 值 | 占比 | 方向 |
|------|------|-----------|------|------|
| 1 | no_action | -0.809496 | 11.2% | ↓ 负贡献 |
| 2 | replace_nearby | -1.259635 | 17.5% | ↓ 负贡献 |
| 3 | repair_value | -1.393638 | 19.3% | ↓ 负贡献 |
| 4 | delete | -3.753890 | 52.0% | ↓ 负贡献 |

**结论**: 最有效的清洗动作是 **no_action**
（no_action, replace_nearby, repair_value, delete 对性能有负面影响，建议减少使用）

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
| 1 | semantic | +1.250578 | 26.5% | 🟡 中 |
| 2 | syntactic | -1.020329 | 21.6% | 🟡 中 |
| 3 | missing | -2.452786 | 51.9% | 🔴 高 |

**结论**: 对下游性能影响最大的错误类型是 **semantic**，检测器应优先识别此类错误。

## 综合清洗建议

