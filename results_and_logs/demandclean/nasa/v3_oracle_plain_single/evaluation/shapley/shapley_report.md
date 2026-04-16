# Shapley Value Analysis Report

- 数据规模: 1503 行 × 5 列
- 错误总数: 1230
- 任务类型: regression (ridge)
- MC 采样次数: 200

## 维度 1: 动作重要性 (Action Importance)

| 排名 | 动作 | Shapley 值 | 占比 | 方向 |
|------|------|-----------|------|------|
| 1 | repair_value | +1.946404 | 73.8% | ↑ 正贡献 |
| 2 | delete | +0.450771 | 17.1% | ↑ 正贡献 |
| 3 | no_action | +0.196694 | 7.5% | ↑ 正贡献 |
| 4 | replace_nearby | +0.043191 | 1.6% | ↑ 正贡献 |

**结论**: 最有效的清洗动作是 **repair_value**

## 维度 2: 特征重要性 (Feature Importance)

| 排名 | 特征 | Shapley 值 | 占比 | 清洗价值 |
|------|------|-----------|------|---------|
| 1 | thickness | +3.984889 | 41.8% | ★ 高价值 |
| 2 | velocity | +0.996585 | 10.4% | ★ 高价值 |
| 3 | chord_length | +0.733043 | 7.7% | ★ 高价值 |
| 4 | frequency | +0.685608 | 7.2% | ★ 高价值 |
| 5 | angle | -3.139254 | 32.9% | ✗ 负价值 |

**结论**: 
- 最值得清洗的特征: **thickness, velocity, chord_length**（优先投入清洗预算）
- 清洗后反而降低性能: angle（建议保留原值不清洗）

## 维度 3: 错误类型重要性 (Error Type Importance)

| 排名 | 错误类型 | Shapley 值 | 占比 | 优先级 |
|------|---------|-----------|------|--------|
| 1 | syntactic | +2.874429 | 79.9% | 🔴 高 |
| 2 | missing | +0.063955 | 1.8% | 🟢 低 |
| 3 | semantic | -0.658659 | 18.3% | 🟢 低 |

**结论**: 对下游性能影响最大的错误类型是 **syntactic**，检测器应优先识别此类错误。

## 综合清洗建议

在有限的真值预算下，建议优先对 **thickness, velocity, chord_length** 列中的 **syntactic** 类型错误执行 **repair_value** 操作，以最大化下游 regression 任务性能。
