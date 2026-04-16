# 数据集：NASA (Airfoil Self-Noise)

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 回归 (Regression) |
| 目标列 | `sound_pressure_level` |
| 数据规模 | 1,503 条记录 × 6 列 (5 特征 + 1 标签) |
| 索引文件 | `clean_index.csv`, `dirty_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共5列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| frequency | 数值 | 频率 (Hz) |
| angle | 数值 | 攻角 (度) |
| chord_length | 数值 | 弦长 (米) |
| velocity | 数值 | 自由流速度 (米/秒) |
| thickness | 数值 | 吸力侧位移厚度 (米) |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| sound_pressure_level | 数值 | 缩放声压级 (分贝) |

## 错误统计

### 总览
| 指标 | 值 |
|------|-----|
| 错误单元格数 | 731 |
| 总单元格数 | 7,515 |
| 单元格错误率 | 9.73% |
| 错误行数 | 731 / 1,503 |
| 行错误率 | 48.6% |
| 标签错误数 | 0 |
| 标签错误率 | 0.0% |

### 错误类型分布
| 类型 | 数量 | 占比 |
|------|------|------|
| Semantic (语义错误) | 475 | 64.98% |
| Syntactic (句法错误) | 256 | 35.02% |
| Missing (缺失值) | 0 | 0.00% |

### 各列错误分布
| 列名 | 错误数 | 错误率 | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| velocity | 157 | 10.45% | 115 | 42 |
| angle | 150 | 9.98% | 108 | 42 |
| chord_length | 148 | 9.85% | 80 | 68 |
| frequency | 138 | 9.18% | 107 | 31 |
| thickness | 138 | 9.18% | 65 | 73 |

## 数据来源
Brooks, T., Pope, D., & Marcolini, M. Airfoil Self-Noise. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise. 1989.
