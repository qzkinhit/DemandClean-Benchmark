# 数据集：SmartFactory

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `labels` |
| 数据规模 | 23,645 条记录 × 19 列 (18 特征 + 1 标签) |
| 索引文件 | `clean_index.csv`, `dirty_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共18列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| i_w_blo_weg | 数值 | 左下传感器输入位移 |
| o_w_blo_power | 数值 | 左下传感器输出功率 |
| o_w_blo_voltage | 数值 | 左下传感器输出电压 |
| i_w_bhl_weg | 数值 | 左后传感器输入位移 |
| o_w_bhl_power | 数值 | 左后传感器输出功率 |
| o_w_bhl_voltage | 数值 | 左后传感器输出电压 |
| i_w_bhr_weg | 数值 | 右后传感器输入位移 |
| o_w_bhr_power | 数值 | 右后传感器输出功率 |
| o_w_bhr_voltage | 数值 | 右后传感器输出电压 |
| i_w_bru_weg | 数值 | 右下传感器输入位移 |
| o_w_bru_power | 数值 | 右下传感器输出功率 |
| o_w_bru_voltage | 数值 | 右下传感器输出电压 |
| i_w_hr_weg | 数值 | 右传感器输入位移 |
| o_w_hr_power | 数值 | 右传感器输出功率 |
| o_w_hr_voltage | 数值 | 右传感器输出电压 |
| i_w_hl_weg | 数值 | 左传感器输入位移 |
| o_w_hl_power | 数值 | 左传感器输出功率 |
| o_w_hl_voltage | 数值 | 左传感器输出电压 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| labels | 多分类 | 设备状态标签 |

## 错误统计

### 总览
| 指标 | 值 |
|------|-----|
| 错误单元格数 | 7,093 |
| 总单元格数 | 425,610 |
| 单元格错误率 | 1.67% |
| 错误行数 | 7,093 / 23,645 |
| 行错误率 | 30.0% |
| 标签错误数 | 0 |
| 标签错误率 | 0.0% |

### 错误类型分布
| 类型 | 数量 | 占比 |
|------|------|------|
| Semantic (语义错误) | 7,093 | 100.00% |
| Syntactic (句法错误) | 0 | 0.00% |
| Missing (缺失值) | 0 | 0.00% |

### 各列错误分布
| 列名 | 错误数 | 错误率 | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| o_w_hl_power | 429 | 1.81% | 429 | 0 |
| o_w_bhl_power | 413 | 1.75% | 413 | 0 |
| o_w_bru_voltage | 413 | 1.75% | 413 | 0 |
| o_w_bru_power | 409 | 1.73% | 409 | 0 |
| o_w_bhl_voltage | 404 | 1.71% | 404 | 0 |
| i_w_hl_weg | 402 | 1.70% | 402 | 0 |
| o_w_blo_power | 398 | 1.68% | 398 | 0 |
| o_w_hr_power | 398 | 1.68% | 398 | 0 |
| o_w_blo_voltage | 396 | 1.67% | 396 | 0 |
| i_w_bru_weg | 391 | 1.65% | 391 | 0 |
| i_w_hr_weg | 390 | 1.65% | 390 | 0 |
| i_w_bhl_weg | 385 | 1.63% | 385 | 0 |
| o_w_hr_voltage | 382 | 1.62% | 382 | 0 |
| o_w_hl_voltage | 382 | 1.62% | 382 | 0 |
| o_w_bhr_voltage | 381 | 1.61% | 381 | 0 |
| o_w_bhr_power | 378 | 1.60% | 378 | 0 |
| i_w_blo_weg | 374 | 1.58% | 374 | 0 |
| i_w_bhr_weg | 368 | 1.56% | 368 | 0 |

## 数据来源
Oliver Birgelen, Alexander; Niggemann. Smart Factory: High Storage System Data for Energy Optimization. https://www.kaggle.com/inIT-OWL/high-storage-system-data-for-energy-optimization. 2018.
