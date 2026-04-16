# 数据集：Breast_Cancer

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `class` |
| 数据规模 | 699 条记录 × 10 列 (9 特征 + 1 标签) |
| 索引文件 | `clean_index.csv`, `dirty_index.csv` |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 特征列 (Features) - 共9列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| Clump Thickness | 数值 | 细胞团厚度 (1-10) |
| Uniformity of Cell Size | 数值 | 细胞大小均匀性 (1-10) |
| Uniformity of Cell Shape | 数值 | 细胞形状均匀性 (1-10) |
| Marginal Adhesion | 数值 | 边缘粘附性 (1-10) |
| Single Epithelial Cell Size | 数值 | 单上皮细胞大小 (1-10) |
| Bare Nuclei | 数值 | 裸核 (1-10) |
| Bland Chromatin | 数值 | 平淡染色质 (1-10) |
| Normal Nucleoli | 数值 | 正常核仁 (1-10) |
| Mitoses | 数值 | 有丝分裂 (1-10) |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| class | 二分类 | 肿瘤类型 (2:良性, 4:恶性) |

## 错误统计

### 总览
| 指标 | 值 |
|------|-----|
| 错误单元格数 | 531 |
| 总单元格数 | 6,291 |
| 单元格错误率 | 8.44% |
| 错误行数 | 387 / 699 |
| 行错误率 | 55.4% |
| 标签错误数 | 15 |
| 标签错误率 | 2.15% |

### 错误类型分布
| 类型 | 数量 | 占比 |
|------|------|------|
| Semantic (语义错误) | 371 | 69.87% |
| Syntactic (句法错误) | 138 | 25.99% |
| Missing (缺失值) | 0 | 0.00% |

### 各列错误分布
| 列名 | 错误数 | 错误率 | Semantic | Syntactic |
|------|--------|--------|----------|-----------|
| Clump Thickness | 73 | 10.44% | 49 | 24 |
| Bare Nuclei | 72 | 10.30% | 50 | 0 |
| Normal Nucleoli | 63 | 9.01% | 48 | 15 |
| Single Epithelial Cell Size | 61 | 8.73% | 44 | 17 |
| Bland Chromatin | 61 | 8.73% | 39 | 22 |
| Uniformity of Cell Shape | 55 | 7.87% | 44 | 11 |
| Uniformity of Cell Size | 53 | 7.58% | 38 | 15 |
| Marginal Adhesion | 51 | 7.30% | 33 | 18 |
| Mitoses | 42 | 6.01% | 26 | 16 |

## 数据来源
Wolberg, W. Breast Cancer Wisconsin (Original). https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original. 1990.
