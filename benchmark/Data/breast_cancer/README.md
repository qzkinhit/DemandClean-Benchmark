# 数据集：Breast_Cancer

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `class` |
| 数据规模 | 699 条记录 × 11 属性 |
| 索引文件 | `clean_with_index.csv`, `dirty_with_index.csv` |

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

## 错误信息
- **错误类型**: 缺失值(Missing Values), 异常值(Outliers), 拼写错误(Typos)
- **错误条目数**: 453
- **错误单元格数**: 631

## 数据来源
Wolberg, W. Breast Cancer Wisconsin (Original). https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original. 1990.

## 运行命令示例

```bash
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/breast_cancer/dirty_with_index.csv \
  --clean_path Data/breast_cancer/clean_with_index.csv \
  --task_name breast_cancer_test \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column class \
  --task_type classification \
  --index_attribute index \
  --verbose
```
