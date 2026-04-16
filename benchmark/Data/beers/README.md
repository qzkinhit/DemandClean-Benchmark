# 数据集：Beers

## 基本信息

| 项目 | 值 |
|------|-----|
| 任务类型 | 分类 (Classification) |
| 目标列 | `style` |
| 数据规模 | 2,410 条记录 × 11 属性 |
| 索引文件 | `clean_index.csv`, `dirty_index.csv` |
| 缺失值标记 | `empty` |

## 文件结构

### 主要数据文件
| 文件名 | 说明 | 用途 |
|--------|------|------|
| `clean.csv` | 干净数据（无索引） | 原始干净数据 |
| `dirty.csv` | 脏数据（无索引） | 原始脏数据 |
| `clean_index.csv` | 干净数据（带index列） | **推荐使用** - 主要评测文件 |
| `dirty_index.csv` | 脏数据（带index列） | **推荐使用** - 主要评测文件 |
| `clean_with_index.csv` | 同clean_index.csv | 兼容旧版本 |
| `dirty_with_index.csv` | 同dirty_index.csv | 兼容旧版本 |

### HoloClean专用文件
| 文件名 | 说明 |
|--------|------|
| `clean_holoclean.csv` | 转置格式 (tid, attribute, correct_val) |
| `dirty_index_holoclean.csv` | 去掉index列的脏数据 |
| `*_ori_empty.csv` | 缺失值统一为空字符串的版本 |

### 规则文件
| 文件名 | 说明 |
|--------|------|
| `dc_rules_holoclean.txt` | HoloClean格式的Denial Constraints |
| `dc_rules-validate-fd-horizon.txt` | Horizon格式的Functional Dependencies |
| `fd_rule.txt` | 函数依赖规则 |
| `rules.txt` | 通用规则文件 |

## 列定义

### 索引列 (Index)
| 列名 | 说明 |
|------|------|
| `index` | 行索引，不参与模型训练 |

### 排除列 (Excluded) - 不参与模型训练
| 列名 | 类型 | 说明 |
|------|------|------|
| id | 数值 | 啤酒ID (标识符) |
| beer_name | 文本 | 啤酒名称 (文本标识) |
| brewery_id | 数值 | 酿酒厂ID (标识符) |
| brewery_name | 文本 | 酿酒厂名称 (文本标识) |
| city | 文本 | 城市 (文本分类，高基数) |
| state | 文本 | 州 (文本分类) |

### 特征列 (Features) - 共3列
| 属性名 | 类型 | 说明 |
|--------|------|------|
| ounces | 数值 | 容量(盎司) |
| abv | 数值 | 酒精度数 |
| ibu | 数值 | 苦度单位 |

### 目标列 (Label)
| 属性名 | 类型 | 说明 |
|--------|------|------|
| style | 多分类 | 啤酒风格类型 |

## 错误信息
- **错误类型**: 缺失值(Missing Value), 规则违例(Rule Violation), 拼写错误(Typos)
- **错误条目数**: 2,410
- **错误单元格数**: 3,357
- **缺失值标记**: `empty`

## 数据来源
J.-N. Hould. Craft beers dataset. https://www.kaggle.com/nickhould/craft-cans. 2017.

## 运行命令示例

```bash
# DeleteAll baseline
python MethodsRunScript/run_deleteall/run_deleteall_base.py \
  --dirty_path Data/beers/dirty_index.csv \
  --clean_path Data/beers/clean_index.csv \
  --task_name beers_deleteall \
  --mode drop_missing \
  --output_path results/deleteall/ \
  --label_column style \
  --task_type classification \
  --index_attribute index \
  --exclude_columns id beer_name brewery_id brewery_name city state
```
