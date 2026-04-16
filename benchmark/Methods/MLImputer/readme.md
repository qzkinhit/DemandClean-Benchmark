# MLImputer

## 简介

MLImputer是基于机器学习模型的缺失值填充方法，通过训练模型预测缺失值。

## 核心特点

- **MICE插补**: Multiple Imputation by Chained Equations
- **KNN插补**: 基于K近邻的插补
- **随机森林插补**: 使用随机森林作为基估计器

## 真值使用情况

**类型**: 全自动执行，无需人工参与 (Type 1)

MLImputer通过已有数据训练模型预测缺失值，不需要额外标注。

## 文件结构

```
MLImputer/
├── __init__.py             # 包初始化
├── mlimputer_wrapper.py    # MLImputer封装类
├── readme.md               # 说明文档
└── requirements.txt        # 依赖
```

## 支持的方法

| 方法 | 说明 |
|------|------|
| `mice` | 多重插补（迭代式） |
| `knn` | K近邻插补 |
| `rf` | 随机森林插补 |

## 使用示例

```python
from Methods.MLImputer import MLImputerWrapper

# 创建插补器
imputer = MLImputerWrapper(
    method='mice',
    max_iter=10,
    verbose=True
)

# 执行插补
repaired_df, info = imputer.clean(
    dirty_path='data/dirty.csv',
    output_path='results/imputed.csv'
)

print(f"填充的单元格数: {info['imputed_cells']}")
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| method | 'mice' | 插补方法 |
| max_iter | 10 | MICE最大迭代次数 |
| n_neighbors | 5 | KNN邻居数 |
| random_state | 42 | 随机种子 |
