# DeleteAll Baseline

## 简介

DeleteAll是一种简单的baseline方法，通过**删除含有问题的行**来"清洗"数据。

## 支持模式

### 1. drop_missing (默认)
- 删除所有含有缺失值（NaN、空字符串、N/A等）的行
- **Type 1**: 全自动执行，无需人工参与
- **真值成本**: 0

### 2. drop_errors
- 删除所有与干净数据不一致的行
- **Type 2**: 需要干净数据对比
- **真值成本**: 删除的行数

## 用途

- 建立一种激进的清洗策略baseline
- 对比保留数据量vs数据质量的权衡
- 验证模型对数据量减少的敏感度

## 使用方式

```python
from Methods.DeleteAll.deleteall_wrapper import DeleteAllWrapper

# 模式1: 仅删除缺失值行
cleaner = DeleteAllWrapper(mode='drop_missing', verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv'
)

# 模式2: 删除所有错误行
cleaner = DeleteAllWrapper(mode='drop_errors', verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv',
    clean_path='path/to/clean.csv'
)
```

## 代码修改说明

本方法为本仓库新实现，无官方代码基础。
