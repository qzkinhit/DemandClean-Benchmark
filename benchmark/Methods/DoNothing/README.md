# DoNothing Baseline

## 简介

DoNothing是最简单的baseline方法，**不对数据做任何清洗操作**，直接返回原始脏数据。

## 用途

- 建立性能下界（lower bound）
- 对比其他清洗方法的改进效果
- 验证评估流程的正确性

## 真值使用

- **Type 1**: 全自动执行，无需人工参与
- **真值成本**: 0

## 使用方式

```python
from Methods.DoNothing.donothing_wrapper import DoNothingWrapper

cleaner = DoNothingWrapper(verbose=True)
repaired_df, info = cleaner.clean(
    dirty_path='path/to/dirty.csv',
    output_path='path/to/output.csv'
)
```

## 代码修改说明

本方法为本仓库新实现，无官方代码基础。
