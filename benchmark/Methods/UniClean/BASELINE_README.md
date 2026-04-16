# UniClean - 多信号融合数据清洗

## 官方信息
- **论文**: UniClean: A Unified Framework for Data Cleaning with Multi-Signal Fusion (VLDB 2025)
- **类型**: Type 1 - 全自动执行

## 方法描述
UniClean通过融合多种清洗信号（约束、统计、模式等）并优化清洗工作流来实现高效的数据清洗。

## 依赖安装

```bash
pip install pyspark==3.1.1
```

或使用本目录的requirements.txt:
```bash
pip install -r Methods/UniClean/requirements.txt
```

## 核心文件

- `Clean.py` - 核心清洗逻辑（CleanonLocalWithnoSmple等函数）
- `SampleScrubber/cleaner/single.py` - 单属性清洗器（Number, Pattern, Outlier）
- `SampleScrubber/cleaner/multiple.py` - 多属性清洗器（AttrRelation）
- `AnalyticsCache/` - 分析和缓存模块

## 数据格式要求

数据需要有index列：
```
Data/{dataset}/
├── dirty_with_index.csv
└── clean_with_index.csv
```

## 运行方式

### 方式1: 使用wrapper
```python
from Methods.UniClean.uniclean_wrapper import UniCleanWrapper, get_beers_cleaners

cleaners = get_beers_cleaners()
wrapper = UniCleanWrapper(cleaners=cleaners)
df, info = wrapper.clean('Data/beers/dirty_with_index.csv')
```

### 方式2: 直接运行官方脚本
```bash
python Methods/UniClean/main_beers.py \
    --file_load Data/beers/dirty_with_index.csv \
    --clean_path Data/beers/clean_with_index.csv \
    --save_path results/uniclean/
```

## 清洗器配置示例

```python
from SampleScrubber.cleaner.single import Number, Pattern, Outlier
from SampleScrubber.cleaner.multiple import AttrRelation

# beers数据集清洗器
cleaners = [
    Number("ounces", name="Number_ounces"),
    Number("abv", name="Number_abv"),
    AttrRelation(["brewery_id"], ["brewery_name"], '0'),
    AttrRelation(["brewery_id"], ["city"], '1'),
    AttrRelation(["brewery_id"], ["state"], '2')
]
```

## 与官方实现的差异

**无差异** - wrapper仅封装官方实现，不含任何简化版本。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| single_max | 10000 | 单次处理最大记录数 |
| batch_size | 500 | 批处理大小 |
| executor_memory | 8g | Spark executor内存 |
| driver_memory | 8g | Spark driver内存 |
