"""
UniClean Wrapper - 多清洗信号融合 + 工作流优化的数据清洗方法

UniClean是VLDB 2025提出的统一数据清洗框架，通过融合多种清洗信号并优化
清洗工作流来实现高效的数据清洗。

论文: UniClean: A Unified Framework for Data Cleaning with Multi-Signal Fusion (VLDB 2025)

真值使用情况: 全自动执行，无需人工参与 (Type 1)

依赖: pyspark>=3.1.1

注意: 本wrapper仅调用官方实现，不含任何简化版本
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional

# 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def _check_pyspark():
    """检查PySpark是否已安装"""
    try:
        from pyspark.sql import SparkSession
        return True
    except ImportError:
        raise ImportError(
            "UniClean需要PySpark。请安装: pip install pyspark==3.1.1\n"
            "详见 Methods/UniClean/requirements.txt"
        )


def _check_clean_module():
    """检查Clean模块是否存在"""
    try:
        from Clean import CleanonLocalWithnoSmple
        return True
    except ImportError:
        raise ImportError(
            "UniClean需要Clean.py模块。\n"
            "请确保 Methods/UniClean/Clean.py 文件存在并包含 CleanonLocalWithnoSmple 函数"
        )


class UniCleanWrapper:
    """
    UniClean清洗方法的封装类

    核心特点:
    1. 多信号融合: 整合多种清洗信号（约束、统计、模式等）
    2. PySpark支持: 使用Spark进行大规模数据处理
    3. 可配置清洗器: 支持自定义清洗规则

    官方实现要求:
    - PySpark环境
    - Clean.py模块中的CleanonLocalWithnoSmple函数
    - 数据需要有index列
    """

    def __init__(self,
                 cleaners: List = None,
                 single_max: int = 10000,
                 batch_size: int = 500,
                 executor_memory: str = "8g",
                 driver_memory: str = "8g",
                 verbose: bool = False):
        """
        初始化UniClean包装器

        Args:
            cleaners: 清洗器列表（如果为None，使用默认配置）
            single_max: 单次处理最大记录数 (官方默认30000)
            batch_size: 批处理大小 (官方默认500)
            executor_memory: Spark executor内存
            driver_memory: Spark driver内存
            verbose: 是否打印详细信息
        """
        self.cleaners = cleaners
        self.single_max = single_max
        self.batch_size = batch_size
        self.executor_memory = executor_memory
        self.driver_memory = driver_memory
        self.verbose = verbose
        self.ground_truth_used = 0
        self.spark = None

    def _init_spark(self, app_name: str = "UniClean"):
        """初始化Spark会话"""
        _check_pyspark()

        from pyspark.sql import SparkSession

        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.executor.memory", self.executor_memory) \
                .config("spark.driver.memory", self.driver_memory) \
                .config("spark.executor.memoryOverhead", "8g") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
        return self.spark

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              output_path: str = None,
              cleaners: List = None,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执行UniClean清洗流程

        Args:
            dirty_path: 脏数据路径 (CSV格式，需要有index列)
            clean_path: 干净数据路径（用于评估，可选）
            output_path: 输出路径
            cleaners: 自定义清洗器列表
            index_attribute: 索引列名 (默认'index')

        Returns:
            修复后的数据和清洗信息

        Raises:
            ImportError: 如果PySpark或Clean模块未安装
        """
        import time

        # 检查依赖
        _check_pyspark()
        _check_clean_module()

        from pyspark.sql.functions import monotonically_increasing_id
        from Clean import CleanonLocalWithnoSmple

        # 初始化Spark
        spark = self._init_spark()

        # 使用提供的清洗器或默认清洗器
        use_cleaners = cleaners or self.cleaners

        if not use_cleaners:
            raise ValueError(
                "UniClean需要配置清洗器(cleaners)。\n"
                "示例:\n"
                "  from SampleScrubber.cleaner.single import Number\n"
                "  from SampleScrubber.cleaner.multiple import AttrRelation\n"
                "  cleaners = [\n"
                "      Number('ounces', name='Number_ounces'),\n"
                "      AttrRelation(['brewery_id'], ['brewery_name'], '0')\n"
                "  ]"
            )

        try:
            # 读取数据
            data = spark.read.csv(dirty_path, header=True, inferSchema=True)

            if index_attribute not in data.columns:
                data = data.withColumn(index_attribute, monotonically_increasing_id())
            data.persist()

            # 执行清洗
            start_time = time.perf_counter()

            # 创建临时输出目录
            if output_path:
                table_path = os.path.dirname(output_path)
            else:
                table_path = os.path.join(_current_dir, 'temp_output')
            os.makedirs(table_path, exist_ok=True)

            if self.verbose:
                print(f"UniClean开始清洗，数据量: {data.count()}")
                print(f"清洗器数量: {len(use_cleaners)}")

            # 调用官方UniClean的清洗函数
            cleaned_data = CleanonLocalWithnoSmple(
                spark,
                use_cleaners,
                data,
                table_path,
                batch_size=self.batch_size,
                single_max=self.single_max
            )

            elapsed_time = time.perf_counter() - start_time

            # 转换为Pandas DataFrame
            result_df = cleaned_data.toPandas()

            # 后处理：将空值统一为 "empty"
            result_df = result_df.fillna('empty')
            result_df = result_df.replace('', 'empty')

            if output_path:
                result_df.to_csv(output_path, index=False)

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'UniClean',
                'type': 'data-oriented',
                'auto_level': 1,
                'cleaners_count': len(use_cleaners),
                'elapsed_time': elapsed_time
            }

            return result_df, info

        finally:
            if self.spark:
                self.spark.stop()
                self.spark = None

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本（UniClean全自动，始终为0）"""
        return self.ground_truth_used


def uniclean_clean(dirty_path: str,
                   cleaners: List,
                   output_path: str = None,
                   **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    UniClean清洗的便捷函数

    Args:
        dirty_path: 脏数据路径
        cleaners: 清洗器列表 (必需)
        output_path: 输出路径
        **kwargs: 传递给UniCleanWrapper的参数

    Returns:
        修复后的数据和清洗信息
    """
    wrapper = UniCleanWrapper(cleaners=cleaners, **kwargs)
    return wrapper.clean(dirty_path, output_path=output_path)


# 预定义的清洗器配置（示例）
def get_beers_cleaners():
    """
    获取beers数据集的清洗器配置

    使用示例:
        cleaners = get_beers_cleaners()
        df, info = uniclean_clean('Data/beers/dirty_with_index.csv', cleaners)
    """
    from SampleScrubber.cleaner.single import Number
    from SampleScrubber.cleaner.multiple import AttrRelation

    return [
        Number("ounces", name="Number_ounces"),
        Number("abv", name="Number_abv"),
        AttrRelation(["brewery_id"], ["brewery_name"], '0'),
        AttrRelation(["brewery_id"], ["city"], '1'),
        AttrRelation(["brewery_id"], ["state"], '2')
    ]


def get_hospitals_cleaners():
    """
    获取hospitals数据集的清洗器配置

    使用示例:
        cleaners = get_hospitals_cleaners()
        df, info = uniclean_clean('Data/hospital/dirty_with_index.csv', cleaners)
    """
    from SampleScrubber.cleaner.single import Number
    from SampleScrubber.cleaner.multiple import AttrRelation

    return [
        Number("Score", name="Number_Score"),
        AttrRelation(["ZIP Code"], ["City"], '0'),
        AttrRelation(["ZIP Code"], ["State"], '1'),
        AttrRelation(["ZIP Code"], ["County Name"], '2')
    ]


def load_cleaners_from_rules(rules_path: str):
    """
    从统一规则文件加载UniClean清洗器

    Args:
        rules_path: rules.txt文件路径 (如 Data/beers/rules.txt)

    Returns:
        清洗器列表

    示例:
        cleaners = load_cleaners_from_rules('Data/beers/rules.txt')
        wrapper = UniCleanWrapper(cleaners=cleaners)
        df, info = wrapper.clean(dirty_path)
    """
    from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
    from SampleScrubber.cleaner.multiple import AttrRelation

    cleaners = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"规则文件不存在: {rules_path}")

    # 解析规则文件
    in_uniclean_section = False
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 检测section
            if line.startswith('[') and line.endswith(']'):
                in_uniclean_section = (line == '[UNICLEAN]')
                continue

            # 解析UNICLEAN section中的cleaner定义
            if in_uniclean_section:
                local_vars = {
                    'Number': Number,
                    'Pattern': Pattern,
                    'Outlier': Outlier,
                    'Date': Date,
                    'AttrRelation': AttrRelation
                }
                try:
                    cleaner = eval(line, {"__builtins__": {}}, local_vars)
                    cleaners.append(cleaner)
                except Exception as e:
                    print(f"Warning: 解析cleaner失败 '{line}': {e}")

    return cleaners


def get_cleaners_for_dataset(dataset_name: str, data_dir: str = 'Data'):
    """
    根据数据集名称从rules.txt加载清洗器

    Args:
        dataset_name: 数据集名称 (如 'beers', 'adult')
        data_dir: 数据根目录

    Returns:
        清洗器列表
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return load_cleaners_from_rules(rules_path)
