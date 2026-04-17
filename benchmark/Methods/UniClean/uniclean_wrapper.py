"""
UniClean Wrapper - 多cleaning信号融合 + 工作流优化datacleaningmethod

UniCleanisVLDB 2025提出统一datacleaning框架，通过融合多种cleaning信号and优化
cleaning工作流来实现高效datacleaning。

论文: UniClean: A Unified Framework for Data Cleaning with Multi-Signal Fusion (VLDB 2025)

ground truthusestats: fully automatic执row，none需人工参and (Type 1)

依赖: pyspark>=3.1.1

Note: 本wrapper仅Invokeofficial implementation，not contain任何简化version
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional

# 添加currentdirectorytopath
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def _check_pyspark():
    """checkPySparkwhether安装"""
    try:
        from pyspark.sql import SparkSession
        return True
    except ImportError:
        raise ImportError(
            "UniCleanrequires PySpark。Please install: pip install pyspark==3.1.1\n"
            "See Methods/UniClean/requirements.txt"
        )


def _check_clean_module():
    """checkCleanmodulewhether存at"""
    try:
        from Clean import CleanonLocalWithnoSmple
        return True
    except ImportError:
        raise ImportError(
            "UniCleanrequires the Clean.py module。\n"
            "Ensure Methods/UniClean/Clean.py exists and contains CleanonLocalWithnoSmple function"
        )


class UniCleanWrapper:
    """
    UniCleancleaningmethod封装class

    核心特点:
    1. 多信号融合: 整合多种cleaning信号（约束、Count、mode等）
    2. PySpark支持: useSpark进row大规模data处理
    3. 可Configurationcleaning器: 支持customcleaning规则

    official implementationneed求:
    - PySparkenvironment
    - Clean.pymoduleCleanonLocalWithnoSmplefunction
    - data需needhasindexcolumn
    """

    def __init__(self,
                 cleaners: List = None,
                 single_max: int = 10000,
                 batch_size: int = 500,
                 executor_memory: str = "8g",
                 driver_memory: str = "8g",
                 verbose: bool = False):
        """
        initializeUniCleanpackage装器

        Args:
            cleaners: cleaning器column list（iftoNone，usedefaultConfiguration）
            single_max: 单次处理最大记录数 (officialdefault30000)
            batch_size: batch size (officialdefault500)
            executor_memory: Spark executor内存
            driver_memory: Spark driver内存
            verbose: whether打印详细信息
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
        """initializeSparkwill话"""
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
        执rowUniCleancleaning流程

        Args:
            dirty_path: 脏datapath (CSVformat，需needhasindexcolumn)
            clean_path: cleandatapath（used forevaluation，optional）
            output_path: outputpath
            cleaners: customcleaning器column list
            index_attribute: 索引column name (default'index')

        Returns:
            repairafterdataandcleaning信息

        Raises:
            ImportError: ifPySparkorCleanmodulenot安装
        """
        import time

        # check依赖
        _check_pyspark()
        _check_clean_module()

        from pyspark.sql.functions import monotonically_increasing_id
        from Clean import CleanonLocalWithnoSmple

        # initializeSpark
        spark = self._init_spark()

        # useprovidecleaning器ordefaultcleaning器
        use_cleaners = cleaners or self.cleaners

        if not use_cleaners:
            raise ValueError(
                "UniCleanrequires configured cleaners(cleaners)。\n"
                "Example:\n"
                "  from SampleScrubber.cleaner.single import Number\n"
                "  from SampleScrubber.cleaner.multiple import AttrRelation\n"
                "  cleaners = [\n"
                "      Number('ounces', name='Number_ounces'),\n"
                "      AttrRelation(['brewery_id'], ['brewery_name'], '0')\n"
                "  ]"
            )

        try:
            # Read data
            data = spark.read.csv(dirty_path, header=True, inferSchema=True)

            if index_attribute not in data.columns:
                data = data.withColumn(index_attribute, monotonically_increasing_id())
            data.persist()

            # Run cleaning
            start_time = time.perf_counter()

            # Create a temporary output directory
            if output_path:
                table_path = os.path.dirname(output_path)
            else:
                table_path = os.path.join(_current_dir, 'temp_output')
            os.makedirs(table_path, exist_ok=True)

            if self.verbose:
                print(f"UniCleanstarts cleaning，row count: {data.count()}")
                print(f"cleaner count: {len(use_cleaners)}")

            # InvokeofficialUniCleancleaningfunction
            cleaned_data = CleanonLocalWithnoSmple(
                spark,
                use_cleaners,
                data,
                table_path,
                batch_size=self.batch_size,
                single_max=self.single_max
            )

            elapsed_time = time.perf_counter() - start_time

            # ConverttoPandas DataFrame
            result_df = cleaned_data.toPandas()

            # after处理：将empty valuenormalize to "empty"
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
        """getground truthuse成本（UniCleanfully automatic，始终to0）"""
        return self.ground_truth_used


def uniclean_clean(dirty_path: str,
                   cleaners: List,
                   output_path: str = None,
                   **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    UniCleancleaning便捷function

    Args:
        dirty_path: 脏datapath
        cleaners: cleaning器column list (必需)
        output_path: outputpath
        **kwargs: 传递toUniCleanWrapperParameter

    Returns:
        repairafterdataandcleaning信息
    """
    wrapper = UniCleanWrapper(cleaners=cleaners, **kwargs)
    return wrapper.clean(dirty_path, output_path=output_path)


# 预definitioncleaning器Configuration（Example）
def get_beers_cleaners():
    """
    getbeersDatasetcleaning器Configuration

    useExample:
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
    gethospitalsDatasetcleaning器Configuration

    useExample:
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
    from统一规则fileloadUniCleancleaning器

    Args:
        rules_path: rules.txtfilepath (如 Data/beers/rules.txt)

    Returns:
        cleaning器column list

    Example:
        cleaners = load_cleaners_from_rules('Data/beers/rules.txt')
        wrapper = UniCleanWrapper(cleaners=cleaners)
        df, info = wrapper.clean(dirty_path)
    """
    from SampleScrubber.cleaner.single import Number, Pattern, Outlier, Date
    from SampleScrubber.cleaner.multiple import AttrRelation

    cleaners = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"规则filedoes not exist: {rules_path}")

    # 解析规则file
    in_uniclean_section = False
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith('#'):
                continue

            # detectionsection
            if line.startswith('[') and line.endswith(']'):
                in_uniclean_section = (line == '[UNICLEAN]')
                continue

            # 解析UNICLEAN sectioncleanerdefinition
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
                    print(f"Warning: failed to parse cleaner '{line}': {e}")

    return cleaners


def get_cleaners_for_dataset(dataset_name: str, data_dir: str = 'Data'):
    """
    根据Dataset名称fromrules.txtloadcleaning器

    Args:
        dataset_name: Dataset名称 (如 'beers', 'adult')
        data_dir: data根directory

    Returns:
        cleaning器column list
    """
    rules_path = os.path.join(data_dir, dataset_name, 'rules.txt')
    return load_cleaners_from_rules(rules_path)
