"""
BoostClean Wrapper - 检测-修复器集成 + 自动迭代优化

BoostClean是一种面向模型的自动数据清洗方法，通过集成多种检测器和修复器，
并使用boosting策略自动优化清洗流程。

论文: BoostClean: Automatic Error Detection and Repair for Machine Learning (SIGMOD 2017)
GitHub: https://github.com/HoloClean/boostclean

真值使用情况: 需少量验证集真值评估清洗效果 (Type 2)

本模块提供调用官方activedetect包的接口
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# 添加当前目录到路径，以便导入官方实现
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入官方实现
try:
    from activedetect.loaders.csv_loader import CSVLoader
    from activedetect.error_detectors.ErrorDetector import ErrorDetector
    from activedetect.learning.BoostClean import BoostClean
    from activedetect.learning.EvaluateCleaning import EvaluateCleaning
    from activedetect.reporting.CSVLogging import CSVLogging
    from activedetect.loaders.type_inference import LoLTypeInference
    from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
    from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
    HAS_ACTIVEDETECT = True
except ImportError as e:
    HAS_ACTIVEDETECT = False
    IMPORT_ERROR = str(e)


class BoostCleanWrapper:
    """
    BoostClean清洗方法的封装类

    封装官方activedetect包的BoostClean类，提供统一接口。

    核心特点:
    1. 检测器集成: 多种错误检测器（统计、规则、ML）
    2. 修复器集成: 多种修复策略（删除、填充、替换）
    3. Boosting优化: 自动选择最优的检测-修复组合
    4. 验证集驱动: 使用小验证集评估清洗效果
    """

    def __init__(self,
                 boosting_rounds: int = 5,
                 quantitative_thresh: int = 10,
                 verbose: bool = False):
        """
        初始化BoostClean包装器

        Args:
            boosting_rounds: Boosting轮数
            quantitative_thresh: 数值异常检测阈值
            verbose: 是否打印详细信息
        """
        self.boosting_rounds = boosting_rounds
        self.quantitative_thresh = quantitative_thresh
        self.verbose = verbose
        self.ground_truth_used = 0
        self.selected_operations = []
        self.ensemble = None

    def _check_dependencies(self):
        """检查依赖是否满足"""
        if not HAS_ACTIVEDETECT:
            raise ImportError(
                f"activedetect包导入失败: {IMPORT_ERROR}\n"
                "请确保BoostClean的activedetect包已正确安装。"
            )

    def _load_data_as_lol(self, filepath: str, convert_empty: bool = True) -> List[List]:
        """
        加载数据为list of lists格式（官方格式）

        Args:
            filepath: CSV文件路径
            convert_empty: 是否将 "empty" 转换为空字符串

        Returns:
            List of lists格式的数据
        """
        loader = CSVLoader()
        data = loader.loadFile(filepath)

        # 将 "empty" 空值标记转换为空字符串（BoostClean 内部使用空字符串表示空值）
        if convert_empty and data:
            data = [['' if cell == 'empty' else cell for cell in row] for row in data]

        return data

    def _get_default_modules(self):
        """获取默认的检测模块配置"""
        modules = [QuantitativeErrorModule, PuncErrorModule]
        config = [{'thresh': self.quantitative_thresh}, {}]
        return modules, config

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              label_column: str = None,
              output_path: str = None,
              base_model=None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行BoostClean流程

        Args:
            dirty_path: 脏数据路径
            clean_path: 干净数据路径（用于获取真值标签）
            label_column: 标签列名（如果为None，使用最后一列）
            output_path: 输出路径
            base_model: sklearn基模型（默认RandomForest）

        Returns:
            清洗后的数据和清洗信息
        """
        self._check_dependencies()

        # 设置基模型
        if base_model is None:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # 加载数据
        loaded_data = self._load_data_as_lol(dirty_path)

        # 移除表头
        header = loaded_data[0] if loaded_data else []
        data = loaded_data[1:] if len(loaded_data) > 1 else loaded_data

        # 分离特征和标签
        features = [row[:-1] for row in data]
        labels = [row[-1] for row in data]

        # 转换标签为数值
        try:
            labels = [float(l) for l in labels]
        except:
            # 如果标签不是数值，进行编码
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(labels).tolist()

        # 设置日志
        log_path = output_path.replace('.csv', '.log') if output_path else 'boostclean.log'
        logging = CSVLogging(log_path)

        # 获取检测模块
        modules, config = self._get_default_modules()

        # 加载干净数据用于评估
        wrong_cells = []
        if clean_path:
            clean_data = self._load_data_as_lol(clean_path)
            clean_data = clean_data[1:] if len(clean_data) > 1 else clean_data
            # 识别错误单元格
            for i, (dirty_row, clean_row) in enumerate(zip(data, clean_data)):
                for j, (d, c) in enumerate(zip(dirty_row, clean_row)):
                    if str(d) != str(c):
                        wrong_cells.append((i, j))
            self.ground_truth_used = len(clean_data)  # 验证集使用

        try:
            # 创建BoostClean实例
            bc = BoostClean(
                modules=modules,
                config=config,
                base_model=base_model,
                features=features,
                labels=labels,
                logging=logging,
                wrong_cells=wrong_cells
            )

            # 运行BoostClean
            self.ensemble, rep_tuples, sel_clf = bc.run(j=self.boosting_rounds)

            # 转换清洗结果为DataFrame
            if rep_tuples:
                cleaned_df = pd.DataFrame(rep_tuples, columns=header if header else None)
            else:
                # 如果没有修复，返回原数据
                cleaned_df = pd.read_csv(dirty_path)

            # 后处理：将空字符串转换回标准的 "empty" 空值表示
            cleaned_df = cleaned_df.replace('', 'empty')

            if self.verbose:
                print(f"BoostClean完成，集成大小: {len(self.ensemble)}")

        except Exception as e:
            if self.verbose:
                print(f"BoostClean执行出错: {e}")
            # 返回原数据
            cleaned_df = pd.read_csv(dirty_path)
            self.ensemble = []

        # 保存结果
        if output_path:
            cleaned_df.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': 'BoostClean',
            'type': 'model-oriented',
            'auto_level': 2,  # 需要验证集
            'ensemble_size': len(self.ensemble) if self.ensemble else 0,
            'boosting_rounds': self.boosting_rounds
        }

        return cleaned_df, info

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本"""
        return self.ground_truth_used


def boostclean_clean(dirty_path: str,
                     clean_path: str = None,
                     label_column: str = None,
                     output_path: str = None,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """BoostClean清洗的便捷函数"""
    wrapper = BoostCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, label_column, output_path)
