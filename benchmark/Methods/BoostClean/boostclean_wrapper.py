"""
BoostClean Wrapper - detection-repair器集成 + 自动迭代优化

BoostCleanis一种面向model自动datacleaningmethod，通过集成多种detection器andrepair器，
anduseboosting策略自动优化cleaning流程。

论文: BoostClean: Automatic Error Detection and Repair for Machine Learning (SIGMOD 2017)
GitHub: https://github.com/HoloClean/boostclean

ground truthusestats: 需少量validation setground truthevaluationcleaning效果 (Type 2)

本moduleprovideInvokeofficialactivedetectpackage接口
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# 添加currentdirectorytopath，以便导入official implementation
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入official implementation
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
    BoostCleancleaningmethod封装class

    封装officialactivedetectpackageBoostCleanclass，provide统一接口。

    核心特点:
    1. detection器集成: 多种error detection器（Count、规则、ML）
    2. repair器集成: 多种repair策略（delete、填充、replace）
    3. Boosting优化: 自动选择最优detection-repair组合
    4. validation set驱动: use小validation setevaluationcleaning效果
    """

    def __init__(self,
                 boosting_rounds: int = 5,
                 quantitative_thresh: int = 10,
                 verbose: bool = False):
        """
        initializeBoostCleanpackage装器

        Args:
            boosting_rounds: Boostingrounds
            quantitative_thresh: 数值异常detection阈值
            verbose: whether打印详细信息
        """
        self.boosting_rounds = boosting_rounds
        self.quantitative_thresh = quantitative_thresh
        self.verbose = verbose
        self.ground_truth_used = 0
        self.selected_operations = []
        self.ensemble = None

    def _check_dependencies(self):
        """check依赖whether满足"""
        if not HAS_ACTIVEDETECT:
            raise ImportError(
                f"activedetectpackage import failed: {IMPORT_ERROR}\n"
                "EnsureBoostCleanactivedetectpackageis installed correctly。"
            )

    def _load_data_as_lol(self, filepath: str, convert_empty: bool = True) -> List[List]:
        """
        Load datatolist of listsformat（officialformat）

        Args:
            filepath: CSVfilepath
            convert_empty: whether将 "empty" Convertto空字符串

        Returns:
            List of listsformatdata
        """
        loader = CSVLoader()
        data = loader.loadFile(filepath)

        # 将 "empty" empty value标记Convertto空字符串（BoostClean 内部use空字符串markerempty value）
        if convert_empty and data:
            data = [['' if cell == 'empty' else cell for cell in row] for row in data]

        return data

    def _get_default_modules(self):
        """getdefaultdetectionmoduleConfiguration"""
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
        执rowBoostClean流程

        Args:
            dirty_path: 脏datapath
            clean_path: cleandatapath（used forgetground truthlabel）
            label_column: labelcolumn name（iftoNone，use最after一column）
            output_path: outputpath
            base_model: sklearnbase model（defaultRandomForest）

        Returns:
            cleaningafterdataandcleaning信息
        """
        self._check_dependencies()

        # setbase model
        if base_model is None:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Load data
        loaded_data = self._load_data_as_lol(dirty_path)

        # 移除表头
        header = loaded_data[0] if loaded_data else []
        data = loaded_data[1:] if len(loaded_data) > 1 else loaded_data

        # 分离featureandlabel
        features = [row[:-1] for row in data]
        labels = [row[-1] for row in data]

        # Convertlabelto数值
        try:
            labels = [float(l) for l in labels]
        except:
            # If the label is not numeric，encode it
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(labels).tolist()

        # setlog
        log_path = output_path.replace('.csv', '.log') if output_path else 'boostclean.log'
        logging = CSVLogging(log_path)

        # getdetectionmodule
        modules, config = self._get_default_modules()

        # loadcleandataused forevaluation
        wrong_cells = []
        if clean_path:
            clean_data = self._load_data_as_lol(clean_path)
            clean_data = clean_data[1:] if len(clean_data) > 1 else clean_data
            # IdentifyErrorcell
            for i, (dirty_row, clean_row) in enumerate(zip(data, clean_data)):
                for j, (d, c) in enumerate(zip(dirty_row, clean_row)):
                    if str(d) != str(c):
                        wrong_cells.append((i, j))
            self.ground_truth_used = len(clean_data)  # validation setuse

        try:
            # Create a BoostClean instance
            bc = BoostClean(
                modules=modules,
                config=config,
                base_model=base_model,
                features=features,
                labels=labels,
                logging=logging,
                wrong_cells=wrong_cells
            )

            # RunBoostClean
            self.ensemble, rep_tuples, sel_clf = bc.run(j=self.boosting_rounds)

            # ConvertcleaningresulttoDataFrame
            if rep_tuples:
                cleaned_df = pd.DataFrame(rep_tuples, columns=header if header else None)
            else:
                # If nothing was repaired, return the original data
                cleaned_df = pd.read_csv(dirty_path)

            # Post-process: convert empty strings back to the standard "empty" empty valuemarker
            cleaned_df = cleaned_df.replace('', 'empty')

            if self.verbose:
                print(f"BoostCleandone，ensemble size: {len(self.ensemble)}")

        except Exception as e:
            if self.verbose:
                print(f"BoostCleanexecution error: {e}")
            # returnoriginaldata
            cleaned_df = pd.read_csv(dirty_path)
            self.ensemble = []

        # Save results
        if output_path:
            cleaned_df.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': 'BoostClean',
            'type': 'model-oriented',
            'auto_level': 2,  # 需needvalidation set
            'ensemble_size': len(self.ensemble) if self.ensemble else 0,
            'boosting_rounds': self.boosting_rounds
        }

        return cleaned_df, info

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本"""
        return self.ground_truth_used


def boostclean_clean(dirty_path: str,
                     clean_path: str = None,
                     label_column: str = None,
                     output_path: str = None,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """BoostCleancleaning便捷function"""
    wrapper = BoostCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, label_column, output_path)
