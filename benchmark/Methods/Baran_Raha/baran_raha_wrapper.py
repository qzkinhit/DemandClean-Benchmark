"""
Baran/Raha Wrapper - 基于迭代标注的错误检测与修复系统

Raha (SIGMOD 2019): 错误检测系统，融合多种检测策略
Baran (SIGMOD 2020): 错误修复系统，基于主动学习的迭代修复

论文:
- Raha: A Configuration-Free Error Detection System (SIGMOD 2019)
- Baran: Effective Error Correction via a Unified Context Representation (SIGMOD 2020)

真值使用情况: 迭代式标注，需要人工参与 (Type 3)
- 真值成本 = LABELING_BUDGET (默认20条)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入官方实现
try:
    import raha
    from raha.dataset import Dataset
    from detection import Detection
    from correction import Correction
    HAS_BARAN_RAHA = True
except ImportError as e:
    HAS_BARAN_RAHA = False
    IMPORT_ERROR = str(e)


class BaranRahaWrapper:
    """
    Baran/Raha清洗方法的封装类

    Raha用于错误检测，Baran用于错误修复。
    这是一个迭代式的系统，需要人工标注。

    真值使用: Type 3 (迭代交互)
    - 默认LABELING_BUDGET=20，即需要人工标注20条记录
    """

    def __init__(self,
                 labeling_budget: int = 20,
                 classification_model: str = "ABC",
                 min_correction_candidate_probability: float = 0.0,
                 min_correction_occurrence: int = 2,
                 max_value_length: int = 50,
                 verbose: bool = True,
                 save_results: bool = False):
        """
        初始化Baran/Raha包装器

        Args:
            labeling_budget: 标注预算（需要人工标注的记录数）
            classification_model: 分类模型 ["ABC", "DTC", "GBC", "GNB", "KNC", "SGDC", "SVC"]
            min_correction_candidate_probability: 最小修正候选概率
            min_correction_occurrence: 最小修正出现次数
            max_value_length: 最大值长度
            verbose: 是否打印详细信息
            save_results: 是否保存中间结果
        """
        self.labeling_budget = labeling_budget
        self.classification_model = classification_model
        self.min_correction_candidate_probability = min_correction_candidate_probability
        self.min_correction_occurrence = min_correction_occurrence
        self.max_value_length = max_value_length
        self.verbose = True
        self.save_results = save_results

        # 真值使用成本 = 标注预算
        self.ground_truth_used = labeling_budget

    def _check_dependencies(self):
        """检查依赖是否满足"""
        if not HAS_BARAN_RAHA:
            raise ImportError(
                f"Baran/Raha模块导入失败: {IMPORT_ERROR}\n"
                "请确保在Methods/Baran_Raha目录下有完整的官方代码，"
                "并安装必要的依赖（raha, mwparserfromhell, py7zr等）"
            )

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              output_path: str = None,
              task_name: str = "baran_task",
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执行Baran/Raha清洗流程

        Args:
            dirty_path: 脏数据路径
            clean_path: 干净数据路径（用于自动标注，可选）
            output_path: 输出路径
            task_name: 任务名称
            index_attribute: 索引列名

        Returns:
            修复后的数据和清洗信息
        """
        self._check_dependencies()

        import time
        import tempfile
        start_time = time.perf_counter()

        # 预处理：将 "empty" 替换为空值，让 Baran 的 fillna("nan") 统一处理
        temp_dirty_path = None
        temp_clean_path = None
        try:
            # 处理脏数据
            dirty_df_raw = pd.read_csv(dirty_path)
            dirty_df_raw = dirty_df_raw.replace('empty', np.nan)
            temp_dirty_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
            dirty_df_raw.to_csv(temp_dirty_path, index=False)

            # 处理干净数据（如果提供）
            if clean_path:
                clean_df_raw = pd.read_csv(clean_path)
                clean_df_raw = clean_df_raw.replace('empty', np.nan)
                temp_clean_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
                clean_df_raw.to_csv(temp_clean_path, index=False)

            # 构建数据集字典（使用临时文件）
            dataset_dictionary = {
                "name": task_name,
                "path": temp_dirty_path,
            }
            if temp_clean_path:
                dataset_dictionary["clean_path"] = temp_clean_path
            # 1. 错误检测 (Raha)
            if self.verbose:
                print("=" * 60)
                print("阶段1: Raha错误检测")
                print("=" * 60)

            detector = Detection()
            detected_cells = detector.run(dataset_dictionary)
            detection_p, detection_r, detection_f = detector.d.get_data_cleaning_evaluation(detected_cells)[:3]

            if self.verbose:
                print(f"检测到错误单元格: {len(detected_cells)}")
                print(f"检测 Precision: {detection_p:.4f}")
                print(f"检测 Recall: {detection_r:.4f}")
                print(f"检测 F1: {detection_f:.4f}")

            # 2. 错误修复 (Baran)
            if self.verbose:
                print("=" * 60)
                print("阶段2: Baran错误修复")
                print("=" * 60)

            corrector = Correction()
            corrector.LABELING_BUDGET = self.labeling_budget
            corrector.CLASSIFICATION_MODEL = self.classification_model
            corrector.MIN_CORRECTION_CANDIDATE_PROBABILITY = self.min_correction_candidate_probability
            corrector.MIN_CORRECTION_OCCURRENCE = self.min_correction_occurrence
            corrector.MAX_VALUE_LENGTH = self.max_value_length
            corrector.VERBOSE = self.verbose
            corrector.SAVE_RESULTS = self.save_results

            correction_dictionary = corrector.run(detector.d)

            if self.verbose:
                print(f"修复单元格数: {len(correction_dictionary)}")

            # 3. 应用修复（使用原始脏数据，保留原始格式）
            repaired_df = pd.read_csv(dirty_path)
            for cell, value in correction_dictionary.items():
                repaired_df.iloc[cell[0], cell[1]] = value

            # 后处理：将 Baran 的 "nan" 空值表示替换为标准的 "empty"
            repaired_df = repaired_df.replace('nan', 'empty')

            elapsed_time = time.perf_counter() - start_time

            # 保存结果
            if output_path:
                # 确保输出目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                repaired_df.to_csv(output_path, index=False)
                if self.verbose:
                    print(f"修复后数据已保存: {output_path}")

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'Baran_Raha',
                'type': 'data-oriented',
                'auto_level': 3,  # 迭代交互
                'labeling_budget': self.labeling_budget,
                'detected_cells': len(detected_cells),
                'corrected_cells': len(correction_dictionary),
                'detection_precision': detection_p,
                'detection_recall': detection_r,
                'detection_f1': detection_f,
                'elapsed_time': elapsed_time
            }

            return repaired_df, info

        except Exception as e:
            raise RuntimeError(f"Baran/Raha清洗失败: {e}")

        finally:
            # 清理临时文件
            if temp_dirty_path and os.path.exists(temp_dirty_path):
                os.remove(temp_dirty_path)
            if temp_clean_path and os.path.exists(temp_clean_path):
                os.remove(temp_clean_path)

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本（= LABELING_BUDGET）"""
        return self.ground_truth_used


def baran_raha_clean(dirty_path: str,
                     clean_path: str = None,
                     output_path: str = None,
                     labeling_budget: int = 20,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Baran/Raha清洗的便捷函数"""
    wrapper = BaranRahaWrapper(labeling_budget=labeling_budget, **kwargs)
    return wrapper.clean(dirty_path, clean_path, output_path)
