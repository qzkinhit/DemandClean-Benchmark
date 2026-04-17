"""
Baran/Raha Wrapper - 基于迭代标注error detectionandrepair系统

Raha (SIGMOD 2019): error detection系统，融合多种detection策略
Baran (SIGMOD 2020): error repair系统，基于主动学习迭代repair

论文:
- Raha: A Configuration-Free Error Detection System (SIGMOD 2019)
- Baran: Effective Error Correction via a Unified Context Representation (SIGMOD 2020)

ground truthusestats: 迭代式标注，需need人工参and (Type 3)
- ground truth成本 = LABELING_BUDGET (default20条)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# 添加currentdirectorytopath
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入official implementation
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
    Baran/Rahacleaningmethod封装class

    Rahaused forerror detection，Baranused forerror repair。
    这is一个迭代式系统，需need人工标注。

    ground truthuse: Type 3 (iterative interactive)
    - defaultLABELING_BUDGET=20，i.e.需need人工标注20条记录
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
        initializeBaran/Rahapackage装器

        Args:
            labeling_budget: 标注预算（需need人工标注记录数）
            classification_model: classificationmodel ["ABC", "DTC", "GBC", "GNB", "KNC", "SGDC", "SVC"]
            min_correction_candidate_probability: 最小修正候选概率
            min_correction_occurrence: 最小修正出现次数
            max_value_length: 最大值长度
            verbose: whether打印详细信息
            save_results: whethersave间result
        """
        self.labeling_budget = labeling_budget
        self.classification_model = classification_model
        self.min_correction_candidate_probability = min_correction_candidate_probability
        self.min_correction_occurrence = min_correction_occurrence
        self.max_value_length = max_value_length
        self.verbose = True
        self.save_results = save_results

        # ground truthuse成本 = 标注预算
        self.ground_truth_used = labeling_budget

    def _check_dependencies(self):
        """check依赖whether满足"""
        if not HAS_BARAN_RAHA:
            raise ImportError(
                f"Baran/Rahamodule import failed: {IMPORT_ERROR}\n"
                "Ensure that underMethods/Baran_Rahathere is a complete copy of the official code，"
                "and install the required dependencies（raha, mwparserfromhell, py7zr等）"
            )

    def clean(self,
              dirty_path: str,
              clean_path: str = None,
              output_path: str = None,
              task_name: str = "baran_task",
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执rowBaran/Rahacleaning流程

        Args:
            dirty_path: 脏datapath
            clean_path: cleandatapath（used forauto-labeling，optional）
            output_path: outputpath
            task_name: task名称
            index_attribute: 索引column name

        Returns:
            repairafterdataandcleaning信息
        """
        self._check_dependencies()

        import time
        import tempfile
        start_time = time.perf_counter()

        # 预处理：将 "empty" replacetoempty value，让 Baran  fillna("nan") 统一处理
        temp_dirty_path = None
        temp_clean_path = None
        try:
            # Handle dirty data
            dirty_df_raw = pd.read_csv(dirty_path)
            dirty_df_raw = dirty_df_raw.replace('empty', np.nan)
            temp_dirty_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
            dirty_df_raw.to_csv(temp_dirty_path, index=False)

            # Handle clean data（ifprovide）
            if clean_path:
                clean_df_raw = pd.read_csv(clean_path)
                clean_df_raw = clean_df_raw.replace('empty', np.nan)
                temp_clean_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
                clean_df_raw.to_csv(temp_clean_path, index=False)

            # Build the dataset dict（using a temp file）
            dataset_dictionary = {
                "name": task_name,
                "path": temp_dirty_path,
            }
            if temp_clean_path:
                dataset_dictionary["clean_path"] = temp_clean_path
            # 1. error detection (Raha)
            if self.verbose:
                print("=" * 60)
                print("Stage1: Rahaerror detection")
                print("=" * 60)

            detector = Detection()
            detected_cells = detector.run(dataset_dictionary)
            detection_p, detection_r, detection_f = detector.d.get_data_cleaning_evaluation(detected_cells)[:3]

            if self.verbose:
                print(f"detected error cells: {len(detected_cells)}")
                print(f"detection Precision: {detection_p:.4f}")
                print(f"detection Recall: {detection_r:.4f}")
                print(f"detection F1: {detection_f:.4f}")

            # 2. error repair (Baran)
            if self.verbose:
                print("=" * 60)
                print("Stage2: Baranerror repair")
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
                print(f"repaired cell count: {len(correction_dictionary)}")

            # 3. Apply repairs（use the raw dirty data to keep the original format）
            repaired_df = pd.read_csv(dirty_path)
            for cell, value in correction_dictionary.items():
                repaired_df.iloc[cell[0], cell[1]] = value

            # Post-process: map Baran's "nan" empty marker to the standard "empty"
            repaired_df = repaired_df.replace('nan', 'empty')

            elapsed_time = time.perf_counter() - start_time

            # Save results
            if output_path:
                # Ensure the output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                repaired_df.to_csv(output_path, index=False)
                if self.verbose:
                    print(f"Repaired data saved: {output_path}")

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'Baran_Raha',
                'type': 'data-oriented',
                'auto_level': 3,  # iterative interactive
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
            raise RuntimeError(f"Baran/Rahacleaningfailure: {e}")

        finally:
            # Clean up temp files
            if temp_dirty_path and os.path.exists(temp_dirty_path):
                os.remove(temp_dirty_path)
            if temp_clean_path and os.path.exists(temp_clean_path):
                os.remove(temp_clean_path)

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本（= LABELING_BUDGET）"""
        return self.ground_truth_used


def baran_raha_clean(dirty_path: str,
                     clean_path: str = None,
                     output_path: str = None,
                     labeling_budget: int = 20,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Baran/Rahacleaning便捷function"""
    wrapper = BaranRahaWrapper(labeling_budget=labeling_budget, **kwargs)
    return wrapper.clean(dirty_path, clean_path, output_path)
