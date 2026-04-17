"""
HoloClean Wrapper - 约束 + Count + 知识库融合datacleaningmethod

HoloCleanis一个基于概率图model全面datacleaning系统，通过融合多种cleaning信号
（完整约束、Count信息、外部知识库）进rowdatarepair。

论文: HoloClean: Holistic Data Repairs with Probabilistic Inference (VLDB 2017)
GitHub: https://github.com/HoloClean/holoclean

ground truthusestats: fully automatic执row，none需人工参and (Type 1)

Note: HoloClean需needPostgreSQLdata库支持
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# 添加currentdirectorytopath
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入official implementation
try:
    from holoclean import HoloClean, Session
    from detect import NullDetector, ViolationDetector
    from repair.featurize import (
        InitAttrFeaturizer, OccurAttrFeaturizer,
        FreqFeaturizer, ConstraintFeaturizer
    )
    HAS_HOLOCLEAN = True
except ImportError as e:
    HAS_HOLOCLEAN = False
    IMPORT_ERROR = str(e)


class HoloCleanWrapper:
    """
    HoloCleancleaningmethod封装class

    该class封装HoloClean核心功can，provide统一接口used fordatacleaningtask。

    Note: HoloClean需needPostgreSQLdata库支持
    - 安装PostgreSQLandcreate用户anddata库
    - setenvironment变量oruseParameter指定连接信息
    """

    def __init__(self,
                 db_user: str = "holocleanuser",
                 db_pwd: str = "abcd1234",
                 db_host: str = "localhost",
                 db_name: str = "holo",
                 domain_thresh_1: float = 0.1,
                 domain_thresh_2: float = 0.0,
                 weak_label_thresh: float = 0.90,
                 max_domain: int = 1000000,
                 cor_strength: float = 0.05,
                 nb_cor_strength: float = 0.3,
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.001,
                 threads: int = 20,
                 batch_size: int = 1,
                 epochs: int = 20,
                 verbose: bool = False,
                 timeout: int = 60000):
        """
        initializeHoloCleanpackage装器

        Args:
            db_user: data库用户名
            db_pwd: data库密码
            db_host: data库主机
            db_name: data库名称
            domain_thresh_1: firstStage域值阈值
            domain_thresh_2: secondStage域值阈值
            weak_label_thresh: 弱label阈值
            max_domain: 最大域size
            cor_strength: 相关强度
            nb_cor_strength: 近邻相关强度
            weight_decay: 权重衰减
            learning_rate: 学习率
            threads: 线程数
            batch_size: batch size
            epochs: trainingrounds
            verbose: whether打印详细信息
            timeout: 超时time（毫秒）
        """
        self.db_user = db_user
        self.db_pwd = db_pwd
        self.db_host = db_host
        self.db_name = db_name
        self.domain_thresh_1 = domain_thresh_1
        self.domain_thresh_2 = domain_thresh_2
        self.weak_label_thresh = weak_label_thresh
        self.max_domain = max_domain
        self.cor_strength = cor_strength
        self.nb_cor_strength = nb_cor_strength
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.threads = threads
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.timeout = timeout

        self.hc = None
        self.session = None
        self.ground_truth_used = 0

    def _check_dependencies(self):
        """check依赖whether满足"""
        if not HAS_HOLOCLEAN:
            raise ImportError(
                f"HoloCleanmodule import failed: {IMPORT_ERROR}\n"
                "Ensure that underMethods/HoloCleanthere is a complete copy of the official code，"
                "and install the required dependencies（torch, psycopg2等）"
            )

    def setup(self) -> bool:
        """
        setHoloCleanenvironment

        Returns:
            whethersuccessinitialize
        """
        self._check_dependencies()

        try:
            self.hc = HoloClean(
                db_user=self.db_user,
                db_pwd=self.db_pwd,
                db_host=self.db_host,
                db_name=self.db_name,
                domain_thresh_1=self.domain_thresh_1,
                domain_thresh_2=self.domain_thresh_2,
                weak_label_thresh=self.weak_label_thresh,
                max_domain=self.max_domain,
                cor_strength=self.cor_strength,
                nb_cor_strength=self.nb_cor_strength,
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                threads=self.threads,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                timeout=self.timeout
            )
            self.session = self.hc.session
            return True
        except Exception as e:
            if self.verbose:
                print(f"HoloCleaninitialization failed: {e}")
            return False

    def clean(self,
              dirty_path: str,
              dc_path: str = None,
              output_path: str = None,
              dataset_name: str = "data") -> Tuple[pd.DataFrame, Dict]:
        """
        执row完整cleaning流程

        Args:
            dirty_path: 脏datapath
            dc_path: 约束filepath（optional）
            output_path: outputpath（optional）
            dataset_name: Dataset名称

        Returns:
            repairafterdataandcleaning信息
        """
        self._check_dependencies()

        # initialize
        if not self.setup():
            raise RuntimeError("HoloCleaninitialization failed，Please checkdata库Configuration")

        try:
            # Load data
            self.session.load_data(dataset_name, dirty_path)

            # Load constraints（ifprovide）
            if dc_path and os.path.exists(dc_path):
                self.session.load_dcs(dc_path)

            # detectionError
            detectors = [NullDetector()]
            if dc_path:
                detectors.append(ViolationDetector())
            self.session.detect_errors(detectors)

            # Set domain
            self.session.setup_domain()

            # feature extractor
            featurizers = [
                InitAttrFeaturizer(),
                OccurAttrFeaturizer(),
                FreqFeaturizer()
            ]
            if dc_path:
                featurizers.append(ConstraintFeaturizer())

            # repair
            self.session.repair_errors(featurizers)

            # getrepairafterdata
            repaired_df = self.session.ds.get_repaired_dataset()

            # Save results
            if output_path:
                repaired_df.to_csv(output_path, index=False)

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'HoloClean',
                'type': 'data-oriented',
                'auto_level': 1,  # fully automatic
                'detectors': len(detectors),
                'featurizers': len(featurizers)
            }

            return repaired_df, info

        except Exception as e:
            raise RuntimeError(f"HoloCleancleaningfailure: {e}")

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本（HoloCleanfully automatic，始终to0）"""
        return self.ground_truth_used


def holoclean_clean(dirty_path: str,
                    dc_path: str = None,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """HoloCleancleaning便捷function"""
    wrapper = HoloCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, dc_path, output_path)
