"""
HoloClean Wrapper - 约束 + 统计 + 知识库融合的数据清洗方法

HoloClean是一个基于概率图模型的全面数据清洗系统，通过融合多种清洗信号
（完整性约束、统计信息、外部知识库）进行数据修复。

论文: HoloClean: Holistic Data Repairs with Probabilistic Inference (VLDB 2017)
GitHub: https://github.com/HoloClean/holoclean

真值使用情况: 全自动执行，无需人工参与 (Type 1)

注意: HoloClean需要PostgreSQL数据库支持
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 尝试导入官方实现
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
    HoloClean清洗方法的封装类

    该类封装了HoloClean的核心功能，提供统一的接口用于数据清洗任务。

    注意: HoloClean需要PostgreSQL数据库支持
    - 安装PostgreSQL并创建用户和数据库
    - 设置环境变量或使用参数指定连接信息
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
        初始化HoloClean包装器

        Args:
            db_user: 数据库用户名
            db_pwd: 数据库密码
            db_host: 数据库主机
            db_name: 数据库名称
            domain_thresh_1: 第一阶段域值阈值
            domain_thresh_2: 第二阶段域值阈值
            weak_label_thresh: 弱标签阈值
            max_domain: 最大域大小
            cor_strength: 相关性强度
            nb_cor_strength: 近邻相关性强度
            weight_decay: 权重衰减
            learning_rate: 学习率
            threads: 线程数
            batch_size: 批大小
            epochs: 训练轮数
            verbose: 是否打印详细信息
            timeout: 超时时间（毫秒）
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
        """检查依赖是否满足"""
        if not HAS_HOLOCLEAN:
            raise ImportError(
                f"HoloClean模块导入失败: {IMPORT_ERROR}\n"
                "请确保在Methods/HoloClean目录下有完整的官方代码，"
                "并安装必要的依赖（torch, psycopg2等）"
            )

    def setup(self) -> bool:
        """
        设置HoloClean环境

        Returns:
            是否成功初始化
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
                print(f"HoloClean初始化失败: {e}")
            return False

    def clean(self,
              dirty_path: str,
              dc_path: str = None,
              output_path: str = None,
              dataset_name: str = "data") -> Tuple[pd.DataFrame, Dict]:
        """
        执行完整的清洗流程

        Args:
            dirty_path: 脏数据路径
            dc_path: 约束文件路径（可选）
            output_path: 输出路径（可选）
            dataset_name: 数据集名称

        Returns:
            修复后的数据和清洗信息
        """
        self._check_dependencies()

        # 初始化
        if not self.setup():
            raise RuntimeError("HoloClean初始化失败，请检查数据库配置")

        try:
            # 加载数据
            self.session.load_data(dataset_name, dirty_path)

            # 加载约束（如果提供）
            if dc_path and os.path.exists(dc_path):
                self.session.load_dcs(dc_path)

            # 检测错误
            detectors = [NullDetector()]
            if dc_path:
                detectors.append(ViolationDetector())
            self.session.detect_errors(detectors)

            # 设置域
            self.session.setup_domain()

            # 特征提取器
            featurizers = [
                InitAttrFeaturizer(),
                OccurAttrFeaturizer(),
                FreqFeaturizer()
            ]
            if dc_path:
                featurizers.append(ConstraintFeaturizer())

            # 修复
            self.session.repair_errors(featurizers)

            # 获取修复后的数据
            repaired_df = self.session.ds.get_repaired_dataset()

            # 保存结果
            if output_path:
                repaired_df.to_csv(output_path, index=False)

            info = {
                'ground_truth_cost': self.ground_truth_used,
                'method': 'HoloClean',
                'type': 'data-oriented',
                'auto_level': 1,  # 全自动
                'detectors': len(detectors),
                'featurizers': len(featurizers)
            }

            return repaired_df, info

        except Exception as e:
            raise RuntimeError(f"HoloClean清洗失败: {e}")

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本（HoloClean全自动，始终为0）"""
        return self.ground_truth_used


def holoclean_clean(dirty_path: str,
                    dc_path: str = None,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """HoloClean清洗的便捷函数"""
    wrapper = HoloCleanWrapper(**kwargs)
    return wrapper.clean(dirty_path, dc_path, output_path)
