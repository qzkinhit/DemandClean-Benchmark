"""
MLImputer Wrapper - 机器学习插补方法

MLImputer是基于机器学习模型的缺失值填充方法，通过训练模型预测缺失值。

真值使用情况: 全自动执行，无需人工参与 (Type 1)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 常见的缺失值占位符（不区分大小写）
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class MLImputerWrapper:
    """
    机器学习插补方法的封装类

    支持多种ML插补策略:
    1. MICE (Multiple Imputation by Chained Equations)
    2. KNN插补
    3. 随机森林插补
    """

    def __init__(self,
                 method: str = 'mice',
                 max_iter: int = 10,
                 n_neighbors: int = 5,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        初始化MLImputer

        Args:
            method: 插补方法 ('mice', 'knn', 'rf')
            max_iter: MICE最大迭代次数
            n_neighbors: KNN邻居数
            random_state: 随机种子
            verbose: 是否打印详细信息
        """
        self.method = method
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose

        self.imputer = None
        self.label_encoders = {}
        self.ground_truth_used = 0

    def setup(self):
        """设置插补器"""
        if self.method == 'mice':
            self.imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=2 if self.verbose else 0
            )
        elif self.method == 'knn':
            self.imputer = KNNImputer(
                n_neighbors=self.n_neighbors
            )
        elif self.method == 'rf':
            self.imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=10,
                    random_state=self.random_state
                ),
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=2 if self.verbose else 0
            )
        return True

    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据并将缺失值占位符转换为NaN"""
        self.data = pd.read_csv(data_path)
        # 将常见占位符转换为NaN
        self.data = self._convert_placeholders_to_nan(self.data)
        return self.data

    def _convert_placeholders_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将常见的缺失值占位符转换为NaN

        支持识别的占位符：empty, null, none, na, n/a, nan, missing, 空字符串等
        """
        result = df.copy()

        for col in result.columns:
            if result[col].dtype == object:  # 仅处理字符串列
                # 创建小写版本用于比较
                col_lower = result[col].astype(str).str.lower().str.strip()

                # 标记需要替换的单元格
                for placeholder in MISSING_VALUE_PLACEHOLDERS:
                    mask = col_lower == placeholder.lower()
                    result.loc[mask, col] = np.nan

        return result

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        预处理：编码分类特征

        Args:
            data: 输入DataFrame

        Returns:
            编码后的数组和列名
        """
        processed = data.copy()
        columns = data.columns.tolist()

        # 编码分类特征
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # 处理缺失值
            mask = processed[col].notna()
            if mask.sum() > 0:
                processed.loc[mask, col] = le.fit_transform(processed.loc[mask, col].astype(str))
                self.label_encoders[col] = le
            processed[col] = pd.to_numeric(processed[col], errors='coerce')

        return processed.values, columns

    def postprocess(self, X: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """
        后处理：解码分类特征

        Args:
            X: 插补后的数组
            columns: 列名

        Returns:
            处理后的DataFrame
        """
        df = pd.DataFrame(X, columns=columns)

        # 解码分类特征
        for col, le in self.label_encoders.items():
            df[col] = df[col].round().astype(int)
            df[col] = df[col].clip(0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(df[col])

        return df

    def impute(self, X: np.ndarray) -> np.ndarray:
        """
        执行插补

        Args:
            X: 输入数组

        Returns:
            插补后的数组
        """
        return self.imputer.fit_transform(X)

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本"""
        return self.ground_truth_used

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行完整的插补流程

        Args:
            dirty_path: 脏数据路径
            output_path: 输出路径

        Returns:
            插补后的数据和信息
        """
        # 初始化
        self.setup()

        # 加载数据
        data = self.load_data(dirty_path)

        # 统计缺失值
        missing_before = data.isnull().sum().sum()

        # 预处理
        X, columns = self.preprocess(data)

        # 插补
        X_imputed = self.impute(X)

        # 后处理
        repaired_df = self.postprocess(X_imputed, columns)

        # 统计填充数
        missing_after = repaired_df.isnull().sum().sum()

        # 后处理：将空值统一为 "empty"
        repaired_df = repaired_df.fillna('empty')
        repaired_df = repaired_df.replace('', 'empty')

        # 保存结果
        if output_path:
            repaired_df.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.get_ground_truth_cost(),
            'method': f'MLImputer-{self.method}',
            'type': 'data-preparation',
            'auto_level': 1,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'imputed_cells': int(missing_before - missing_after)
        }

        return repaired_df, info


def mlimputer_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """MLImputer清洗的便捷函数"""
    wrapper = MLImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
