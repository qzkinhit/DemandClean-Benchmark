"""
MLImputer Wrapper - 机器学习插补method

MLImputeris基于机器学习model缺失值填充method，通过trainingmodel预测缺失值。

ground truthusestats: fully automatic执row，none需人工参and (Type 1)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 常见缺失值占位符（不区分size写）
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class MLImputerWrapper:
    """
    机器学习插补method封装class

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
        initializeMLImputer

        Args:
            method: 插补method ('mice', 'knn', 'rf')
            max_iter: MICE最大迭代次数
            n_neighbors: KNN邻居数
            random_state: 随机种子
            verbose: whether打印详细信息
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
        """set插补器"""
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
        """Load dataand将缺失值占位符ConverttoNaN"""
        self.data = pd.read_csv(data_path)
        # 将常见占位符ConverttoNaN
        self.data = self._convert_placeholders_to_nan(self.data)
        return self.data

    def _convert_placeholders_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将常见缺失值占位符ConverttoNaN

        支持Identify占位符：empty, null, none, na, n/a, nan, missing, 空字符串等
        """
        result = df.copy()

        for col in result.columns:
            if result[col].dtype == object:  # 仅处理字符串column
                # Create a lowercase copy for comparison
                col_lower = result[col].astype(str).str.lower().str.strip()

                # Mark cells that need to be replaced
                for placeholder in MISSING_VALUE_PLACEHOLDERS:
                    mask = col_lower == placeholder.lower()
                    result.loc[mask, col] = np.nan

        return result

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        预处理：编码classificationfeature

        Args:
            data: inputDataFrame

        Returns:
            编码afterarrayandcolumn name
        """
        processed = data.copy()
        columns = data.columns.tolist()

        # 编码classificationfeature
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # Handle missing values
            mask = processed[col].notna()
            if mask.sum() > 0:
                processed.loc[mask, col] = le.fit_transform(processed.loc[mask, col].astype(str))
                self.label_encoders[col] = le
            processed[col] = pd.to_numeric(processed[col], errors='coerce')

        return processed.values, columns

    def postprocess(self, X: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """
        after处理：解码classificationfeature

        Args:
            X: post-imputationarray
            columns: column name

        Returns:
            处理afterDataFrame
        """
        df = pd.DataFrame(X, columns=columns)

        # 解码classificationfeature
        for col, le in self.label_encoders.items():
            df[col] = df[col].round().astype(int)
            df[col] = df[col].clip(0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(df[col])

        return df

    def impute(self, X: np.ndarray) -> np.ndarray:
        """
        执row插补

        Args:
            X: inputarray

        Returns:
            post-imputationarray
        """
        return self.imputer.fit_transform(X)

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本"""
        return self.ground_truth_used

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执row完整插补流程

        Args:
            dirty_path: 脏datapath
            output_path: outputpath

        Returns:
            post-imputationdataand信息
        """
        # initialize
        self.setup()

        # Load data
        data = self.load_data(dirty_path)

        # Count缺失值
        missing_before = data.isnull().sum().sum()

        # 预处理
        X, columns = self.preprocess(data)

        # 插补
        X_imputed = self.impute(X)

        # after处理
        repaired_df = self.postprocess(X_imputed, columns)

        # Count填充数
        missing_after = repaired_df.isnull().sum().sum()

        # after处理：将empty valuenormalize to "empty"
        repaired_df = repaired_df.fillna('empty')
        repaired_df = repaired_df.replace('', 'empty')

        # Save results
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
    """MLImputercleaning便捷function"""
    wrapper = MLImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
