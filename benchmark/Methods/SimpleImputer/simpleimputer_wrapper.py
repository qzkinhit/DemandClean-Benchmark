"""
SimpleImputer Wrapper - 简单Count填充method

SimpleImputeris基于简单Count量（均值/位数/众数）缺失值填充method。

ground truthusestats: fully automatic执row，none需人工参and (Type 1)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.preprocessing import LabelEncoder

# 常见缺失值占位符（不区分size写）
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class SimpleImputerWrapper:
    """
    简单插补method封装class

    支持多种简单插补策略:
    1. 均值填充 (mean)
    2. 位数填充 (median)
    3. 众数填充 (most_frequent)
    4. 常数填充 (constant)
    """

    def __init__(self,
                 strategy: str = 'mean',
                 fill_value: Optional[str] = None,
                 verbose: bool = False):
        """
        initializeSimpleImputer

        Args:
            strategy: 填充策略 ('mean', 'median', 'most_frequent', 'constant')
            fill_value: 当strategy='constant'时use填充值
            verbose: whether打印详细信息
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose

        self.numeric_imputer = None
        self.categorical_imputer = None
        self.ground_truth_used = 0

    def setup(self):
        """set插补器"""
        # 数值column插补器
        if self.strategy in ['mean', 'median']:
            self.numeric_imputer = SklearnSimpleImputer(strategy=self.strategy)
        elif self.strategy == 'constant':
            self.numeric_imputer = SklearnSimpleImputer(
                strategy='constant',
                fill_value=self.fill_value if self.fill_value else 0
            )
        else:
            self.numeric_imputer = SklearnSimpleImputer(strategy='mean')

        # classificationcolumn插补器
        if self.strategy == 'most_frequent':
            self.categorical_imputer = SklearnSimpleImputer(strategy='most_frequent')
        elif self.strategy == 'constant':
            self.categorical_imputer = SklearnSimpleImputer(
                strategy='constant',
                fill_value=self.fill_value if self.fill_value else 'missing'
            )
        else:
            self.categorical_imputer = SklearnSimpleImputer(strategy='most_frequent')

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

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执row插补

        Args:
            data: inputDataFrame

        Returns:
            post-imputationDataFrame
        """
        result = data.copy()

        # get数值columnandclassificationcolumn
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        # 填充数值column
        if numeric_cols:
            result[numeric_cols] = self.numeric_imputer.fit_transform(data[numeric_cols])

        # 填充classificationcolumn
        if categorical_cols:
            # Convert classification columns to strings，handle possible None values
            cat_data = data[categorical_cols].astype(str).replace('nan', np.nan)
            result[categorical_cols] = self.categorical_imputer.fit_transform(cat_data)

        return result

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

        # 插补
        repaired_df = self.impute(data)

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
            'method': f'SimpleImputer-{self.strategy}',
            'type': 'data-preparation',
            'auto_level': 1,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'imputed_cells': int(missing_before - missing_after)
        }

        return repaired_df, info


def simpleimputer_clean(dirty_path: str,
                        output_path: str = None,
                        **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """SimpleImputercleaning便捷function"""
    wrapper = SimpleImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
