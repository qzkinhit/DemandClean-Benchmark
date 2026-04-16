"""
SimpleImputer Wrapper - 简单统计填充方法

SimpleImputer是基于简单统计量（均值/中位数/众数）的缺失值填充方法。

真值使用情况: 全自动执行，无需人工参与 (Type 1)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.preprocessing import LabelEncoder

# 常见的缺失值占位符（不区分大小写）
MISSING_VALUE_PLACEHOLDERS = [
    'empty', 'null', 'none', 'na', 'n/a', 'nan', 'missing',
    '', ' ', '?', '-', '--', 'unknown', 'undefined'
]


class SimpleImputerWrapper:
    """
    简单插补方法的封装类

    支持多种简单插补策略:
    1. 均值填充 (mean)
    2. 中位数填充 (median)
    3. 众数填充 (most_frequent)
    4. 常数填充 (constant)
    """

    def __init__(self,
                 strategy: str = 'mean',
                 fill_value: Optional[str] = None,
                 verbose: bool = False):
        """
        初始化SimpleImputer

        Args:
            strategy: 填充策略 ('mean', 'median', 'most_frequent', 'constant')
            fill_value: 当strategy='constant'时使用的填充值
            verbose: 是否打印详细信息
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose

        self.numeric_imputer = None
        self.categorical_imputer = None
        self.ground_truth_used = 0

    def setup(self):
        """设置插补器"""
        # 数值列插补器
        if self.strategy in ['mean', 'median']:
            self.numeric_imputer = SklearnSimpleImputer(strategy=self.strategy)
        elif self.strategy == 'constant':
            self.numeric_imputer = SklearnSimpleImputer(
                strategy='constant',
                fill_value=self.fill_value if self.fill_value else 0
            )
        else:
            self.numeric_imputer = SklearnSimpleImputer(strategy='mean')

        # 分类列插补器
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

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行插补

        Args:
            data: 输入DataFrame

        Returns:
            插补后的DataFrame
        """
        result = data.copy()

        # 获取数值列和分类列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

        # 填充数值列
        if numeric_cols:
            result[numeric_cols] = self.numeric_imputer.fit_transform(data[numeric_cols])

        # 填充分类列
        if categorical_cols:
            # 将分类列转换为字符串，处理可能的None值
            cat_data = data[categorical_cols].astype(str).replace('nan', np.nan)
            result[categorical_cols] = self.categorical_imputer.fit_transform(cat_data)

        return result

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

        # 插补
        repaired_df = self.impute(data)

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
    """SimpleImputer清洗的便捷函数"""
    wrapper = SimpleImputerWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
