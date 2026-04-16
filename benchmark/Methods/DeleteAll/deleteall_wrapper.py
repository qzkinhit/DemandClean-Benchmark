"""
DeleteAll Wrapper - 删除所有含缺失值/异常值的行

DeleteAll是一种简单的baseline方法，通过删除所有包含缺失值的行来"清洗"数据。
也可以选择删除所有与干净数据不一致的行。

真值使用情况:
- 仅删除缺失值: Type 1 (全自动)
- 删除所有错误行: Type 2 (需要真值对比)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class DeleteAllWrapper:
    """
    DeleteAll清洗器 - 删除含有问题的行

    支持三种模式:
    1. drop_missing: 仅删除含缺失值的行 (Type 1)
    2. drop_errors: 删除所有与干净数据不一致的行 (Type 2)
    3. drop_feature_errors: 删除特征列有错误的行，如果全删就只删50% (Type 2)
    """

    def __init__(self,
                 mode: str = 'drop_missing',
                 verbose: bool = False,
                 max_deletion_ratio: float = 0.8,
                 feature_columns: list = None,
                 label_column: str = None):
        """
        初始化DeleteAll

        Args:
            mode: 删除模式 ('drop_missing', 'drop_errors', 'drop_feature_errors')
            verbose: 是否打印详细信息
            max_deletion_ratio: 最大删除比例 (仅drop_feature_errors模式)
            feature_columns: 特征列列表 (仅drop_feature_errors模式)
            label_column: 标签列名 (仅drop_feature_errors模式)
        """
        self.mode = mode
        self.verbose = verbose
        self.ground_truth_used = 0
        self.max_deletion_ratio = max_deletion_ratio
        self.feature_columns = feature_columns
        self.label_column = label_column

    def setup(self):
        """设置（无操作）"""
        return True

    def clean(self,
              dirty_path: str,
              output_path: str = None,
              clean_path: str = None,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执行删除操作

        Args:
            dirty_path: 脏数据路径
            output_path: 输出路径
            clean_path: 干净数据路径（drop_errors模式需要）
            index_attribute: 索引列名

        Returns:
            清洗后的数据和信息
        """
        # 读取脏数据
        dirty_data = pd.read_csv(dirty_path)
        original_rows = len(dirty_data)

        if self.verbose:
            print(f"DeleteAll: 读取数据 {original_rows} 行, {len(dirty_data.columns)} 列")

        if self.mode == 'drop_missing':
            # 模式1: 删除所有含缺失值的行
            # 将空字符串和特定值也视为缺失
            result = dirty_data.copy()

            # 替换常见的缺失值表示
            result = result.replace(['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None', 'none', '?',
                                     'empty', 'EMPTY', 'Empty', 'missing', 'MISSING', 'Missing',
                                     'nan', 'NaN', 'NAN', '-', '--', 'unknown', 'Unknown', 'UNKNOWN'], np.nan)

            # 删除含有缺失值的行
            result = result.dropna()

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = 0  # 不需要真值

            if self.verbose:
                print(f"DeleteAll (drop_missing): 删除了 {deleted_rows} 行含缺失值的数据")

        elif self.mode == 'drop_errors':
            # 模式2: 删除所有与干净数据不一致的行
            if clean_path is None:
                raise ValueError("drop_errors模式需要提供clean_path")

            clean_data = pd.read_csv(clean_path)

            # 找出不一致的行
            dirty_aligned = dirty_data.set_index(index_attribute, drop=False)
            clean_aligned = clean_data.set_index(index_attribute, drop=False)

            # 比较每行是否一致
            error_rows = []
            for idx in dirty_aligned.index:
                if idx in clean_aligned.index:
                    dirty_row = dirty_aligned.loc[idx].astype(str).str.lower().str.strip()
                    clean_row = clean_aligned.loc[idx].astype(str).str.lower().str.strip()
                    if not dirty_row.equals(clean_row):
                        error_rows.append(idx)

            # 删除不一致的行
            result = dirty_data.set_index(index_attribute, drop=False)
            result = result.drop(index=error_rows, errors='ignore')
            result = result.reset_index(drop=True)

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = deleted_rows  # 使用了真值来判断哪些行有错

            if self.verbose:
                print(f"DeleteAll (drop_errors): 删除了 {deleted_rows} 行含错误的数据")

        elif self.mode == 'drop_feature_errors':
            # 模式3: 删除特征列有错误的行，如果全删就只删50%（缺失值一定要删除）
            if clean_path is None:
                raise ValueError("drop_feature_errors模式需要提供clean_path")

            clean_data = pd.read_csv(clean_path)

            # 确定特征列
            exclude_cols = [index_attribute]
            if self.label_column:
                exclude_cols.append(self.label_column)

            if self.feature_columns:
                feature_cols = [c for c in self.feature_columns if c in dirty_data.columns]
            else:
                # 默认：所有列除了index和label
                feature_cols = [c for c in dirty_data.columns if c not in exclude_cols]

            if self.verbose:
                print(f"特征列: {feature_cols}")

            # 对齐数据
            dirty_aligned = dirty_data.set_index(index_attribute, drop=False)
            clean_aligned = clean_data.set_index(index_attribute, drop=False)

            # 统计每行的错误情况
            missing_rows = set()  # 含缺失值的行
            feature_error_rows = set()  # 特征列有错误的行

            missing_markers = ['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None', 'none', '?',
                             'empty', 'EMPTY', 'Empty', 'missing', 'MISSING', 'Missing',
                             'nan', 'NaN', 'NAN', '-', '--', 'unknown', 'Unknown', 'UNKNOWN']

            for idx in dirty_aligned.index:
                if idx not in clean_aligned.index:
                    continue

                for col in feature_cols:
                    dirty_val = str(dirty_aligned.loc[idx, col]).strip()
                    clean_val = str(clean_aligned.loc[idx, col]).strip()

                    # 检查是否是缺失值
                    if dirty_val in missing_markers or pd.isna(dirty_aligned.loc[idx, col]):
                        missing_rows.add(idx)
                    # 检查是否有错误（与干净数据不一致）
                    elif dirty_val.lower() != clean_val.lower():
                        feature_error_rows.add(idx)

            # 所有需要删除的行
            all_error_rows = missing_rows.union(feature_error_rows)

            if self.verbose:
                print(f"含缺失值的行: {len(missing_rows)}")
                print(f"特征列有错误的行: {len(feature_error_rows)}")
                print(f"总共需要删除的行: {len(all_error_rows)}")

            # 检查是否会删除超过max_deletion_ratio
            deletion_ratio = len(all_error_rows) / original_rows if original_rows > 0 else 0

            if deletion_ratio > self.max_deletion_ratio:
                # 超过最大删除比例，只删除缺失值行，再随机删除部分错误行
                if self.verbose:
                    print(f"删除比例 {deletion_ratio:.2%} 超过限制 {self.max_deletion_ratio:.2%}")

                # 缺失值行必须删除
                rows_to_delete = set(missing_rows)

                # 计算还能删除多少
                max_delete = int(original_rows * self.max_deletion_ratio)
                remaining_quota = max_delete - len(rows_to_delete)

                if remaining_quota > 0 and feature_error_rows - missing_rows:
                    # 从非缺失值的错误行中随机选择
                    pure_error_rows = list(feature_error_rows - missing_rows)
                    import random
                    random.seed(42)
                    additional_delete = random.sample(pure_error_rows,
                                                     min(remaining_quota, len(pure_error_rows)))
                    rows_to_delete.update(additional_delete)

                if self.verbose:
                    print(f"实际删除行数: {len(rows_to_delete)} (限制为{self.max_deletion_ratio:.0%})")
            else:
                rows_to_delete = all_error_rows

            # 执行删除
            result = dirty_data.set_index(index_attribute, drop=False)
            result = result.drop(index=list(rows_to_delete), errors='ignore')
            result = result.reset_index(drop=True)

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = deleted_rows

            if self.verbose:
                print(f"DeleteAll (drop_feature_errors): 删除了 {deleted_rows} 行")

        else:
            raise ValueError(f"未知模式: {self.mode}")

        # 保存结果
        if output_path:
            result.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': f'DeleteAll-{self.mode}',
            'type': 'baseline',
            'auto_level': 1 if self.mode == 'drop_missing' else 2,
            'original_rows': original_rows,
            'remaining_rows': len(result),
            'deleted_rows': deleted_rows,
            'deletion_ratio': deleted_rows / original_rows if original_rows > 0 else 0
        }

        return result, info

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本"""
        return self.ground_truth_used


def deleteall_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """DeleteAll清洗的便捷函数"""
    wrapper = DeleteAllWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path, **kwargs)
