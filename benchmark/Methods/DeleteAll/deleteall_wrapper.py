"""
DeleteAll Wrapper - deleteall含缺失值/异常值row

DeleteAllis一种简单baselinemethod，通过deleteallcontainrows with missing values来"cleaning"data。
也can选择deletealldiffers from the clean datarow。

ground truthusestats:
- 仅delete缺失值: Type 1 (fully automatic)
- deleteallErrorrow: Type 2 (需needground truth对比)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class DeleteAllWrapper:
    """
    DeleteAllcleaning器 - delete含has问题row

    支持三种mode:
    1. drop_missing: 仅delete含rows with missing values (Type 1)
    2. drop_errors: deletealldiffers from the clean datarow (Type 2)
    3. drop_feature_errors: deleterows with errors in feature columns，ifif all would be dropped, drop only50% (Type 2)
    """

    def __init__(self,
                 mode: str = 'drop_missing',
                 verbose: bool = False,
                 max_deletion_ratio: float = 0.8,
                 feature_columns: list = None,
                 label_column: str = None):
        """
        initializeDeleteAll

        Args:
            mode: deletemode ('drop_missing', 'drop_errors', 'drop_feature_errors')
            verbose: whether打印详细信息
            max_deletion_ratio: 最大deleteratio (仅drop_feature_errorsmode)
            feature_columns: feature columnscolumn list (仅drop_feature_errorsmode)
            label_column: labelcolumn name (仅drop_feature_errorsmode)
        """
        self.mode = mode
        self.verbose = verbose
        self.ground_truth_used = 0
        self.max_deletion_ratio = max_deletion_ratio
        self.feature_columns = feature_columns
        self.label_column = label_column

    def setup(self):
        """set（none操作）"""
        return True

    def clean(self,
              dirty_path: str,
              output_path: str = None,
              clean_path: str = None,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        Run deletion操作

        Args:
            dirty_path: 脏datapath
            output_path: outputpath
            clean_path: cleandatapath（drop_errorsmode需need）
            index_attribute: 索引column name

        Returns:
            cleaningafterdataand信息
        """
        # Read脏data
        dirty_data = pd.read_csv(dirty_path)
        original_rows = len(dirty_data)

        if self.verbose:
            print(f"DeleteAll: Read data {original_rows} row, {len(dirty_data.columns)} column")

        if self.mode == 'drop_missing':
            # mode1: deleteall含rows with missing values
            # Treat empty strings and certain sentinels as missing too
            result = dirty_data.copy()

            # Replace common missing-value markers
            result = result.replace(['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None', 'none', '?',
                                     'empty', 'EMPTY', 'Empty', 'missing', 'MISSING', 'Missing',
                                     'nan', 'NaN', 'NAN', '-', '--', 'unknown', 'Unknown', 'UNKNOWN'], np.nan)

            # deleterows containing missing values
            result = result.dropna()

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = 0  # 不需needground truth

            if self.verbose:
                print(f"DeleteAll (drop_missing): delete {deleted_rows} rowrows with missing values")

        elif self.mode == 'drop_errors':
            # mode2: deletealldiffers from the clean datarow
            if clean_path is None:
                raise ValueError("drop_errors mode requires clean_path")

            clean_data = pd.read_csv(clean_path)

            # Find inconsistent rows
            dirty_aligned = dirty_data.set_index(index_attribute, drop=False)
            clean_aligned = clean_data.set_index(index_attribute, drop=False)

            # Compare each rowwhetherconsistent
            error_rows = []
            for idx in dirty_aligned.index:
                if idx in clean_aligned.index:
                    dirty_row = dirty_aligned.loc[idx].astype(str).str.lower().str.strip()
                    clean_row = clean_aligned.loc[idx].astype(str).str.lower().str.strip()
                    if not dirty_row.equals(clean_row):
                        error_rows.append(idx)

            # delete不consistentrow
            result = dirty_data.set_index(index_attribute, drop=False)
            result = result.drop(index=error_rows, errors='ignore')
            result = result.reset_index(drop=True)

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = deleted_rows  # useground truth来判断哪些rowhas错

            if self.verbose:
                print(f"DeleteAll (drop_errors): delete {deleted_rows} rowerror-containing rows")

        elif self.mode == 'drop_feature_errors':
            # mode3: deleterows with errors in feature columns，ifif all would be dropped, drop only50%（缺失值must be deleted）
            if clean_path is None:
                raise ValueError("drop_feature_errors mode requires clean_path")

            clean_data = pd.read_csv(clean_path)

            # Determinefeature columns
            exclude_cols = [index_attribute]
            if self.label_column:
                exclude_cols.append(self.label_column)

            if self.feature_columns:
                feature_cols = [c for c in self.feature_columns if c in dirty_data.columns]
            else:
                # Default: every column except index and label
                feature_cols = [c for c in dirty_data.columns if c not in exclude_cols]

            if self.verbose:
                print(f"feature columns: {feature_cols}")

            # Align data
            dirty_aligned = dirty_data.set_index(index_attribute, drop=False)
            clean_aligned = clean_data.set_index(index_attribute, drop=False)

            # Count每rowErrorstats
            missing_rows = set()  # 含rows with missing values
            feature_error_rows = set()  # rows with errors in feature columns

            missing_markers = ['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None', 'none', '?',
                             'empty', 'EMPTY', 'Empty', 'missing', 'MISSING', 'Missing',
                             'nan', 'NaN', 'NAN', '-', '--', 'unknown', 'Unknown', 'UNKNOWN']

            for idx in dirty_aligned.index:
                if idx not in clean_aligned.index:
                    continue

                for col in feature_cols:
                    dirty_val = str(dirty_aligned.loc[idx, col]).strip()
                    clean_val = str(clean_aligned.loc[idx, col]).strip()

                    # Check whether the value is missing
                    if dirty_val in missing_markers or pd.isna(dirty_aligned.loc[idx, col]):
                        missing_rows.add(idx)
                    # Check whether any error exists（differs from the clean data）
                    elif dirty_val.lower() != clean_val.lower():
                        feature_error_rows.add(idx)

            # All rows to delete
            all_error_rows = missing_rows.union(feature_error_rows)

            if self.verbose:
                print(f"含rows with missing values: {len(missing_rows)}")
                print(f"rows with errors in feature columns: {len(feature_error_rows)}")
                print(f"Total rows to delete: {len(all_error_rows)}")

            # checkwhetherwilldeleteexceedsmax_deletion_ratio
            deletion_ratio = len(all_error_rows) / original_rows if original_rows > 0 else 0

            if deletion_ratio > self.max_deletion_ratio:
                # exceeds最大deleteratio，only delete rows with missing values，then randomly delete error rows
                if self.verbose:
                    print(f"deleteratio {deletion_ratio:.2%} exceeds the limit {self.max_deletion_ratio:.2%}")

                # rows with missing values must be deleted
                rows_to_delete = set(missing_rows)

                # Compute how many more can be deleted
                max_delete = int(original_rows * self.max_deletion_ratio)
                remaining_quota = max_delete - len(rows_to_delete)

                if remaining_quota > 0 and feature_error_rows - missing_rows:
                    # Randomly pick error rows whose cells are not missing
                    pure_error_rows = list(feature_error_rows - missing_rows)
                    import random
                    random.seed(42)
                    additional_delete = random.sample(pure_error_rows,
                                                     min(remaining_quota, len(pure_error_rows)))
                    rows_to_delete.update(additional_delete)

                if self.verbose:
                    print(f"Actual deleted rows: {len(rows_to_delete)} (limit{self.max_deletion_ratio:.0%})")
            else:
                rows_to_delete = all_error_rows

            # Run deletion
            result = dirty_data.set_index(index_attribute, drop=False)
            result = result.drop(index=list(rows_to_delete), errors='ignore')
            result = result.reset_index(drop=True)

            deleted_rows = original_rows - len(result)
            self.ground_truth_used = deleted_rows

            if self.verbose:
                print(f"DeleteAll (drop_feature_errors): delete {deleted_rows} row")

        else:
            raise ValueError(f"not知mode: {self.mode}")

        # Save results
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
        """getground truthuse成本"""
        return self.ground_truth_used


def deleteall_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """DeleteAllcleaning便捷function"""
    wrapper = DeleteAllWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path, **kwargs)
