"""
RepairAll Wrapper

RepairAll是一个上界baseline，直接使用干净数据替换脏数据。
相当于"完美修复"所有错误，用于建立性能上界。

特点:
- 直接返回干净数据作为"修复"结果
- Type 2方法（需要真值）
- 用于建立性能上界baseline
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple


class RepairAllWrapper:
    """
    RepairAll清洗方法 - 完美修复baseline

    直接使用干净数据替换脏数据，相当于100%正确修复所有错误。
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.method_name = "RepairAll"
        self.method_type = 2  # 需要真值

    def clean(self,
              dirty_path: str,
              clean_path: str,
              output_path: str,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执行"完美修复"：直接返回干净数据

        Args:
            dirty_path: 脏数据路径
            clean_path: 干净数据路径
            output_path: 输出文件路径
            index_attribute: 索引列名

        Returns:
            (修复后的数据DataFrame, 清洗信息字典)
        """
        # 读取数据
        dirty_data = pd.read_csv(dirty_path)
        clean_data = pd.read_csv(clean_path)

        if self.verbose:
            print(f"RepairAll: 读取脏数据 {len(dirty_data)} 行, {len(dirty_data.columns)} 列")
            print(f"RepairAll: 读取干净数据 {len(clean_data)} 行, {len(clean_data.columns)} 列")

        # 计算修复统计信息
        total_cells = len(dirty_data) * len(dirty_data.columns)
        repaired_cells = 0

        # 比较脏数据和干净数据，计算修复了多少单元格
        common_indices = dirty_data.index.intersection(clean_data.index)
        common_columns = [c for c in dirty_data.columns if c in clean_data.columns]

        for col in common_columns:
            if col == index_attribute:
                continue
            for idx in common_indices:
                try:
                    dirty_val = dirty_data.loc[idx, col]
                    clean_val = clean_data.loc[idx, col]
                    if str(dirty_val) != str(clean_val):
                        repaired_cells += 1
                except:
                    pass

        # 直接使用干净数据作为输出
        result = clean_data.copy()

        # 保存结果
        result.to_csv(output_path, index=False)

        if self.verbose:
            print(f"RepairAll: 修复了 {repaired_cells} 个单元格")
            print(f"RepairAll: 输出保存到 {output_path}")

        clean_info = {
            'original_rows': len(dirty_data),
            'output_rows': len(result),
            'repaired_cells': repaired_cells,
            'total_cells': total_cells,
            'repair_ratio': repaired_cells / total_cells if total_cells > 0 else 0,
            'ground_truth_cost': repaired_cells,  # 使用了所有真值来修复
            'method_type': self.method_type
        }

        return result, clean_info


if __name__ == '__main__':
    # 简单测试
    import sys
    if len(sys.argv) >= 3:
        wrapper = RepairAllWrapper(verbose=True)
        result, info = wrapper.clean(
            dirty_path=sys.argv[1],
            clean_path=sys.argv[2],
            output_path='test_repairall_output.csv'
        )
        print(f"结果: {info}")
