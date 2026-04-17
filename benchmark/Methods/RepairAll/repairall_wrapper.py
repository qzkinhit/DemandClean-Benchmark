"""
RepairAll Wrapper

RepairAllis一个上界baseline，直接usecleandatareplace脏data。
相当于"完美repair"allError，used for建立can上界。

特点:
- 直接returncleandata作to"repair"result
- Type 2method（需needground truth）
- used for建立can上界baseline
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple


class RepairAllWrapper:
    """
    RepairAllcleaningmethod - 完美repairbaseline

    直接usecleandatareplace脏data，相当于100%正确repairallError。
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.method_name = "RepairAll"
        self.method_type = 2  # 需needground truth

    def clean(self,
              dirty_path: str,
              clean_path: str,
              output_path: str,
              index_attribute: str = 'index') -> Tuple[pd.DataFrame, Dict]:
        """
        执row"完美repair"：直接returncleandata

        Args:
            dirty_path: 脏datapath
            clean_path: cleandatapath
            output_path: outputfilepath
            index_attribute: 索引column name

        Returns:
            (repairafterdataDataFrame, cleaning信息dict)
        """
        # Read data
        dirty_data = pd.read_csv(dirty_path)
        clean_data = pd.read_csv(clean_path)

        if self.verbose:
            print(f"RepairAll: Read脏data {len(dirty_data)} row, {len(dirty_data.columns)} column")
            print(f"RepairAll: Readcleandata {len(clean_data)} row, {len(clean_data.columns)} column")

        # 计算repairCount信息
        total_cells = len(dirty_data) * len(dirty_data.columns)
        repaired_cells = 0

        # 比较脏dataandcleandata，计算repair多少cell
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

        # 直接usecleandata作tooutput
        result = clean_data.copy()

        # Save results
        result.to_csv(output_path, index=False)

        if self.verbose:
            print(f"RepairAll: repair {repaired_cells} 个cell")
            print(f"RepairAll: outputsaveto {output_path}")

        clean_info = {
            'original_rows': len(dirty_data),
            'output_rows': len(result),
            'repaired_cells': repaired_cells,
            'total_cells': total_cells,
            'repair_ratio': repaired_cells / total_cells if total_cells > 0 else 0,
            'ground_truth_cost': repaired_cells,  # useallground truth来repair
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
        print(f"result: {info}")
