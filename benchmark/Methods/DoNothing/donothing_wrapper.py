"""
DoNothing Wrapper - 不做任何cleaning基准method

DoNothingis最简单baseline，直接return脏data不做任何处理。
used for对比其他cleaningmethod效果。

ground truthusestats: none (Type 1 - fully automatic)
"""

import os
import pandas as pd
from typing import Dict, Tuple


class DoNothingWrapper:
    """
    DoNothingcleaning器 - 不做任何cleaning

    这is一个baselinemethod，直接returninputdata，不做任何修改。
    used for建立can下界。
    """

    def __init__(self, verbose: bool = False):
        """
        initializeDoNothing

        Args:
            verbose: whether打印详细信息
        """
        self.verbose = verbose
        self.ground_truth_used = 0

    def setup(self):
        """set（none操作）"""
        return True

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执row"cleaning"（实际上不做任何操作）

        Args:
            dirty_path: 脏datapath
            output_path: outputpath

        Returns:
            original始dataand信息
        """
        # 直接Read脏data
        data = pd.read_csv(dirty_path)

        if self.verbose:
            print(f"DoNothing: Read data {len(data)} row, {len(data.columns)} column")
            print("DoNothing: 不做任何cleaning操作")

        # Save results（andinput相同）
        if output_path:
            data.to_csv(output_path, index=False)

        info = {
            'ground_truth_cost': 0,
            'method': 'DoNothing',
            'type': 'baseline',
            'auto_level': 1,
            'rows': len(data),
            'columns': len(data.columns),
            'changes_made': 0
        }

        return data, info

    def get_ground_truth_cost(self) -> int:
        """getground truthuse成本"""
        return self.ground_truth_used


def donothing_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """DoNothingcleaning便捷function"""
    wrapper = DoNothingWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
