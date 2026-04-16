"""
DoNothing Wrapper - 不做任何清洗的基准方法

DoNothing是最简单的baseline，直接返回脏数据不做任何处理。
用于对比其他清洗方法的效果。

真值使用情况: 无 (Type 1 - 全自动)
"""

import os
import pandas as pd
from typing import Dict, Tuple


class DoNothingWrapper:
    """
    DoNothing清洗器 - 不做任何清洗

    这是一个baseline方法，直接返回输入数据，不做任何修改。
    用于建立性能下界。
    """

    def __init__(self, verbose: bool = False):
        """
        初始化DoNothing

        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.ground_truth_used = 0

    def setup(self):
        """设置（无操作）"""
        return True

    def clean(self,
              dirty_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行"清洗"（实际上不做任何操作）

        Args:
            dirty_path: 脏数据路径
            output_path: 输出路径

        Returns:
            原始数据和信息
        """
        # 直接读取脏数据
        data = pd.read_csv(dirty_path)

        if self.verbose:
            print(f"DoNothing: 读取数据 {len(data)} 行, {len(data.columns)} 列")
            print("DoNothing: 不做任何清洗操作")

        # 保存结果（与输入相同）
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
        """获取真值使用成本"""
        return self.ground_truth_used


def donothing_clean(dirty_path: str,
                    output_path: str = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """DoNothing清洗的便捷函数"""
    wrapper = DoNothingWrapper(**kwargs)
    return wrapper.clean(dirty_path, output_path)
