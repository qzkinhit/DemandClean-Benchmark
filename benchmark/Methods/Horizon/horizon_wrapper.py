"""
Horizon Wrapper - 基于功能依赖模式的数据清洗方法

Horizon (VLDB 2021): 通过分析功能依赖(FD)约束来修复数据中的错误。

论文: Pattern Selection for Data Quality and Beyond (VLDB 2021)

真值使用情况: 全自动执行，但需要提供功能依赖规则 (Type 1)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def load_fds_from_rules(rules_path: str) -> List[Tuple[str, str]]:
    """
    从统一规则文件加载Horizon的功能依赖规则

    Args:
        rules_path: rules.txt文件路径 (如 Data/beers/rules.txt)

    Returns:
        FD列表，每项为(lhs, rhs)元组
    """
    fds = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"规则文件不存在: {rules_path}")

    in_horizon_section = False
    with open(rules_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('[') and line.endswith(']'):
                in_horizon_section = (line == '[HORIZON_FD]')
                continue

            if in_horizon_section:
                # 支持 => 和 ⇒
                if '=>' in line:
                    parts = line.split('=>')
                elif '⇒' in line:
                    parts = line.split('⇒')
                else:
                    continue

                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()
                    fds.append((lhs, rhs))

    return fds


def write_horizon_rule_file(rules_path: str, output_path: str) -> str:
    """
    从统一规则文件提取FD并写入Horizon格式文件

    Args:
        rules_path: 统一规则文件路径
        output_path: 输出FD文件路径

    Returns:
        输出文件路径
    """
    fds = load_fds_from_rules(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} ⇒ {rhs}\n")

    return output_path


class HorizonWrapper:
    """
    Horizon清洗方法的封装类

    Horizon基于功能依赖(FD)进行数据修复。
    需要提供功能依赖规则文件，格式为: "属性A ⇒ 属性B"

    真值使用: Type 1 (全自动，但需要规则文件)
    """

    def __init__(self,
                 verbose: bool = False):
        """
        初始化Horizon包装器

        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.ground_truth_used = 0  # Horizon是全自动的

    def clean(self,
              dirty_path: str,
              clean_path: str,
              rule_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执行Horizon清洗流程

        Args:
            dirty_path: 脏数据路径
            clean_path: 干净数据路径（用于评估）
            rule_path: 功能依赖规则文件路径
            output_path: 输出路径

        Returns:
            修复后的数据和清洗信息
        """
        # 导入Horizon模块
        try:
            from .horizon import Horizon, dirty_cells
        except ImportError:
            from horizon import Horizon, dirty_cells

        if self.verbose:
            print("=" * 60)
            print("Horizon 数据清洗")
            print("=" * 60)
            print(f"脏数据: {dirty_path}")
            print(f"规则文件: {rule_path}")

        # 执行Horizon
        pattern_expressions, dirty_c, elapsed_time = Horizon(dirty_path, rule_path, clean_path)

        # 将pattern_expressions应用到数据
        repaired_df = pd.read_csv(dirty_path)
        # 预处理：将 "empty" 和空字符串统一为 NaN 方便处理
        repaired_df = repaired_df.replace({'empty': '', 'Empty': '', 'EMPTY': ''})
        repaired_df = repaired_df.fillna("")

        corrected_cells = 0
        for i, expr in enumerate(pattern_expressions):
            for col, val in expr.items():
                if col in repaired_df.columns:
                    old_val = str(repaired_df.iloc[i][col])
                    if old_val != str(val) and val != 'empty' and val != '':
                        repaired_df.iloc[i, repaired_df.columns.get_loc(col)] = val
                        corrected_cells += 1

        # 后处理：将空字符串统一为 "empty"
        repaired_df = repaired_df.replace('', 'empty')

        if self.verbose:
            print(f"识别脏单元格: {len(dirty_c)}")
            print(f"修复单元格: {corrected_cells}")
            print(f"执行时间: {elapsed_time:.2f}秒")

        # 保存结果
        if output_path:
            repaired_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"修复后数据已保存: {output_path}")

        info = {
            'ground_truth_cost': self.ground_truth_used,
            'method': 'Horizon',
            'type': 'data-oriented',
            'auto_level': 1,
            'dirty_cells': len(dirty_c),
            'corrected_cells': corrected_cells,
            'elapsed_time': elapsed_time
        }

        return repaired_df, info

    def get_ground_truth_cost(self) -> int:
        """获取真值使用成本"""
        return self.ground_truth_used


def horizon_clean(dirty_path: str,
                  clean_path: str,
                  rule_path: str,
                  output_path: str = None,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Horizon清洗的便捷函数"""
    wrapper = HorizonWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, rule_path, output_path)
