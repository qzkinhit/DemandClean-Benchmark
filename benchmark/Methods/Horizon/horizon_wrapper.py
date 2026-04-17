"""
Horizon Wrapper - 基于功can依赖modedatacleaningmethod

Horizon (VLDB 2021): 通过分析功can依赖(FD)约束来repairdataError。

论文: Pattern Selection for Data Quality and Beyond (VLDB 2021)

ground truthusestats: fully automatic执row，但需needprovide功can依赖规则 (Type 1)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

# 添加currentdirectorytopath
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def load_fds_from_rules(rules_path: str) -> List[Tuple[str, str]]:
    """
    from统一规则fileloadHorizon功can依赖规则

    Args:
        rules_path: rules.txtfilepath (如 Data/beers/rules.txt)

    Returns:
        FDcolumn list，每项to(lhs, rhs)元组
    """
    fds = []

    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"规则filedoes not exist: {rules_path}")

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
                # Accept both => and ⇒
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
    from统一规则file提取FDand写入Horizonformatfile

    Args:
        rules_path: 统一规则filepath
        output_path: outputFDfilepath

    Returns:
        outputfilepath
    """
    fds = load_fds_from_rules(rules_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for lhs, rhs in fds:
            f.write(f"{lhs} ⇒ {rhs}\n")

    return output_path


class HorizonWrapper:
    """
    Horizoncleaningmethod封装class

    Horizon基于功can依赖(FD)进rowdatarepair。
    需needprovide功can依赖规则file，formatto: "属A ⇒ 属B"

    ground truthuse: Type 1 (fully automatic，但需need规则file)
    """

    def __init__(self,
                 verbose: bool = False):
        """
        initializeHorizonpackage装器

        Args:
            verbose: whether打印详细信息
        """
        self.verbose = verbose
        self.ground_truth_used = 0  # Horizonisfully automatic

    def clean(self,
              dirty_path: str,
              clean_path: str,
              rule_path: str,
              output_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        执rowHorizoncleaning流程

        Args:
            dirty_path: 脏datapath
            clean_path: cleandatapath（used forevaluation）
            rule_path: 功can依赖规则filepath
            output_path: outputpath

        Returns:
            repairafterdataandcleaning信息
        """
        # 导入Horizonmodule
        try:
            from .horizon import Horizon, dirty_cells
        except ImportError:
            from horizon import Horizon, dirty_cells

        if self.verbose:
            print("=" * 60)
            print("Horizon datacleaning")
            print("=" * 60)
            print(f"脏data: {dirty_path}")
            print(f"规则file: {rule_path}")

        # 执rowHorizon
        pattern_expressions, dirty_c, elapsed_time = Horizon(dirty_path, rule_path, clean_path)

        # 将pattern_expressionsshould用todata
        repaired_df = pd.read_csv(dirty_path)
        # 预处理：将 "empty" and空字符串normalize to NaN 方便处理
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

        # after处理：将空字符串normalize to "empty"
        repaired_df = repaired_df.replace('', 'empty')

        if self.verbose:
            print(f"Identify脏cell: {len(dirty_c)}")
            print(f"repaircell: {corrected_cells}")
            print(f"执rowtime: {elapsed_time:.2f}秒")

        # Save results
        if output_path:
            repaired_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"Repaired data saved: {output_path}")

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
        """getground truthuse成本"""
        return self.ground_truth_used


def horizon_clean(dirty_path: str,
                  clean_path: str,
                  rule_path: str,
                  output_path: str = None,
                  **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Horizoncleaning便捷function"""
    wrapper = HorizonWrapper(**kwargs)
    return wrapper.clean(dirty_path, clean_path, rule_path, output_path)
