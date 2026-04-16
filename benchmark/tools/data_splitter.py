"""
data_splitter.py - DemandClean 对齐的数据划分模块

提供与 DemandClean (run_demandclean_base.py:2659-2663) 完全相同的
seed=42, 60/20/20 训练/验证/测试集划分。

所有 run_xxx_base.py 脚本均可调用此模块，确保划分一致性。

Usage:
    from tools.data_splitter import get_demandclean_split

    split = get_demandclean_split(dirty_path, clean_path)
    dirty_train = split['dirty_train']
    clean_val = split['clean_val']
    clean_test = split['clean_test']
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Optional


SEED = 42
TRAIN_RATIO = 0.6  # 60% train, 20% val, 20% test


def compute_split_indices(n_total: int, seed: int = SEED):
    """
    计算与 DemandClean 完全相同的 60/20/20 划分索引。

    对应 run_demandclean_base.py:2659-2663:
        all_idx = np.arange(n_total)
        train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    Args:
        n_total: 数据集总行数
        seed: 随机种子（默认42）

    Returns:
        (train_idx, val_idx, test_idx) 三个 numpy 数组
    """
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


def get_demandclean_split(dirty_path: str, clean_path: str,
                           seed: int = SEED) -> Dict:
    """
    读取数据并返回与 DemandClean 完全相同的 60/20/20 划分。

    Args:
        dirty_path: 脏数据 CSV 路径
        clean_path: 干净数据 CSV 路径
        seed: 随机种子

    Returns:
        字典包含:
        - train_idx, val_idx, test_idx: 划分索引
        - dirty_train: 脏数据训练集 (60%)
        - clean_train: 干净数据训练集 (60%)
        - clean_val: 干净数据验证集 (20%)
        - clean_test: 干净数据测试集 (20%)
        - dirty_full: 全量脏数据
        - clean_full: 全量干净数据
        - n_total: 总行数
    """
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    assert len(dirty_df) == len(clean_df), \
        f"脏数据({len(dirty_df)}行)与干净数据({len(clean_df)}行)行数不一致"

    n_total = len(dirty_df)
    train_idx, val_idx, test_idx = compute_split_indices(n_total, seed)

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'dirty_train': dirty_df.iloc[train_idx].reset_index(drop=True),
        'dirty_val': dirty_df.iloc[val_idx].reset_index(drop=True),
        'dirty_test': dirty_df.iloc[test_idx].reset_index(drop=True),
        'clean_train': clean_df.iloc[train_idx].reset_index(drop=True),
        'clean_val': clean_df.iloc[val_idx].reset_index(drop=True),
        'clean_test': clean_df.iloc[test_idx].reset_index(drop=True),
        'dirty_full': dirty_df,
        'clean_full': clean_df,
        'n_total': n_total,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
    }


def save_split_csvs(split: Dict, output_dir: str, prefix: str = ''):
    """将划分后的子集保存为 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)

    for key in ['dirty_train', 'clean_train', 'clean_val', 'clean_test']:
        filename = f"{prefix}{key}.csv" if prefix else f"{key}.csv"
        split[key].to_csv(os.path.join(output_dir, filename), index=False)


def verify_split_consistency(dirty_path: str, clean_path: str,
                              demandclean_dirty_train_path: Optional[str] = None):
    """
    验证划分与 DemandClean 一致。

    如果 DemandClean 已生成 dirty_train_60pct.csv，可以传入进行对比。
    """
    split = get_demandclean_split(dirty_path, clean_path)
    print(f"总行数: {split['n_total']}")
    print(f"训练集: {split['n_train']} ({split['n_train']/split['n_total']*100:.1f}%)")
    print(f"验证集: {split['n_val']} ({split['n_val']/split['n_total']*100:.1f}%)")
    print(f"测试集: {split['n_test']} ({split['n_test']/split['n_total']*100:.1f}%)")

    if demandclean_dirty_train_path and os.path.exists(demandclean_dirty_train_path):
        dc_train = pd.read_csv(demandclean_dirty_train_path)
        our_train = split['dirty_train']
        if len(dc_train) == len(our_train):
            match = (dc_train.values == our_train.values).all()
            print(f"\n与 DemandClean dirty_train_60pct.csv 对比: {'完全一致 ✓' if match else '不一致 ✗'}")
        else:
            print(f"\n行数不一致: DemandClean={len(dc_train)}, ours={len(our_train)}")

    return split


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='验证 DemandClean 数据划分')
    parser.add_argument('--dirty', required=True, help='脏数据路径')
    parser.add_argument('--clean', required=True, help='干净数据路径')
    parser.add_argument('--dc_train', default=None, help='DemandClean 的 dirty_train_60pct.csv 路径（可选）')
    args = parser.parse_args()

    verify_split_consistency(args.dirty, args.clean, args.dc_train)
