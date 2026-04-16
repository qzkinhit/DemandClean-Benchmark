"""
Baran/Raha 运行脚本

Baran/Raha是基于迭代标注的数据清洗系统：
- Raha (SIGMOD 2019): 错误检测
- Baran (SIGMOD 2020): 错误修复

特点:
- 迭代式主动学习
- 需要人工标注 (Type 3)
- 真值成本 = LABELING_BUDGET

高维数据集处理:
- 当列数超过 MAX_COLUMNS_PER_SPLIT 时，自动拆分数据集
- 对每个子集分别运行 Baran/Raha 清洗
- 最后合并所有子集的清洗结果

用法:
    python run_raha_baran_base.py --dirty_path <脏数据路径> --clean_path <干净数据路径>

示例:
    python run_raha_baran_base.py \\
        --dirty_path ../../Data/hospital/dirty_index.csv \\
        --clean_path ../../Data/hospital/clean_index.csv \\
        --task_name hospital_baran \\
        --labeling_budget 20 \\
        --output_path ../../results/raha_baran/
"""

import os
import sys
import argparse
import time
import logging
import tempfile
import shutil
import multiprocessing
import gc
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

# 添加项目根目录到路径（使用 insert(0) 确保优先级高于 Methods/Baran_Raha/tools/ 同名包）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'Methods/Baran_Raha/'))

# 格式预清洗：与 DemandClean 使用相同的输入预处理，避免格式差异被误判为句法错误
from tools.csv_normalizer import normalize_dirty_format

# 高维数据集拆分阈值ca
# Baran/Raha的vicinity_models是O(n²)复杂度，超过此阈值需要拆分
MAX_COLUMNS_PER_SPLIT = 30

# 默认最大并行进程数（Raha的multiprocessing.Pool默认使用所有CPU核心，高维数据集容易OOM）
DEFAULT_MAX_WORKERS = 12


def limit_pool_workers(max_workers: int):
    """
    限制multiprocessing.Pool的默认并行进程数以控制内存使用。
    不修改官方代码，通过替换os.cpu_count()返回值实现。
    Pool()在processes=None时调用os.cpu_count()获取默认进程数。
    """
    _original_cpu_count = os.cpu_count

    def _limited_cpu_count():
        real_count = _original_cpu_count()
        return min(max_workers, real_count) if real_count else max_workers

    os.cpu_count = _limited_cpu_count


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """设置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def split_columns(columns: List[str], index_col: str, max_cols: int) -> List[List[str]]:
    """
    将列拆分成多个组，每组不超过max_cols列

    Args:
        columns: 所有列名列表
        index_col: 索引列名（每个组都需要包含）
        max_cols: 每组最大列数（不含索引列）

    Returns:
        列组列表，每组包含索引列和一部分特征列
    """
    # 排除索引列
    feature_cols = [c for c in columns if c != index_col]

    # 按max_cols拆分
    groups = []
    for i in range(0, len(feature_cols), max_cols):
        group_cols = [index_col] + feature_cols[i:i + max_cols]
        groups.append(group_cols)

    return groups


def clean_single_split(
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    columns: List[str],
    split_idx: int,
    task_name: str,
    temp_dir: str,
    cleaner,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗单个列子集

    Args:
        dirty_df: 完整脏数据
        clean_df: 完整干净数据
        columns: 要处理的列（包含索引列）
        split_idx: 拆分索引
        task_name: 任务名称
        temp_dir: 临时目录
        cleaner: BaranRahaWrapper实例
        logger: 日志记录器

    Returns:
        (清洗后的子集DataFrame, 清洗信息)
    """
    index_col = columns[0]  # 第一列是索引列

    # 提取子集
    dirty_subset = dirty_df[columns].copy()
    clean_subset = clean_df[columns].copy()

    # 保存临时文件
    split_name = f"{task_name}_split{split_idx}"
    dirty_path = os.path.join(temp_dir, f"{split_name}_dirty.csv")
    clean_path = os.path.join(temp_dir, f"{split_name}_clean.csv")
    output_path = os.path.join(temp_dir, f"{split_name}_cleaned.csv")

    dirty_subset.to_csv(dirty_path, index=False)
    clean_subset.to_csv(clean_path, index=False)

    logger.info(f"  Split {split_idx}: 处理 {len(columns)-1} 列 ({columns[1]}...{columns[-1]})")

    # 调用Baran/Raha清洗（抑制stderr避免dBoost的BrokenPipeError）
    try:
        import io
        import contextlib

        # 抑制stderr（dBoost的"Discarding"消息会触发BrokenPipeError）
        with contextlib.redirect_stderr(io.StringIO()):
            repaired_df, info = cleaner.clean(
                dirty_path=dirty_path,
                clean_path=clean_path,
                output_path=output_path,
                task_name=split_name,
                index_attribute=index_col
            )
        return repaired_df, info
    except Exception as e:
        logger.warning(f"  Split {split_idx} 清洗失败: {e}，使用原始脏数据")
        return dirty_subset, {'error': str(e)}


def merge_splits(
    base_df: pd.DataFrame,
    split_results: List[pd.DataFrame],
    column_groups: List[List[str]],
    index_col: str
) -> pd.DataFrame:
    """
    合并所有拆分的清洗结果

    Args:
        base_df: 原始数据（用于保留列顺序）
        split_results: 各子集清洗结果
        column_groups: 各子集的列组
        index_col: 索引列名

    Returns:
        合并后的完整DataFrame
    """
    # 以第一个split为基础
    result = split_results[0].set_index(index_col)

    # 合并其余splits
    for i, (split_df, cols) in enumerate(zip(split_results[1:], column_groups[1:]), 1):
        split_df = split_df.set_index(index_col)
        # 只取非索引列
        feature_cols = [c for c in cols if c != index_col]
        for col in feature_cols:
            if col in split_df.columns:
                result[col] = split_df[col]

    # 重置索引并恢复原始列顺序
    result = result.reset_index()
    original_cols = [c for c in base_df.columns if c in result.columns]
    result = result[original_cols]

    return result


def run_with_column_split(
    dirty_path: str,
    clean_path: str,
    output_file: str,
    task_name: str,
    index_attribute: str,
    cleaner,
    logger: logging.Logger,
    max_columns: int = MAX_COLUMNS_PER_SPLIT
) -> Tuple[pd.DataFrame, Dict]:
    """
    对高维数据集进行列拆分清洗

    Args:
        dirty_path: 脏数据路径
        clean_path: 干净数据路径
        output_file: 输出文件路径
        task_name: 任务名称
        index_attribute: 索引列名
        cleaner: BaranRahaWrapper实例
        logger: 日志记录器
        max_columns: 每个子集最大列数

    Returns:
        (合并后的清洗结果, 清洗信息汇总)
    """
    # 读取数据
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    n_cols = len(dirty_df.columns)
    logger.info(f"检测到高维数据集: {n_cols} 列，将拆分处理")

    # 拆分列
    column_groups = split_columns(
        list(dirty_df.columns),
        index_attribute,
        max_columns
    )
    n_splits = len(column_groups)
    logger.info(f"拆分为 {n_splits} 个子集，每组最多 {max_columns} 列")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix=f"baran_split_{task_name}_")
    logger.info(f"临时目录: {temp_dir}")

    try:
        split_results = []
        total_detected = 0
        total_corrected = 0

        for i, cols in enumerate(column_groups):
            repaired_df, info = clean_single_split(
                dirty_df, clean_df, cols, i, task_name,
                temp_dir, cleaner, logger
            )
            split_results.append(repaired_df)
            total_detected += info.get('detected_cells', 0)
            total_corrected += info.get('corrected_cells', 0)

            # 显式垃圾回收，释放 Baran/Raha 内部的大型对象（vicinity_models 等）
            gc.collect()

        # 合并结果
        logger.info("合并所有子集结果...")
        merged_df = merge_splits(dirty_df, split_results, column_groups, index_attribute)

        # 保存结果
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        logger.info(f"合并结果已保存: {output_file}")

        # 汇总信息
        combined_info = {
            'ground_truth_cost': cleaner.get_ground_truth_cost() * n_splits,
            'method': 'Baran_Raha_ColumnSplit',
            'type': 'data-oriented',
            'auto_level': 3,
            'n_splits': n_splits,
            'total_detected_cells': total_detected,
            'total_corrected_cells': total_corrected,
            'columns_per_split': max_columns
        }

        return merged_df, combined_info

    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Baran/Raha data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_raha_baran_base.py --dirty_path ../../Data/hospital/dirty.csv --clean_path ../../Data/hospital/clean.csv

参数说明:
  --labeling_budget: 标注预算，即需要人工标注的记录数（默认20）
  --max_columns: 每个拆分子集的最大列数（默认40）

注意:
  - 这是一个迭代式系统，真值成本 = labeling_budget
  - 如果提供clean_path，系统会自动使用真值进行标注
  - 对于高维数据集（>40列），会自动拆分处理
        """
    )

    # 数据路径参数
    parser.add_argument('--dirty_path', type=str, default='../data/breast_cancer/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--clean_path', type=str, default='../data/breast_cancer/clean_index.csv',
                        help='干净数据路径（用于自动标注和评估）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='breast_cancer_raha_baran',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/raha_baran/',
                        help='结果输出路径')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')

    # Baran/Raha参数
    parser.add_argument('--labeling_budget', type=int, default=20,
                        help='标注预算（默认20）')
    parser.add_argument('--classification_model', type=str, default='ABC',
                        choices=['ABC', 'DTC', 'GBC', 'GNB', 'KNC', 'SGDC', 'SVC'],
                        help='分类模型（默认ABC）')
    parser.add_argument('--min_correction_probability', type=float, default=0.0,
                        help='最小修正候选概率（默认0.0）')
    parser.add_argument('--min_correction_occurrence', type=int, default=2,
                        help='最小修正出现次数（默认2）')

    # 高维数据集参数
    parser.add_argument('--max_columns', type=int, default=MAX_COLUMNS_PER_SPLIT,
                        help=f'每个拆分子集的最大列数（默认{MAX_COLUMNS_PER_SPLIT}）')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'Raha并行进程数上限，控制内存使用（默认{DEFAULT_MAX_WORKERS}）')

    # 评估参数
    parser.add_argument('--label_column', type=str, default='style',
                        help='标签列名（用于下游任务评估）')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression', 'clustering'],
                        help='下游任务类型（默认classification）')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'lr'],
                        help='评估模型列表（默认rf lr）')
    parser.add_argument('--mse_attributes', type=str, nargs='*', default=[],
                        help='需要计算MSE的属性列表')

    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细信息')
    parser.add_argument('--use_split', action='store_true',
                        help='使用 DemandClean 对齐的 60/20/20 数据划分（seed=42）')

    args = parser.parse_args()

    # 限制Raha的并行进程数，防止OOM（在导入Baran/Raha之前执行）
    limit_pool_workers(args.max_workers)

    # 使用命令行参数的列拆分阈值
    max_columns = args.max_columns

    # 创建输出目录
    result_path = os.path.join(args.output_path, args.task_name)
    os.makedirs(result_path, exist_ok=True)

    # 设置日志
    logger = setup_logging(result_path, args.task_name)

    # 记录开始时间
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"运行开始时间: {start_datetime}")
    logger.info("=" * 60)
    logger.info("Baran/Raha 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"干净数据: {args.clean_path or '未提供'}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info(f"标注预算: {args.labeling_budget}")
    logger.info(f"列拆分阈值: {max_columns}")
    logger.info(f"并行进程数限制: {args.max_workers}")
    logger.info("-" * 60)

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': args.labeling_budget}

    # =========================================================================
    # 格式预清洗：统一 dirty CSV 的数值格式，与 DemandClean 输入保持一致
    # 作用：将 "1.0" → "1" 等整数浮点表示统一为整数格式
    # 目的：避免 RAHA 将格式差异（如 dirty "1.0" vs clean "1"）误判为句法错误
    # =========================================================================
    normalized_dirty_path = None
    actual_dirty_path = args.dirty_path  # 实际传给 Baran 的 dirty 路径

    try:
        dirty_raw = pd.read_csv(args.dirty_path, dtype=str, keep_default_na=False)
        dirty_normalized = normalize_dirty_format(dirty_raw, verbose=True)

        # 检查是否有变化
        n_changed = (dirty_raw != dirty_normalized).sum().sum()
        if n_changed > 0:
            # 写入临时归一化文件
            normalized_dirty_path = os.path.join(result_path, f"{args.task_name}_dirty_normalized.csv")
            dirty_normalized.to_csv(normalized_dirty_path, index=False)
            actual_dirty_path = normalized_dirty_path
            logger.info(f"[格式预清洗] 归一化 {n_changed} 个单元格，使用归一化后的 dirty: {normalized_dirty_path}")
        else:
            logger.info("[格式预清洗] 无需归一化，dirty 数据格式已统一")
    except Exception as e:
        logger.warning(f"[格式预清洗] 失败，使用原始 dirty: {e}")

    try:
        from Methods.Baran_Raha.baran_raha_wrapper import BaranRahaWrapper

        cleaner = BaranRahaWrapper(
            labeling_budget=args.labeling_budget,
            classification_model=args.classification_model,
            min_correction_candidate_probability=args.min_correction_probability,
            min_correction_occurrence=args.min_correction_occurrence,
            verbose=args.verbose
        )

        # 检查列数，决定是否需要拆分
        dirty_df = pd.read_csv(actual_dirty_path)
        n_cols = len(dirty_df.columns)

        if n_cols > max_columns + 1:  # +1 for index column
            # 高维数据集：使用列拆分策略
            logger.info(f"数据集列数 ({n_cols}) 超过阈值 ({max_columns})，启用列拆分模式")
            repaired_df, clean_info = run_with_column_split(
                dirty_path=actual_dirty_path,
                clean_path=args.clean_path,
                output_file=output_file,
                task_name=args.task_name,
                index_attribute=args.index_attribute,
                cleaner=cleaner,
                logger=logger,
                max_columns=max_columns
            )
        else:
            # 正常处理
            logger.info(f"数据集列数 ({n_cols}) 在阈值内，正常处理")
            repaired_df, clean_info = cleaner.clean(
                dirty_path=actual_dirty_path,
                clean_path=args.clean_path,
                output_path=output_file,
                task_name=args.task_name,
                index_attribute=args.index_attribute
            )
        success = True

    except ImportError as e:
        logger.error(f"依赖导入失败: {e}")
        logger.error("请确保Baran/Raha模块已正确安装")
        logger.error("需要安装: raha, mwparserfromhell, py7zr等")
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")

    if 'n_splits' in clean_info:
        logger.info(f"拆分数量: {clean_info['n_splits']}")
        logger.info(f"总检测单元格: {clean_info.get('total_detected_cells', 'N/A')}")
        logger.info(f"总修复单元格: {clean_info.get('total_corrected_cells', 'N/A')}")

    logger.info(f"标注预算: {clean_info.get('labeling_budget', args.labeling_budget)}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', args.labeling_budget)} (Type 3: 迭代交互)")

    # 调用统一测评模块
    if success and args.clean_path and os.path.exists(args.clean_path):
        logger.info("\n" + "=" * 60)
        logger.info("调用统一测评模块 getScoreML")
        logger.info("=" * 60)

        try:
            # 重新确保项目根目录在 sys.path 最前面（Baran/Raha 执行时会 insert Methods/Baran_Raha/）
            # 使用绝对路径确保正确匹配
            if PROJECT_ROOT not in sys.path:
                sys.path.insert(0, PROJECT_ROOT)
            else:
                # 如果已存在，移到最前面
                sys.path.remove(PROJECT_ROOT)
                sys.path.insert(0, PROJECT_ROOT)

            # 强制重新加载tools模块（避免缓存的Baran_Raha/tools）
            if 'tools' in sys.modules:
                del sys.modules['tools']
            if 'tools.getScoreML' in sys.modules:
                del sys.modules['tools.getScoreML']

            from tools.getScoreML import run_all_evaluation

            eval_results = run_all_evaluation(
                dirty_path=actual_dirty_path,  # 使用归一化后的 dirty，确保评测基准一致
                cleaned_path=output_file,
                clean_path=args.clean_path,
                output_path=result_path,
                task_name=args.task_name,
                label_column=args.label_column,
                task_type=args.task_type,
                models=args.models,
                method_type=3,  # Baran/Raha是Type 3迭代交互
                ground_truth_used=clean_info.get('ground_truth_cost', args.labeling_budget),
                index_attribute=args.index_attribute,
                mse_attributes=args.mse_attributes,
                verbose=args.verbose
            )

            # 合并结果
            clean_info.update(eval_results)

        except ImportError as e:
            logger.warning(f"警告: 无法导入getScoreML模块: {e}")
        except Exception as e:
            logger.error(f"统一测评出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not args.clean_path:
            logger.info("\n未提供干净数据路径，跳过统一测评")


    logger.info(f"\n结果已保存到: {result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
