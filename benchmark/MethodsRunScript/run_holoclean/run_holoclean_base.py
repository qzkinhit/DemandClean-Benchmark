"""
HoloClean 运行脚本

HoloClean是基于概率图模型的数据清洗系统（VLDB 2017）。

特点:
- 融合约束、统计和知识库信号
- 全自动执行
- 需要PostgreSQL数据库支持

高维/大规模数据集处理:
- 当列数超过 MAX_COLUMNS 时，自动列拆分
- 当行数超过 MAX_ROWS 时，自动行拆分
- 对每个子集分别运行 HoloClean 清洗
- 最后合并所有子集的清洗结果

用法:
    python run_holoclean_base.py --dirty_path <脏数据路径> --rule_path <约束文件>

示例:
    python run_holoclean_base.py \\
        --dirty_path ../../Data/hospital/dirty_index.csv \\
        --rule_path ../../Data/hospital/dc_rules_holoclean.txt \\
        --clean_path ../../Data/hospital/clean_index.csv \\
        --task_name hospital_holoclean \\
        --output_path ../../results/holoclean/
"""

import os
import sys
import argparse
import time
import logging
import tempfile
import shutil
import gc
import pandas as pd
from typing import List, Tuple, Dict, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../Methods/HoloClean/')

# 拆分阈值
MAX_COLUMNS_PER_SPLIT = 40  # 列拆分阈值
MAX_ROWS_PER_SPLIT = 5000   # 行拆分阈值

# HoloClean 内部空值表示
HOLOCLEAN_NULL_REPR = '_nan_'
# 标准空值表示
STANDARD_NULL_REPR = 'empty'


def normalize_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """将 HoloClean 的 _nan_ 空值表示替换为标准的 empty"""
    return df.replace(HOLOCLEAN_NULL_REPR, STANDARD_NULL_REPR)


try:
    from tools.get_T_table import transform_csv_file
    from tools.insert_null import inject_missing_values
    import Methods.HoloClean as holoclean
    from Methods.HoloClean.detect import NullDetector, ViolationDetector
    from Methods.HoloClean.repair.featurize import (
        InitAttrFeaturizer, OccurAttrFeaturizer,
        FreqFeaturizer, ConstraintFeaturizer
    )
    HAS_HOLOCLEAN = True
except ImportError as e:
    HAS_HOLOCLEAN = False
    IMPORT_ERROR = str(e)


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


def parse_holoclean_dc_rules(rules_path):
    """
    从rules.txt文件中解析HOLOCLEAN_DC section的规则

    Args:
        rules_path: 规则文件路径

    Returns:
        list: DC规则列表
    """
    dc_rules = []
    in_holoclean_section = False
    has_sections = False

    with open(rules_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查是否有section标记
    for line in lines:
        if line.strip().startswith('['):
            has_sections = True
            break

    if has_sections:
        # 有section，只提取HOLOCLEAN_DC部分
        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('[HOLOCLEAN_DC]'):
                in_holoclean_section = True
                continue
            elif line_stripped.startswith('[') and in_holoclean_section:
                # 遇到下一个section，停止
                in_holoclean_section = False
                continue

            if in_holoclean_section and line_stripped and not line_stripped.startswith('#'):
                dc_rules.append(line_stripped)
    else:
        # 没有section，假设所有非空非注释行都是DC规则
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                dc_rules.append(line_stripped)

    return dc_rules


def create_temp_dc_file(dc_rules, output_dir):
    """
    创建临时的DC规则文件供HoloClean使用

    Args:
        dc_rules: DC规则列表
        output_dir: 输出目录

    Returns:
        str: 临时文件路径
    """
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    temp_path = os.path.join(output_dir, 'temp_dc_rules.txt')

    with open(temp_path, 'w', encoding='utf-8') as f:
        for rule in dc_rules:
            f.write(rule + '\n')

    return temp_path


def filter_rules_for_columns(dc_rules: List[str], columns: List[str]) -> List[str]:
    """
    过滤DC规则，只保留涉及当前列子集的规则

    Args:
        dc_rules: 所有DC规则
        columns: 当前子集包含的列名

    Returns:
        适用于当前子集的规则列表
    """
    filtered = []
    col_set = set(columns)
    for rule in dc_rules:
        # DC规则格式: t1&t2&EQ(t1.col1,t2.col1)&...
        # 提取规则中涉及的列名
        rule_cols = set()
        for part in rule.split('&'):
            part = part.strip()
            if '(' in part and '.' in part:
                # 提取 t1.colname 或 t2.colname 中的 colname
                import re
                col_refs = re.findall(r't\d+\.(\w+)', part)
                rule_cols.update(col_refs)
        # 只保留所有涉及列都在当前子集中的规则
        if rule_cols and rule_cols.issubset(col_set):
            filtered.append(rule)
    return filtered


def _cleanup_holoclean_db(db_user: str, db_name: str, logger: logging.Logger,
                          db_host: str = '127.0.0.1', db_port: int = 5432,
                          db_pwd: str = 'abcd1234'):
    """
    清理 HoloClean 在数据库中残留的辅助表。
    HoloClean 的辅助表（cell_distr, dk_cells 等）使用固定表名（无任务名前缀），
    多个子集顺序执行时会产生冲突。
    """
    # HoloClean 辅助表的固定名称
    aux_tables = [
        'c_cells', 'dk_cells', 'cell_domain', 'pos_values',
        'cell_distr', 'inf_values_idx', 'inf_values_dom',
        'grounding', 'Feature', 'pos_values_idx'
    ]

    try:
        import psycopg2
        conn = psycopg2.connect(
            user=db_user, password=db_pwd,
            dbname=db_name, host=db_host, port=db_port
        )
        conn.autocommit = True
        cur = conn.cursor()
        for table in aux_tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"    数据库清理警告（不影响后续执行）: {e}")


def run_holoclean_on_subset(
    dirty_path: str,
    clean_path: str,
    output_path: str,
    task_name: str,
    index_attribute: str,
    dc_rules: List[str],
    args,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    在单个数据子集上运行 HoloClean

    Args:
        dirty_path: 子集脏数据路径
        clean_path: 子集干净数据路径
        output_path: 输出文件路径
        task_name: 唯一任务名称（用于数据库表名）
        index_attribute: 索引列名
        dc_rules: 适用于该子集的DC规则
        args: 命令行参数
        logger: 日志记录器

    Returns:
        修复后的 DataFrame，失败时返回 None
    """
    from tools.get_T_table import transform_csv_file
    from tools.insert_null import inject_missing_values

    temp_dir = tempfile.mkdtemp(prefix=f"holo_{task_name}_")

    try:
        # 清理数据库残留的辅助表（HoloClean 辅助表用固定名，多次运行会冲突）
        _cleanup_holoclean_db(args.db_user, args.db_name, logger)

        # 预处理：统一缺失值
        ori_dirty = os.path.join(temp_dir, "dirty_ori.csv")
        inject_missing_values(
            csv_file=dirty_path, output_file=ori_dirty,
            missing_value_in_ori_data='empty',
            missing_value_representation='', attributes_error_ratio=None
        )
        ori_clean = os.path.join(temp_dir, "clean_ori.csv")
        inject_missing_values(
            csv_file=clean_path, output_file=ori_clean,
            missing_value_in_ori_data='empty',
            missing_value_representation='', attributes_error_ratio=None
        )

        # 移除索引列
        dirty_data = pd.read_csv(ori_dirty)
        if index_attribute in dirty_data.columns:
            dirty_no_idx = dirty_data.drop(columns=[index_attribute])
        else:
            dirty_no_idx = dirty_data

        dirty_holo_path = os.path.join(temp_dir, "dirty_holo.csv")
        dirty_no_idx.to_csv(dirty_holo_path, index=False, encoding='utf-8')

        # 转换clean数据格式
        clean_holo_path = os.path.join(temp_dir, "clean_holo.csv")
        transform_csv_file(ori_clean, clean_holo_path)

        clean_data = pd.read_csv(clean_path)
        clean_data_attributes = clean_data.columns.tolist()

        # 初始化 HoloClean
        hc = holoclean.HoloClean(
            db_user=args.db_user, db_name=args.db_name,
            domain_thresh_1=0, domain_thresh_2=0,
            weak_label_thresh=args.weak_label_thresh,
            max_domain=10000, cor_strength=0.6, nb_cor_strength=0.8,
            epochs=args.epochs, weight_decay=0.01,
            learning_rate=args.learning_rate, threads=args.threads,
            batch_size=1, verbose=False,
            timeout=3 * 60000, feature_norm=False,
            weight_norm=False, print_fw=True
        ).session

        # 加载数据
        safe_name = task_name.replace('-', '_').replace('.', '_')
        if safe_name[0].isdigit():
            safe_name = 't_' + safe_name
        hc.load_data(safe_name, dirty_holo_path)

        # 加载规则
        dc_file_path = None
        if dc_rules:
            dc_file_path = os.path.join(temp_dir, "dc_rules.txt")
            with open(dc_file_path, 'w') as f:
                for rule in dc_rules:
                    f.write(rule + '\n')
            hc.load_dcs(dc_file_path)
            hc.ds.set_constraints(hc.get_dcs())

        # 检测错误
        detectors = [NullDetector()]
        if dc_file_path:
            detectors.append(ViolationDetector())

        try:
            hc.detect_errors(detectors)
        except (ValueError, Exception) as e:
            if "Wrong number of items" in str(e) or "empty" in str(e).lower():
                logger.info(f"    子集无错误，返回原始数据")
                return dirty_data
            raise

        # 设置域、特征提取、修复
        hc.setup_domain()
        featurizers = [InitAttrFeaturizer(), OccurAttrFeaturizer(), FreqFeaturizer()]
        if dc_file_path:
            featurizers.append(ConstraintFeaturizer())
        hc.repair_errors(featurizers)

        # 获取修复结果
        try:
            hc.evaluate(
                fpath=clean_holo_path, tid_col='tid',
                attr_col='attribute', val_col='correct_val',
                output_csv_path=output_path,
                attrubte_list=clean_data_attributes,
                clean_data=clean_data, index_attribute=index_attribute
            )
            return pd.read_csv(output_path)
        except Exception:
            # 降级：从数据库查询
            query = "SELECT * FROM {} ORDER BY _tid_".format(
                hc.ds.raw_data.name + '_repaired')
            data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
            repaired_df = pd.DataFrame(data, columns=columns)
            if '_tid_' in repaired_df.columns:
                repaired_df.drop(columns=['_tid_'], inplace=True)
            if clean_data is not None and index_attribute in clean_data.columns:
                if len(clean_data) == len(repaired_df):
                    repaired_df[index_attribute] = clean_data[index_attribute].values
            repaired_df.to_csv(output_path, index=False, encoding='utf-8')
            return repaired_df

    except Exception as e:
        err_msg = str(e).lower()
        # 当没有需要修复的单元格时，HoloClean不会创建_vid_列或cell_distr表
        # 这种情况下返回原始脏数据（因为实际上没有错误需要修复）
        if '_vid_' in err_msg or 'cell_distr' in err_msg or 'inferring on 0' in err_msg:
            logger.info(f"    子集无需修复（无错误单元格），返回原始数据")
            return pd.read_csv(dirty_path)  # 返回原始脏数据
        logger.warning(f"    子集清洗失败: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_with_column_split(
    dirty_path: str, clean_path: str, output_file: str,
    task_name: str, index_attribute: str,
    dc_rules: List[str], args, logger: logging.Logger,
    max_columns: int
) -> Tuple[pd.DataFrame, Dict]:
    """对高维数据集进行列拆分清洗"""
    dirty_df = pd.read_csv(dirty_path)
    n_cols = len([c for c in dirty_df.columns if c != index_attribute])
    logger.info(f"列拆分模式: {n_cols} 特征列，拆分为每组最多 {max_columns} 列")

    # 拆分列
    feature_cols = [c for c in dirty_df.columns if c != index_attribute]
    column_groups = []
    for i in range(0, len(feature_cols), max_columns):
        group = [index_attribute] + feature_cols[i:i + max_columns]
        column_groups.append(group)

    n_splits = len(column_groups)
    logger.info(f"拆分为 {n_splits} 个子集")

    temp_dir = tempfile.mkdtemp(prefix=f"holo_colsplit_{task_name}_")
    split_results = []

    try:
        for i, cols in enumerate(column_groups):
            logger.info(f"  列子集 {i}/{n_splits}: {len(cols)-1} 列 ({cols[1]}...{cols[-1]})")

            # 保存子集
            split_dirty = os.path.join(temp_dir, f"split{i}_dirty.csv")
            split_clean = os.path.join(temp_dir, f"split{i}_clean.csv")
            split_output = os.path.join(temp_dir, f"split{i}_cleaned.csv")

            dirty_df[cols].to_csv(split_dirty, index=False)
            pd.read_csv(clean_path)[cols].to_csv(split_clean, index=False)

            # 过滤适用于当前列子集的规则
            sub_rules = filter_rules_for_columns(dc_rules, cols)

            split_name = f"{task_name}_cs{i}"
            result = run_holoclean_on_subset(
                split_dirty, split_clean, split_output, split_name,
                index_attribute, sub_rules, args, logger
            )

            if result is not None:
                split_results.append((i, result))
                logger.info(f"  列子集 {i} 完成")
            else:
                # 失败时用脏数据
                split_results.append((i, dirty_df[cols].copy()))
                logger.warning(f"  列子集 {i} 失败，使用原始脏数据")

            gc.collect()

        # 合并列子集：按索引列 join
        logger.info("合并所有列子集...")
        merged = split_results[0][1].set_index(index_attribute)
        for idx, split_df in split_results[1:]:
            split_df = split_df.set_index(index_attribute)
            for col in split_df.columns:
                if col != index_attribute:
                    merged[col] = split_df[col]
        merged = merged.reset_index()
        # 恢复原始列顺序
        original_cols = [c for c in dirty_df.columns if c in merged.columns]
        merged = merged[original_cols]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 统一空值表示：将 _nan_ 替换为 empty
    merged = normalize_null_values(merged)
    merged.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"合并结果已保存: {output_file}")

    return merged, {
        'ground_truth_cost': 0,
        'split_mode': 'column',
        'n_splits': n_splits
    }


def run_with_row_split(
    dirty_path: str, clean_path: str, output_file: str,
    task_name: str, index_attribute: str,
    dc_rules: List[str], args, logger: logging.Logger,
    max_rows: int
) -> Tuple[pd.DataFrame, Dict]:
    """对大规模数据集进行行拆分清洗"""
    dirty_df = pd.read_csv(dirty_path)
    n_rows = len(dirty_df)
    logger.info(f"行拆分模式: {n_rows} 行，拆分为每组最多 {max_rows} 行")

    clean_df = pd.read_csv(clean_path)

    # 拆分行
    row_groups = []
    for start in range(0, n_rows, max_rows):
        end = min(start + max_rows, n_rows)
        row_groups.append((start, end))

    n_splits = len(row_groups)
    logger.info(f"拆分为 {n_splits} 个子集")

    temp_dir = tempfile.mkdtemp(prefix=f"holo_rowsplit_{task_name}_")
    split_results = []

    try:
        for i, (start, end) in enumerate(row_groups):
            logger.info(f"  行子集 {i}/{n_splits}: 行 {start}~{end} ({end-start} 行)")

            # 保存子集
            split_dirty = os.path.join(temp_dir, f"split{i}_dirty.csv")
            split_clean = os.path.join(temp_dir, f"split{i}_clean.csv")
            split_output = os.path.join(temp_dir, f"split{i}_cleaned.csv")

            dirty_df.iloc[start:end].to_csv(split_dirty, index=False)
            clean_df.iloc[start:end].to_csv(split_clean, index=False)

            split_name = f"{task_name}_rs{i}"
            result = run_holoclean_on_subset(
                split_dirty, split_clean, split_output, split_name,
                index_attribute, dc_rules, args, logger
            )

            if result is not None:
                split_results.append(result)
                logger.info(f"  行子集 {i} 完成")
            else:
                split_results.append(dirty_df.iloc[start:end].copy())
                logger.warning(f"  行子集 {i} 失败，使用原始脏数据")

            gc.collect()

        # 合并行子集：直接 concat
        logger.info("合并所有行子集...")
        merged = pd.concat(split_results, ignore_index=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 统一空值表示：将 _nan_ 替换为 empty
    merged = normalize_null_values(merged)
    merged.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"合并结果已保存: {output_file}")

    return merged, {
        'ground_truth_cost': 0,
        'split_mode': 'row',
        'n_splits': n_splits
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run HoloClean data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python holoclean_run.py --dirty_path ../../Data/hospital/dirty.csv --rule_path ../../Data/hospital/rules.txt

注意:
  - 需要PostgreSQL数据库支持
  - 约束文件格式为Denial Constraints
  - 全自动执行，无需人工参与
        """
    )

    # 数据路径参数
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--rule_path', type=str, default=None,
                        help='约束文件路径（Denial Constraints）')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于评估，可选）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='beers_holoclean',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/holoclean/',
                        help='结果输出路径')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')

    # 数据库参数
    parser.add_argument('--db_user', type=str, default='holocleanuser',
                        help='数据库用户名')
    parser.add_argument('--db_name', type=str, default='holo',
                        help='数据库名称')

    # HoloClean参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数（默认10）')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率（默认0.001）')
    parser.add_argument('--threads', type=int, default=1,
                        help='线程数（默认1）')
    parser.add_argument('--weak_label_thresh', type=float, default=0.99,
                        help='弱标签阈值（默认0.99）')

    # 拆分参数
    parser.add_argument('--max_columns', type=int, default=MAX_COLUMNS_PER_SPLIT,
                        help=f'列拆分阈值（默认{MAX_COLUMNS_PER_SPLIT}）')
    parser.add_argument('--max_rows', type=int, default=MAX_ROWS_PER_SPLIT,
                        help=f'行拆分阈值（默认{MAX_ROWS_PER_SPLIT}）')

    # 评估参数
    parser.add_argument('--label_column', type=str, default='style',
                        help='标签列名（用于下游任务评估）')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression', 'clustering'],
                        help='下游任务类型（默认classification）')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'lr'],
                        help='评估模型列表（默认rf lr）')

    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细信息')
    parser.add_argument('--use_split', action='store_true',
                        help='使用 DemandClean 对齐的 60/20/20 数据划分（seed=42）')

    args = parser.parse_args()

    # 创建输出目录
    result_path = os.path.join(args.output_path, args.task_name)
    os.makedirs(result_path, exist_ok=True)

    # 设置日志
    logger = setup_logging(result_path, args.task_name)

    # 检查依赖
    if not HAS_HOLOCLEAN:
        logger.error(f"HoloClean模块导入失败: {IMPORT_ERROR}")
        logger.error("请确保HoloClean模块已正确安装，并配置好PostgreSQL数据库")
        return

    # 记录开始时间
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"运行开始时间: {start_datetime}")
    logger.info("=" * 60)
    logger.info("HoloClean 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"约束文件: {args.rule_path or '未提供'}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info(f"数据库: {args.db_name}")
    logger.info(f"列拆分阈值: {args.max_columns}")
    logger.info(f"行拆分阈值: {args.max_rows}")
    logger.info("-" * 60)

    # 预先解析DC规则（拆分模式也需要）
    dc_rules = []
    if args.rule_path and os.path.exists(args.rule_path):
        dc_rules = parse_holoclean_dc_rules(args.rule_path)
        logger.info(f"加载 {len(dc_rules)} 条DC规则")

    # 检查数据集规模，决定是否需要拆分
    dirty_df = pd.read_csv(args.dirty_path)
    n_rows, n_cols = dirty_df.shape
    feature_cols = n_cols - 1  # 去掉索引列
    logger.info(f"数据集规模: {n_rows} 行 × {n_cols} 列 ({n_rows * feature_cols} cells)")

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': 0}

    need_col_split = feature_cols > args.max_columns
    need_row_split = n_rows > args.max_rows

    if need_col_split or need_row_split:
        # 拆分模式
        try:
            if need_col_split:
                logger.info(f"列数 ({feature_cols}) 超过阈值 ({args.max_columns})，启用列拆分")
                merged_df, clean_info = run_with_column_split(
                    args.dirty_path, args.clean_path, output_file,
                    args.task_name, args.index_attribute,
                    dc_rules, args, logger, args.max_columns
                )
            else:
                logger.info(f"行数 ({n_rows}) 超过阈值 ({args.max_rows})，启用行拆分")
                merged_df, clean_info = run_with_row_split(
                    args.dirty_path, args.clean_path, output_file,
                    args.task_name, args.index_attribute,
                    dc_rules, args, logger, args.max_rows
                )
            success = True
        except Exception as e:
            logger.error(f"拆分模式执行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 正常模式（原有逻辑）
        logger.info(f"数据集规模在阈值内，正常处理")

        try:
            # Step 1: 统一缺失值标记为 'empty'
            logger.info("Step 1: 统一缺失值标记...")
            ori_empty_dirty_path = os.path.join(result_path, "dirty_ori_empty.csv")
            inject_missing_values(
                csv_file=args.dirty_path,
                output_file=ori_empty_dirty_path,
                missing_value_in_ori_data='empty',
                missing_value_representation='',
                attributes_error_ratio=None
            )

            if args.clean_path and os.path.exists(args.clean_path):
                ori_empty_clean_path = os.path.join(result_path, "clean_ori_empty.csv")
                inject_missing_values(
                    csv_file=args.clean_path,
                    output_file=ori_empty_clean_path,
                    missing_value_in_ori_data='empty',
                    missing_value_representation='',
                    attributes_error_ratio=None
                )
            else:
                ori_empty_clean_path = None

            # Step 2: 读取脏数据并移除索引列
            logger.info("Step 2: 移除索引列...")
            dirty_data = pd.read_csv(ori_empty_dirty_path)
            if args.index_attribute in dirty_data.columns:
                dirty_data_no_index = dirty_data.drop(columns=[args.index_attribute])
            else:
                dirty_data_no_index = dirty_data

            dirty_holoclean_path = os.path.join(result_path, "dirty_holoclean.csv")
            dirty_data_no_index.to_csv(dirty_holoclean_path, index=False, encoding='utf-8')

            # Step 3: 转换clean数据为HoloClean格式 (tid, attribute, correct_val)
            clean_holoclean_path = None
            if ori_empty_clean_path:
                logger.info("Step 3: 转换clean数据为HoloClean格式...")
                clean_holoclean_path = os.path.join(result_path, "clean_holoclean.csv")
                transform_csv_file(ori_empty_clean_path, clean_holoclean_path)

            clean_data = pd.read_csv(args.clean_path) if args.clean_path else None
            clean_data_attributes = clean_data.columns.tolist() if clean_data is not None else []

            # Step 4: 初始化HoloClean
            logger.info("Step 4: 初始化HoloClean...")
            hc = holoclean.HoloClean(
                db_user=args.db_user,
                db_name=args.db_name,
                domain_thresh_1=0,
                domain_thresh_2=0,
                weak_label_thresh=args.weak_label_thresh,
                max_domain=10000,
                cor_strength=0.6,
                nb_cor_strength=0.8,
                epochs=args.epochs,
                weight_decay=0.01,
                learning_rate=args.learning_rate,
                threads=args.threads,
                batch_size=1,
                verbose=args.verbose,
                timeout=3 * 60000,
                feature_norm=False,
                weight_norm=False,
                print_fw=True
            ).session

            # Step 5: 加载数据
            logger.info("Step 5: 加载数据到HoloClean...")
            safe_task_name = args.task_name.replace('-', '_').replace('.', '_')
            if safe_task_name[0].isdigit():
                safe_task_name = 't_' + safe_task_name
            hc.load_data(safe_task_name, dirty_holoclean_path)

            # Step 6: 加载约束
            dc_file_path = None
            if args.rule_path and os.path.exists(args.rule_path):
                logger.info("Step 6: 加载约束规则...")
                local_dc_rules = parse_holoclean_dc_rules(args.rule_path)
                if local_dc_rules:
                    dc_file_path = create_temp_dc_file(local_dc_rules, result_path)
                    logger.info(f"解析到 {len(local_dc_rules)} 条DC规则")
                    hc.load_dcs(dc_file_path)
                    hc.ds.set_constraints(hc.get_dcs())
                else:
                    logger.warning("警告: 规则文件中没有找到HOLOCLEAN_DC规则")

            # Step 7: 检测错误
            logger.info("Step 7: 检测错误...")
            detectors = [NullDetector()]
            if dc_file_path:
                detectors.append(ViolationDetector())

            no_errors_detected = False
            try:
                hc.detect_errors(detectors)
            except (ValueError, Exception) as detect_err:
                if "Wrong number of items" in str(detect_err) or "empty" in str(detect_err).lower():
                    logger.warning(f"未检测到错误（数据可能已经干净）: {detect_err}")
                    no_errors_detected = True
                else:
                    raise

            if no_errors_detected:
                logger.info("数据无需修复，直接输出原始数据...")
                output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
                dirty_data.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"已保存: {output_file}")
                success = True
                clean_info = {
                    'ground_truth_cost': 0,
                    'detectors': len(detectors),
                    'featurizers': 0,
                    'no_errors_detected': True
                }
            else:
                # Step 8-10: 设置域、特征提取、修复
                numerical_error = False
                try:
                    logger.info("Step 8: 设置域...")
                    hc.setup_domain()

                    logger.info("Step 9: 特征提取...")
                    featurizers = [
                        InitAttrFeaturizer(),
                        OccurAttrFeaturizer(),
                        FreqFeaturizer()
                    ]
                    if dc_file_path:
                        featurizers.append(ConstraintFeaturizer())

                    logger.info("Step 10: 修复错误...")
                    hc.repair_errors(featurizers)

                except ZeroDivisionError as zde:
                    logger.warning(f"数值计算错误（高维数据限制）: {zde}")
                    logger.info("HoloClean无法处理此数据集，输出原始数据作为结果...")
                    numerical_error = True

                # Step 11: 保存修复结果
                logger.info("Step 11: 保存修复结果...")
                output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")

                if numerical_error:
                    dirty_data.to_csv(output_file, index=False, encoding='utf-8')
                    logger.info(f"已保存（原始数据）: {output_file}")
                    success = True
                    clean_info = {
                        'ground_truth_cost': 0,
                        'detectors': len(detectors),
                        'featurizers': 0,
                        'numerical_error': True
                    }
                elif clean_holoclean_path and clean_data is not None:
                    try:
                        hc.evaluate(
                            fpath=clean_holoclean_path,
                            tid_col='tid',
                            attr_col='attribute',
                            val_col='correct_val',
                            output_csv_path=output_file,
                            attrubte_list=clean_data_attributes,
                            clean_data=clean_data,
                            index_attribute=args.index_attribute
                        )
                        logger.info(f"修复后数据已保存: {output_file}")
                    except (AttributeError, Exception) as eval_err:
                        logger.warning(f"evaluate失败: {eval_err}")
                        logger.info("尝试使用替代方法导出修复数据...")
                        try:
                            query = "SELECT * FROM {} ORDER BY _tid_".format(
                                hc.ds.raw_data.name + '_repaired')
                            data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
                            repaired_df = pd.DataFrame(data, columns=columns)
                            if '_tid_' in repaired_df.columns:
                                repaired_df.drop(columns=['_tid_'], inplace=True)
                            if clean_data is not None and args.index_attribute in clean_data.columns:
                                if len(clean_data) == len(repaired_df):
                                    repaired_df[args.index_attribute] = clean_data[args.index_attribute].values
                            if clean_data_attributes:
                                attr_list = [c for c in clean_data_attributes if c != args.index_attribute]
                                missing_cols = [c for c in attr_list if c not in repaired_df.columns]
                                for col in missing_cols:
                                    repaired_df[col] = ''
                                ordered_cols = [args.index_attribute] + attr_list
                                repaired_df = repaired_df[ordered_cols]
                            # 统一空值表示：将 _nan_ 替换为 empty
                            repaired_df = normalize_null_values(repaired_df)
                            repaired_df.to_csv(output_file, index=False, encoding='utf-8')
                            logger.info(f"修复后数据已保存（替代方式）: {output_file}")
                        except Exception as fallback_err:
                            logger.error(f"替代方式也失败: {fallback_err}")
                            import traceback
                            traceback.print_exc()
                            output_file = None
                else:
                    try:
                        query = "SELECT * FROM {} ORDER BY _tid_".format(
                            hc.ds.raw_data.name + '_repaired')
                        data, columns = hc.ds.engine.execute_query_with_attribute_list(query)
                        repaired_df = pd.DataFrame(data, columns=columns)
                        if '_tid_' in repaired_df.columns:
                            repaired_df.drop(columns=['_tid_'], inplace=True)
                        if args.index_attribute in dirty_data.columns:
                            repaired_df[args.index_attribute] = dirty_data[args.index_attribute].values
                        # 统一空值表示：将 _nan_ 替换为 empty
                        repaired_df = normalize_null_values(repaired_df)
                        repaired_df.to_csv(output_file, index=False)
                        logger.info(f"修复后数据已保存: {output_file}")
                    except Exception as save_err:
                        logger.warning(f"警告: 无法保存修复数据: {save_err}")
                        output_file = None

                if not numerical_error:
                    success = True
                    clean_info = {
                        'ground_truth_cost': 0,
                        'detectors': len(detectors),
                        'featurizers': len(featurizers)
                    }

        except Exception as e:
            logger.error(f"执行出错: {e}")
            import traceback
            traceback.print_exc()
            success = False
            clean_info = {'ground_truth_cost': 0}
            output_file = None

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', 0)} (全自动)")

    # 调用统一测评模块
    if success and output_file and args.clean_path and os.path.exists(args.clean_path):
        logger.info("\n" + "=" * 60)
        logger.info("调用统一测评模块 getScoreML")
        logger.info("=" * 60)

        try:
            from tools.getScoreML import run_all_evaluation

            eval_results = run_all_evaluation(
                dirty_path=args.dirty_path,
                cleaned_path=output_file,
                clean_path=args.clean_path,
                output_path=result_path,
                task_name=args.task_name,
                label_column=args.label_column,
                task_type=args.task_type,
                models=args.models,
                method_type=1,  # HoloClean是全自动Type 1
                ground_truth_used=clean_info.get('ground_truth_cost', 0),
                index_attribute=args.index_attribute,
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
