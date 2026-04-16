"""
Lopster 运行脚本

Lopster是基于潜在空间表示学习的数据清洗方法（VLDB 2024）。

特点:
- 使用VAE学习数据表示进行错误检测和修复
- 需要clean.csv用于训练VAE模型
- 支持TensorFlow官方实现
- 自动预处理数据格式（处理带单位数值、百分号、缺失值标记等）

用法:
    python run_lopster_base.py --dataset adult --data_path ../../Data

示例:
    python run_lopster_base.py \
        --dataset adult \
        --data_path ../../Data \
        --task_name adult_lopster \
        --output_path ../../results/lopster/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd
import numpy as np
import shutil
import re

# 添加项目根目录到路径
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 通用数据预处理函数 - 为 Lopster VAE 准备纯数值格式数据
# ============================================================================

def clean_numeric_value(val):
    """
    清洗数值字段，处理带单位、百分号等情况

    支持格式:
    - "12.0 oz", "12.0 oz.", "12 ounce" → 12.0
    - "0.05", "0.09%", "5%" → 0.05, 0.09, 5.0
    - "empty", "null", "none", "na", "" → NaN
    - 纯数值 → 保持不变
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().strip()

    # 缺失值标记
    if val_str in ['empty', 'null', 'none', 'na', 'n/a', '', ' ', 'nan', '-']:
        return np.nan

    # 去除百分号
    val_str = val_str.replace('%', '')

    # 提取数值部分（支持小数和负数）
    match = re.search(r'(-?\d+\.?\d*)', val_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan

    return np.nan


def prepare_data_for_lopster(input_path, output_path, logger=None):
    """
    为 Lopster 准备数据：转换为纯数值格式

    处理内容:
    1. 移除 index 列（Lopster 不需要）
    2. 将缺失值标记统一转为 NaN
    3. 尝试将可转换的列转为数值类型
    4. 处理带单位的数值（如 "12.0 oz" → 12.0）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        logger: 日志记录器（可选）

    Returns:
        处理后的 DataFrame
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    df = pd.read_csv(input_path)
    original_shape = df.shape

    # 1. 移除 index 列
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        log(f"  - 移除 index 列")

    # 2. 统一缺失值标记
    missing_markers = ['empty', 'Empty', 'EMPTY', 'null', 'NULL', 'Null',
                       'none', 'None', 'NONE', 'na', 'NA', 'N/A', 'n/a',
                       '', ' ', 'nan', 'NaN', 'NAN', '-']
    df = df.replace(missing_markers, np.nan)

    # 3. 对每列尝试转换为数值
    converted_cols = []
    for col in df.columns:
        # 跳过已经是数值类型的列
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 检查是否有非数值内容（排除 NaN）
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        # 尝试直接转换
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            converted_cols.append(col)
            continue
        except:
            pass

        # 检查是否是带单位的数值列（更严格的判断）
        # 格式: "12.0 oz", "0.05%", "12 ounces" 等
        # 关键: 整个值应该主要是数字，后面可选地跟着单位
        sample_values = non_null.head(20).astype(str).tolist()

        # 严格匹配: 数字（可选小数）+ 可选空格 + 可选单位/百分号
        # 单位只能是短的（1-7个字符），避免匹配 "10 Barrel Brewing Company"
        unit_pattern = r'^\s*-?\d+\.?\d*\s*(%|oz\.?|ounces?|ml|l|kg|g|lb\.?|lbs?|in|ft|m|cm|mm)?\s*$'

        # 检查大部分样本值是否符合带单位数值的格式
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0

        # 只有当 80% 以上的值符合格式时才转换
        if match_ratio >= 0.8:
            df[col] = df[col].apply(clean_numeric_value)
            converted_cols.append(f"{col}(带单位)")

    if converted_cols:
        log(f"  - 转换为数值列: {converted_cols}")

    # 4. 保存
    df.to_csv(output_path, index=False)
    log(f"  - 保存到: {output_path}")
    log(f"  - 形状: {original_shape} → {df.shape}")

    return df


def prepare_lopster_dataset(data_path, dataset, result_path, logger=None):
    """
    为 Lopster 准备整个数据集（不修改原始 Data 目录！）

    生成文件到 result_path:
    - lopster_data/clean.csv: Lopster 训练用（无 index，编码后）
    - lopster_data/dirty01.csv: Lopster 清洗用（无 index，编码后）
    - clean_encoded.csv: 测评用（带 index，编码后）
    - dirty_encoded.csv: 测评用（带 index，编码后）

    Args:
        data_path: 数据集根目录（只读）
        dataset: 数据集名称
        result_path: 结果输出目录
        logger: 日志记录器

    Returns:
        dict: 包含预处理信息
            - dirty_src: 原始脏数据路径
            - clean_src: 原始干净数据路径
            - lopster_data_dir: Lopster 数据目录
            - dirty_encoded_path: 编码后脏数据路径（带 index，测评用）
            - clean_encoded_path: 编码后干净数据路径（带 index，测评用）
            - index_column: index 列数据（如果有）
            - n_rows: 数据行数
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    dataset_dir = os.path.join(data_path, dataset)

    # 确定源文件（只读）
    clean_src = None
    dirty_src = None

    # 优先使用 _index.csv 版本
    if os.path.exists(os.path.join(dataset_dir, 'clean_index.csv')):
        clean_src = os.path.join(dataset_dir, 'clean_index.csv')
    elif os.path.exists(os.path.join(dataset_dir, 'clean.csv')):
        clean_src = os.path.join(dataset_dir, 'clean.csv')

    if os.path.exists(os.path.join(dataset_dir, 'dirty_index.csv')):
        dirty_src = os.path.join(dataset_dir, 'dirty_index.csv')
    elif os.path.exists(os.path.join(dataset_dir, 'dirty.csv')):
        dirty_src = os.path.join(dataset_dir, 'dirty.csv')

    if not clean_src:
        raise FileNotFoundError(f"找不到干净数据文件: {dataset_dir}/clean_index.csv 或 clean.csv")
    if not dirty_src:
        raise FileNotFoundError(f"找不到脏数据文件: {dataset_dir}/dirty_index.csv 或 dirty.csv")

    # 创建 Lopster 数据目录（在 results 下）
    lopster_data_dir = os.path.join(result_path, 'lopster_data', dataset)
    os.makedirs(lopster_data_dir, exist_ok=True)

    # 读取原始数据，保存元信息
    original_dirty = pd.read_csv(dirty_src)
    original_clean = pd.read_csv(clean_src)

    index_column = original_dirty['index'].values if 'index' in original_dirty.columns else None

    log(f"\n准备 Lopster 数据格式（输出到 results 目录，不修改原始数据）...")
    log(f"  源文件: {os.path.basename(clean_src)}, {os.path.basename(dirty_src)}")
    log(f"  Lopster 数据目录: {lopster_data_dir}")

    # ========== 1. 生成 Lopster 用的数据（无 index，编码后） ==========
    lopster_clean = os.path.join(lopster_data_dir, 'clean.csv')
    lopster_dirty = os.path.join(lopster_data_dir, 'dirty01.csv')

    log(f"\n生成 Lopster 训练数据 (无 index):")
    prepare_data_for_lopster(clean_src, lopster_clean, logger)
    log(f"\n生成 Lopster 清洗数据 (无 index):")
    prepare_data_for_lopster(dirty_src, lopster_dirty, logger)

    # ========== 2. 生成测评用的数据（带 index，编码后） ==========
    clean_encoded_path = os.path.join(result_path, 'clean_encoded.csv')
    dirty_encoded_path = os.path.join(result_path, 'dirty_encoded.csv')

    log(f"\n生成测评用数据 (带 index，编码后):")

    # 严格的单位匹配模式（与 prepare_data_for_lopster 保持一致）
    unit_pattern = r'^\s*-?\d+\.?\d*\s*(%|oz\.?|ounces?|ml|l|kg|g|lb\.?|lbs?|in|ft|m|cm|mm)?\s*$'

    # 处理 clean 数据
    clean_df = original_clean.copy()
    missing_markers = ['empty', 'Empty', 'EMPTY', 'null', 'NULL', 'Null',
                       'none', 'None', 'NONE', 'na', 'NA', 'N/A', 'n/a',
                       '', ' ', 'nan', 'NaN', 'NAN', '-']
    clean_df = clean_df.replace(missing_markers, np.nan)

    # 对每列尝试转换为数值
    for col in clean_df.columns:
        if col == 'index':
            continue
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            continue
        non_null = clean_df[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='raise')
            continue
        except:
            pass
        # 使用严格的单位匹配
        sample_values = non_null.head(20).astype(str).tolist()
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0
        if match_ratio >= 0.8:
            clean_df[col] = clean_df[col].apply(clean_numeric_value)

    clean_df.to_csv(clean_encoded_path, index=False)
    log(f"  - clean_encoded.csv: {clean_encoded_path}")

    # 处理 dirty 数据
    dirty_df = original_dirty.copy()
    dirty_df = dirty_df.replace(missing_markers, np.nan)

    for col in dirty_df.columns:
        if col == 'index':
            continue
        if pd.api.types.is_numeric_dtype(dirty_df[col]):
            continue
        non_null = dirty_df[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            dirty_df[col] = pd.to_numeric(dirty_df[col], errors='raise')
            continue
        except:
            pass
        # 使用严格的单位匹配
        sample_values = non_null.head(20).astype(str).tolist()
        matches = sum(1 for v in sample_values if re.match(unit_pattern, str(v), re.IGNORECASE))
        match_ratio = matches / len(sample_values) if sample_values else 0
        if match_ratio >= 0.8:
            dirty_df[col] = dirty_df[col].apply(clean_numeric_value)

    dirty_df.to_csv(dirty_encoded_path, index=False)
    log(f"  - dirty_encoded.csv: {dirty_encoded_path}")

    log(f"\nLopster 数据准备完成!")

    return {
        'dirty_src': dirty_src,
        'clean_src': clean_src,
        'lopster_data_dir': lopster_data_dir,
        'dirty_encoded_path': dirty_encoded_path,
        'clean_encoded_path': clean_encoded_path,
        'index_column': index_column,
        'n_rows': len(original_dirty),
        'original_columns': list(original_dirty.columns)
    }


def postprocess_lopster_output(lopster_output_path, original_info, output_path, logger=None):
    """
    后处理 Lopster 输出，还原为与原始数据格式一致

    处理内容:
    1. 还原 index 列
    2. 还原 ID 列（beer_name 等，Lopster 不应修改这些列）
    3. 确保行数与原数据一致
    4. 保持列顺序与原数据一致

    Args:
        lopster_output_path: Lopster 输出文件路径
        original_info: prepare_lopster_dataset 返回的原始数据信息
        output_path: 最终输出路径
        logger: 日志记录器

    Returns:
        处理后的 DataFrame
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    log(f"\n后处理 Lopster 输出...")

    # 读取 Lopster 输出
    lopster_df = pd.read_csv(lopster_output_path)
    log(f"  Lopster 输出行数: {len(lopster_df)}")
    log(f"  原始数据行数: {original_info['n_rows']}")

    # 读取原始脏数据（用于还原 ID 列等）
    original_dirty = pd.read_csv(original_info['dirty_src'])

    # 检查行数是否一致
    if len(lopster_df) != original_info['n_rows']:
        log(f"  ⚠️ 警告: 行数不一致! Lopster={len(lopster_df)}, 原始={original_info['n_rows']}")
        # 如果少了行，用原始脏数据补齐
        if len(lopster_df) < original_info['n_rows']:
            # 移除 index 列以便比较
            if 'index' in original_dirty.columns:
                original_dirty_no_idx = original_dirty.drop(columns=['index'])
            else:
                original_dirty_no_idx = original_dirty
            # 补齐缺失的行（使用原始脏数据的最后几行）
            missing_rows = original_info['n_rows'] - len(lopster_df)
            log(f"  补齐 {missing_rows} 行缺失数据")
            extra_rows = original_dirty_no_idx.iloc[len(lopster_df):].copy()
            # 确保列对齐
            for col in lopster_df.columns:
                if col not in extra_rows.columns:
                    extra_rows[col] = np.nan
            extra_rows = extra_rows[lopster_df.columns]
            lopster_df = pd.concat([lopster_df, extra_rows], ignore_index=True)

    # ========== 还原 ID 列（Lopster 不应修改这些列） ==========
    # 读取数据集配置，获取 id_cols
    try:
        import json
        config_path = os.path.join(PROJECT_ROOT, 'Methods', 'Lopster', 'dataset_configuration.json')
        with open(config_path, 'r') as f:
            dataset_config = json.load(f)

        dataset_name = os.path.basename(os.path.dirname(original_info['dirty_src']))
        if dataset_name in dataset_config:
            id_cols = dataset_config[dataset_name].get('id_cols', [])
            date_cols = dataset_config[dataset_name].get('date_cols', [])
            restore_cols = id_cols + date_cols

            if restore_cols:
                log(f"  还原 ID/日期列: {restore_cols}")
                for col in restore_cols:
                    if col in original_dirty.columns and col in lopster_df.columns:
                        # 用原始数据覆盖 Lopster 输出（ID 列不应被修改）
                        lopster_df[col] = original_dirty[col].values[:len(lopster_df)]
    except Exception as e:
        log(f"  ⚠️ 无法读取数据集配置: {e}")

    # 还原 index 列
    if original_info['index_column'] is not None:
        if 'index' in lopster_df.columns:
            lopster_df['index'] = original_info['index_column'][:len(lopster_df)]
        else:
            lopster_df.insert(0, 'index', original_info['index_column'][:len(lopster_df)])
        log(f"  还原 index 列")

    # 重新排列列顺序，与原始数据一致
    original_columns = original_info['original_columns']
    final_columns = []
    for col in original_columns:
        if col in lopster_df.columns:
            final_columns.append(col)
    # 添加 lopster 可能新增的列
    for col in lopster_df.columns:
        if col not in final_columns:
            final_columns.append(col)

    lopster_df = lopster_df[final_columns]
    log(f"  最终列顺序: {final_columns}")

    # 保存
    lopster_df.to_csv(output_path, index=False)
    log(f"  保存到: {output_path}")
    log(f"  最终行数: {len(lopster_df)}")

    return lopster_df


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


def main():
    parser = argparse.ArgumentParser(
        description='Run Lopster data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_lopster_base.py --dataset adult --data_path ../../Data

注意:
  - Lopster需要clean.csv用于训练VAE模型
  - 需要TensorFlow环境
        """
    )

    # 数据路径参数
    parser.add_argument('--dataset', type=str, default='beers',
                        help='数据集名称（如adult, beers等）')
    parser.add_argument('--data_path', type=str, default='../../Data',
                        help='数据集根目录（默认../../Data）')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于评估，可选，默认自动推断）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='beers_lopster',
                        help='任务名称（默认使用数据集名称）')
    parser.add_argument('--output_path', type=str, default='../../results/lospter',
                        help='结果输出路径')

    # Lopster参数
    parser.add_argument('--latent_dim', type=int, default=120,
                        help='潜在空间维度（默认120）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（默认100）')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率（默认0.001）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批大小（默认256）')
    parser.add_argument('--K', type=int, default=12,
                        help='K参数（默认12）')
    parser.add_argument('--clean_ratio', type=float, default=1.0,
                        help='clean数据使用比例（0.01-1.0，默认1.0=100%%）')

    # 评估参数
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')
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

    # 自动设置任务名称
    if args.task_name is None:
        args.task_name = f"{args.dataset}_lopster"

    # 自动推断clean_path
    if args.clean_path is None:
        args.clean_path = os.path.join(args.data_path, args.dataset, 'clean.csv')

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
    logger.info("Lopster 数据清洗")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info(f"潜在维度: {args.latent_dim}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info("-" * 60)

    # 计算绝对路径
    original_cwd = os.getcwd()
    abs_data_path = os.path.join(original_cwd, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    abs_output_path = os.path.join(original_cwd, args.output_path) if not os.path.isabs(args.output_path) else args.output_path
    abs_result_path = os.path.join(abs_output_path, args.task_name)

    # 自动预处理数据（转换为 Lopster 需要的格式，输出到 results 目录）
    preprocess_info = None
    try:
        preprocess_info = prepare_lopster_dataset(abs_data_path, args.dataset, abs_result_path, logger)
    except Exception as e:
        logger.warning(f"数据预处理警告: {e}")
        logger.warning("将尝试使用原始数据文件...")
        import traceback
        traceback.print_exc()

    # 切换到Lopster目录执行（因为datasets.py中的配置文件路径是相对路径）
    lopster_dir = os.path.join(PROJECT_ROOT, 'Methods', 'Lopster')

    try:
        os.chdir(lopster_dir)
        # 将Lopster目录添加到路径
        if lopster_dir not in sys.path:
            sys.path.insert(0, lopster_dir)

        # 导入并执行Lopster
        from lopster_wrapper import LopsterWrapper

        # 创建清洗器
        cleaner = LopsterWrapper(
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            K=args.K,
            clean_ratio=args.clean_ratio,
            verbose=args.verbose
        )

        # 执行清洗 - 使用 results 目录下的预处理数据
        output_file = os.path.join(abs_result_path, f"{args.task_name}_cleaned.csv")

        # 确定 Lopster 数据路径
        if preprocess_info and 'lopster_data_dir' in preprocess_info:
            # 使用预处理后的数据目录（在 results 下）
            lopster_data_path = os.path.dirname(preprocess_info['lopster_data_dir'])  # lopster_data/
            logger.info(f"\n使用预处理数据: {preprocess_info['lopster_data_dir']}")
        else:
            # 回退到原始数据目录
            lopster_data_path = abs_data_path
            logger.info(f"\n使用原始数据: {lopster_data_path}")

        repaired_df, clean_info = cleaner.clean(
            dataset=args.dataset,
            path=lopster_data_path,
            output_path=output_file
        )
        success = True

    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
        success = False
        clean_info = {'ground_truth_cost': 0}
        repaired_df = None
    finally:
        os.chdir(original_cwd)

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', 0)} 行 (用于训练VAE)")

    # 后处理：添加 index 列
    if success and preprocess_info and os.path.exists(output_file):
        try:
            postprocess_lopster_output(output_file, preprocess_info, output_file, logger)
        except Exception as e:
            logger.warning(f"后处理警告: {e}")
            import traceback
            traceback.print_exc()

    # 调用统一测评模块（使用编码后的数据进行测评！）
    if success and preprocess_info:
        logger.info("\n" + "=" * 60)
        logger.info("调用统一测评模块 getScoreML")
        logger.info("=" * 60)
        logger.info("使用编码后的数据进行测评（确保格式一致）:")
        logger.info(f"  - dirty: {preprocess_info.get('dirty_encoded_path', 'N/A')}")
        logger.info(f"  - cleaned: {output_file}")
        logger.info(f"  - clean: {preprocess_info.get('clean_encoded_path', 'N/A')}")

        try:
            # 确保项目根目录在路径最前面（避免和Lopster目录下的utils.py冲突）
            if PROJECT_ROOT in sys.path:
                sys.path.remove(PROJECT_ROOT)
            sys.path.insert(0, PROJECT_ROOT)
            # 移除Lopster目录（如果在路径中）
            if lopster_dir in sys.path:
                sys.path.remove(lopster_dir)

            from tools.getScoreML import run_all_evaluation

            # 使用编码后的数据进行测评
            dirty_encoded_path = preprocess_info.get('dirty_encoded_path')
            clean_encoded_path = preprocess_info.get('clean_encoded_path')

            if dirty_encoded_path and clean_encoded_path and os.path.exists(dirty_encoded_path) and os.path.exists(clean_encoded_path):
                eval_results = run_all_evaluation(
                    dirty_path=dirty_encoded_path,
                    cleaned_path=output_file,
                    clean_path=clean_encoded_path,
                    output_path=abs_result_path,
                    task_name=args.task_name,
                    label_column=args.label_column,
                    task_type=args.task_type,
                    models=args.models,
                    method_type=2,  # Lopster是Type 2 - 需要训练数据
                    ground_truth_used=clean_info.get('ground_truth_cost', 0),
                    index_attribute=args.index_attribute,
                    mse_attributes=args.mse_attributes,
                    verbose=args.verbose
                )

                # 合并结果
                clean_info.update(eval_results)
            else:
                logger.warning("编码后的数据文件不存在，跳过测评")

        except ImportError as e:
            logger.warning(f"警告: 无法导入getScoreML模块: {e}")
        except Exception as e:
            logger.error(f"统一测评出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not success:
            logger.info("\n清洗失败，跳过测评")
        elif not preprocess_info:
            logger.info("\n预处理信息缺失，跳过测评")

    logger.info(f"\n结果已保存到: {abs_result_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
