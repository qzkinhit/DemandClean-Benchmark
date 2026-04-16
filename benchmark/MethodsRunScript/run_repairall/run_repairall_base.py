"""
RepairAll 运行脚本

RepairAll是完美修复baseline，直接使用干净数据替换脏数据。

特点:
- 直接返回干净数据（100%正确修复）
- Type 2方法（需要真值）
- 用于建立性能上界

用法:
    python run_repairall_base.py --dirty_path <脏数据路径> --clean_path <干净数据路径>

示例:
    python run_repairall_base.py \\
        --dirty_path ../../Data/beers/dirty.csv \\
        --clean_path ../../Data/beers/clean.csv \\
        --task_name beers_repairall \\
        --output_path ../../results/repairall/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """设置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


from Methods.RepairAll.repairall_wrapper import RepairAllWrapper


def main():
    parser = argparse.ArgumentParser(
        description='Run RepairAll baseline (perfect repair using ground truth).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_repairall_base.py --dirty_path ../../Data/beers/dirty.csv --clean_path ../../Data/beers/clean.csv

说明:
  - RepairAll直接使用干净数据替换脏数据
  - 相当于100%正确修复所有错误
  - 用于建立性能上界baseline
        """
    )

    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于修复和评估）')
    parser.add_argument('--task_name', type=str, default='beers_repairall',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/repairall/',
                        help='结果输出路径')
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
    parser.add_argument('--exclude_columns', type=str, nargs='*', default=[],
                        help='需要排除的非特征列（如id, datetime等）')
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

    # 记录开始时间
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"运行开始时间: {start_datetime}")
    logger.info("=" * 60)
    logger.info("RepairAll Baseline (完美修复)")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"干净数据: {args.clean_path}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info("-" * 60)

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': 0}

    try:
        cleaner = RepairAllWrapper(verbose=args.verbose)
        repaired_df, clean_info = cleaner.clean(
            dirty_path=args.dirty_path,
            clean_path=args.clean_path,
            output_path=output_file,
            index_attribute=args.index_attribute
        )
        success = True
        logger.info(f"输出数据已保存: {output_file}")
        logger.info(f"原始行数: {clean_info.get('original_rows', 'N/A')}")
        logger.info(f"修复单元格数: {clean_info.get('repaired_cells', 'N/A')}")

    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', 0)}")

    # 调用统一测评模块
    if success and os.path.exists(args.clean_path):
        logger.info("")
        logger.info("=" * 60)
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
                method_type=2,  # 需要真值
                ground_truth_used=clean_info.get('ground_truth_cost', 0),
                index_attribute=args.index_attribute,
                exclude_columns=args.exclude_columns if args.exclude_columns else None,
                mse_attributes=args.mse_attributes,
                verbose=args.verbose
            )
            clean_info.update(eval_results)

        except ImportError as e:
            logger.warning(f"无法导入getScoreML模块: {e}")
        except Exception as e:
            logger.error(f"统一测评出错: {e}")
            import traceback
            logger.error(traceback.format_exc())


    logger.info(f"结果已保存到: {result_path}")
    logger.info(f"日志文件: {os.path.join(result_path, f'{args.task_name}.log')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
