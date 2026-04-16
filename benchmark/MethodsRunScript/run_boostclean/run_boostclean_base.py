"""
BoostClean 运行脚本

BoostClean是面向模型的自动数据清洗方法，通过Boosting策略集成多种检测-修复器。

注意:
- BoostClean使用activedetect包进行错误检测
- 最后一列被视为标签列
- 需要验证集真值来评估清洗效果

用法:
    python run_boostclean_base.py --dirty_path <脏数据路径> --clean_path <干净数据路径>

示例:
    python run_boostclean_base.py \\
        --dirty_path ../../Data/adult/dirty.csv \\
        --clean_path ../../Data/adult/clean.csv \\
        --task_name adult_boostclean \\
        --output_path ../../results/boostclean/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


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


from Methods.BoostClean.boostclean_wrapper import BoostCleanWrapper


def main():
    parser = argparse.ArgumentParser(
        description='Run BoostClean data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_boostclean_base.py --dirty_path ../../Data/adult/dirty.csv --clean_path ../../Data/adult/clean.csv

注意:
  - 使用activedetect包进行错误检测
  - 最后一列被视为标签列
  - 需要验证集真值来评估清洗效果
        """
    )

    # 数据路径参数
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于评估，可选）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='beers_boostclean',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/boostclean/',
                        help='结果输出路径')
    parser.add_argument('--label_column', type=str, default='style',
                        help='标签列名（如果不指定，使用最后一列）')

    # BoostClean参数
    parser.add_argument('--boosting_rounds', type=int, default=5,
                        help='Boosting轮数（默认5）')
    parser.add_argument('--quantitative_thresh', type=int, default=10,
                        help='数值异常检测阈值（默认10）')

    # 评估参数
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')
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
    logger.info("BoostClean 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"干净数据: {args.clean_path or '未提供'}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info(f"Boosting轮数: {args.boosting_rounds}")
    logger.info("-" * 60)

    # 创建清洗器
    cleaner = BoostCleanWrapper(
        boosting_rounds=args.boosting_rounds,
        quantitative_thresh=args.quantitative_thresh,
        verbose=args.verbose
    )

    # 执行清洗
    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    try:
        repaired_df, clean_info = cleaner.clean(
            dirty_path=args.dirty_path,
            clean_path=args.clean_path,
            label_column=args.label_column,
            output_path=output_file
        )
        success = True
    except ImportError as e:
        logger.error(f"依赖导入失败: {e}")
        logger.error("请确保activedetect包已正确安装")
        success = False
        clean_info = {'ground_truth_cost': 0, 'ensemble_size': 0}
    except Exception as e:
        logger.error(f"执行出错: {e}")
        success = False
        clean_info = {'ground_truth_cost': 0, 'ensemble_size': 0}

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
    logger.info(f"集成大小: {clean_info.get('ensemble_size', 0)}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', 0)}")

    # 调用统一测评模块
    if success and args.clean_path and os.path.exists(args.clean_path):
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
                method_type=2,  # BoostClean是Type 2需要验证集
                ground_truth_used=clean_info.get('ground_truth_cost', 0),
                index_attribute=args.index_attribute,
                mse_attributes=args.mse_attributes,
                verbose=args.verbose
            )

            # 合并结果
            clean_info.update(eval_results)

        except ImportError as e:
            logger.warning(f"无法导入getScoreML模块: {e}")
        except Exception as e:
            logger.error(f"统一测评出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        if not args.clean_path:
            logger.info("未提供干净数据路径，跳过统一测评")

    logger.info(f"结果已保存到: {result_path}")
    logger.info(f"日志文件: {os.path.join(result_path, f'{args.task_name}.log')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
