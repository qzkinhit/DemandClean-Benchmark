"""
ActiveClean 运行脚本

ActiveClean是面向模型的数据清洗方法，通过模型梯度选择最有价值的样本进行清洗。

注意:
- ActiveClean要求输入数据是向量化的（所有列都是数值型）
- 最后一列必须是标签列
- 适用于分类任务

用法:
    python run_activeclean_base.py --dirty_path <脏数据路径> --clean_path <干净数据路径>

示例:
    python run_activeclean_base.py \\
        --dirty_path ../../Data/adult/adult_vectorized_dirty.csv \\
        --clean_path ../../Data/adult/adult_vectorized_clean.csv \\
        --task_name adult_activeclean \\
        --output_path ../../results/activeclean/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """设置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有handlers

    # 文件handler
    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


from Methods.ActiveClean.activeclean_wrapper import run_activeclean


def main():
    parser = argparse.ArgumentParser(
        description='Run ActiveClean data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_activeclean_base.py --dirty_path ../../Data/beers/dirty.csv --clean_path ../../Data/beers/clean.csv

注意:
  - 数据必须是向量化的（所有列都是数值型）
  - 最后一列是标签列
  - ActiveClean主要输出是模型性能报告，而不是修复后的数据
        """
    )

    # 数据路径参数 - 默认使用beers数据集
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径（向量化CSV，最后一列是标签）')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于模拟人工清洗）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='activeclean_task',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/ActiveClean/',
                        help='结果输出路径')

    # ActiveClean参数
    parser.add_argument('--batch_size', type=int, default=50,
                        help='每次清洗的批大小（默认50）')
    parser.add_argument('--total_budget', type=int, default=10000,
                        help='最大清洗样本数（默认10000）')

    # 评估参数
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')
    parser.add_argument('--label_column', type=str, default=None,
                        help='标签列名（默认为最后一列）')
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
    logger.info("ActiveClean 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"干净数据: {args.clean_path}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info("-" * 60)

    # 调用 ActiveClean wrapper
    # run_activeclean(clean_path, dirty_path, batchsize, total)
    txt, cleaned_data, ground_truth_cost = run_activeclean(
        args.clean_path,
        args.dirty_path,
        batchsize=args.batch_size,
        total=args.total_budget
    )

    # 记录时间
    elapsed_time = time.time() - start_time

    # 保存清洗后的数据
    output_file = os.path.join(result_path, f"{args.task_name}_output.csv")
    cleaned_data.to_csv(output_file, index=False)
    logger.info(f"清洗后数据已保存到: {output_file}")

    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"真值使用量 (Ground Truth Cost): {ground_truth_cost}")

    # 调用统一测评模块
    logger.info("")
    logger.info("=" * 60)
    logger.info("调用统一测评模块 getScoreML")
    logger.info("=" * 60)

    try:
        from tools.getScoreML import run_all_evaluation

        # 获取标签列名（默认为最后一列）
        label_col = args.label_column
        if label_col is None:
            label_col = cleaned_data.columns[-1]
            logger.info(f"自动检测标签列: {label_col}")

        # 使用清洗后的数据进行评估
        eval_results = run_all_evaluation(
            dirty_path=args.dirty_path,
            cleaned_path=output_file,  # 使用清洗后的数据
            clean_path=args.clean_path,
            output_path=result_path,
            task_name=args.task_name,
            label_column=label_col,
            task_type=args.task_type,
            models=args.models,
            method_type=3,  # ActiveClean是Type 3迭代交互
            ground_truth_used=ground_truth_cost,
            index_attribute=args.index_attribute,
            mse_attributes=args.mse_attributes,
            verbose=args.verbose
        )

    except ImportError as e:
        logger.warning(f"无法导入getScoreML模块: {e}")
        eval_results = {}
    except Exception as e:
        logger.error(f"统一测评出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        eval_results = {}

    # 保存完整结果
    results_file = os.path.join(result_path, f"{args.task_name}_summary.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ActiveClean 清洗结果评估\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"执行时间: {elapsed_time:.2f} 秒\n")
        f.write(f"方法类型: model-oriented\n")
        f.write(f"自动化级别: 3 (需要迭代交互)\n")
        f.write(f"真值使用量 (Ground Truth Cost): {ground_truth_cost}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ActiveClean 详细报告:\n")
        f.write("-" * 60 + "\n")
        f.write(txt)

    # 保存评估信息
    eval_file = os.path.join(result_path, f"{args.task_name}_total_evaluation.txt")
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write("ActiveClean-specific Metrics:\n")
        f.write(f"Ground Truth Cost: {ground_truth_cost}\n")
        f.write(f"Execution Time: {elapsed_time:.2f}s\n")

    logger.info(f"结果已保存到: {result_path}")
    logger.info(f"日志文件: {os.path.join(result_path, f'{args.task_name}.log')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
