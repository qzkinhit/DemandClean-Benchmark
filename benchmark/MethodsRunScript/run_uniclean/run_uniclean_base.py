"""
UniClean 运行脚本

UniClean是多信号融合的数据清洗框架（VLDB 2025）。

特点:
- 融合多种清洗信号
- 基于PySpark的大规模处理
- 支持自定义清洗规则

用法:
    python run_uniclean_base.py --dirty_path <脏数据路径> --dataset <数据集名称>

示例:
    python run_uniclean_base.py \\
        --dirty_path ../../Data/beers/dirty_index.csv \\
        --clean_path ../../Data/beers/clean_index.csv \\
        --dataset beers \\
        --task_name beers_uniclean \\
        --output_path ../../results/uniclean/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../Methods/UniClean/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../tools/')


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
        description='Run UniClean data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_uniclean_base.py --dirty_path ../../Data/beers/dirty.csv --dataset beers

支持的数据集:
  beers, hospitals, flights, tax, rayyan, soccer, cloud, commercial

注意:
  - 需要PySpark环境
  - 全自动执行，无需人工参与
        """
    )

    # 数据路径参数
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于评估，可选）')
    parser.add_argument('--dataset', type=str, default='beers',
                        help='数据集名称（用于自动从rules.txt加载清洗器配置）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='beers_uniclean',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/uniclean/',
                        help='结果输出路径')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')

    # 评估参数
    parser.add_argument('--label_column', type=str, default='style',
                        help='标签列名（用于下游任务评估）')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression', 'clustering'],
                        help='下游任务类型（默认classification）')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'lr'],
                        help='评估模型列表（默认rf lr）')

    # UniClean参数
    parser.add_argument('--single_max', type=int, default=10000,
                        help='单次处理最大记录数（默认10000）')
    parser.add_argument('--executor_memory', type=str, default='8g',
                        help='Spark executor内存（默认8g）')
    parser.add_argument('--driver_memory', type=str, default='8g',
                        help='Spark driver内存（默认8g）')
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
    logger.info("UniClean 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"数据集: {args.dataset or '未指定'}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info("-" * 60)

    # 获取清洗器配置 - 优先从rules.txt读取
    cleaners = None
    if args.dataset:
        # 推断数据目录和规则文件路径
        data_dir = os.path.dirname(args.dirty_path)
        rules_path = os.path.join(data_dir, 'rules.txt')

        try:
            from Methods.UniClean.uniclean_wrapper import load_cleaners_from_rules

            if os.path.exists(rules_path):
                logger.info(f"从规则文件加载清洗器: {rules_path}")
                cleaners = load_cleaners_from_rules(rules_path)
                logger.info(f"成功加载 {len(cleaners)} 个清洗器")
            else:
                logger.warning(f"警告: 规则文件不存在: {rules_path}")
                # 回退到预定义配置
                from Methods.UniClean.uniclean_wrapper import get_beers_cleaners, get_hospitals_cleaners
                if args.dataset == 'beers':
                    cleaners = get_beers_cleaners()
                elif args.dataset == 'hospitals':
                    cleaners = get_hospitals_cleaners()
        except ImportError as e:
            logger.error(f"无法导入清洗器模块: {e}")

    # 尝试使用UniClean
    try:
        from Methods.UniClean.uniclean_wrapper import UniCleanWrapper

        cleaner = UniCleanWrapper(
            cleaners=cleaners,
            single_max=args.single_max,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            verbose=args.verbose
        )

        output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
        repaired_df, clean_info = cleaner.clean(
            dirty_path=args.dirty_path,
            clean_path=args.clean_path,
            output_path=output_file,
            index_attribute=args.index_attribute
        )
        success = True

    except ImportError as e:
        logger.error(f"依赖导入失败: {e}")
        logger.error("请确保PySpark已安装，且UniClean模块在正确路径下")
        success = False
        clean_info = {'ground_truth_cost': 0, 'cleaners_count': 0}
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
        success = False
        clean_info = {'ground_truth_cost': 0, 'cleaners_count': 0}

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
    logger.info(f"清洗器数量: {clean_info.get('cleaners_count', 0)}")
    logger.info(f"真值使用成本: {clean_info.get('ground_truth_cost', 0)} (全自动)")

    # 调用统一测评模块
    if success and args.clean_path and os.path.exists(args.clean_path):
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
                method_type=1,  # UniClean是全自动Type 1
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
