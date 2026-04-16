"""
Horizon 运行脚本

Horizon是基于约束的数据清洗方法。

特点:
- 基于denial constraints进行错误检测和修复
- 使用图模型推理
- 全自动执行 (Type 1)

用法:
    python run_horizon_base.py --dirty_path <脏数据路径> --rule_path <约束文件>

示例:
    python run_horizon_base.py \\
        --dirty_path ../../Data/hospital/dirty_index.csv \\
        --rule_path ../../Data/hospital/dc_rules.txt \\
        --clean_path ../../Data/hospital/clean_index.csv \\
        --task_name hospital_horizon \\
        --output_path ../../results/horizon/
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from Methods.Horizon.horizon import Horizon
from tools.getScore import calculate_all_metrics


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
        description='Run Horizon data cleaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_horizon_base.py --dirty_path ../../Data/hospital/dirty.csv --rule_path ../../Data/hospital/rules.txt

注意:
  - 需要提供denial constraints文件
  - 全自动执行，无需人工参与
        """
    )

    # 数据路径参数
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='脏数据路径')
    parser.add_argument('--rule_path', type=str, default='../../Data/beers/rules.txt',
                        help='约束文件路径')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='干净数据路径（用于评估）')

    # 任务参数
    parser.add_argument('--task_name', type=str, default='beers_horizon',
                        help='任务名称')
    parser.add_argument('--output_path', type=str, default='../../results/horizon/',
                        help='结果输出路径')
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='索引列名')
    parser.add_argument('--mse_attributes', type=str, nargs='*', default=[],
                        help='需要计算MSE的属性列表')

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

    # 记录开始时间
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"运行开始时间: {start_datetime}")
    logger.info("=" * 60)
    logger.info("Horizon 数据清洗")
    logger.info("=" * 60)
    logger.info(f"脏数据: {args.dirty_path}")
    logger.info(f"约束文件: {args.rule_path}")
    logger.info(f"任务名称: {args.task_name}")
    logger.info("-" * 60)

    output_file = os.path.join(result_path, f"{args.task_name}_cleaned.csv")
    success = False
    clean_info = {'ground_truth_cost': 0}

    try:
        # 执行Horizon清洗
        logger.info("执行Horizon清洗...")
        pattern_expressions, dirty_c, _ = Horizon(
            args.dirty_path, args.rule_path, args.clean_path
        )

        # 应用修复
        res_df = pd.read_csv(args.dirty_path)
        for i in range(len(res_df)):
            for v in pattern_expressions[i]:
                value_to_assign = pattern_expressions[i][v]
                try:
                    value_to_assign = int(value_to_assign)
                except ValueError:
                    pass
                res_df.loc[i, v] = str(value_to_assign)

        res_df.to_csv(output_file, index=False)
        success = True
        clean_info = {
            'ground_truth_cost': 0,
            'method': 'Horizon',
            'type': 'constraint-based',
            'auto_level': 1
        }
        logger.info(f"修复后数据已保存: {output_file}")

    except ImportError as e:
        logger.error(f"依赖导入失败: {e}")
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

    # 记录时间
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"执行时间: {elapsed_time:.2f} 秒")
    logger.info(f"执行状态: {'成功' if success else '失败'}")
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
                method_type=1,  # Horizon是全自动Type 1
                ground_truth_used=0,
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
