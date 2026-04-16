"""
CtxPipe 运行脚本（Clean4MLBaseline 集成）

功能：
- RL 模式：上下文感知 + 预训练权重推理，生成数据准备管道（列可能改变）
- 保模式（--schema_preserving）：仅缺失值填充（数值=中位数、类别=众数），列不变，适合传统评估

兼容性：
- GTE 模型优先从环境变量 `GTE_MODEL_PATH` 读取；无可用模型时回退零向量（可运行，效果下降）
- 预训练权重强制映射到 CPU 加载
- Multiprocessing 不可用时自动退化为单进程
"""
import os
import sys
import argparse
import time
import logging
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from tools.getScore import calculate_all_metrics
from tools.insert_null import inject_missing_values
from ctxpipe_adapter import CtxPipeAdapter


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """设置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有handlers
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
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='Run CtxPipe - Context-aware Data Preparation Pipeline Construction.'
    )

    # 定义命令行参数
    parser.add_argument(
        '--dirty_path', type=str,
        default='../../Data/beers/dirty_index.csv',
        help='Path to the input dirty CSV file.'
    )
    parser.add_argument(
        '--clean_path', type=str,
        default='../../Data/beers/clean_index.csv',
        help='Path to the input clean CSV file (for evaluation).'
    )
    parser.add_argument(
        '--task_name', type=str,
        default='beers_ctxpipe',
        help='Task name for the pipeline construction process.'
    )
    parser.add_argument(
        '--output_path', type=str,
        default='../../results/ctxpipe/',
        help='Path to save the output results.'
    )
    parser.add_argument(
        '--index_attribute', type=str,
        default='index',
        help='Index attribute of data'
    )
    parser.add_argument(
        '--mse_attributes', type=str, nargs='*',
        default=[],
        help='List of attributes to calculate MSE, separated by space.'
    )
    parser.add_argument(
        '--label_index', type=int,
        default=None,
        help='Label column index (0-based). If not specified, will auto-detect.'
    )
    parser.add_argument(
        '--model_tag', type=str,
        default='ctx_50000',
        help='CtxPipe pretrained model tag (default: ctx_50000)'
    )
    parser.add_argument(
        '--skip_evaluation', action='store_true',
        help='Skip traditional data cleaning evaluation (only report ML metrics)'
    )
    parser.add_argument(
        '--schema_preserving', action='store_true',
        help='Output schema-preserving cleaned data (imputation only). Skips RL pipeline output to keep same columns.'
    )

    # 评估参数
    parser.add_argument(
        '--label_column', type=str, default='style',
        help='标签列名（用于下游任务评估）'
    )
    parser.add_argument(
        '--task_type', type=str, default='classification',
        choices=['classification', 'regression', 'clustering'],
        help='下游任务类型（默认classification）'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', default=['rf', 'lr'],
        help='评估模型列表（默认rf lr）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='是否打印详细信息'
    )
    parser.add_argument(
        '--use_split', action='store_true',
        help='使用 DemandClean 对齐的 60/20/20 数据划分（seed=42）'
    )

    # 解析命令行参数
    args = parser.parse_args()
    mse_attributes = args.mse_attributes
    stra_path = os.path.join(args.output_path, f"{args.task_name}")
    index_attribute = args.index_attribute

    # 创建输出目录
    if not os.path.exists(stra_path):
        os.makedirs(stra_path)

    # 设置日志
    logger = setup_logging(stra_path, args.task_name)

    # 处理空值表示（统一转换为empty）
    logger.info("Preprocessing missing values...")
    inject_missing_values(
        csv_file=args.clean_path,
        output_file=args.clean_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )
    inject_missing_values(
        csv_file=args.dirty_path,
        output_file=args.dirty_path,
        attributes_error_ratio=None,
        missing_value_in_ori_data='NULL',
        missing_value_representation='empty'
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Running CtxPipe with dirty file: {args.dirty_path}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Model tag: {args.model_tag}")
    logger.info(f"{'='*60}\n")

    # 记录开始时间
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"运行开始时间: {start_datetime}")

    # 创建CtxPipe适配器
    adapter = CtxPipeAdapter(model_tag=args.model_tag)

    # 运行CtxPipe生成管道
    try:
        cleaned_data, ctxpipe_results = adapter.run_ctxpipe(
            dirty_path=args.dirty_path,
            task_name=args.task_name,
            label_index=args.label_index,
            output_path=stra_path,
            schema_preserving=args.schema_preserving
        )
    except Exception as e:
        logger.error(f"\n{'!'*60}")
        logger.error(f"Error running CtxPipe: {e}")
        logger.error(f"{'!'*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 保存CtxPipe处理后的数据
    # 后处理：将空值统一为 "empty"（NaN、空字符串等）
    cleaned_data = cleaned_data.fillna('empty')
    cleaned_data = cleaned_data.replace('', 'empty')

    res_path = os.path.join(stra_path, f"{args.task_name}_cleaned.csv")
    cleaned_data.to_csv(res_path, index=False)
    logger.info(f"\nCtxPipe output saved to: {res_path}")

    # 保存CtxPipe的管道信息
    pipeline_info_path = os.path.join(stra_path, f"{args.task_name}_pipeline_info.txt")
    with open(pipeline_info_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CtxPipe Pipeline Information\n")
        f.write("="*60 + "\n\n")
        f.write(f"Task Name: {args.task_name}\n")
        f.write(f"Model Tag: {args.model_tag}\n")
        f.write(f"Execution Time: {elapsed_time:.2f} seconds\n\n")
        f.write(f"Selected AI Sequence:\n")
        for i, primitive in enumerate(ctxpipe_results.get('ai_sequence', []), 1):
            f.write(f"  {i}. {primitive}\n")
        ml_score_val = ctxpipe_results.get('ml_score', None)
        if isinstance(ml_score_val, (int, float)):
            f.write(f"\nML Model Score: {ml_score_val:.4f}\n")
        else:
            f.write(f"\nML Model Score: N/A\n")
        f.write(f"Logical Pipeline: {ctxpipe_results.get('logical_pipeline', 'N/A')}\n")

    logger.info(f"Pipeline info saved to: {pipeline_info_path}")

    # 打印CtxPipe结果
    logger.info(f"\n{'='*60}")
    logger.info("CtxPipe Results:")
    logger.info(f"{'='*60}")
    logger.info(f"AI Sequence: {ctxpipe_results.get('ai_sequence', [])}")
    ml_score_val = ctxpipe_results.get('ml_score', None)
    if isinstance(ml_score_val, (int, float)):
        logger.info(f"ML Score: {ml_score_val:.4f}")
    else:
        logger.info("ML Score: N/A")
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")
    logger.info(f"{'='*60}\n")

    # 可选：传统数据清洗评估
    # 注意：CtxPipe主要用于ML数据准备，可能不适合传统的数据清洗评估
    if not args.skip_evaluation:
        logger.info("="*60)
        logger.info("Traditional Data Cleaning Evaluation")
        logger.info("Note: CtxPipe is designed for ML data preparation,")
        logger.info("      traditional cleaning metrics may not be applicable.")
        logger.info("="*60)

        try:
            # 处理评估数据的空值
            inject_missing_values(
                csv_file=res_path,
                output_file=res_path,
                attributes_error_ratio=None,
                missing_value_in_ori_data='NULL',
                missing_value_representation='empty'
            )

            # 读取数据
            clean_data = pd.read_csv(args.clean_path)
            dirty_data = pd.read_csv(args.dirty_path)
            cleaned_result = pd.read_csv(res_path)

            # 检查列是否匹配
            if set(cleaned_result.columns) != set(clean_data.columns):
                logger.warning("\nWarning: Column names do not match between cleaned and original data.")
                logger.warning(f"Original columns: {clean_data.columns.tolist()}")
                logger.warning(f"Cleaned columns: {cleaned_result.columns.tolist()}")
                logger.warning("Skipping traditional evaluation.\n")
            else:
                # 计算评估指标
                attributes = clean_data.columns.tolist()
                results = calculate_all_metrics(
                    clean_data, dirty_data, cleaned_result, attributes,
                    stra_path, args.task_name,
                    index_attribute=index_attribute,
                    mse_attributes=mse_attributes
                )

                # 保存评估结果
                results_path = os.path.join(stra_path, f"{args.task_name}_total_evaluation.txt")
                original_stdout = sys.stdout

                with open(results_path, 'w', encoding='utf-8') as f:
                    try:
                        sys.stdout = f
                        print("Traditional Data Cleaning Evaluation Results:")
                        print(f"Accuracy: {results.get('accuracy')}")
                        print(f"Recall: {results.get('recall')}")
                        print(f"F1 Score: {results.get('f1_score')}")
                        print(f"EDR: {results.get('edr')}")
                        print(f"Hybrid Distance: {results.get('hybrid_distance')}")
                        print(f"R-EDR: {results.get('r_edr')}")
                        print(f"Time: {elapsed_time}")
                        print(f"Speed: {100*float(elapsed_time)/clean_data.shape[0]} seconds/100num")
                        print(f"\nCtxPipe-specific Metrics:")
                        ml_score_val = ctxpipe_results.get('ml_score', None)
                        if isinstance(ml_score_val, (int, float)):
                            print(f"ML Score: {ml_score_val:.4f}")
                        else:
                            print("ML Score: N/A")
                    finally:
                        sys.stdout = original_stdout

                # 打印到控制台
                logger.info("\nTraditional Evaluation Results:")
                logger.info(f"Accuracy: {results.get('accuracy')}")
                logger.info(f"Recall: {results.get('recall')}")
                logger.info(f"F1 Score: {results.get('f1_score')}")
                logger.info(f"EDR: {results.get('edr')}")
                logger.info(f"Hybrid Distance: {results.get('hybrid_distance')}")
                logger.info(f"R-EDR: {results.get('r_edr')}")
                logger.info(f"Time: {elapsed_time:.2f}s")
                logger.info(f"Speed: {100 * float(elapsed_time) / clean_data.shape[0]:.4f} seconds/100num")

        except Exception as e:
            logger.warning(f"\nWarning: Traditional evaluation failed: {e}")
            logger.warning("This is expected if CtxPipe significantly transformed the data.")

    # 调用统一测评模块 getScoreML
    if not args.skip_evaluation and os.path.exists(res_path):
        logger.info(f"\n{'='*60}")
        logger.info("调用统一测评模块 getScoreML")
        logger.info("="*60)

        try:
            from tools.getScoreML import run_all_evaluation

            eval_results = run_all_evaluation(
                dirty_path=args.dirty_path,
                cleaned_path=res_path,
                clean_path=args.clean_path,
                output_path=stra_path,
                task_name=args.task_name,
                label_column=args.label_column,
                task_type=args.task_type,
                models=args.models,
                method_type=1,  # CtxPipe是全自动Type 1
                ground_truth_used=0,
                index_attribute=index_attribute,
                mse_attributes=mse_attributes,
                verbose=getattr(args, 'verbose', False)
            )

            logger.info("\ngetScoreML统一测评完成")

        except ImportError as e:
            logger.warning(f"警告: 无法导入getScoreML模块: {e}")
        except Exception as e:
            logger.error(f"统一测评出错: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info(f"CtxPipe finished successfully!")
    logger.info(f"Results saved to: {stra_path}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
