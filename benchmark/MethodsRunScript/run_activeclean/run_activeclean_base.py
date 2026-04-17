"""
ActiveClean runner

ActiveClean is a model-oriented data cleaning method that selects the most informative samples to clean based on model gradients.

Notes:
- ActiveClean expects vectorized inputs (all columns numeric).
- The last column must be the label column.
- Classification tasks only.

Usage:
    python run_activeclean_base.py --dirty_path <dirty_path> --clean_path <clean_path>

Example:
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

# Add the project root to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and a file."""
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear any existing handlers

    # File handler.
    log_file = os.path.join(result_path, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter.
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
Example:
  python run_activeclean_base.py --dirty_path ../../Data/beers/dirty.csv --clean_path ../../Data/beers/clean.csv

Notes:
  - Input data must be vectorized (all columns numeric).
  - The last column must be the label column.
  - ActiveClean primarily emits a model performance report, not a repaired dataset.
        """
    )

    # Data paths (default: beers dataset).
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='Path to the dirty CSV (vectorized; last column is the label)')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='Path to the clean CSV (used to simulate human labeling)')

    # Task arguments.
    parser.add_argument('--task_name', type=str, default='activeclean_task',
                        help='Task name')
    parser.add_argument('--output_path', type=str, default='../../results/ActiveClean/',
                        help='Output directory')

    # ActiveClean arguments.
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size per cleaning round (default 50)')
    parser.add_argument('--total_budget', type=int, default=10000,
                        help='Maximum number of samples to clean (default 10000)')

    # Evaluation arguments.
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='Index column name')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Label column name (default: last column)')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression', 'clustering'],
                        help='Downstream task type (default classification)')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'lr'],
                        help='Evaluation models (default rf lr)')
    parser.add_argument('--mse_attributes', type=str, nargs='*', default=[],
                        help='Attributes to evaluate with MSE')

    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information')
    parser.add_argument('--use_split', action='store_true',
                        help='Use the DemandClean-aligned 60/20/20 split (seed=42)')

    args = parser.parse_args()

    # Create the output directory.
    result_path = os.path.join(args.output_path, args.task_name)
    os.makedirs(result_path, exist_ok=True)

    # Set up logging.
    logger = setup_logging(result_path, args.task_name)

    # Record the start time.
    start_time = time.time()
    from datetime import datetime
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Run start time: {start_datetime}")
    logger.info("=" * 60)
    logger.info("ActiveClean data cleaning")
    logger.info("=" * 60)
    logger.info(f"Dirty data: {args.dirty_path}")
    logger.info(f"Clean data: {args.clean_path}")
    logger.info(f"Task name: {args.task_name}")
    logger.info("-" * 60)

    # Invoke the ActiveClean wrapper.
    # run_activeclean(clean_path, dirty_path, batchsize, total)
    txt, cleaned_data, ground_truth_cost = run_activeclean(
        args.clean_path,
        args.dirty_path,
        batchsize=args.batch_size,
        total=args.total_budget
    )

    # Record elapsed time.
    elapsed_time = time.time() - start_time

    # Save the cleaned data.
    output_file = os.path.join(result_path, f"{args.task_name}_output.csv")
    cleaned_data.to_csv(output_file, index=False)
    logger.info(f"Cleaned data saved to: {output_file}")

    logger.info("-" * 60)
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")
    logger.info(f"Ground Truth Cost: {ground_truth_cost}")

    # Invoke the unified evaluation module.
    logger.info("")
    logger.info("=" * 60)
    logger.info("Invoking unified evaluation module getScoreML")
    logger.info("=" * 60)

    try:
        from tools.getScoreML import run_all_evaluation

        # Resolve the label column (default: last column).
        label_col = args.label_column
        if label_col is None:
            label_col = cleaned_data.columns[-1]
            logger.info(f"Auto-detected label column: {label_col}")

        # Evaluate using the cleaned data.
        eval_results = run_all_evaluation(
            dirty_path=args.dirty_path,
            cleaned_path=output_file,  # use the cleaned data
            clean_path=args.clean_path,
            output_path=result_path,
            task_name=args.task_name,
            label_column=label_col,
            task_type=args.task_type,
            models=args.models,
            method_type=3,  # ActiveClean is Type 3 iterative interactive
            ground_truth_used=ground_truth_cost,
            index_attribute=args.index_attribute,
            mse_attributes=args.mse_attributes,
            verbose=args.verbose
        )

    except ImportError as e:
        logger.warning(f"Failed to import getScoreML: {e}")
        eval_results = {}
    except Exception as e:
        logger.error(f"Unified evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        eval_results = {}

    # Save the complete result summary.
    results_file = os.path.join(result_path, f"{args.task_name}_summary.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ActiveClean cleaning result evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Method type: model-oriented\n")
        f.write(f"Automation level: 3 (requires iterative interaction)\n")
        f.write(f"Ground Truth Cost: {ground_truth_cost}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ActiveClean detailed report:\n")
        f.write("-" * 60 + "\n")
        f.write(txt)

    # Save evaluation info.
    eval_file = os.path.join(result_path, f"{args.task_name}_total_evaluation.txt")
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write("ActiveClean-specific Metrics:\n")
        f.write(f"Ground Truth Cost: {ground_truth_cost}\n")
        f.write(f"Execution Time: {elapsed_time:.2f}s\n")

    logger.info(f"Results saved to: {result_path}")
    logger.info(f"Log file: {os.path.join(result_path, f'{args.task_name}.log')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
