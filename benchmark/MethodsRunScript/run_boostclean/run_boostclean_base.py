"""
BoostClean runner

BoostClean is a model-oriented automatic data-cleaning method that ensembles multiple detector-repairer pairs with a Boosting strategy.

Notes:
- BoostClean uses the activedetect package for error detection.
- The last column is treated as the label column.
- Requires a validation-set ground truth to evaluate cleaning quality.

Usage:
    python run_boostclean_base.py --dirty_path <dirty_path> --clean_path <clean_path>

Example:
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

# Add the project root to sys.path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


def setup_logging(result_path: str, task_name: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and a file."""
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
Example:
  python run_boostclean_base.py --dirty_path ../../Data/adult/dirty.csv --clean_path ../../Data/adult/clean.csv

Notes:
  - Uses the activedetect package for error detection.
  - The last column is treated as the label column.
  - Requires a validation-set ground truth to evaluate cleaning quality.
        """
    )

    # Data path arguments.
    parser.add_argument('--dirty_path', type=str, default='../../Data/beers/dirty_index.csv',
                        help='Path to the dirty data')
    parser.add_argument('--clean_path', type=str, default='../../Data/beers/clean_index.csv',
                        help='Path to the clean data (used for evaluation; optional)')

    # Task arguments.
    parser.add_argument('--task_name', type=str, default='beers_boostclean',
                        help='Task name')
    parser.add_argument('--output_path', type=str, default='../../results/boostclean/',
                        help='Output directory')
    parser.add_argument('--label_column', type=str, default='style',
                        help='Label column name (defaults to the last column if unspecified)')

    # BoostClean arguments.
    parser.add_argument('--boosting_rounds', type=int, default=5,
                        help='Number of Boosting rounds (default 5)')
    parser.add_argument('--quantitative_thresh', type=int, default=10,
                        help='Numeric outlier threshold (default 10)')

    # Evaluation arguments.
    parser.add_argument('--index_attribute', type=str, default='index',
                        help='Index column name')
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
    logger.info("BoostClean data cleaning")
    logger.info("=" * 60)
    logger.info(f"Dirty data: {args.dirty_path}")
    logger.info(f"Clean data: {args.clean_path or 'not provided'}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Boosting rounds: {args.boosting_rounds}")
    logger.info("-" * 60)

    # Instantiate the cleaner.
    cleaner = BoostCleanWrapper(
        boosting_rounds=args.boosting_rounds,
        quantitative_thresh=args.quantitative_thresh,
        verbose=args.verbose
    )

    # Run the cleaner.
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
        logger.error(f"Dependency import failed: {e}")
        logger.error("Ensure the activedetect package is installed correctly.")
        success = False
        clean_info = {'ground_truth_cost': 0, 'ensemble_size': 0}
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        success = False
        clean_info = {'ground_truth_cost': 0, 'ensemble_size': 0}

    # Record elapsed time.
    elapsed_time = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"Execution Time: {elapsed_time:.2f} seconds")
    logger.info(f"Status: {'success' if success else 'failed'}")
    logger.info(f"Ensemble size: {clean_info.get('ensemble_size', 0)}")
    logger.info(f"Ground Truth Cost: {clean_info.get('ground_truth_cost', 0)}")

    # Invoke the unified evaluation module.
    if success and args.clean_path and os.path.exists(args.clean_path):
        logger.info("")
        logger.info("=" * 60)
        logger.info("Invoking unified evaluation module getScoreML")
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
                method_type=2,  # BoostClean is Type 2 (validation set required)
                ground_truth_used=clean_info.get('ground_truth_cost', 0),
                index_attribute=args.index_attribute,
                mse_attributes=args.mse_attributes,
                verbose=args.verbose
            )

            # Merge results.
            clean_info.update(eval_results)

        except ImportError as e:
            logger.warning(f"Failed to import getScoreML: {e}")
        except Exception as e:
            logger.error(f"Unified evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        if not args.clean_path:
            logger.info("No clean path provided; skipping unified evaluation.")

    logger.info(f"Results saved to: {result_path}")
    logger.info(f"Log file: {os.path.join(result_path, f'{args.task_name}.log')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
