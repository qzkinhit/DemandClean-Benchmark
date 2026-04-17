#!/usr/bin/env python3
"""
Unified report generation script.
Regenerates reports for every baseline's existing cleaned outputs.

Usage:
    python generate_reports.py                    # generate reports for all baselines
    python generate_reports.py holoclean          # generate only the holoclean report
    python generate_reports.py holoclean ctxpipe  # generate reports for several named baselines
    python generate_reports.py --list             # list all available baselines
"""

import os
import sys
import re
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Deferred import to avoid loading heavy dependencies when only --list is used.
run_all_evaluation = None

def _get_run_all_evaluation():
    global run_all_evaluation
    if run_all_evaluation is None:
        from tools.getScoreML import run_all_evaluation as _run_all_evaluation
        run_all_evaluation = _run_all_evaluation
    return run_all_evaluation

# ============================================================================
# Dataset configuration (shared across all baselines).
# ============================================================================
DATASETS = [
    {
        'dataset': 'adult',
        'label_column': 'income',
        'task_type': 'classification',
        'models': ['rf', 'lr', 'svm', 'knn', 'dt', 'gb'],
    },
    {
        'dataset': 'beers',
        'label_column': 'style',
        'task_type': 'classification',
        'models': ['rf', 'lr', 'svm', 'knn', 'dt', 'gb'],
    },
    {
        'dataset': 'bike',
        'label_column': 'cnt',
        'task_type': 'regression',
        'models': ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb'],
    },
    {
        'dataset': 'breast_cancer',
        'label_column': 'class',
        'task_type': 'classification',
        'models': ['rf', 'lr', 'svm', 'knn', 'dt', 'gb'],
    },
    {
        'dataset': 'har',
        'label_column': 'gt',
        'task_type': 'clustering',
        'models': ['kmeans', 'agglomerative'],
    },
    {
        'dataset': 'mercedes',
        'label_column': 'y',
        'task_type': 'regression',
        'models': ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb'],
    },
    {
        'dataset': 'nasa',
        'label_column': 'sound_pressure_level',
        'task_type': 'regression',
        'models': ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb'],
    },
    {
        'dataset': 'smartfactory',
        'label_column': 'labels',
        'task_type': 'classification',
        'models': ['rf', 'lr', 'svm', 'knn', 'dt', 'gb'],
    },
    {
        'dataset': 'soilmoisture',
        'label_column': 'soil_moisture',
        'task_type': 'regression',
        'models': ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb'],
    },
]

# ============================================================================
# Baseline configuration.
# ============================================================================
BASELINES = {
    # Type 1: fully automatic methods.
    'donothing': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'No cleaning (baseline control)',
    },
    'deleteall': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Delete every row containing an error',
    },
    'simpleimputer': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Simple statistical imputation (mean/mode)',
    },
    'mlimputer': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'ML-based missing value imputation',
    },
    'holoclean': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Probabilistic graphical model-based repair',
    },
    'horizon': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Rule-based data cleaning',
    },
    'lopster': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Constraint-based data repair',
    },
    'uniclean': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Unified data cleaning framework',
    },
    'ctxpipe': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': 'Context-aware data cleaning pipeline',
    },
    'repairall': {
        'method_type': 1,
        'ground_truth_used': 100,
        'description': 'Repair using ground truth directly (ideal upper bound)',
    },
    # Type 2: model-driven methods.
    'activeclean': {
        'method_type': 2,
        'ground_truth_used': 50,
        'description': 'Model-driven active-learning cleaning',
    },
    'boostclean': {
        'method_type': 2,
        'ground_truth_used': 50,
        'description': 'Model-driven iterative cleaning',
    },
    # Type 3: methods that require human labeling.
    'raha_baran': {
        'method_type': 3,
        'ground_truth_used': 20,
        'description': 'Semi-supervised cleaning with error detection',
    },
}


def parse_log_for_time(log_path):
    """Parse the execution time from a log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'Execution Time:\s*([\d.]+)\s*seconds', content)
    if match:
        return float(match.group(1))
    return None


def generate_reports_for_baseline(baseline_name, datasets=None, verbose=True):
    """Generate reports for the given baseline."""
    if baseline_name not in BASELINES:
        print(f"Error: unknown baseline '{baseline_name}'")
        print(f"Available baselines: {', '.join(BASELINES.keys())}")
        return False

    config = BASELINES[baseline_name]
    results_base = f'results/{baseline_name}'

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Baseline: {baseline_name}")
        print(f"# {config['description']}")
        print(f"# Method Type: {config['method_type']}, Ground Truth: {config['ground_truth_used']}")
        print(f"{'#'*70}")

    datasets_to_process = datasets or DATASETS
    success_count = 0
    skip_count = 0
    fail_count = 0

    for ds in datasets_to_process:
        dataset = ds['dataset']
        task_name = f"{dataset}_{baseline_name}_vzekai"

        result_dir = os.path.join(results_base, task_name)
        cleaned_path = os.path.join(result_dir, f'{task_name}_cleaned.csv')
        log_path = os.path.join(result_dir, f'{task_name}.log')
        print(cleaned_path)

        if not os.path.exists(cleaned_path):
            if verbose:
                print(f"  Skipping {task_name}: cleaned file not found")
            skip_count += 1
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {task_name}")
            print(f"{'='*60}")

        dirty_path = f'Data/{dataset}/dirty_index.csv'
        clean_path = f'Data/{dataset}/clean_index.csv'

        exec_time = parse_log_for_time(log_path)
        if exec_time and verbose:
            print(f"Original execution time: {exec_time:.2f} seconds")

        try:
            _get_run_all_evaluation()(
                dirty_path=dirty_path,
                cleaned_path=cleaned_path,
                clean_path=clean_path,
                output_path=result_dir,
                task_name=task_name,
                label_column=ds['label_column'],
                task_type=ds['task_type'],
                models=ds['models'],
                method_type=config['method_type'],
                ground_truth_used=config['ground_truth_used'],
                index_attribute='index',
                verbose=verbose
            )
            if verbose:
                print(f"[OK] Report generated: {result_dir}/{task_name}_report.txt")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {task_name} - {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            fail_count += 1

    if verbose:
        print(f"\n[{baseline_name}] done: success {success_count}, skipped {skip_count}, failed {fail_count}")

    return fail_count == 0


def list_baselines():
    """List all available baselines."""
    print("\nAvailable Baselines:")
    print("=" * 70)
    print(f"{'Name':<15} {'Type':<10} {'GT Used':<10} {'Description'}")
    print("-" * 70)
    for name, config in BASELINES.items():
        type_str = f"Type {config['method_type']}"
        gt_str = f"{config['ground_truth_used']}%"
        print(f"{name:<15} {type_str:<10} {gt_str:<10} {config['description']}")
    print("=" * 70)
    print(f"\nTotal: {len(BASELINES)} baselines")


def main():
    parser = argparse.ArgumentParser(
        description='Unified report generation script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_reports.py                    # generate reports for all baselines
    python generate_reports.py holoclean          # generate only the holoclean report
    python generate_reports.py holoclean ctxpipe  # generate reports for several named baselines
    python generate_reports.py --list             # list all available baselines
        """
    )
    parser.add_argument('baselines', nargs='*', help='baseline names to generate reports for (default: all)')
    parser.add_argument('--list', '-l', action='store_true', help='list all available baselines')
    parser.add_argument('--quiet', '-q', action='store_true', help='quiet mode, reduce output')

    args = parser.parse_args()

    if args.list:
        list_baselines()
        return

    baselines_to_run = args.baselines if args.baselines else list(BASELINES.keys())

    # Validate baseline names.
    invalid = [b for b in baselines_to_run if b not in BASELINES]
    if invalid:
        print(f"Error: unknown baseline: {', '.join(invalid)}")
        print(f"Use --list to see all available baselines")
        sys.exit(1)

    print(f"\nGenerating reports for: {', '.join(baselines_to_run)}")

    total_success = 0
    total_fail = 0

    for baseline in baselines_to_run:
        success = generate_reports_for_baseline(baseline, verbose=not args.quiet)
        if success:
            total_success += 1
        else:
            total_fail += 1

    print(f"\n{'='*70}")
    print(f"All done: {total_success} baselines succeeded, {total_fail} had failures")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
