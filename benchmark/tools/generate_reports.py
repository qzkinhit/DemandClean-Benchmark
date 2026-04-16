#!/usr/bin/env python3
"""
统一的报告生成脚本
为所有 baseline 的现有 cleaned 文件重新生成 report

用法:
    python generate_reports.py                    # 生成所有 baseline 的报告
    python generate_reports.py holoclean          # 只生成 holoclean 的报告
    python generate_reports.py holoclean ctxpipe  # 生成多个指定 baseline 的报告
    python generate_reports.py --list             # 列出所有可用的 baseline
"""

import os
import sys
import re
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 延迟导入，避免 --list 时加载不必要的依赖
run_all_evaluation = None

def _get_run_all_evaluation():
    global run_all_evaluation
    if run_all_evaluation is None:
        from tools.getScoreML import run_all_evaluation as _run_all_evaluation
        run_all_evaluation = _run_all_evaluation
    return run_all_evaluation

# ============================================================================
# 数据集配置（所有 baseline 共用）
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
# Baseline 配置
# ============================================================================
BASELINES = {
    # Type 1: 全自动方法
    'donothing': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '不做任何清洗（基线对照）',
    },
    'deleteall': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '删除所有含错误的行',
    },
    'simpleimputer': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '简单统计填充（均值/众数）',
    },
    'mlimputer': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '基于ML模型的缺失值填充',
    },
    'holoclean': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '基于概率图模型的数据修复',
    },
    'horizon': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '基于规则的数据清洗',
    },
    'lopster': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '基于约束的数据修复',
    },
    'uniclean': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '统一数据清洗框架',
    },
    'ctxpipe': {
        'method_type': 1,
        'ground_truth_used': 0,
        'description': '上下文感知的数据清洗管道',
    },
    'repairall': {
        'method_type': 1,
        'ground_truth_used': 100,
        'description': '使用ground truth直接修复（理想上界）',
    },
    # Type 2: 模型驱动方法
    'activeclean': {
        'method_type': 2,
        'ground_truth_used': 50,
        'description': '模型驱动的主动学习清洗',
    },
    'boostclean': {
        'method_type': 2,
        'ground_truth_used': 50,
        'description': '模型驱动的迭代清洗',
    },
    # Type 3: 需要人工标注
    'raha_baran': {
        'method_type': 3,
        'ground_truth_used': 20,
        'description': '基于错误检测的半监督清洗',
    },
}


def parse_log_for_time(log_path):
    """从 log 文件中解析执行时间"""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'执行时间:\s*([\d.]+)\s*秒', content)
    if match:
        return float(match.group(1))
    match = re.search(r'Execution Time:\s*([\d.]+)\s*seconds', content)
    if match:
        return float(match.group(1))
    return None


def generate_reports_for_baseline(baseline_name, datasets=None, verbose=True):
    """为指定 baseline 生成报告"""
    if baseline_name not in BASELINES:
        print(f"错误: 未知的 baseline '{baseline_name}'")
        print(f"可用的 baseline: {', '.join(BASELINES.keys())}")
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
                print(f"  跳过 {task_name}: cleaned 文件不存在")
            skip_count += 1
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"处理: {task_name}")
            print(f"{'='*60}")

        dirty_path = f'Data/{dataset}/dirty_index.csv'
        clean_path = f'Data/{dataset}/clean_index.csv'

        exec_time = parse_log_for_time(log_path)
        if exec_time and verbose:
            print(f"原执行时间: {exec_time:.2f} 秒")

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
                print(f"✓ 成功生成报告: {result_dir}/{task_name}_report.txt")
            success_count += 1
        except Exception as e:
            print(f"✗ 失败: {task_name} - {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            fail_count += 1

    if verbose:
        print(f"\n[{baseline_name}] 完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}")

    return fail_count == 0


def list_baselines():
    """列出所有可用的 baseline"""
    print("\n可用的 Baseline 列表:")
    print("=" * 70)
    print(f"{'名称':<15} {'类型':<10} {'GT用量':<10} {'描述'}")
    print("-" * 70)
    for name, config in BASELINES.items():
        type_str = f"Type {config['method_type']}"
        gt_str = f"{config['ground_truth_used']}%"
        print(f"{name:<15} {type_str:<10} {gt_str:<10} {config['description']}")
    print("=" * 70)
    print(f"\n共 {len(BASELINES)} 个 baseline")


def main():
    parser = argparse.ArgumentParser(
        description='统一的报告生成脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python generate_reports.py                    # 生成所有 baseline 的报告
    python generate_reports.py holoclean          # 只生成 holoclean 的报告
    python generate_reports.py holoclean ctxpipe  # 生成多个指定 baseline 的报告
    python generate_reports.py --list             # 列出所有可用的 baseline
        """
    )
    parser.add_argument('baselines', nargs='*', help='要生成报告的 baseline 名称（不指定则生成全部）')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用的 baseline')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式，减少输出')

    args = parser.parse_args()

    if args.list:
        list_baselines()
        return

    baselines_to_run = args.baselines if args.baselines else list(BASELINES.keys())

    # 验证 baseline 名称
    invalid = [b for b in baselines_to_run if b not in BASELINES]
    if invalid:
        print(f"错误: 未知的 baseline: {', '.join(invalid)}")
        print(f"使用 --list 查看所有可用的 baseline")
        sys.exit(1)

    print(f"\n将为以下 baseline 生成报告: {', '.join(baselines_to_run)}")

    total_success = 0
    total_fail = 0

    for baseline in baselines_to_run:
        success = generate_reports_for_baseline(baseline, verbose=not args.quiet)
        if success:
            total_success += 1
        else:
            total_fail += 1

    print(f"\n{'='*70}")
    print(f"全部完成: {total_success} 个 baseline 成功, {total_fail} 个有失败")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
