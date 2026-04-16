#!/usr/bin/env python3
"""
DemandClean CSV 报告生成器

从 report.json 提取实验结果，生成与 Clean4MLBaseline 格式一致的 CSV。

Usage:
    python generate_csv.py --versions v3,v6
    python generate_csv.py --versions v3 --datasets beers,adult
    python generate_csv.py --versions v3,v6 --output results.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

# ============================================================================
# 常量
# ============================================================================

SHORT_NAME_MAP = {
    'v1': 'v1_oracle_dueling_single',
    'v2': 'v2_oracle_dueling_two',
    'v3': 'v3_oracle_plain_single',
    'v4': 'v4_oracle_plain_two',
    'v5': 'v5_auto_dueling_single',
    'v6': 'v6_auto_dueling_two',
    'v7': 'v7_auto_plain_single',
    'v8': 'v8_auto_plain_two',
    'ngt': 'v5_auto_dueling_single_ngt',
}

ALL_DATASETS = [
    'adult', 'beers', 'bike', 'breast_cancer', 'har',
    'mercedes', 'nasa', 'smartfactory', 'soilmoisture',
]

CLASSIFICATION_MODELS = ['rf', 'lr', 'svm', 'knn', 'dt', 'gb']
REGRESSION_MODELS = ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb']
CLUSTERING_MODELS = ['kmeans', 'agglomerative', 'spectral']


def resolve_version(name: str) -> str:
    name = name.strip()
    return SHORT_NAME_MAP.get(name, name)


def version_display(full_name: str) -> str:
    """全名转显示名 (如 Demandclean_v3)"""
    for short, full in SHORT_NAME_MAP.items():
        if full == full_name:
            return f'Demandclean_{short}'
    return full_name


def find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


# ============================================================================
# 数据加载
# ============================================================================

def load_report(dataset: str, version_full: str, results_dir: str) -> dict:
    """加载单个 report.json"""
    report_path = os.path.join(
        results_dir, dataset, version_full, 'report', f'{version_full}_report.json'
    )
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'  [警告] {dataset}/{version_full}: 读取失败 - {e}')
        return None


# ============================================================================
# 格式化 ML 结果
# ============================================================================

def format_ml_cell(model_name: str, metrics: dict, task_type: str) -> str:
    """格式化单个 ML 模型结果"""
    name_map = {
        'rf': 'RF', 'lr': 'LR', 'svm': 'SVM', 'knn': 'KNN', 'dt': 'DT', 'gb': 'GB',
        'ridge': 'RIDGE', 'lasso': 'LASSO',
        'kmeans': 'KMeans', 'agglomerative': 'Agglomerative', 'spectral': 'Spectral',
    }
    display = name_map.get(model_name, model_name.upper())

    if not metrics:
        return f"{display}\n    (no data)"

    lines = [display]
    if task_type == 'classification':
        for k in ['accuracy', 'f1', 'precision', 'recall']:
            if k in metrics:
                lines.append(f"    {k:<15s}: {metrics[k]:.6f}")
    elif task_type == 'regression':
        for k in ['mse', 'r2']:
            if k in metrics:
                lines.append(f"    {k:<15s}: {metrics[k]:.6f}")
    elif task_type == 'clustering':
        for k in ['silhouette', 'ari']:
            if k in metrics:
                lines.append(f"    {k:<15s}: {metrics[k]:.6f}")
    return '\n'.join(lines)


def extract_ml_results(report: dict) -> dict:
    """从 report.json 提取 ML 结果"""
    gs = report.get('getscoreml_results', {})
    task_type = report.get('task_type', 'classification')

    results = {}

    if task_type == 'classification':
        for model in CLASSIFICATION_MODELS:
            metrics = {}
            acc = gs.get(f'ml_{model}_accuracy')
            f1 = gs.get(f'ml_{model}_f1')
            prec = gs.get(f'ml_{model}_precision')
            rec = gs.get(f'ml_{model}_recall')
            if acc is not None:
                metrics['accuracy'] = acc
            if f1 is not None:
                metrics['f1'] = f1
            if prec is not None:
                metrics['precision'] = prec
            if rec is not None:
                metrics['recall'] = rec
            if metrics:
                results[model] = metrics

    elif task_type == 'regression':
        model_map = {'lr': 'lr', 'ridge': 'ridge', 'lasso': 'lasso'}
        for model in REGRESSION_MODELS:
            metrics = {}
            mse = gs.get(f'ml_{model}_mse')
            r2 = gs.get(f'ml_{model}_r2')
            if mse is not None:
                metrics['mse'] = mse
            if r2 is not None:
                metrics['r2'] = r2
            if metrics:
                results[model] = metrics

    elif task_type == 'clustering':
        for model in CLUSTERING_MODELS:
            metrics = {}
            sil = gs.get(f'ml_{model}_silhouette')
            ari = gs.get(f'ml_{model}_ari')
            if sil is not None:
                metrics['silhouette'] = sil
            if ari is not None:
                metrics['ari'] = ari
            if metrics:
                results[model] = metrics

    return results


# ============================================================================
# 生成 CSV 行
# ============================================================================

def report_to_row(dataset: str, version_full: str, report: dict) -> dict:
    """将 report.json 转换为 CSV 行"""
    gs = report.get('getscoreml_results', {})
    task_type = report.get('task_type', 'classification')

    # 基础指标
    elapsed = report.get('elapsed_time', 0)
    f1_score = gs.get('f1_score', 0)
    r_edr = gs.get('r_edr', 0)
    hybrid_dist = gs.get('hybrid_distance', 0)
    edr = gs.get('edr', 0)
    col_avg_rmse = gs.get('col_avg_rmse', '')
    col_avg_f1 = gs.get('col_avg_f1', '')

    # 容忍度指标
    p_do_nothing = gs.get('tolerance_P_do_nothing', gs.get('P_do_nothing', ''))
    p_demand_clean = gs.get('tolerance_P_demand_clean', gs.get('P_demand_clean', ''))
    p_repair_all = gs.get('tolerance_P_repair_all', gs.get('P_repair_all', ''))

    # Snoopy 上界
    ub_dirty = gs.get('snoopy_upper_bound_dirty', '')
    ub_cleaned = gs.get('snoopy_upper_bound_cleaned', '')
    ub_clean = gs.get('snoopy_upper_bound_clean', '')
    ub_improvement = gs.get('snoopy_upper_bound_improvement', '')

    # 真值成本
    truth_cost = report.get('ground_truth_used', gs.get('ground_truth_cost', 0))

    # ML 结果
    ml_results = extract_ml_results(report)

    return {
        'Baseline': version_display(version_full),
        'Dataset': dataset,
        'time': f'{elapsed:.2f}' if elapsed else '',
        'f1_score': f1_score,
        'r_edr': r_edr,
        'hybrid_distance': hybrid_dist,
        'edr': edr,
        'col_avg_rmse': col_avg_rmse,
        'col_avg_f1': col_avg_f1,
        'P_do_nothing': p_do_nothing,
        'P_demand_clean': p_demand_clean,
        'P_repair_all': p_repair_all,
        'upper_bound_dirty': ub_dirty,
        'upper_bound_cleaned': ub_cleaned,
        'upper_bound_clean': ub_clean,
        'upper_bound_improvement': ub_improvement,
        'truth_cost': truth_cost,
        'ml_results': ml_results,
        'task_type': task_type,
    }


# ============================================================================
# CSV 输出
# ============================================================================

def write_csv(rows: list, output_path: str):
    """写出 CSV"""
    headers = [
        'Baseline', 'Dataset', '时间（单位 s）', '传统清洗的F1值', 'r_edr',
        'hybrid_distance', 'edr', 'col_avg_rmse', 'col_avg_f1',
        'P_do_nothing', 'P_demand_clean', 'P_repair_all',
        'upper_bound_dirty', 'upper_bound_cleaned', 'upper_bound_clean',
        'upper_bound_improvement', '真值使用单元格数',
        'ML_1', 'ML_2', 'ML_3', 'ML_4', 'ML_5', 'ML_6'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            ml = row['ml_results']
            task = row['task_type']

            if task == 'classification':
                model_order = CLASSIFICATION_MODELS
            elif task == 'regression':
                model_order = REGRESSION_MODELS
            else:
                model_order = CLUSTERING_MODELS

            ml_cells = []
            for m in model_order:
                ml_cells.append(format_ml_cell(m, ml.get(m, {}), task))
            while len(ml_cells) < 6:
                ml_cells.append('')

            csv_row = [
                row['Baseline'], row['Dataset'], row['time'],
                row['f1_score'], row['r_edr'], row['hybrid_distance'], row['edr'],
                row['col_avg_rmse'], row['col_avg_f1'],
                row['P_do_nothing'], row['P_demand_clean'], row['P_repair_all'],
                row['upper_bound_dirty'], row['upper_bound_cleaned'],
                row['upper_bound_clean'], row['upper_bound_improvement'],
                row['truth_cost'],
            ] + ml_cells[:6]

            writer.writerow(csv_row)

    print(f'已写出: {output_path} ({len(rows)} 行)')


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DemandClean CSV 报告生成器')
    parser.add_argument('--versions', type=str, required=True,
                        help='版本列表，逗号分隔 (如 v3,v6)')
    parser.add_argument('--datasets', type=str, default='',
                        help='数据集列表，逗号分隔 (默认全部)')
    parser.add_argument('--output', type=str, default='',
                        help='输出 CSV 路径 (默认自动命名)')
    args = parser.parse_args()

    # 解析参数
    versions = [v.strip() for v in args.versions.split(',') if v.strip()]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()] if args.datasets else ALL_DATASETS

    project_root = find_project_root()
    results_dir = os.path.join(project_root, 'results', 'demandclean')

    # 输出路径
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_root, output_path)
    else:
        ver_tag = '_'.join(versions)
        output_path = os.path.join(project_root, f'demandclean_{ver_tag}_results.csv')

    print(f'DemandClean CSV 生成器')
    print(f'  版本: {versions}')
    print(f'  数据集: {datasets}')
    print(f'  输出: {output_path}')
    print()

    # 加载并转换
    rows = []
    for ver in versions:
        ver_full = resolve_version(ver)
        for ds in datasets:
            report = load_report(ds, ver_full, results_dir)
            if report:
                row = report_to_row(ds, ver_full, report)
                rows.append(row)
                print(f'  {ver}/{ds}: OK')
            else:
                print(f'  {ver}/{ds}: 跳过 (无报告)')

    if rows:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        write_csv(rows, output_path)
    else:
        print('没有找到任何有效报告')
        sys.exit(1)


if __name__ == '__main__':
    main()
