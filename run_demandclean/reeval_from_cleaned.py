#!/usr/bin/env python3
"""
从已保存的 cleaned.csv 重新跑测评（不重新训练）

Usage:
    python run_demandclean/reeval_from_cleaned.py
    python run_demandclean/reeval_from_cleaned.py --dataset beers
    python run_demandclean/reeval_from_cleaned.py --dataset all
"""

import os
import sys
import json
import argparse
import traceback

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'tools'))

from tools.getScoreML import run_all_evaluation

# ============================================================================
# 数据集配置（与 run_demandclean_base.py 保持一致）
# ============================================================================
DATASETS = {
    'beers': {
        'task_type': 'classification',
        'label_col': 'style',
    },
    'adult': {
        'task_type': 'classification',
        'label_col': 'income',
    },
    'bike': {
        'task_type': 'regression',
        'label_col': 'cnt',
    },
    'breast_cancer': {
        'task_type': 'classification',
        'label_col': 'class',
    },
    'har': {
        'task_type': 'clustering',
        'label_col': 'gt',
    },
    'mercedes': {
        'task_type': 'regression',
        'label_col': 'y',
    },
    'nasa': {
        'task_type': 'regression',
        'label_col': 'sound_pressure_level',
    },
    'smartfactory': {
        'task_type': 'classification',
        'label_col': 'labels',
    },
    'soilmoisture': {
        'task_type': 'regression',
        'label_col': 'soil_moisture',
    },
    'hospitals': {
        'task_type': 'classification',
        'label_col': 'Condition',
    },
    'flights': {
        'task_type': 'classification',
        'label_col': 'arrival_delay_bucket',
    },
    'soccer': {
        'task_type': 'classification',
        'label_col': 'manager',
    },
}

VERSION = 'v3_oracle_plain_single'


def get_ml_models(task_type: str):
    """根据任务类型返回评估模型列表"""
    if task_type == 'classification':
        return ['rf', 'lr', 'svm', 'knn', 'dt', 'gb']
    elif task_type == 'clustering':
        return ['kmeans', 'agglomerative', 'spectral']
    else:
        return ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb']


def get_mse_attributes(dirty_path: str, label_col: str):
    """从数据文件推断数值列（用于 hybrid_distance 的 MSE 计算）"""
    import pandas as pd
    df = pd.read_csv(dirty_path, nrows=100)
    drop_cols = {'index', 'id', label_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    mse_attrs = []
    for col in feature_cols:
        vals = df[col].dropna()
        # 过滤常见缺失值标记
        vals = vals[~vals.astype(str).str.strip().isin(['?', '', 'N/A', 'NA', 'nan', 'NaN', 'null', 'None', '-'])]
        if len(vals) == 0:
            continue
        try:
            pd.to_numeric(vals, errors='raise')
            mse_attrs.append(col)
        except (ValueError, TypeError):
            pass
    return mse_attrs


def reeval_dataset(dataset_name: str):
    """对单个数据集重新跑测评"""
    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']
    label_col = ds_cfg['label_col']

    # 路径
    data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
    results_dir = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name, VERSION)
    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    cleaned_path = os.path.join(results_dir, 'data', f'{VERSION}_cleaned_original.csv')
    eval_dir = os.path.join(results_dir, 'evaluation')

    # 检查文件存在
    for path, desc in [
        (dirty_path, 'dirty.csv'),
        (clean_path, 'clean.csv'),
        (cleaned_path, 'cleaned_original.csv'),
    ]:
        if not os.path.exists(path):
            print(f"  [跳过] {desc} 不存在: {path}")
            return None

    ml_models = get_ml_models(task_type)
    mse_attrs = get_mse_attributes(dirty_path, label_col)

    print(f"\n{'='*70}")
    print(f"重新测评: {dataset_name}")
    print(f"  任务类型: {task_type}")
    print(f"  标签列: {label_col}")
    print(f"  评估模型: {ml_models}")
    print(f"  MSE属性: {mse_attrs}")
    print(f"  cleaned.csv: {cleaned_path}")
    print(f"  输出目录: {eval_dir}")
    print(f"{'='*70}")

    try:
        # 尝试从现有 report.json 获取 ground_truth_used
        report_path = os.path.join(results_dir, 'report', f'{VERSION}_report.json')
        ground_truth_used = 0
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    old_report = json.load(f)
                ground_truth_used = old_report.get('ground_truth_used', 0)
            except Exception:
                pass

        results = run_all_evaluation(
            dirty_path=dirty_path,
            cleaned_path=cleaned_path,
            clean_path=clean_path,
            output_path=eval_dir,
            task_name=VERSION,
            label_column=label_col,
            task_type=task_type,
            models=ml_models,
            method_type=1,  # oracle + single_phase = Type 1
            ground_truth_used=ground_truth_used,
            index_attribute='index',
            mse_attributes=mse_attrs,
            verbose=True,
        )

        # 更新 report.json 中的 getscoreml_results
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                # 清理 None 值
                clean_results = {}
                for k, v in results.items():
                    clean_results[k] = 0.0 if v is None else v
                report['getscoreml_results'] = clean_results
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                print(f"  ✓ report.json 已更新: {report_path}")
            except Exception as e:
                print(f"  [警告] 更新 report.json 失败: {e}")

        print(f"\n✓ {dataset_name} 测评完成")
        return results
    except Exception as e:
        print(f"\n✗ {dataset_name} 测评失败: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='从已保存 cleaned.csv 重新测评')
    parser.add_argument(
        '--dataset', type=str, default='all',
        help='数据集名称，或 "all" 跑全部 (默认: all)',
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        datasets_to_run = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            print(f"未知数据集: {args.dataset}")
            print(f"可选: {list(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run = [args.dataset]

    print(f"将对以下数据集重新测评: {datasets_to_run}")

    all_results = {}
    for ds in datasets_to_run:
        result = reeval_dataset(ds)
        if result is not None:
            all_results[ds] = result

    # 汇总
    print(f"\n{'='*70}")
    print("测评汇总")
    print(f"{'='*70}")
    for ds, res in all_results.items():
        task_type = DATASETS[ds]['task_type']
        print(f"\n--- {ds} ({task_type}) ---")

        # 传统指标
        for key in ['precision', 'accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
            if key in res:
                print(f"  {key}: {res[key]}")

        # 下游指标
        ml_keys = sorted([k for k in res if k.startswith('ml_')])
        if ml_keys:
            for k in ml_keys:
                print(f"  {k}: {res[k]}")

        # 容忍度
        tol_keys = sorted([k for k in res if k.startswith('tolerance_')])
        if tol_keys:
            for k in tol_keys:
                print(f"  {k}: {res[k]}")

        # Snoopy
        snp_keys = sorted([k for k in res if k.startswith('snoopy_')])
        if snp_keys:
            for k in snp_keys:
                print(f"  {k}: {res[k]}")

    print(f"\n完成 {len(all_results)}/{len(datasets_to_run)} 个数据集的测评")


if __name__ == '__main__':
    main()
