#!/usr/bin/env python3
"""
Reference Strategies 评估脚本（使用DC pipeline一致预处理）
==========================================================
重新评估 NoFix / DeleteAll / ReplaceAll / RepairAll 四个reference策略，
保证与 DemandClean 主管道使用完全相同的编码和标准化。

预处理规则（与 run_demandclean_base.py 一致）:
  - LabelEncoder fit on dirty 60% train data only
  - StandardScaler fit on dirty 60% train data only
  - Clean/Cleaned data reuse same encoders
  - 60/20/20 split, seed=42
"""

import sys, os, json, time, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, silhouette_score, adjusted_rand_score)

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

DATASETS = {
    'beers':         {'task': 'classification', 'label': 'style',
                      'cat_cols': {'beer_name','brewery_name','city','state'}},
    'adult':         {'task': 'classification', 'label': 'income',
                      'cat_cols': {'workclass','education','marital_status','occupation',
                                   'relationship','race','gender','native_country'}},
    'breast_cancer': {'task': 'classification', 'label': 'class', 'cat_cols': set()},
    'smartfactory':  {'task': 'classification', 'label': 'labels', 'cat_cols': set()},
    'bike':          {'task': 'regression', 'label': 'cnt', 'cat_cols': set()},
    'mercedes':      {'task': 'regression', 'label': 'y',
                      'cat_cols': {'X0','X1','X2','X3','X4','X5','X6','X8'}},
    'nasa':          {'task': 'regression', 'label': 'sound_pressure_level', 'cat_cols': set()},
    'soilmoisture':  {'task': 'regression', 'label': 'soil_moisture', 'cat_cols': set()},
    'har':           {'task': 'clustering', 'label': 'gt', 'cat_cols': set()},
}


def load_data(dataset_name):
    data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
    for dirty_name in ['dirty_index.csv', 'dirty_with_index.csv']:
        p = os.path.join(data_dir, dirty_name)
        if os.path.exists(p):
            dirty_df = pd.read_csv(p); break
    for clean_name in ['clean_index.csv', 'clean_with_index.csv']:
        p = os.path.join(data_dir, clean_name)
        if os.path.exists(p):
            clean_df = pd.read_csv(p); break
    return dirty_df, clean_df


def split_data(n, seed=42):
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


def encode(df, feature_cols, label_col, cat_cols, le_dict=None, scaler=None, label_le=None, fit=False):
    """与DC pipeline完全一致的编码逻辑"""
    df_enc = df.copy()
    if le_dict is None:
        le_dict = {}

    for col in feature_cols:
        if col in cat_cols:
            if fit:
                le = LabelEncoder()
                df_enc[col] = df_enc[col].fillna('__MISSING__').astype(str)
                le.fit(df_enc[col])
                le_dict[col] = le
            else:
                le = le_dict.get(col)
                if le is None: continue
                df_enc[col] = df_enc[col].fillna('__MISSING__').astype(str)
                known = set(le.classes_)
                df_enc[col] = df_enc[col].apply(lambda v: v if v in known else '__UNKNOWN__')
                if '__UNKNOWN__' not in le.classes_:
                    le.classes_ = np.append(le.classes_, '__UNKNOWN__')
            df_enc[col] = le.transform(df_enc[col])
        else:
            df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce')

    X = df_enc[feature_cols].values.astype(float)
    X_filled = np.nan_to_num(X, nan=0.0)

    if fit:
        scaler = StandardScaler()
        scaler.fit(X_filled)

    nan_mask = np.isnan(X)
    X_scaled = scaler.transform(X_filled)
    X_scaled[nan_mask] = 0.0

    # Label
    y_series = df_enc[label_col]
    if y_series.dtype == 'object' or label_col in cat_cols:
        if fit:
            label_le = LabelEncoder()
            y_series = y_series.fillna('__MISSING__').astype(str)
            label_le.fit(y_series)
        else:
            if label_le:
                y_series = y_series.fillna('__MISSING__').astype(str)
                known = set(label_le.classes_)
                y_series = y_series.apply(lambda v: v if v in known else '__UNKNOWN__')
                if '__UNKNOWN__' not in label_le.classes_:
                    label_le.classes_ = np.append(label_le.classes_, '__UNKNOWN__')
        if label_le:
            y = label_le.transform(y_series)
        else:
            y = pd.to_numeric(y_series, errors='coerce').fillna(0).values
    else:
        y = pd.to_numeric(y_series, errors='coerce').fillna(0).values

    return X_scaled, y, le_dict, scaler, label_le


def evaluate_all_models(X_train, y_train, X_test, y_test, task_type):
    results = {}
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    if task_type == 'classification':
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'dt': DecisionTreeClassifier(random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                yp = model.predict(X_test)
                results[f'{name}_accuracy'] = accuracy_score(y_test, yp)
                results[f'{name}_f1'] = f1_score(y_test, yp, average='weighted', zero_division=0)
            except Exception as e:
                results[f'{name}_accuracy'] = 0.0
                results[f'{name}_f1'] = 0.0

    elif task_type == 'regression':
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'lr': Ridge(alpha=1.0, solver='svd'),
            'ridge': Ridge(alpha=1.0, solver='svd'),
            'lasso': Lasso(alpha=0.1, random_state=42, max_iter=5000),
            'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                yp = model.predict(X_test)
                results[f'{name}_mse'] = mean_squared_error(y_test, yp)
                results[f'{name}_r2'] = r2_score(y_test, yp)
            except Exception as e:
                results[f'{name}_mse'] = 0.0
                results[f'{name}_r2'] = 0.0

    elif task_type == 'clustering':
        n_clusters = max(len(np.unique(y_test[~np.isnan(y_test.astype(float))])), 2)
        for name, Cls in [('kmeans', KMeans), ('agglomerative', AgglomerativeClustering)]:
            try:
                if name == 'kmeans':
                    m = Cls(n_clusters=n_clusters, random_state=42, n_init=10)
                else:
                    m = Cls(n_clusters=n_clusters)
                yp = m.fit_predict(X_train)
                sil = silhouette_score(X_train, yp, sample_size=min(len(X_train), 10000), random_state=42)
                ari = adjusted_rand_score(y_train[:len(yp)].astype(int), yp)
                results[f'{name}_silhouette'] = sil
                results[f'{name}_ari'] = ari
            except Exception as e:
                results[f'{name}_silhouette'] = 0.0
                results[f'{name}_ari'] = 0.0

    return results


def run_dataset(ds_name):
    cfg = DATASETS[ds_name]
    task = cfg['task']
    label_col = cfg['label']
    cat_cols = cfg['cat_cols']

    print(f"\n{'='*60}")
    print(f"  {ds_name} ({task})")
    print(f"{'='*60}")

    dirty_df, clean_df = load_data(ds_name)
    train_idx, val_idx, test_idx = split_data(len(dirty_df))

    dirty_train = dirty_df.iloc[train_idx].reset_index(drop=True)
    clean_train = clean_df.iloc[train_idx].reset_index(drop=True)
    clean_test = clean_df.iloc[test_idx].reset_index(drop=True)

    exclude = {label_col, 'index', 'Unnamed: 0'}
    feature_cols = [c for c in dirty_df.columns if c not in exclude]

    # Fit encoders on DIRTY train (与DC pipeline一致)
    _, _, le_dict, scaler, label_le = encode(
        dirty_train, feature_cols, label_col, cat_cols, fit=True)

    # Encode test (clean)
    X_test, y_test, _, _, _ = encode(
        clean_test, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)

    results = {}

    # --- NoFix: train on dirty ---
    print("  NoFix...")
    X_nf, y_nf, _, _, _ = encode(
        dirty_train, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)
    results['nofix'] = evaluate_all_models(X_nf, y_nf, X_test, y_test, task)

    # --- RepairAll / FullFix: train on clean ---
    print("  RepairAll (FullFix)...")
    X_ff, y_ff, _, _, _ = encode(
        clean_train, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)
    results['repairall'] = evaluate_all_models(X_ff, y_ff, X_test, y_test, task)

    # --- DeleteAll: delete rows where dirty != clean ---
    print("  DeleteAll...")
    diff_mask = (dirty_train[feature_cols] != clean_train[feature_cols]).any(axis=1)
    keep_mask = ~diff_mask
    # 保留至少20%
    if keep_mask.sum() < len(dirty_train) * 0.2:
        keep_mask = pd.Series([True] * len(dirty_train))
    del_train = clean_train[keep_mask].reset_index(drop=True)
    X_da, y_da, _, _, _ = encode(
        del_train, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)
    results['deleteall'] = evaluate_all_models(X_da, y_da, X_test, y_test, task)
    results['deleteall_n_rows'] = len(del_train)

    # --- ReplaceAll: use VEC on all detected errors (reuse run_replaceall_baseline logic) ---
    print("  ReplaceAll (VEC on all errors)...")
    # 简化版: 用KNN替换所有dirty!=clean的单元格
    X_dirty_arr, y_dirty_arr, _, _, _ = encode(
        dirty_train, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)
    X_clean_arr, y_clean_arr, _, _, _ = encode(
        clean_train, feature_cols, label_col, cat_cols,
        le_dict=le_dict, scaler=scaler, label_le=label_le, fit=False)

    X_replaced = X_dirty_arr.copy()
    error_cells = []
    for j in range(X_dirty_arr.shape[1]):
        for i in range(X_dirty_arr.shape[0]):
            if abs(X_dirty_arr[i,j] - X_clean_arr[i,j]) > 1e-6 or (np.isnan(X_dirty_arr[i,j]) and not np.isnan(X_clean_arr[i,j])):
                error_cells.append((i, j))
    print(f"    {len(error_cells)} error cells detected")

    if error_cells:
        from sklearn.neighbors import NearestNeighbors
        X_fill = np.nan_to_num(X_replaced, nan=0.0)
        nn = NearestNeighbors(n_neighbors=min(6, len(X_fill)), algorithm='auto', n_jobs=-1)
        nn.fit(X_fill)
        col_groups = {}
        for (r, c) in error_cells:
            col_groups.setdefault(c, []).append(r)
        for col_idx, rows in col_groups.items():
            for row_idx in rows:
                _, indices = nn.kneighbors(X_fill[row_idx:row_idx+1])
                neighbors = [idx for idx in indices[0] if idx != row_idx][:5]
                if neighbors:
                    X_replaced[row_idx, col_idx] = np.mean(X_fill[neighbors, col_idx])
                else:
                    X_replaced[row_idx, col_idx] = np.nanmean(X_fill[:, col_idx])
    X_replaced = np.nan_to_num(X_replaced, nan=0.0)
    results['replaceall'] = evaluate_all_models(X_replaced, y_dirty_arr, X_test, y_test, task)
    results['replaceall_n_errors'] = len(error_cells)

    # Print summary
    print(f"\n  Summary ({ds_name}):")
    metric = 'rf_accuracy' if task == 'classification' else ('rf_r2' if task == 'regression' else 'kmeans_silhouette')
    for strategy in ['nofix', 'deleteall', 'replaceall', 'repairall']:
        val = results[strategy].get(metric, 0)
        print(f"    {strategy:12s}: {metric}={val:.4f}")

    results['dataset'] = ds_name
    results['task_type'] = task
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.dataset.split(',')] if args.dataset else list(DATASETS.keys())

    print("Reference Strategies Evaluation (DC-consistent preprocessing)")
    print(f"Datasets: {ds_list}")
    print(f"Time: {datetime.now()}")

    all_results = []
    for ds in ds_list:
        if ds not in DATASETS:
            print(f"Unknown: {ds}"); continue
        try:
            r = run_dataset(ds)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_dir = os.path.join(_PROJECT_ROOT, 'results', 'reference_strategies')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'reference_results_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
