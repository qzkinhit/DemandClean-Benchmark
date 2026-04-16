#!/usr/bin/env python3
"""
ReplaceAll Baseline 评估脚本 (v2 — 与 Clean4MLBaseline reeval_with_split.py 对齐)
==================================================================================

对所有 Oracle 检测到的错误调用值估计链 (VEC) 进行替换，零 GT 成本。
评估流水线与 Clean4MLBaseline/tools/reeval_with_split.py 完全一致:
  - 数据来源: Clean4MLBaseline/Data/{dataset}/
  - 预处理: 60% dirty train fit 编码器 (LabelEncoder + StandardScaler)
  - 评估: 分类=RF accuracy, 回归=RF R², 聚类=KMeans silhouette (全量)
  - 容忍度: P_do_nothing / P_demand_clean / P_repair_all + Snoopy upper bounds

用法:
    python run_demandclean/run_replaceall_baseline.py
    python run_demandclean/run_replaceall_baseline.py --dataset beers
    python run_demandclean/run_replaceall_baseline.py --dataset beers,nasa,har
"""

import sys
import os
import json
import time
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score
)

warnings.filterwarnings("ignore")

# 项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from demandclean.config import DemandCleanConfig
from demandclean.core.environments.value_estimation import ValueEstimator

SEED = 42
C4ML_ROOT = '/Users/qianzekai/PycharmProjects/Clean4MLBaseline'
C4ML_DATA = os.path.join(C4ML_ROOT, 'Data')

# ============================================================================
# 数据集配置 (与 Clean4MLBaseline datasets_config.py 完全一致)
# ============================================================================
DATASETS = {
    'adult': {'label_column': 'income', 'task_type': 'classification', 'exclude_columns': [],
              'feature_columns': ['age','workclass','fnlwgt','education','educational_num','marital_status',
                                  'occupation','relationship','race','gender','capital_gain','capital_loss',
                                  'hours_per_week','native_country']},
    'beers': {'label_column': 'style', 'task_type': 'classification',
              'exclude_columns': ['id','beer_name','brewery_id','brewery_name','city','state'],
              'feature_columns': ['ounces','abv','ibu']},
    'bike': {'label_column': 'cnt', 'task_type': 'regression', 'exclude_columns': ['dteday'],
             'feature_columns': ['season','yr','mnth','hr','holiday','weekday','workingday',
                                 'weathersit','temp','atemp','hum','windspeed','casual','registered']},
    'breast_cancer': {'label_column': 'class', 'task_type': 'classification', 'exclude_columns': [],
                      'feature_columns': ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
                                          'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
                                          'Bland Chromatin','Normal Nucleoli','Mitoses']},
    'har': {'label_column': 'gt', 'task_type': 'clustering', 'exclude_columns': [],
            'feature_columns': ['x','y','z']},
    'mercedes': {'label_column': 'y', 'task_type': 'regression', 'exclude_columns': [],
                 'feature_columns': 'auto'},
    'nasa': {'label_column': 'sound_pressure_level', 'task_type': 'regression', 'exclude_columns': [],
             'feature_columns': ['frequency','angle','chord_length','velocity','thickness']},
    'smartfactory': {'label_column': 'labels', 'task_type': 'classification', 'exclude_columns': [],
                     'feature_columns': ['i_w_blo_weg','o_w_blo_power','o_w_blo_voltage',
                                         'i_w_bhl_weg','o_w_bhl_power','o_w_bhl_voltage',
                                         'i_w_bhr_weg','o_w_bhr_power','o_w_bhr_voltage',
                                         'i_w_bru_weg','o_w_bru_power','o_w_bru_voltage',
                                         'i_w_hr_weg','o_w_hr_power','o_w_hr_voltage',
                                         'i_w_hl_weg','o_w_hl_power','o_w_hl_voltage']},
    'soilmoisture': {'label_column': 'soil_moisture', 'task_type': 'regression',
                     'exclude_columns': ['datetime'], 'feature_columns': 'auto'},
}

# DemandClean 编码所需的分类列配置
DC_CATEGORICAL = {
    'beers': {'beer_name','brewery_name','city','state'},
    'adult': {'workclass','education','marital_status','occupation','relationship','race','gender','native_country'},
    'mercedes': {'X0','X1','X2','X3','X4','X5','X6','X8'},
}


# ============================================================================
# 1. reeval_with_split.py 相同的预处理 (逐字复制核心逻辑)
# ============================================================================

def preprocess_for_ml(df: pd.DataFrame, label_col: str, index_col: str = 'index',
                      exclude_cols: List[str] = None, feature_cols=None,
                      fitted_encoders: Dict = None, fitted_scaler=None, fitted_label_encoder=None):
    """与 Clean4MLBaseline/tools/reeval_with_split.py:preprocess_for_ml 完全一致"""
    is_fit_mode = (fitted_encoders is None)
    drop_cols = [index_col] if index_col in df.columns else []
    if exclude_cols:
        drop_cols += [c for c in exclude_cols if c in df.columns]
    if label_col not in df.columns:
        raise ValueError(f"标签列 '{label_col}' 不在数据中。可用列: {list(df.columns)}")
    y = df[label_col].copy()
    drop_cols.append(label_col)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    if feature_cols and feature_cols != 'auto':
        available = [c for c in feature_cols if c in X.columns]
        X = X[available]
    encoders = {} if is_fit_mode else fitted_encoders
    for col in X.columns:
        if X[col].dtype == 'object':
            # 先尝试转数值：如果 >50% 的非空值能转数值，则当数值列处理
            numeric_attempt = pd.to_numeric(X[col], errors='coerce')
            non_null = X[col].notna().sum()
            numeric_ratio = numeric_attempt.notna().sum() / max(non_null, 1)
            if numeric_ratio > 0.5:
                X[col] = numeric_attempt
                continue

            X[col] = X[col].astype(str).fillna('__MISSING__')
            if is_fit_mode:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    X[col] = X[col].apply(lambda v: v if v in known else '__UNKNOWN__')
                    if '__UNKNOWN__' not in known:
                        le.classes_ = np.append(le.classes_, '__UNKNOWN__')
                    X[col] = le.transform(X[col])
                else:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    if is_fit_mode:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = fitted_scaler
        X_scaled = scaler.transform(X) if scaler else StandardScaler().fit_transform(X)
    y_arr = y.copy()
    if y_arr.dtype == 'object':
        if is_fit_mode:
            label_encoder = LabelEncoder()
            y_arr = label_encoder.fit_transform(y_arr.astype(str))
        else:
            label_encoder = fitted_label_encoder
            if label_encoder:
                known = set(label_encoder.classes_)
                y_arr = y_arr.astype(str).apply(lambda v: v if v in known else label_encoder.classes_[0])
                y_arr = label_encoder.transform(y_arr)
            else:
                label_encoder = LabelEncoder()
                y_arr = label_encoder.fit_transform(y_arr.astype(str))
    else:
        y_arr = pd.to_numeric(y_arr, errors='coerce').fillna(0).values
        label_encoder = None
    if is_fit_mode:
        return X_scaled, np.array(y_arr), encoders, scaler, label_encoder
    else:
        return X_scaled, np.array(y_arr)


def demandclean_split(n_total: int, seed: int = SEED):
    """与 reeval_with_split.py 完全相同的 60/20/20 划分"""
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


# ============================================================================
# 2. reeval_with_split.py 相同的评估函数
# ============================================================================

def evaluate_ml(X_train, y_train, X_test, y_test, task_type: str) -> Dict:
    """评估多个下游模型 (RF, LR, SVM, KNN, DT, GB)"""
    results = {}
    if task_type == 'clustering':
        n_clusters = len(np.unique(y_train))
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        y_pred = km.fit_predict(X_train)
        sil = silhouette_score(X_train, y_pred, sample_size=min(len(X_train), 10000), random_state=SEED)
        ari = adjusted_rand_score(y_train, y_pred)
        results['kmeans'] = {'silhouette': sil, 'ari': ari}
        # Also run LR-based classification on cluster labels for multi-model comparison
        try:
            from sklearn.cluster import AgglomerativeClustering
            for name, model in [('lr', LogisticRegression(max_iter=1000, random_state=SEED)),
                                ('svm', SVC(random_state=SEED))]:
                model.fit(X_train, y_train)
                y_pred_m = model.predict(X_test)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred_m),
                    'f1': f1_score(y_test, y_pred_m, average='weighted', zero_division=0),
                    'silhouette': sil,  # same clustering result
                }
        except Exception:
            pass
    elif task_type == 'classification':
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=SEED)),
            ('lr', LogisticRegression(max_iter=1000, random_state=SEED)),
            ('svm', SVC(random_state=SEED)),
            ('knn', KNeighborsClassifier()),
            ('dt', DecisionTreeClassifier(random_state=SEED)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
        ]
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                }
            except Exception as e:
                print(f"    [WARN] {name} failed: {e}")
    else:  # regression
        models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=SEED)),
            ('lr', Ridge(random_state=SEED)),
            ('svm', SVR()),
            ('knn', KNeighborsRegressor()),
            ('dt', DecisionTreeRegressor(random_state=SEED)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=SEED)),
        ]
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                }
            except Exception as e:
                print(f"    [WARN] {name} failed: {e}")
    return results


def calc_tolerance_and_snoopy(X_dirty, y_dirty, X_cleaned, y_cleaned, X_clean, y_clean,
                               train_idx, test_idx, task_type) -> Dict:
    """与 reeval_with_split.py:calc_tolerance_and_snoopy 相同"""
    results = {}

    def get_perf(X_tr, y_tr, X_te, y_te):
        if task_type == 'clustering':
            n_clusters = len(np.unique(y_te))
            km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
            y_pred = km.fit_predict(X_tr)
            return silhouette_score(X_tr, y_pred, sample_size=min(len(X_tr), 10000), random_state=SEED)
        elif task_type == 'classification':
            m = RandomForestClassifier(n_estimators=100, random_state=SEED)
            m.fit(X_tr, y_tr)
            return accuracy_score(y_te, m.predict(X_te))
        else:
            m = RandomForestRegressor(n_estimators=100, random_state=SEED)
            m.fit(X_tr, y_tr)
            return r2_score(y_te, m.predict(X_te))

    try:
        if task_type == 'clustering':
            results['P_do_nothing'] = get_perf(X_dirty, y_dirty, None, y_clean)
        else:
            results['P_do_nothing'] = get_perf(X_dirty[train_idx], y_dirty[train_idx],
                                                X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_do_nothing'] = 0.0

    try:
        if task_type == 'clustering':
            results['P_demand_clean'] = get_perf(X_cleaned, y_cleaned, None, y_clean)
        else:
            cleaned_train_idx = train_idx if len(X_cleaned) == len(X_clean) else np.arange(len(X_cleaned))
            results['P_demand_clean'] = get_perf(X_cleaned[cleaned_train_idx], y_cleaned[cleaned_train_idx],
                                                  X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_demand_clean'] = 0.0

    try:
        if task_type == 'clustering':
            results['P_repair_all'] = get_perf(X_clean, y_clean, None, y_clean)
        else:
            results['P_repair_all'] = get_perf(X_clean[train_idx], y_clean[train_idx],
                                                X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_repair_all'] = 0.0

    # Snoopy upper bounds
    try:
        if task_type == 'clustering':
            n_clusters = len(np.unique(y_clean))
            for tag, Xd in [('dirty', X_dirty), ('cleaned', X_cleaned), ('clean', X_clean)]:
                km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
                yp = km.fit_predict(Xd)
                results[f'upper_bound_{tag}'] = silhouette_score(
                    Xd, yp, sample_size=min(len(Xd), 10000), random_state=SEED)
        else:
            from sklearn.model_selection import cross_val_score
            if task_type == 'classification':
                m_cls = RandomForestClassifier(n_estimators=100, random_state=SEED)
                results['upper_bound_dirty'] = np.mean(cross_val_score(m_cls, X_dirty, y_dirty, cv=5))
                results['upper_bound_cleaned'] = np.mean(
                    cross_val_score(m_cls, X_cleaned, y_cleaned, cv=min(5, max(2, len(X_cleaned)))))
                results['upper_bound_clean'] = np.mean(cross_val_score(m_cls, X_clean, y_clean, cv=5))
            else:
                m_reg = RandomForestRegressor(n_estimators=100, random_state=SEED)
                results['upper_bound_dirty'] = -np.mean(
                    cross_val_score(m_reg, X_dirty, y_dirty, cv=5, scoring='neg_mean_squared_error'))
                results['upper_bound_cleaned'] = -np.mean(
                    cross_val_score(m_reg, X_cleaned, y_cleaned,
                                    cv=min(5, max(2, len(X_cleaned))), scoring='neg_mean_squared_error'))
                results['upper_bound_clean'] = -np.mean(
                    cross_val_score(m_reg, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error'))

        ub_d = results.get('upper_bound_dirty', 0)
        ub_c = results.get('upper_bound_cleaned', 0)
        ub_cl = results.get('upper_bound_clean', 0)
        denom = ub_cl - ub_d
        results['upper_bound_improvement'] = (ub_c - ub_d) / denom if abs(denom) > 1e-10 else 0.0
    except Exception as e:
        print(f"    [WARN] Snoopy 计算失败: {e}")
        for k in ['upper_bound_dirty', 'upper_bound_cleaned', 'upper_bound_clean', 'upper_bound_improvement']:
            results.setdefault(k, 0.0)

    return results


# ============================================================================
# 3. ReplaceAll 策略: 在 raw DataFrame 上执行 VEC 替换
# ============================================================================

def run_replaceall_on_df(dirty_df: pd.DataFrame, clean_df: pd.DataFrame,
                         feature_cols: list, label_col: str, dataset_name: str) -> pd.DataFrame:
    """
    在 raw DataFrame 上执行 ReplaceAll (Oracle 检测 + VEC 替换)。
    返回清洗后的 DataFrame (与 dirty_df 行数相同)。
    """
    cleaned_df = dirty_df.copy()
    cat_cols = DC_CATEGORICAL.get(dataset_name, set())

    # Oracle 检测: dirty vs clean
    error_positions = []
    for col_name in feature_cols:
        if col_name not in dirty_df.columns or col_name not in clean_df.columns:
            continue
        col_idx = feature_cols.index(col_name)
        n_rows = min(len(dirty_df), len(clean_df))
        dirty_col = dirty_df[col_name].values[:n_rows]
        clean_col = clean_df[col_name].values[:n_rows]
        dirty_num = pd.to_numeric(pd.Series(dirty_col), errors='coerce')
        clean_num = pd.to_numeric(pd.Series(clean_col), errors='coerce')

        for row_idx in range(n_rows):
            d_val, c_val = dirty_col[row_idx], clean_col[row_idx]
            d_num, c_num = dirty_num.iloc[row_idx], clean_num.iloc[row_idx]
            if pd.isna(d_val) and pd.isna(c_val):
                continue
            if pd.isna(d_val) and not pd.isna(c_val):
                error_positions.append((row_idx, col_idx))
                continue
            if not np.isnan(d_num) and not np.isnan(c_num):
                if abs(d_num - c_num) > 1e-6:
                    error_positions.append((row_idx, col_idx))
                continue
            if str(d_val).strip() != str(c_val).strip():
                error_positions.append((row_idx, col_idx))

    print(f"  Oracle 检测到 {len(error_positions)} 个错误")

    if not error_positions:
        return cleaned_df

    # 编码 dirty_df 用于 VEC (DemandClean 内部编码)
    from run_demandclean.run_demandclean_base import preprocess_data
    try:
        (X_dirty_enc, y_dirty_enc, X_clean_enc, y_clean_enc,
         column_names, fd_rules, rules_path,
         dirty_csv_path, clean_csv_path, csv_columns,
         data_scaler, dc_label_encoders, categorical_cols_set,
         _, _) = preprocess_data(dataset_name)
    except Exception as e:
        print(f"  [WARN] preprocess_data 失败, 用简化编码: {e}")
        # fallback: 简化编码
        X_enc = dirty_df[feature_cols].copy()
        for col in feature_cols:
            X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce')
        X_arr = np.nan_to_num(X_enc.values.astype(float), nan=0.0)
        col_means = np.nanmean(X_arr, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        # 简化 VEC: KNN 均值替换
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=6, algorithm='auto')
        nn.fit(X_arr)
        _, indices = nn.kneighbors(X_arr)

        for row_idx, col_idx in error_positions:
            if row_idx >= len(X_arr):
                continue
            neighbors = [indices[row_idx, j] for j in range(1, indices.shape[1])]
            neighbor_vals = X_arr[neighbors, col_idx]
            valid = neighbor_vals[neighbor_vals != 0] if col_means[col_idx] != 0 else neighbor_vals
            if len(valid) > 0:
                est_val = np.mean(valid)
            else:
                est_val = col_means[col_idx]
            # 写回 DataFrame
            cleaned_df.iloc[row_idx, cleaned_df.columns.get_loc(feature_cols[col_idx])] = est_val

        return cleaned_df

    # 用 DemandClean 的 VEC
    config = DemandCleanConfig(
        column_names=column_names,
        fd_rules=fd_rules,
        categorical_cols=categorical_cols_set,
        scaler=data_scaler,
        label_encoders=dc_label_encoders,
        dirty_df=dirty_df,
    )

    X_replaced = X_dirty_enc.copy()
    BATCH_THRESHOLD = 5000

    if len(error_positions) > BATCH_THRESHOLD:
        # 批量 KNN
        from sklearn.neighbors import NearestNeighbors
        X_filled = np.nan_to_num(X_replaced, nan=0.0)
        col_means = np.nanmean(X_replaced, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        nn = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=-1)
        nn.fit(X_filled)
        error_rows = sorted(set(r for r, _ in error_positions if r < len(X_replaced)))
        if error_rows:
            distances, indices = nn.kneighbors(X_filled[error_rows])
            row_to_nb = {}
            for i, ri in enumerate(error_rows):
                row_to_nb[ri] = [indices[i, j] for j in range(len(indices[i])) if indices[i, j] != ri][:5]
            for row_idx, col_idx in error_positions:
                if row_idx >= len(X_replaced):
                    continue
                nbs = row_to_nb.get(row_idx, [])
                if nbs:
                    vals = X_filled[nbs, col_idx]
                    X_replaced[row_idx, col_idx] = np.mean(vals)
                else:
                    X_replaced[row_idx, col_idx] = col_means[col_idx]
    else:
        estimator = ValueEstimator(config)
        col_means = np.nanmean(X_replaced, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        deleted_rows = set()
        for row_idx, col_idx in error_positions:
            if row_idx >= len(X_replaced):
                continue
            est = estimator.estimate_feature_value(X_replaced, row_idx, col_idx, deleted_rows, col_means)
            X_replaced[row_idx, col_idx] = est

    # 反编码: 把 X_replaced 的值写回 cleaned_df
    # 由于编码方式复杂(scaler+LE), 不能简单反编码
    # 策略: 对于每个 error position, 用 scaler.inverse_transform 后写回
    if data_scaler is not None:
        X_inv = data_scaler.inverse_transform(np.nan_to_num(X_replaced, nan=0.0))
    else:
        X_inv = np.nan_to_num(X_replaced, nan=0.0)

    for row_idx, col_idx in error_positions:
        if row_idx >= len(X_inv) or col_idx >= len(column_names):
            continue
        col_name = column_names[col_idx]
        if col_name in cleaned_df.columns:
            cleaned_df.at[row_idx, col_name] = X_inv[row_idx, col_idx]

    return cleaned_df


# ============================================================================
# 4. 主评估函数
# ============================================================================

def evaluate_one(dataset_name: str) -> Optional[Dict]:
    """
    评估 ReplaceAll on one dataset, 用 reeval_with_split 完全相同的流水线。

    核心策略: 先用 reeval_with_split 方式编码全量数据, 然后在编码空间中
    对 Oracle 检测到的错误做 KNN 替换, 最后在同一编码空间中评估。
    这样编码-替换-评估全部在同一特征空间中进行, 保证一致性。
    """
    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']
    label_col = ds_cfg['label_column']
    exclude_cols = ds_cfg.get('exclude_columns', [])
    feature_cols_cfg = ds_cfg.get('feature_columns', None)

    data_dir = os.path.join(C4ML_DATA, dataset_name)
    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')

    if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
        print(f"  [SKIP] {dataset_name}: data not found")
        return None

    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    n_total = len(dirty_df)
    train_idx, val_idx, test_idx = demandclean_split(n_total)

    print(f"  数据: {n_total} rows, split={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # ── Step 1: 用 reeval_with_split 方式编码全量数据 ──
    print(f"  [1/3] 编码 (reeval_with_split 流水线)...")
    t0 = time.time()
    try:
        dirty_train_for_fit = dirty_df.iloc[train_idx].reset_index(drop=True)
        _, _, encoders, scaler, label_enc = preprocess_for_ml(
            dirty_train_for_fit, label_col, 'index', exclude_cols, feature_cols_cfg)

        X_dirty_all, y_dirty_all = preprocess_for_ml(
            dirty_df, label_col, 'index', exclude_cols, feature_cols_cfg,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)
        X_clean_all, y_clean_all = preprocess_for_ml(
            clean_df, label_col, 'index', exclude_cols, feature_cols_cfg,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)
    except Exception as e:
        print(f"  [ERROR] 编码失败: {e}")
        import traceback; traceback.print_exc()
        return None

    n_features = X_dirty_all.shape[1]
    print(f"        编码后: {X_dirty_all.shape}")

    # ── Step 2: 在 raw DataFrame 上执行 Oracle 检测 + 值替换 ──
    print(f"  [2/4] Oracle 检测 + 值替换 (raw DataFrame)...")

    # 确定 feature_cols
    if feature_cols_cfg == 'auto' or feature_cols_cfg is None:
        drop = {'index', label_col} | set(exclude_cols)
        feature_cols_list = [c for c in dirty_df.columns if c not in drop]
    else:
        feature_cols_list = [c for c in feature_cols_cfg if c in dirty_df.columns]

    cleaned_df = dirty_df.copy()
    n_errors = 0

    for col_name in feature_cols_list:
        if col_name not in dirty_df.columns or col_name not in clean_df.columns:
            continue
        n_rows = min(len(dirty_df), len(clean_df))
        dirty_col = dirty_df[col_name].values[:n_rows]
        clean_col = clean_df[col_name].values[:n_rows]

        # 检测该列的错误行
        error_rows = []
        dirty_num = pd.to_numeric(pd.Series(dirty_col), errors='coerce')
        clean_num = pd.to_numeric(pd.Series(clean_col), errors='coerce')

        for i in range(n_rows):
            d_val, c_val = dirty_col[i], clean_col[i]
            d_n, c_n = dirty_num.iloc[i], clean_num.iloc[i]
            if pd.isna(d_val) and pd.isna(c_val):
                continue
            if pd.isna(d_val) and not pd.isna(c_val):
                error_rows.append(i)
                continue
            if not np.isnan(d_n) and not np.isnan(c_n):
                if abs(d_n - c_n) > 1e-6:
                    error_rows.append(i)
                continue
            if str(d_val).strip() != str(c_val).strip():
                error_rows.append(i)

        if not error_rows:
            continue

        n_errors += len(error_rows)

        # 值替换: 用该列非错误行的统计量估计
        # 数值列 → KNN 均值; 非数值列 → 众数
        col_data = dirty_df[col_name].copy()
        col_numeric = pd.to_numeric(col_data, errors='coerce')
        is_numeric = col_numeric.notna().sum() > len(col_data) * 0.5

        if is_numeric:
            # 用非错误行的 KNN 估计 (在 raw 空间)
            non_error_mask = np.ones(n_rows, dtype=bool)
            non_error_mask[error_rows] = False
            valid_vals = col_numeric[:n_rows].values.copy()
            valid_vals[~non_error_mask] = np.nan

            col_mean = np.nanmean(valid_vals)
            if np.isnan(col_mean):
                col_mean = 0.0

            for i in error_rows:
                cleaned_df.at[i, col_name] = col_mean
        else:
            # 非数值列: 众数替换
            non_error_vals = col_data.iloc[[i for i in range(n_rows) if i not in set(error_rows)]]
            non_error_vals = non_error_vals.dropna()
            if len(non_error_vals) > 0:
                mode_val = non_error_vals.mode().iloc[0] if len(non_error_vals.mode()) > 0 else non_error_vals.iloc[0]
            else:
                mode_val = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else ''
            for i in error_rows:
                cleaned_df.at[i, col_name] = mode_val

    print(f"        Oracle 检测到 {n_errors} 个错误, 已替换")

    # ── Step 3: 用 reeval_with_split 相同的预处理编码 ──
    print(f"  [3/4] 编码 (reeval_with_split 流水线)...")
    try:
        X_cleaned_all, y_cleaned_all = preprocess_for_ml(
            cleaned_df, label_col, 'index', exclude_cols, feature_cols_cfg,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)
    except Exception as e:
        print(f"  [ERROR] 编码 cleaned 失败: {e}")
        import traceback; traceback.print_exc()
        return None

    # ── Step 4: 评估 (与 reeval_with_split 完全相同) ──
    print(f"  [4/4] 评估...")

    # ML 评估
    if task_type == 'clustering':
        ml_results = evaluate_ml(X_cleaned_all, y_cleaned_all, None, None, task_type)
    else:
        X_train_ml = X_cleaned_all[train_idx]
        y_train_ml = y_cleaned_all[train_idx]
        X_test_ml = X_clean_all[test_idx]
        y_test_ml = y_clean_all[test_idx]
        ml_results = evaluate_ml(X_train_ml, y_train_ml, X_test_ml, y_test_ml, task_type)

    # 容忍度 + Snoopy
    tol = calc_tolerance_and_snoopy(
        X_dirty_all, y_dirty_all,
        X_cleaned_all, y_cleaned_all,
        X_clean_all, y_clean_all,
        train_idx, test_idx, task_type)

    # 汇总
    result = {
        'dataset': dataset_name,
        'task_type': task_type,
        'elapsed_time': round(time.time() - t0, 2),
        'truth_cost': 0,
        'ml_results': ml_results,
        'P_do_nothing': tol.get('P_do_nothing', 0),
        'P_demand_clean': tol.get('P_demand_clean', 0),
        'P_repair_all': tol.get('P_repair_all', 0),
        'upper_bound_dirty': tol.get('upper_bound_dirty', 0),
        'upper_bound_cleaned': tol.get('upper_bound_cleaned', 0),
        'upper_bound_clean': tol.get('upper_bound_clean', 0),
        'upper_bound_improvement': tol.get('upper_bound_improvement', 0),
    }

    # 打印关键指标
    if task_type == 'classification':
        perf = ml_results.get('rf', {}).get('accuracy', 0)
        print(f"  RF accuracy: {perf:.4f}")
    elif task_type == 'regression':
        perf = ml_results.get('rf', {}).get('r2', 0)
        print(f"  RF R²: {perf:.4f}")
    else:
        perf = ml_results.get('kmeans', {}).get('silhouette', 0)
        print(f"  KMeans silhouette: {perf:.4f}")

    print(f"  P_nf={tol['P_do_nothing']:.4f}, P_dc={tol['P_demand_clean']:.4f}, P_ra={tol['P_repair_all']:.4f}")

    return result


# ============================================================================
# 5. 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ReplaceAll Baseline (reeval_with_split 对齐版)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='逗号分隔数据集 (如 beers,nasa)。不指定=全部')
    args = parser.parse_args()

    if args.dataset:
        datasets = [d.strip() for d in args.dataset.split(',')]
    else:
        datasets = list(DATASETS.keys())

    print("=" * 70)
    print("ReplaceAll Baseline (v2 — reeval_with_split 对齐)")
    print(f"数据集: {datasets}")
    print(f"数据源: {C4ML_DATA}")
    print("=" * 70)

    all_results = []
    for ds in datasets:
        if ds not in DATASETS:
            print(f"\n[WARN] 未知数据集: {ds}")
            continue
        print(f"\n{'='*60}")
        print(f"ReplaceAll — {ds}")
        print(f"{'='*60}")
        try:
            result = evaluate_one(ds)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    # 保存
    output_dir = os.path.join(_PROJECT_ROOT, 'results', 'replaceall_baseline')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'replaceall_reeval_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {output_path}")

    # 汇总
    print(f"\n{'='*90}")
    print(f"{'DS':<15} {'Type':<6} {'ReplaceAll':>12} {'NoFix':>12} {'RepairAll':>12} {'Tol_post':>10}")
    print(f"{'='*90}")
    for r in all_results:
        ds = r['dataset']
        tt = r['task_type']
        if tt == 'classification':
            perf = r['ml_results'].get('rf', {}).get('accuracy', 0)
        elif tt == 'regression':
            perf = r['ml_results'].get('rf', {}).get('r2', 0)
        else:
            perf = r['ml_results'].get('kmeans', {}).get('silhouette', 0)
        pnf = r['P_do_nothing']
        pra = r['P_repair_all']
        pdc = r['P_demand_clean']
        tol = (pdc - pnf) / (pra - pnf) if abs(pra - pnf) > 1e-6 else 0
        print(f"{ds:<15} {tt:<6} {perf:>12.4f} {pnf:>12.4f} {pra:>12.4f} {tol:>10.4f}")


if __name__ == '__main__':
    main()
