#!/usr/bin/env python3
"""
reeval_with_split.py - 快速重评估脚本

利用 Clean4MLBaseline 中已有的 cleaned CSV，
按照 DemandClean 相同的 seed=42, 60/20/20 划分重新评估，
输出与 DemandClean 格式完全一致的指标表 CSV。

Usage:
    python tools/reeval_with_split.py
    python tools/reeval_with_split.py --baselines horizon donothing --datasets beers adult
    python tools/reeval_with_split.py --output results/reeval_results.csv
    python tools/reeval_with_split.py --baselines repairall --datasets har
"""

import os
import sys
import csv
import re
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

warnings.filterwarnings('ignore')

# ─── 项目路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # Clean4MLBaseline 根目录
sys.path.insert(0, _SCRIPT_DIR)

from datasets_config import DATASETS_CONFIG

# ─── 常量 ───
SEED = 42
ALL_BASELINES = [
    'activeclean', 'boostclean', 'ctxpipe', 'deleteall', 'donothing',
    'holoclean', 'horizon', 'lopster', 'mlimputer', 'repairall',
    'simpleimputer', 'uniclean', 'raha_baran'
]
ALL_DATASETS = list(DATASETS_CONFIG.keys())

CLASSIFICATION_MODELS = ['rf', 'lr', 'svm', 'knn', 'dt', 'gb']
REGRESSION_MODELS = ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb']
CLUSTERING_MODELS = ['kmeans', 'agglomerative', 'spectral']

# 自动方法 → 真值成本=0
AUTO_METHODS = {
    'donothing', 'deleteall', 'simpleimputer', 'mlimputer',
    'horizon', 'holoclean', 'uniclean', 'ctxpipe'
}


# ═══════════════════════════════════════════════════════════════
# 1. 数据划分（与 DemandClean 完全一致）
# ═══════════════════════════════════════════════════════════════

def demandclean_split(n_total: int, seed: int = SEED):
    """返回与 DemandClean run_demandclean_base.py:2659-2663 完全相同的划分索引"""
    all_idx = np.arange(n_total)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════
# 2. 特征预处理
# ═══════════════════════════════════════════════════════════════

def preprocess_for_ml(df: pd.DataFrame, label_col: str, index_col: str = 'index',
                      exclude_cols: List[str] = None, feature_cols=None,
                      fitted_encoders: Dict = None, fitted_scaler=None, fitted_label_encoder=None):
    """
    预处理 DataFrame → (X_scaled, y)，与 getScoreML.py 对齐。

    如果提供 fitted_encoders/fitted_scaler/fitted_label_encoder，则使用它们进行 transform；
    否则创建新的编码器并 fit_transform。

    Returns:
        如果是 fit 模式：(X_scaled, y, encoders, scaler, label_encoder)
        如果是 transform 模式：(X_scaled, y)
    """
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

    # 编码分类列
    encoders = {} if is_fit_mode else fitted_encoders
    for col in X.columns:
        if X[col].dtype == 'object':
            # 先尝试转数值：如果 >50% 的非空值能转数值，则当数值列处理
            numeric_attempt = pd.to_numeric(X[col], errors='coerce')
            non_null = X[col].notna().sum()
            numeric_ratio = numeric_attempt.notna().sum() / max(non_null, 1)
            if numeric_ratio > 0.5:
                # 数值列混入了字符串脏值（如 'empty'）→ to_numeric + fillna(0)
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
                    # 处理未见过的类别
                    known = set(le.classes_)
                    X[col] = X[col].apply(lambda v: v if v in known else '__UNKNOWN__')
                    if '__UNKNOWN__' not in known:
                        le.classes_ = np.append(le.classes_, '__UNKNOWN__')
                    X[col] = le.transform(X[col])
                else:
                    # 回退：创建新编码器
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

    # 标准化
    if is_fit_mode:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = fitted_scaler
        X_scaled = scaler.transform(X) if scaler else StandardScaler().fit_transform(X)

    # 编码标签
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


# ═══════════════════════════════════════════════════════════════
# 3. ML 评估（使用预划分的 train/test）
# ═══════════════════════════════════════════════════════════════

def get_model(name: str, task_type: str):
    if task_type == 'classification':
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=SEED),
            'lr': LogisticRegression(max_iter=1000, random_state=SEED),
            'svm': SVC(random_state=SEED),
            'knn': KNeighborsClassifier(),
            'dt': DecisionTreeClassifier(random_state=SEED),
            'gb': GradientBoostingClassifier(random_state=SEED),
        }
    elif task_type == 'regression':
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=SEED),
            'lr': LinearRegression(),
            'ridge': Ridge(solver='lsqr', random_state=SEED),
            'lasso': Lasso(random_state=SEED),
            'knn': KNeighborsRegressor(),
            'gb': GradientBoostingRegressor(random_state=SEED),
        }
    else:
        return None
    return models.get(name)


def evaluate_ml_split(X_train, y_train, X_test, y_test, task_type: str, models: List[str]) -> Dict:
    """在预划分数据上评估多个模型"""
    MAX_CLUSTERING_SAMPLES = 10000  # 聚类任务采样上限
    results = {}
    for name in models:
        try:
            if task_type == 'clustering':
                # 对聚类任务进行采样（避免大数据集上的性能问题）
                if len(X_train) > MAX_CLUSTERING_SAMPLES:
                    rng = np.random.RandomState(SEED)
                    idx = rng.choice(len(X_train), MAX_CLUSTERING_SAMPLES, replace=False)
                    X_sample, y_sample = X_train[idx], y_train[idx]
                else:
                    X_sample, y_sample = X_train, y_train
                n_clusters = len(np.unique(y_sample))
                if name == 'kmeans':
                    m = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
                elif name == 'agglomerative':
                    m = AgglomerativeClustering(n_clusters=n_clusters)
                elif name == 'spectral':
                    m = SpectralClustering(n_clusters=n_clusters, random_state=SEED, affinity='nearest_neighbors')
                else:
                    continue
                y_pred = m.fit_predict(X_sample)
                sil = silhouette_score(X_sample, y_pred,
                                       sample_size=min(len(X_sample), 10000), random_state=SEED)
                ari = adjusted_rand_score(y_sample, y_pred)  # y_train 是 cleaned 标签
                results[name] = {'silhouette': sil, 'ari': ari}
            else:
                model = get_model(name, task_type)
                if model is None:
                    continue
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if task_type == 'classification':
                    results[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred, average='weighted'),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                    }
                else:
                    results[name] = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                    }
        except Exception as e:
            print(f"    [WARN] {name} 评估失败: {e}")
            results[name] = {}
    return results


# ═══════════════════════════════════════════════════════════════
# 4. 传统清洗指标
# ═══════════════════════════════════════════════════════════════

def calc_traditional_metrics(dirty_df, cleaned_df, clean_df, index_col='index', label_col=None,
                              mse_attributes=None):
    """计算传统清洗指标: precision, recall, f1, edr, r_edr, hybrid_distance, col_avg_rmse, col_avg_f1

    与 DemandClean 的 getScore.py 对齐：
    - hybrid_distance = 0.5 * avg_mse + 0.5 * avg_jaccard
    - col_avg_rmse: 数值列的平均 RMSE
    - col_avg_f1: 类别列的平均 F1（按列计算）
    """
    metrics = {
        'f1_score': 0.0, 'edr': 0.0, 'r_edr': 0.0, 'hybrid_distance': 0.0,
        'col_avg_rmse': 0.0, 'col_avg_f1': 0.0,
    }

    cols = [c for c in dirty_df.columns if c != index_col]
    n = min(len(dirty_df), len(cleaned_df), len(clean_df))
    if n == 0:
        return metrics

    # 向量化比较: 转为小写字符串
    dirty_vals = dirty_df[cols].iloc[:n].astype(str).apply(lambda x: x.str.strip().str.lower())
    cleaned_vals = cleaned_df[cols].iloc[:n].astype(str).apply(lambda x: x.str.strip().str.lower())
    clean_vals = clean_df[cols].iloc[:n].astype(str).apply(lambda x: x.str.strip().str.lower())

    is_error = (dirty_vals != clean_vals)
    is_changed = (dirty_vals != cleaned_vals)
    is_correct = (cleaned_vals == clean_vals)
    is_repaired_wrong = ~is_correct

    tp = (is_changed & is_correct).values.sum()
    fp = (is_changed & ~is_correct).values.sum()
    fn = (is_error & ~is_changed).values.sum()

    dist_dirty_to_clean = is_error.values.sum()
    dist_repaired_to_clean = is_repaired_wrong.values.sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    edr = (dist_dirty_to_clean - dist_repaired_to_clean) / dist_dirty_to_clean if dist_dirty_to_clean > 0 else 0

    # r_edr: 仅在修改过的单元格上
    changed_mask = is_changed
    r_dist_dirty = (changed_mask & is_error).values.sum()
    r_dist_repaired = (changed_mask & is_repaired_wrong).values.sum()
    r_edr = (r_dist_dirty - r_dist_repaired) / r_dist_dirty if r_dist_dirty > 0 else 0

    # hybrid_distance: 与 DemandClean 对齐 (0.5 * MSE + 0.5 * Jaccard)
    # 自动检测数值列（mse_attributes）
    if mse_attributes is None:
        mse_attributes = []
        for col in cols:
            try:
                pd.to_numeric(clean_df[col].iloc[:n], errors='raise')
                mse_attributes.append(col)
            except (ValueError, TypeError):
                pass

    mse_cols = [c for c in mse_attributes if c in cols]
    jaccard_cols = [c for c in cols if c not in mse_cols]

    # 计算 RMSE（数值列）→ col_avg_rmse
    rmse_list = []
    mse_list = []
    for col in mse_cols:
        try:
            cleaned_num = pd.to_numeric(cleaned_df[col].iloc[:n], errors='coerce').fillna(0)
            clean_num = pd.to_numeric(clean_df[col].iloc[:n], errors='coerce').fillna(0)
            col_mse = ((cleaned_num - clean_num) ** 2).mean()
            col_rmse = np.sqrt(col_mse)
            mse_list.append(col_mse)
            rmse_list.append(col_rmse)
        except Exception:
            pass
    avg_mse = np.mean(mse_list) if mse_list else 0.0
    col_avg_rmse = np.mean(rmse_list) if rmse_list else 0.0

    # 计算按列 F1（类别列）→ col_avg_f1
    # 对每列：TP=修改后正确, FP=修改后错误, FN=未修改但是错误
    col_f1_list = []
    jaccard_list = []
    for col in jaccard_cols:
        col_is_error = (dirty_vals[col] != clean_vals[col])
        col_is_changed = (dirty_vals[col] != cleaned_vals[col])
        col_is_correct = (cleaned_vals[col] == clean_vals[col])

        col_tp = (col_is_changed & col_is_correct).sum()
        col_fp = (col_is_changed & ~col_is_correct).sum()
        col_fn = (col_is_error & ~col_is_changed).sum()

        col_prec = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else 0
        col_rec = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 0
        col_f1 = 2 * col_prec * col_rec / (col_prec + col_rec) if (col_prec + col_rec) > 0 else 0
        # 如果没有错误，F1=1（完美）
        if col_is_error.sum() == 0:
            col_f1 = 1.0
        col_f1_list.append(col_f1)

        diff = (~col_is_correct).sum()
        jaccard_list.append(diff / n)

    avg_jaccard = np.mean(jaccard_list) if jaccard_list else 0.0
    col_avg_f1 = np.mean(col_f1_list) if col_f1_list else 1.0

    # 混合距离: w1=0.5, w2=0.5（与 DemandClean 一致）
    if mse_list and jaccard_list:
        hybrid_distance = 0.5 * avg_mse + 0.5 * avg_jaccard
    elif mse_list:
        hybrid_distance = avg_mse
    else:
        hybrid_distance = avg_jaccard

    metrics['f1_score'] = f1
    metrics['edr'] = edr
    metrics['r_edr'] = r_edr
    metrics['hybrid_distance'] = hybrid_distance
    metrics['col_avg_rmse'] = col_avg_rmse
    metrics['col_avg_f1'] = col_avg_f1
    return metrics


# ═══════════════════════════════════════════════════════════════
# 5. 模型容忍度 + Snoopy 上界（使用预划分索引）
# ═══════════════════════════════════════════════════════════════

def calc_tolerance_and_snoopy(X_dirty, y_dirty, X_cleaned, y_cleaned, X_clean, y_clean,
                               train_idx, test_idx, task_type, model_name='rf',
                               cleaned_train_idx=None):
    """
    计算 P_do_nothing, P_demand_clean, P_repair_all 以及 snoopy upper_bound。
    使用预划分的 train/test 索引。

    Args:
        cleaned_train_idx: 可选，cleaned 数据的训练索引。当 cleaned 行数与原始不同时使用。
    """
    if cleaned_train_idx is None:
        cleaned_train_idx = train_idx
    from sklearn.model_selection import cross_val_score

    results = {}

    def get_perf(X_train, y_train, X_test, y_test):
        if task_type == 'clustering':
            n_clusters = len(np.unique(y_test))
            km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
            y_pred = km.fit_predict(X_train)
            try:
                return silhouette_score(X_train, y_pred,
                                        sample_size=min(len(X_train), 10000), random_state=SEED)
            except Exception:
                return 0.0
        elif task_type == 'classification':
            m = get_model(model_name, 'classification')
            m.fit(X_train, y_train)
            return accuracy_score(y_test, m.predict(X_test))
        else:
            m = get_model(model_name, 'regression')
            m.fit(X_train, y_train)
            return r2_score(y_test, m.predict(X_test))

    # P_do_nothing: 脏数据训练，干净数据测试
    try:
        if task_type == 'clustering':
            results['P_do_nothing'] = get_perf(X_dirty, y_dirty, None, y_clean)
        else:
            results['P_do_nothing'] = get_perf(
                X_dirty[train_idx], y_dirty[train_idx],
                X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_do_nothing'] = 0.0

    # P_demand_clean: 清洗后数据训练，干净数据测试
    try:
        if task_type == 'clustering':
            results['P_demand_clean'] = get_perf(X_cleaned, y_cleaned, None, y_clean)
        else:
            # 使用 cleaned_train_idx（可能与 train_idx 不同，当 deleteall 等删除了行时）
            results['P_demand_clean'] = get_perf(
                X_cleaned[cleaned_train_idx], y_cleaned[cleaned_train_idx],
                X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_demand_clean'] = 0.0

    # P_repair_all: 干净数据训练，干净数据测试
    try:
        if task_type == 'clustering':
            results['P_repair_all'] = get_perf(X_clean, y_clean, None, y_clean)
        else:
            results['P_repair_all'] = get_perf(
                X_clean[train_idx], y_clean[train_idx],
                X_clean[test_idx], y_clean[test_idx])
    except Exception:
        results['P_repair_all'] = 0.0

    # Snoopy upper bounds (cross-validation，与 DemandClean 对齐：无采样)
    try:
        if task_type == 'classification':
            m_cls = RandomForestClassifier(n_estimators=100, random_state=SEED)
            results['upper_bound_dirty'] = np.mean(cross_val_score(m_cls, X_dirty, y_dirty, cv=5))
            results['upper_bound_cleaned'] = np.mean(
                cross_val_score(m_cls, X_cleaned, y_cleaned, cv=min(5, max(2, len(X_cleaned)))))
            results['upper_bound_clean'] = np.mean(cross_val_score(m_cls, X_clean, y_clean, cv=5))
        elif task_type == 'regression':
            m_reg = RandomForestRegressor(n_estimators=100, random_state=SEED)
            results['upper_bound_dirty'] = -np.mean(
                cross_val_score(m_reg, X_dirty, y_dirty, cv=5, scoring='neg_mean_squared_error'))
            results['upper_bound_cleaned'] = -np.mean(
                cross_val_score(m_reg, X_cleaned, y_cleaned,
                                cv=min(5, max(2, len(X_cleaned))), scoring='neg_mean_squared_error'))
            results['upper_bound_clean'] = -np.mean(
                cross_val_score(m_reg, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error'))
        else:  # clustering
            n_clusters = len(np.unique(y_clean))
            for tag, Xd, yd in [('dirty', X_dirty, y_dirty),
                                 ('cleaned', X_cleaned, y_cleaned),
                                 ('clean', X_clean, y_clean)]:
                km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
                yp = km.fit_predict(Xd)
                results[f'upper_bound_{tag}'] = silhouette_score(
                    Xd, yp, sample_size=min(len(Xd), 10000), random_state=SEED)

        ub_dirty = results.get('upper_bound_dirty', 0)
        ub_cleaned = results.get('upper_bound_cleaned', 0)
        ub_clean = results.get('upper_bound_clean', 0)
        denom = ub_clean - ub_dirty
        results['upper_bound_improvement'] = (ub_cleaned - ub_dirty) / denom if abs(denom) > 1e-10 else 0.0
    except Exception as e:
        print(f"    [WARN] Snoopy 上界计算失败: {e}")
        for k in ['upper_bound_dirty', 'upper_bound_cleaned', 'upper_bound_clean', 'upper_bound_improvement']:
            results.setdefault(k, 0.0)

    return results


# ═══════════════════════════════════════════════════════════════
# 6. 日志解析：提取执行时间和真值成本
# ═══════════════════════════════════════════════════════════════

def extract_from_logs(result_dir: str, baseline: str) -> Tuple[Optional[float], Optional[int]]:
    """从 baseline 结果目录中提取执行时间和真值成本"""
    elapsed_time = None
    truth_cost = None

    log_files = glob.glob(os.path.join(result_dir, '*.log'))
    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # 执行时间
            m = re.search(r'执行时间:\s*([\d.]+)\s*秒', content)
            if m:
                elapsed_time = float(m.group(1))
            # 真值成本
            m = re.search(r'真值使用[量成本]*[^:]*:\s*(\d+)', content)
            if m:
                truth_cost = int(m.group(1))
        except Exception:
            pass

    # 自动方法 → 真值 = 0
    if baseline in AUTO_METHODS and truth_cost is None:
        truth_cost = 0

    return elapsed_time, truth_cost


# ═══════════════════════════════════════════════════════════════
# 7. 主流程：单个 (baseline, dataset) 评估
# ═══════════════════════════════════════════════════════════════

def find_cleaned_csv(baseline: str, dataset: str) -> Optional[str]:
    """查找 cleaned CSV 文件路径"""
    results_root = os.path.join(_PROJECT_ROOT, 'results', baseline)
    # 尝试多种命名模式
    for suffix in ['_vzekai', '']:
        dir_name = f"{dataset}_{baseline}{suffix}"
        result_dir = os.path.join(results_root, dir_name)
        if os.path.isdir(result_dir):
            # 尝试 *_cleaned.csv（大多数baseline）和 *_output.csv（activeclean）
            for pattern in ['*_cleaned.csv', '*_output.csv']:
                csvs = glob.glob(os.path.join(result_dir, pattern))
                if csvs:
                    return csvs[0]
    return None


def find_result_dir(baseline: str, dataset: str) -> Optional[str]:
    """查找结果目录"""
    results_root = os.path.join(_PROJECT_ROOT, 'results', baseline)
    for suffix in ['_vzekai', '']:
        dir_name = f"{dataset}_{baseline}{suffix}"
        result_dir = os.path.join(results_root, dir_name)
        if os.path.isdir(result_dir):
            return result_dir
    return None


def evaluate_one(baseline: str, dataset: str, train_only: bool = False) -> Optional[Dict]:
    """评估单个 (baseline, dataset) 组合

    Args:
        train_only: 如果为 True，假设 cleaned CSV 只包含 60% 训练集的数据
    """
    ds_cfg = DATASETS_CONFIG[dataset]
    data_dir = os.path.join(_PROJECT_ROOT, 'Data', dataset)
    dirty_path = os.path.join(data_dir, ds_cfg['dirty_file'])
    clean_path = os.path.join(data_dir, ds_cfg['clean_file'])

    cleaned_csv = find_cleaned_csv(baseline, dataset)
    if cleaned_csv is None:
        return None

    if not os.path.exists(dirty_path) or not os.path.exists(clean_path):
        print(f"  [SKIP] {dataset}: dirty/clean CSV 不存在")
        return None

    # 读取数据
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)
    cleaned_df = pd.read_csv(cleaned_csv)

    n_total = len(dirty_df)
    label_col = ds_cfg['label_column']
    index_col = ds_cfg.get('index_column', 'index')
    exclude_cols = ds_cfg.get('exclude_columns', [])
    feature_cols = ds_cfg.get('feature_columns', None)
    task_type = ds_cfg['task_type']

    # 60/20/20 划分
    train_idx, val_idx, test_idx = demandclean_split(n_total)
    n_train = len(train_idx)

    # ── 处理 train_only 模式 ──
    if train_only:
        # cleaned CSV 只包含训练集数据，需要检查是否有 index 列来对齐
        if 'index' in cleaned_df.columns:
            # 有 index 列：用 dirty 数据为基础，用 cleaned 覆盖对应行
            full_cleaned = dirty_df.copy()
            data_cols = [c for c in cleaned_df.columns if c != 'index' and c in full_cleaned.columns]
            for _, row in cleaned_df.iterrows():
                idx = int(row['index'])
                if idx < len(full_cleaned):
                    for col in data_cols:
                        full_cleaned.at[idx, col] = row[col]
            cleaned_df = full_cleaned
        elif len(cleaned_df) == n_train:
            # 无 index 列但行数等于训练集：假设按 train_idx 顺序排列
            full_cleaned = dirty_df.copy()
            common_cols = [c for c in cleaned_df.columns if c in full_cleaned.columns]
            for i, orig_idx in enumerate(train_idx):
                if i < len(cleaned_df):
                    for col in common_cols:
                        full_cleaned.at[orig_idx, col] = cleaned_df.iloc[i][col]
            cleaned_df = full_cleaned
        # 否则保持原样（可能是部分清洗）

    # ── 传统清洗指标（仅在 60% 训练集上计算）──
    dirty_train = dirty_df.iloc[train_idx].reset_index(drop=True)
    clean_train = clean_df.iloc[train_idx].reset_index(drop=True)

    # cleaned 可能行数不同（deleteall 删除了行）
    if len(cleaned_df) == n_total:
        cleaned_train = cleaned_df.iloc[train_idx].reset_index(drop=True)
    else:
        # 行数不一致: 整体使用（无法精确对齐索引）
        cleaned_train = cleaned_df.copy()

    trad_metrics = calc_traditional_metrics(dirty_train, cleaned_train, clean_train,
                                            index_col=index_col, label_col=label_col)

    # ── 预处理特征（使用统一的编码器确保特征对齐）──
    # 与 DemandClean 对齐：在 60% dirty train set 上 fit 编码器
    try:
        # 1. 在 60% dirty train 上 fit 编码器（与 DemandClean 一致）
        dirty_train_for_fit = dirty_df.iloc[train_idx].reset_index(drop=True)
        _, _, encoders, scaler, label_enc = preprocess_for_ml(
            dirty_train_for_fit, label_col, index_col, exclude_cols, feature_cols)

        # 2. 用相同的编码器 transform 全量数据
        X_dirty_all, y_dirty_all = preprocess_for_ml(
            dirty_df, label_col, index_col, exclude_cols, feature_cols,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)

        X_clean_all, y_clean_all = preprocess_for_ml(
            clean_df, label_col, index_col, exclude_cols, feature_cols,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)

        X_cleaned_all, y_cleaned_all = preprocess_for_ml(
            cleaned_df, label_col, index_col, exclude_cols, feature_cols,
            fitted_encoders=encoders, fitted_scaler=scaler, fitted_label_encoder=label_enc)
    except Exception as e:
        print(f"  [ERROR] {dataset}/{baseline} 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ── 下游 ML 评估 ──
    if task_type == 'classification':
        model_list = CLASSIFICATION_MODELS
    elif task_type == 'regression':
        model_list = REGRESSION_MODELS
    else:
        model_list = CLUSTERING_MODELS

    if len(cleaned_df) == n_total:
        X_train_ml = X_cleaned_all[train_idx]
        y_train_ml = y_cleaned_all[train_idx]
    else:
        X_train_ml = X_cleaned_all
        y_train_ml = y_cleaned_all

    X_test_ml = X_clean_all[test_idx]
    y_test_ml = y_clean_all[test_idx]

    ml_results = evaluate_ml_split(X_train_ml, y_train_ml, X_test_ml, y_test_ml,
                                    task_type, model_list)

    # ── 容忍度 + Snoopy ──
    # 注意：P_do_nothing 和 P_repair_all 始终使用原始的 60% train_idx
    # 当 cleaned 行数不同时，P_demand_clean 使用 cleaned 的全量数据
    cleaned_train_idx = train_idx if len(cleaned_df) == n_total else np.arange(len(X_cleaned_all))
    tol = calc_tolerance_and_snoopy(
        X_dirty_all, y_dirty_all,
        X_cleaned_all, y_cleaned_all,
        X_clean_all, y_clean_all,
        train_idx, test_idx, task_type,
        cleaned_train_idx=cleaned_train_idx)

    # ── 日志提取 ──
    result_dir = find_result_dir(baseline, dataset)
    elapsed_time, truth_cost = extract_from_logs(result_dir, baseline) if result_dir else (None, None)

    # ── 组装结果 ──
    row = {
        'Baseline': baseline,
        'Dataset': dataset,
        'time': elapsed_time if elapsed_time is not None else 'N/A',
        'f1_score': trad_metrics['f1_score'],
        'r_edr': trad_metrics['r_edr'],
        'hybrid_distance': trad_metrics['hybrid_distance'],
        'edr': trad_metrics['edr'],
        'col_avg_rmse': trad_metrics.get('col_avg_rmse', ''),
        'col_avg_f1': trad_metrics.get('col_avg_f1', ''),
        'P_do_nothing': tol.get('P_do_nothing', ''),
        'P_demand_clean': tol.get('P_demand_clean', ''),
        'P_repair_all': tol.get('P_repair_all', ''),
        'upper_bound_dirty': tol.get('upper_bound_dirty', ''),
        'upper_bound_cleaned': tol.get('upper_bound_cleaned', ''),
        'upper_bound_clean': tol.get('upper_bound_clean', ''),
        'upper_bound_improvement': tol.get('upper_bound_improvement', ''),
        'truth_cost': truth_cost if truth_cost is not None else 'N/A',
        'ml_results': ml_results,
        'task_type': task_type,
    }
    return row


# ═══════════════════════════════════════════════════════════════
# 8. CSV 输出
# ═══════════════════════════════════════════════════════════════

def format_ml_cell(model_name: str, metrics: Dict, task_type: str) -> str:
    """格式化单个 ML 模型结果为表格单元格"""
    name_map = {
        'rf': 'RF', 'lr': 'LR', 'svm': 'SVM', 'knn': 'KNN', 'dt': 'DT', 'gb': 'GB',
        'ridge': 'RIDGE', 'lasso': 'LASSO',
        'kmeans': 'KMeans', 'agglomerative': 'Agglomerative', 'spectral': 'Spectral',
    }
    display = name_map.get(model_name, model_name.upper())
    if not metrics:
        return f"{display}\n    (failed)"

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


def write_csv(rows: List[Dict], output_path: str):
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
            # 补齐到 6 列
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

    print(f"\n已写出: {output_path} ({len(rows)} 行)")


# ═══════════════════════════════════════════════════════════════
# 9. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='快速重评估 Clean4MLBaseline 的全部 cleaned CSV')
    parser.add_argument('--baselines', nargs='+', default=ALL_BASELINES,
                        help=f'要评估的 baseline 列表（默认全部）')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        help=f'要评估的数据集列表（默认全部）')
    parser.add_argument('--output', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'results', 'reeval_with_split_results.csv'),
                        help='输出 CSV 路径')
    parser.add_argument('--train_only', action='store_true',
                        help='训练集模式：假设 baseline 只清洗了 60%% 训练集（cleaned CSV 行数 = 训练集行数）')
    args = parser.parse_args()

    mode_str = "训练集模式（60%% cleaned）" if args.train_only else "全量模式（100%% cleaned）"
    print("=" * 70)
    print("Clean4MLBaseline 快速重评估（DemandClean 对齐版）")
    print(f"划分方式: seed={SEED}, 60/20/20")
    print(f"评估模式: {mode_str}")
    print(f"Baselines: {args.baselines}")
    print(f"Datasets: {args.datasets}")
    print("=" * 70)

    all_rows = []
    total = len(args.baselines) * len(args.datasets)
    done = 0

    # 按数据集循环（确保 P_do_nothing 等指标一致）
    for dataset in args.datasets:
        print(f"\n=== 数据集: {dataset} ===")
        for baseline in args.baselines:
            done += 1
            print(f"[{done}/{total}] {baseline} × {dataset} ...", end='', flush=True)

            try:
                result = evaluate_one(baseline, dataset, train_only=args.train_only)
                if result is not None:
                    all_rows.append(result)
                    print(f" OK (ML models: {len(result['ml_results'])})")
                else:
                    print(f" SKIP (no cleaned CSV)")
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback
                traceback.print_exc()

    # 写出
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_csv(all_rows, args.output)

    # 统计
    print(f"\n总计: {len(all_rows)} / {total} 个组合评估成功")
    missing = total - len(all_rows)
    if missing > 0:
        print(f"缺失: {missing} 个组合（无 cleaned CSV 或评估失败）")


if __name__ == '__main__':
    main()
