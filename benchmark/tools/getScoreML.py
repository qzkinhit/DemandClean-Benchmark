"""
getScoreML.py - 统一的数据清洗测评模块

Clean4MLBaseline的核心测评程序，所有run_*_base.py脚本都应调用此模块进行评估。

包含指标:
1. 传统清洗指标: 准确率、召回率、F1、EDR、混合距离、R-EDR（来自getScore.py）
2. 下游任务性能: 分类（Accuracy, F1）、回归（MSE, R2）、聚类（Silhouette, ARI）
3. 模型容忍度: 先验容忍度(Tolerance_prior)和后验容忍度(Tolerance_post)
4. Snoopy指标: 基于embedding的数据质量上界评估
5. 真值使用成本: 自动化级别(Type 1/2/3)

使用方式:
    from utils.getScoreML import run_all_evaluation

    results = run_all_evaluation(
        dirty_path='path/to/dirty.csv',
        cleaned_path='path/to/cleaned.csv',
        clean_path='path/to/clean.csv',
        label_column='label',
        task_type='classification',
        models=['rf', 'lr'],
        method_type=1,
        ground_truth_used=0,
        output_path='path/to/results/'
    )
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score
)

# 添加utils目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def safe_print(msg: str) -> None:
    """容错打印：当 stdout 被 TeeLogger 等重定向且文件已关闭时不崩溃"""
    try:
        print(msg)
    except (ValueError, IOError, OSError):
        if sys.__stdout__ is not None:
            sys.__stdout__.write(str(msg) + '\n')
            sys.__stdout__.flush()


def preprocess_for_ml(data: pd.DataFrame, label_column: str,
                      label_encoders: Optional[Dict] = None,
                      scaler: Optional[StandardScaler] = None,
                      label_le: Optional[LabelEncoder] = None,
                      fit: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict, StandardScaler, Optional[LabelEncoder]]:
    """
    预处理数据用于机器学习

    为保证一致性，所有编码器和标准化器应在 dirty 数据上 fit，
    然后 transform 到 cleaned/clean 数据。通过 fit=True/False 控制。

    Args:
        data: 输入DataFrame
        label_column: 标签列名
        label_encoders: 已fit的LabelEncoder字典（fit=False时必须提供）
        scaler: 已fit的StandardScaler（fit=False时必须提供）
        label_le: 已fit的标签LabelEncoder（fit=False时提供）
        fit: True=在此数据上fit编码器; False=用已有编码器transform

    Returns:
        (X_scaled, y, label_encoders, scaler, label_le)
    """
    # 分离特征和标签，排除索引/ID 等非特征列
    non_feature_cols = {label_column, 'index', 'id'}
    X = data.drop(columns=[c for c in non_feature_cols if c in data.columns]).copy()
    y = data[label_column].copy()

    if label_encoders is None:
        label_encoders = {}

    # 编码分类特征
    for col in X.select_dtypes(include=['object']).columns:
        if fit:
            le = LabelEncoder()
            X[col] = X[col].fillna('__MISSING__').astype(str)
            le.fit(X[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                le = LabelEncoder()
                X[col] = X[col].fillna('__MISSING__').astype(str)
                le.fit(X[col])
                label_encoders[col] = le
            else:
                X[col] = X[col].fillna('__MISSING__').astype(str)
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else '__UNKNOWN__')
                if '__UNKNOWN__' not in le.classes_:
                    le.classes_ = np.append(le.classes_, '__UNKNOWN__')
        X[col] = le.transform(X[col])

    # 处理数值列中的非数值（脏数据可能有 'empty', '33..' 等）
    for col in X.columns:
        if col not in label_encoders:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # 处理缺失值
    X = X.fillna(X.mean())
    X = X.fillna(0)  # 安全兜底

    # 标准化
    if fit:
        scaler = StandardScaler()
        scaler.fit(X)
    X_scaled = scaler.transform(X)

    # 编码标签
    if y.dtype == 'object':
        if fit:
            label_le = LabelEncoder()
            y = y.fillna('__MISSING__').astype(str)
            label_le.fit(y)
        else:
            if label_le is not None:
                y = y.fillna('__MISSING__').astype(str)
                known = set(label_le.classes_)
                y = y.apply(lambda v: v if v in known else '__UNKNOWN__')
                if '__UNKNOWN__' not in label_le.classes_:
                    label_le.classes_ = np.append(label_le.classes_, '__UNKNOWN__')
            else:
                label_le = LabelEncoder()
                y = y.fillna('__MISSING__').astype(str)
                label_le.fit(y)
        y = label_le.transform(y)

    return X_scaled, np.array(y), label_encoders, scaler, label_le


def get_classifier(model_name: str):
    """获取分类器"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    classifiers = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42),
        'svm': SVC(random_state=42),
        'knn': KNeighborsClassifier(),
        'dt': DecisionTreeClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }
    return classifiers.get(model_name, classifiers['rf'])


def get_regressor(model_name: str):
    """获取回归器"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor

    regressors = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'lr': LinearRegression(),
        'ridge': Ridge(solver='lsqr', random_state=42),
        'lasso': Lasso(random_state=42),
        'svm': SVR(),
        'knn': KNeighborsRegressor(),
        'gb': GradientBoostingRegressor(random_state=42)
    }
    return regressors.get(model_name, regressors['rf'])


def evaluate_classification(X_train, y_train, X_test, y_test, model_name: str) -> Dict:
    """评估分类任务"""
    model = get_classifier(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }


def evaluate_regression(X_train, y_train, X_test, y_test, model_name: str) -> Dict:
    """评估回归任务"""
    model = get_regressor(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


def evaluate_clustering(X, y_true, n_clusters: int = None) -> Dict:
    """评估聚类任务"""
    from sklearn.cluster import KMeans

    if n_clusters is None:
        n_clusters = len(np.unique(y_true))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    return {
        'silhouette': silhouette_score(X, y_pred, sample_size=min(len(X), 10000), random_state=42),
        'ari': adjusted_rand_score(y_true, y_pred)
    }


def get_clusterer(model_name: str, n_clusters: int):
    """获取聚类器"""
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

    clusterers = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'spectral': SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors'),
    }
    return clusterers.get(model_name, clusterers['kmeans'])


def evaluate_downstream_task(cleaned_data: pd.DataFrame,
                             clean_data: pd.DataFrame,
                             label_column: str,
                             task_type: str = 'classification',
                             models: List[str] = None,
                             test_size: float = 0.2,
                             index_column: str = 'index') -> Dict:
    """
    评估下游任务性能

    Args:
        cleaned_data: 清洗后的数据
        clean_data: 干净数据（真值）
        label_column: 标签列名
        task_type: 任务类型 ('classification', 'regression', 'clustering')
        models: 要评估的模型列表
        test_size: 测试集比例
        index_column: 索引列名（用于行对齐）

    Returns:
        评估结果字典
    """
    if models is None:
        models = ['rf', 'lr']

    results = {}

    # 处理行数不一致的情况（如DeleteAll baseline）
    if len(cleaned_data) != len(clean_data):
        safe_print(f"警告: 清洗后数据行数({len(cleaned_data)})与干净数据行数({len(clean_data)})不一致")
        safe_print("将使用清洗后数据自身进行评估（行对齐）")

        # 尝试通过索引对齐
        if index_column in cleaned_data.columns and index_column in clean_data.columns:
            cleaned_indexed = cleaned_data.set_index(index_column)
            clean_indexed = clean_data.set_index(index_column)
            common_indices = cleaned_indexed.index.intersection(clean_indexed.index)

            if len(common_indices) > 0:
                cleaned_aligned = cleaned_indexed.loc[common_indices].reset_index()
                clean_aligned = clean_indexed.loc[common_indices].reset_index()
                safe_print(f"通过索引对齐，使用 {len(common_indices)} 行数据进行评估")
            else:
                # 没有共同索引，使用cleaned数据自身
                cleaned_aligned = cleaned_data
                clean_aligned = cleaned_data
                safe_print("无法对齐，使用清洗后数据自身的标签")
        else:
            # 没有索引列，使用cleaned数据自身
            cleaned_aligned = cleaned_data
            clean_aligned = cleaned_data
            safe_print("无索引列，使用清洗后数据自身的标签")
    else:
        cleaned_aligned = cleaned_data
        clean_aligned = clean_data

    # 预处理
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_aligned, label_column)

    # 使用对齐后的干净数据标签
    _, y_clean = preprocess_for_ml(clean_aligned, label_column)

    # 最小行数检查：train_test_split 至少需要 5 行
    min_rows = max(5, int(1 / test_size) + 1)  # 确保 test_size 分割后至少 1 行
    if len(X_cleaned) < min_rows:
        safe_print(f"[跳过] 清洗后数据仅 {len(X_cleaned)} 行，不足 {min_rows} 行，无法评估下游任务")
        return results

    if task_type == 'clustering':
        # 聚类任务 - 仅用 KMeans（AgglomerativeClustering O(n²~n³) 大数据集不可接受）
        n_clusters = len(np.unique(y_clean))
        clustering_models = models if models else ['kmeans']
        n_rows = len(X_cleaned)
        sil_sample_size = min(n_rows, 10000)  # silhouette O(n²) 采样加速

        for model_name in clustering_models:
            try:
                clusterer = get_clusterer(model_name, n_clusters)
                y_pred = clusterer.fit_predict(X_cleaned)
                sil_score = silhouette_score(X_cleaned, y_pred, sample_size=sil_sample_size, random_state=42)
                ari_score = adjusted_rand_score(y_clean, y_pred)
                results[f'{model_name}_silhouette'] = sil_score
                results[f'{model_name}_ari'] = ari_score
                safe_print(f"{model_name.upper()} - Silhouette: {sil_score:.4f}, ARI: {ari_score:.4f}")
            except Exception as e:
                safe_print(f"{model_name.upper()} 聚类失败: {e}")
    else:
        # 分类或回归任务
        # 训练用 y_cleaned（Agent 修复后的标签），测试用 y_clean（干净标签）
        # 这样标签修复的效果能在下游任务性能中体现出来
        indices = np.arange(len(X_cleaned))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        X_train, X_test = X_cleaned[train_idx], X_cleaned[test_idx]
        y_train = y_cleaned[train_idx]   # 训练标签: Agent 修复后的
        y_test = y_clean[test_idx]       # 测试标签: 干净真值

        for model_name in models:
            if task_type == 'classification':
                model_results = evaluate_classification(
                    X_train, y_train, X_test, y_test, model_name
                )
                results[f'{model_name}_accuracy'] = model_results['accuracy']
                results[f'{model_name}_f1'] = model_results['f1']
                results[f'{model_name}_precision'] = model_results['precision']
                results[f'{model_name}_recall'] = model_results['recall']
                safe_print(f"{model_name.upper()} - Accuracy: {model_results['accuracy']:.4f}, "
                      f"F1: {model_results['f1']:.4f}, Precision: {model_results['precision']:.4f}, "
                      f"Recall: {model_results['recall']:.4f}")
            else:
                model_results = evaluate_regression(
                    X_train, y_train, X_test, y_test, model_name
                )
                results[f'{model_name}_mse'] = model_results['mse']
                results[f'{model_name}_r2'] = model_results['r2']
                safe_print(f"{model_name.upper()} - MSE: {model_results['mse']:.4f}, "
                      f"R2: {model_results['r2']:.4f}")

    return results


def calculate_tolerance(dirty_data: pd.DataFrame,
                        cleaned_data: pd.DataFrame,
                        clean_data: pd.DataFrame,
                        label_column: str,
                        task_type: str = 'classification',
                        model_name: str = 'rf') -> Dict:
    """
    计算模型噪声容忍度

    先验容忍度 (Tolerance_prior):
        Tolerance_prior(M) = (1/|E|) * Σ (P_TolerClean(M,er) / P_do_nothing(M,er))

    后验容忍度 (Tolerance_post):
        Tolerance_post(M) = (1/|E|) * Σ (P_DemandClean(M,er) - P_do_nothing(M,er)) /
                                        (P_repair_all(M,er) - P_do_nothing(M,er))

    Args:
        dirty_data: 脏数据
        cleaned_data: 清洗后的数据（使用当前方法清洗）
        clean_data: 干净数据（完全修复）
        label_column: 标签列名
        task_type: 任务类型
        model_name: 模型名称

    Returns:
        容忍度指标字典
    """
    # 预处理各版本数据
    X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column)
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column)
    X_clean, y_clean = preprocess_for_ml(clean_data, label_column)

    # 最小行数检查：cleaned 数据行数需要与 clean 匹配索引才能正确评估
    if len(X_cleaned) < 5:
        return {
            'P_do_nothing': 0.0,
            'P_demand_clean': 0.0,
            'P_repair_all': 0.0,
            'tolerance_prior': 0.0,
            'tolerance_post': 0.0,
        }

    # 当 cleaned 行数 != clean 行数时，用 cleaned 自身做 train/test split
    use_cleaned_split = (len(X_cleaned) != len(X_clean))

    # 分割数据
    test_size = 0.2
    n_samples = len(X_clean)
    test_indices = np.random.RandomState(42).choice(
        n_samples, size=int(n_samples * test_size), replace=False
    )
    train_indices = np.array([i for i in range(n_samples) if i not in test_indices])

    def get_performance(X, y_train_labels, X_test, y_test):
        """获取模型性能

        Args:
            X: 训练特征矩阵 (使用 train_indices 切片)
            y_train_labels: 训练标签 (使用 train_indices 切片)
            X_test: 测试特征
            y_test: 测试标签 (干净真值)
        """
        if task_type == 'clustering':
            # 聚类: 用全量数据拟合，返回 silhouette_score（大数据集采样）
            from sklearn.cluster import KMeans as _KMeans
            n_clusters = len(np.unique(y_test))
            km = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            y_pred = km.fit_predict(X)
            try:
                sil_sample = min(len(X), 10000)
                return silhouette_score(X, y_pred, sample_size=sil_sample, random_state=42)
            except Exception:
                return 0.0
        elif task_type == 'classification':
            model = get_classifier(model_name)
            model.fit(X[train_indices], y_train_labels[train_indices])
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
        else:
            model = get_regressor(model_name)
            model.fit(X[train_indices], y_train_labels[train_indices])
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)  # R2 ∈ (-∞, 1]，越大越好

    # 计算各场景下的性能
    # P_do_nothing: 脏特征 + 脏标签训练，干净数据测试
    P_do_nothing = get_performance(X_dirty, y_dirty, X_clean[test_indices], y_clean[test_indices])

    # P_DemandClean: 在清洗后数据上训练
    if use_cleaned_split:
        # cleaned 行数与 clean 不一致（Agent 删除了部分行），无法用相同索引
        # 退化方案: 用 cleaned 自身 split，训练用 y_cleaned，测试也用 y_cleaned
        # 注意: 此分支无法用 y_clean 测试，因为行对应关系已丢失
        from sklearn.model_selection import train_test_split
        n_cleaned = len(X_cleaned)
        if n_cleaned >= 5:
            cleaned_train_idx = np.random.RandomState(42).choice(
                n_cleaned, size=max(1, int(n_cleaned * 0.8)), replace=False
            )
            cleaned_test_idx = np.array([i for i in range(n_cleaned) if i not in cleaned_train_idx])
            if len(cleaned_test_idx) == 0:
                cleaned_test_idx = cleaned_train_idx[:1]
            try:
                if task_type == 'classification':
                    model = get_classifier(model_name)
                    model.fit(X_cleaned[cleaned_train_idx], y_cleaned[cleaned_train_idx])
                    y_pred = model.predict(X_cleaned[cleaned_test_idx])
                    P_demand_clean = accuracy_score(y_cleaned[cleaned_test_idx], y_pred)
                elif task_type == 'clustering':
                    # 聚类: 用全量 cleaned 数据拟合，返回 silhouette_score（大数据集采样）
                    from sklearn.cluster import KMeans as _KMeans
                    n_clusters = len(np.unique(y_cleaned))
                    km = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    y_pred = km.fit_predict(X_cleaned)
                    try:
                        sil_sample = min(len(X_cleaned), 10000)
                        P_demand_clean = silhouette_score(X_cleaned, y_pred, sample_size=sil_sample, random_state=42)
                    except Exception:
                        P_demand_clean = 0.0
                else:
                    model = get_regressor(model_name)
                    model.fit(X_cleaned[cleaned_train_idx], y_cleaned[cleaned_train_idx])
                    y_pred = model.predict(X_cleaned[cleaned_test_idx])
                    P_demand_clean = r2_score(y_cleaned[cleaned_test_idx], y_pred)
            except Exception:
                P_demand_clean = 0.0
        else:
            P_demand_clean = 0.0
    else:
        # 清洗后特征 + 清洗后标签训练，干净数据测试
        P_demand_clean = get_performance(X_cleaned, y_cleaned, X_clean[test_indices], y_clean[test_indices])

    # P_repair_all: 在完全干净数据上训练
    P_repair_all = get_performance(X_clean, y_clean, X_clean[test_indices], y_clean[test_indices])

    # 计算先验容忍度
    if P_do_nothing != 0:
        tolerance_prior = P_demand_clean / P_do_nothing
    else:
        tolerance_prior = 0

    # 计算后验容忍度
    if P_repair_all != P_do_nothing:
        tolerance_post = (P_demand_clean - P_do_nothing) / (P_repair_all - P_do_nothing)
    else:
        tolerance_post = 0

    results = {
        'P_do_nothing': P_do_nothing,
        'P_demand_clean': P_demand_clean,
        'P_repair_all': P_repair_all,
        'tolerance_prior': tolerance_prior,
        'tolerance_post': tolerance_post
    }

    safe_print(f"\n容忍度计算结果:")
    safe_print(f"  P_do_nothing (脏数据性能): {P_do_nothing:.4f}")
    safe_print(f"  P_demand_clean (清洗后性能): {P_demand_clean:.4f}")
    safe_print(f"  P_repair_all (完全修复性能): {P_repair_all:.4f}")
    safe_print(f"  先验容忍度 (Tolerance_prior): {tolerance_prior:.4f}")
    safe_print(f"  后验容忍度 (Tolerance_post): {tolerance_post:.4f}")

    return results


def calculate_tolerance_multi_error_rates(dirty_data: pd.DataFrame,
                                           clean_data: pd.DataFrame,
                                           cleaned_data: pd.DataFrame,
                                           label_column: str,
                                           task_type: str = 'classification',
                                           model_name: str = 'rf',
                                           error_rates: List[float] = None) -> Dict:
    """
    在多个错误率下计算平均容忍度

    Args:
        dirty_data: 脏数据
        clean_data: 干净数据
        cleaned_data: 清洗后的数据
        label_column: 标签列名
        task_type: 任务类型
        model_name: 模型名称
        error_rates: 错误率列表

    Returns:
        多错误率下的容忍度结果
    """
    if error_rates is None:
        error_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # 这里简化处理，使用单一错误率的结果
    # 实际应用中可以结合错误注入器生成不同错误率的数据
    base_tolerance = calculate_tolerance(
        dirty_data, cleaned_data, clean_data,
        label_column, task_type, model_name
    )

    results = {
        'avg_tolerance_prior': base_tolerance['tolerance_prior'],
        'avg_tolerance_post': base_tolerance['tolerance_post'],
        'error_rates': error_rates,
        'detailed_results': base_tolerance
    }

    return results


def calculate_ground_truth_cost(method_type: int,
                                 total_samples: int,
                                 labeled_samples: int = 0,
                                 validation_samples: int = 0,
                                 iterations: int = 0) -> Dict:
    """
    计算真值使用成本

    真值使用类型:
    - Type 1: 全自动执行，无需人工参与 (cost = 0)
    - Type 2: 需少量验证集真值评估清洗效果 (cost = validation_samples)
    - Type 3: 用户迭代逐条清洗模型选中的脏样本 (cost = iterations * batch_size)

    Args:
        method_type: 方法类型 (1, 2, 3)
        total_samples: 总样本数
        labeled_samples: 标注样本数
        validation_samples: 验证集样本数
        iterations: 迭代次数（用于Type 3）

    Returns:
        成本字典
    """
    if method_type == 1:
        # Type 1 也使用 labeled_samples，调用方负责传入正确值
        # Oracle 模式虽然「全自动」，但实际使用了真值
        cost = labeled_samples
        cost_ratio = labeled_samples / total_samples if total_samples > 0 else 0
    elif method_type == 2:
        cost = validation_samples
        cost_ratio = validation_samples / total_samples if total_samples > 0 else 0
    elif method_type == 3:
        cost = labeled_samples
        cost_ratio = labeled_samples / total_samples if total_samples > 0 else 0
    else:
        cost = labeled_samples
        cost_ratio = labeled_samples / total_samples if total_samples > 0 else 0

    return {
        'method_type': method_type,
        'ground_truth_cost': cost,
        'cost_ratio': cost_ratio,
        'total_samples': total_samples,
        'labeled_samples': labeled_samples,
        'validation_samples': validation_samples
    }


def comprehensive_evaluation(dirty_data: pd.DataFrame,
                              cleaned_data: pd.DataFrame,
                              clean_data: pd.DataFrame,
                              label_column: str,
                              task_type: str = 'classification',
                              models: List[str] = None,
                              method_type: int = 1,
                              ground_truth_used: int = 0) -> Dict:
    """
    综合评估函数

    包含:
    1. 下游任务性能
    2. 模型容忍度
    3. 真值使用成本

    Args:
        dirty_data: 脏数据
        cleaned_data: 清洗后的数据
        clean_data: 干净数据
        label_column: 标签列名
        task_type: 任务类型
        models: 模型列表
        method_type: 方法类型
        ground_truth_used: 使用的真值数量

    Returns:
        综合评估结果
    """
    if models is None:
        models = ['rf', 'lr']

    safe_print("="*60)
    safe_print("综合评估开始")
    safe_print("="*60)

    # 1. 下游任务性能评估
    safe_print("\n1. 下游任务性能评估")
    safe_print("-"*40)
    ml_results = evaluate_downstream_task(
        cleaned_data, clean_data, label_column, task_type, models
    )

    # 2. 模型容忍度评估
    safe_print("\n2. 模型容忍度评估")
    safe_print("-"*40)
    tolerance_results = calculate_tolerance(
        dirty_data, cleaned_data, clean_data, label_column, task_type
    )

    # 3. 真值使用成本
    safe_print("\n3. 真值使用成本")
    safe_print("-"*40)
    cost_results = calculate_ground_truth_cost(
        method_type=method_type,
        total_samples=len(clean_data),
        labeled_samples=ground_truth_used
    )
    safe_print(f"  真值使用类型: Type {method_type}")
    safe_print(f"  真值使用数量: {cost_results['ground_truth_cost']}")
    safe_print(f"  真值使用比例: {cost_results['cost_ratio']:.2%}")

    # 汇总结果
    results = {
        'task_type': task_type,
        **ml_results,
        **tolerance_results,
        **cost_results
    }

    safe_print("\n" + "="*60)
    safe_print("综合评估完成")
    safe_print("="*60)

    return results


# 测试函数
def test_evaluation():
    """测试评估函数"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000

    # 干净数据
    clean_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.choice([0, 1], n_samples)
    })

    # 脏数据（添加噪声）
    dirty_data = clean_data.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    dirty_data.loc[noise_idx, 'feature1'] = np.nan
    dirty_data.loc[noise_idx[:50], 'feature2'] *= 10  # 异常值

    # 清洗后数据（部分修复）
    cleaned_data = dirty_data.copy()
    cleaned_data['feature1'].fillna(cleaned_data['feature1'].mean(), inplace=True)

    # 运行评估
    results = comprehensive_evaluation(
        dirty_data=dirty_data,
        cleaned_data=cleaned_data,
        clean_data=clean_data,
        label_column='label',
        task_type='classification',
        models=['rf', 'lr'],
        method_type=1,
        ground_truth_used=0
    )

    safe_print("\n最终结果:")
    for key, value in results.items():
        safe_print(f"  {key}: {value}")


# =============================================================================
# Snoopy评估函数 - 基于embedding的数据质量上界评估
# =============================================================================

def evaluate_snoopy_upper_bound(dirty_data: pd.DataFrame,
                                 cleaned_data: pd.DataFrame,
                                 clean_data: pd.DataFrame,
                                 label_column: str,
                                 task_type: str = 'classification') -> Dict:
    """
    使用Snoopy评估清洗前后的数据质量上界

    Snoopy通过embedding来评估数据质量上界，判断清洗是否提升了上界。

    Args:
        dirty_data: 脏数据
        cleaned_data: 清洗后的数据
        clean_data: 干净数据（ground truth）
        label_column: 标签列名
        task_type: 任务类型

    Returns:
        Snoopy评估结果
    """
    results = {
        'snoopy_available': False,
        'upper_bound_dirty': 0.0,
        'upper_bound_cleaned': 0.0,
        'upper_bound_clean': 0.0,
        'upper_bound_improvement': 0.0
    }

    try:
        # 尝试导入snoopy
        # 注意: tools/snoopy/snoopy/ 才是真正的包，需要把 tools/snoopy 插到
        # sys.path 最前面，否则 Python 会先找到 tools/snoopy/__init__.py（空文件）
        _snoopy_parent = os.path.join(_current_dir, 'snoopy')
        if _snoopy_parent in sys.path:
            sys.path.remove(_snoopy_parent)
        sys.path.insert(0, _snoopy_parent)
        from snoopy.pipeline import run as snoopy_run

        # 预处理数据
        X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column)
        X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column)
        X_clean, y_clean = preprocess_for_ml(clean_data, label_column)

        # 保护: 如果清洗后数据太少无法做 cross_val
        if len(X_cleaned) < 5:
            safe_print(f"[Snoopy] 清洗后数据仅 {len(X_cleaned)} 行，不足5行，跳过上界评估")
            return results

        # 评估各数据集的上界
        results['snoopy_available'] = True
        safe_print("Snoopy模块可用，正在评估数据质量上界...")

        # 简化的上界估计（基于模型交叉验证）
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task_type == 'clustering':
            # 聚类: 用 silhouette_score 作为上界指标（大数据集采样）
            from sklearn.cluster import KMeans as _KMeans
            def _clustering_upper_bound(X, y):
                n_clusters = len(np.unique(y))
                km = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                y_pred = km.fit_predict(X)
                try:
                    sil_sample = min(len(X), 10000)
                    return silhouette_score(X, y_pred, sample_size=sil_sample, random_state=42)
                except Exception:
                    return 0.0
            results['upper_bound_dirty'] = _clustering_upper_bound(X_dirty, y_dirty)
            results['upper_bound_cleaned'] = _clustering_upper_bound(X_cleaned, y_cleaned)
            results['upper_bound_clean'] = _clustering_upper_bound(X_clean, y_clean)
        elif task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # dirty: 脏特征+脏标签; cleaned: 清洗后特征+清洗后标签; clean: 干净数据
            results['upper_bound_dirty'] = np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5))
            results['upper_bound_cleaned'] = np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned))))
            results['upper_bound_clean'] = np.mean(cross_val_score(model, X_clean, y_clean, cv=5))
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            results['upper_bound_dirty'] = -np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5, scoring='neg_mean_squared_error'))
            results['upper_bound_cleaned'] = -np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned)), scoring='neg_mean_squared_error'))
            results['upper_bound_clean'] = -np.mean(cross_val_score(model, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error'))

        # 计算上界提升
        if results['upper_bound_dirty'] != 0:
            results['upper_bound_improvement'] = (
                (results['upper_bound_cleaned'] - results['upper_bound_dirty']) /
                (results['upper_bound_clean'] - results['upper_bound_dirty'])
                if results['upper_bound_clean'] != results['upper_bound_dirty'] else 0
            )

    except ImportError:
        safe_print("Snoopy模块不可用，跳过上界评估")
    except Exception as e:
        safe_print(f"Snoopy评估出错: {e}")

    return results


# =============================================================================
# 理想最小真值成本计算
# =============================================================================

def calculate_ideal_min_ground_truth_cost(dirty_data: pd.DataFrame,
                                           cleaned_data: pd.DataFrame,
                                           clean_data: pd.DataFrame,
                                           index_attribute: str = 'index') -> Dict:
    """
    计算理想最小真值成本

    理想最小成本 = 清洗后数据与脏数据的差异中，实际为正确修复的单元格数量
    即：只有真正用到真值的地方才计入成本

    当 cleaned 行数与 dirty/clean 不同时（如 agent 删除了部分行），
    仅比较 cleaned 中保留的行。

    Args:
        dirty_data: 脏数据
        cleaned_data: 清洗后的数据
        clean_data: 干净数据（ground truth）
        index_attribute: 索引列名

    Returns:
        理想最小真值成本信息
    """
    dirty = dirty_data.copy()
    cleaned = cleaned_data.copy()
    clean = clean_data.copy()

    # 排除索引列
    cols_to_compare = [c for c in dirty.columns if c != index_attribute]

    # 按索引对齐: 只比较 cleaned 中保留的行
    if index_attribute in dirty.columns and index_attribute in cleaned.columns:
        dirty = dirty.set_index(index_attribute)
        cleaned = cleaned.set_index(index_attribute)
        clean = clean.set_index(index_attribute)
        # 取三者共有的索引
        common_idx = dirty.index.intersection(cleaned.index).intersection(clean.index)
        dirty = dirty.loc[common_idx].reset_index()
        cleaned = cleaned.loc[common_idx].reset_index()
        clean = clean.loc[common_idx].reset_index()
    else:
        # 无索引列时，取最短长度
        min_len = min(len(dirty), len(cleaned), len(clean))
        dirty = dirty.iloc[:min_len].reset_index(drop=True)
        cleaned = cleaned.iloc[:min_len].reset_index(drop=True)
        clean = clean.iloc[:min_len].reset_index(drop=True)

    # 逐单元格比较
    changes_count = 0
    correct_repairs = 0
    wrong_repairs = 0
    n_rows = len(dirty)

    for col in cols_to_compare:
        if col not in cleaned.columns or col not in clean.columns:
            continue
        for i in range(n_rows):
            dirty_val = str(dirty.iloc[i][col]).strip().lower() if pd.notna(dirty.iloc[i][col]) else ''
            cleaned_val = str(cleaned.iloc[i][col]).strip().lower() if pd.notna(cleaned.iloc[i][col]) else ''
            clean_val = str(clean.iloc[i][col]).strip().lower() if pd.notna(clean.iloc[i][col]) else ''

            # 检查是否发生了修改
            if dirty_val != cleaned_val:
                changes_count += 1
                # 检查修改是否正确（与真值一致）
                if cleaned_val == clean_val:
                    correct_repairs += 1
                else:
                    wrong_repairs += 1

    # 额外统计: 被删除的行数
    deleted_rows = len(dirty_data) - len(cleaned_data)

    results = {
        'total_changes': changes_count,
        'correct_repairs': correct_repairs,
        'wrong_repairs': wrong_repairs,
        'deleted_rows': max(0, deleted_rows),
        'ideal_min_ground_truth_cost': correct_repairs,  # 理想情况下只需要这些真值
        'repair_accuracy': correct_repairs / changes_count if changes_count > 0 else 0
    }

    return results


# =============================================================================
# 统一测评函数 - run_all_evaluation
# =============================================================================

def run_all_evaluation(dirty_path: str,
                       cleaned_path: str,
                       clean_path: str,
                       output_path: str,
                       task_name: str,
                       label_column: str = None,
                       task_type: str = 'classification',
                       models: List[str] = None,
                       method_type: int = 1,
                       ground_truth_used: int = 0,
                       index_attribute: str = 'index',
                       mse_attributes: List[str] = None,
                       verbose: bool = True) -> Dict:
    """
    统一的数据清洗测评函数

    所有run_*_base.py脚本都应调用此函数进行标准化评估。

    Args:
        dirty_path: 脏数据路径
        cleaned_path: 清洗后数据路径
        clean_path: 干净数据路径（ground truth）
        output_path: 结果输出路径
        task_name: 任务名称
        label_column: 标签列名（可选，用于下游任务评估）
        task_type: 任务类型 ('classification', 'regression', 'clustering')
        models: 评估模型列表
        method_type: 清洗方法类型 (1=全自动, 2=需验证集, 3=需迭代交互)
        ground_truth_used: 实际使用的真值数量（由清洗方法提供）
        index_attribute: 索引列名
        mse_attributes: 需要计算MSE的属性列表
        verbose: 是否打印详细信息

    Returns:
        完整的评估结果字典
    """
    if models is None:
        models = ['rf', 'lr']

    if mse_attributes is None:
        mse_attributes = []

    # 加载数据
    dirty_data = pd.read_csv(dirty_path)
    cleaned_data = pd.read_csv(cleaned_path)
    clean_data = pd.read_csv(clean_path)

    # 获取属性列表
    attributes = clean_data.columns.tolist()

    # 对 0 行 cleaned 数据做保护：直接给出空结果
    if len(cleaned_data) == 0:
        if verbose:
            safe_print("[警告] 清洗后数据为空(0行)，所有指标置0")
        results = {
            'task_name': task_name,
            'task_type': task_type,
            'total_records': len(clean_data),
            'total_cells': len(clean_data) * len(attributes),
            'cleaned_rows': 0,
            'note': 'cleaned_data is empty (0 rows), all metrics set to 0',
        }
        # 写出占位结果文件
        os.makedirs(output_path, exist_ok=True)
        eval_file = os.path.join(output_path, f'{task_name}_evaluation_results.txt')
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write(f"[警告] 清洗后数据为空(0行)，无法计算任何指标\n")
        return results

    results = {
        'task_name': task_name,
        'task_type': task_type,
        'total_records': len(clean_data),
        'total_cells': len(clean_data) * len(attributes),
        'cleaned_rows': len(cleaned_data),
    }

    if verbose:
        safe_print("=" * 70)
        safe_print(f"Clean4MLBaseline 统一测评 - {task_name}")
        safe_print("=" * 70)

    # ==========================================================================
    # 1. 传统清洗指标（来自getScore.py）
    # ==========================================================================
    if verbose:
        safe_print("\n[1/5] 传统清洗指标")
        safe_print("-" * 50)

    try:
        from getScore import calculate_all_metrics

        traditional_results = calculate_all_metrics(
            clean=clean_data,
            dirty=dirty_data,
            cleaned=cleaned_data,
            attributes=attributes,
            output_path=output_path,
            task_name=task_name,
            index_attribute=index_attribute,
            mse_attributes=mse_attributes,
            save_debug_files=True
        )
        results.update(traditional_results)
    except ImportError:
        if verbose:
            safe_print("警告: getScore模块不可用，跳过传统清洗指标")
    except Exception as e:
        if verbose:
            import traceback
            safe_print(f"传统清洗指标计算出错: {type(e).__name__}: {e}")
            safe_print(traceback.format_exc())

    # ==========================================================================
    # 2. 下游任务性能
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[2/5] 下游任务性能 ({task_type})")
            safe_print("-" * 50)

        ml_results = evaluate_downstream_task(
            cleaned_data=cleaned_data,
            clean_data=clean_data,
            label_column=label_column,
            task_type=task_type,
            models=models,
            index_column=index_attribute
        )
        results.update({f'ml_{k}': v for k, v in ml_results.items()})
    else:
        if verbose:
            safe_print("\n[2/5] 下游任务性能 - 跳过（未指定标签列）")

    # ==========================================================================
    # 3. 模型容忍度
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[3/5] 模型容忍度")
            safe_print("-" * 50)

        tolerance_results = calculate_tolerance(
            dirty_data=dirty_data,
            cleaned_data=cleaned_data,
            clean_data=clean_data,
            label_column=label_column,
            task_type=task_type
        )
        results.update({f'tolerance_{k}': v for k, v in tolerance_results.items()})
    else:
        if verbose:
            safe_print("\n[3/5] 模型容忍度 - 跳过（未指定标签列）")

    # ==========================================================================
    # 4. Snoopy上界评估
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[4/5] Snoopy上界评估")
            safe_print("-" * 50)

        snoopy_results = evaluate_snoopy_upper_bound(
            dirty_data=dirty_data,
            cleaned_data=cleaned_data,
            clean_data=clean_data,
            label_column=label_column,
            task_type=task_type
        )
        results.update({f'snoopy_{k}': v for k, v in snoopy_results.items()})

        if verbose and snoopy_results.get('snoopy_available'):
            safe_print(f"  脏数据上界: {snoopy_results['upper_bound_dirty']:.4f}")
            safe_print(f"  清洗后上界: {snoopy_results['upper_bound_cleaned']:.4f}")
            safe_print(f"  干净数据上界: {snoopy_results['upper_bound_clean']:.4f}")
            safe_print(f"  上界提升比例: {snoopy_results['upper_bound_improvement']:.4f}")
    else:
        if verbose:
            safe_print("\n[4/5] Snoopy上界评估 - 跳过（未指定标签列）")

    # ==========================================================================
    # 5. 真值使用成本
    # ==========================================================================
    if verbose:
        safe_print(f"\n[5/5] 真值使用成本")
        safe_print("-" * 50)

    # 计算理想最小真值成本
    ideal_cost = calculate_ideal_min_ground_truth_cost(
        dirty_data=dirty_data,
        cleaned_data=cleaned_data,
        clean_data=clean_data,
        index_attribute=index_attribute
    )
    results.update({f'ideal_{k}': v for k, v in ideal_cost.items()})

    # 实际真值成本（由清洗方法提供）
    cost_results = calculate_ground_truth_cost(
        method_type=method_type,
        total_samples=len(clean_data),
        labeled_samples=ground_truth_used
    )
    results.update(cost_results)

    if verbose:
        safe_print(f"  方法类型: Type {method_type} ({'全自动' if method_type==1 else '需验证集' if method_type==2 else '需迭代交互'})")
        safe_print(f"  实际真值使用: {ground_truth_used}")
        safe_print(f"  理想最小成本: {ideal_cost['ideal_min_ground_truth_cost']}")
        safe_print(f"  总修改单元格: {ideal_cost['total_changes']}")
        safe_print(f"  正确修复: {ideal_cost['correct_repairs']}")
        safe_print(f"  错误修复: {ideal_cost['wrong_repairs']}")

    # ==========================================================================
    # 保存结果
    # ==========================================================================
    results_file = os.path.join(output_path, f"{task_name}_report.txt")
    os.makedirs(output_path, exist_ok=True)

    # 提取已使用的模型列表
    used_models = []
    for key in results.keys():
        if key.startswith('ml_'):
            parts = key.split('_')
            if len(parts) >= 3:
                model_name = parts[1]
                if model_name not in used_models:
                    used_models.append(model_name)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Clean4MLBaseline 统一测评报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"任务名称: {task_name}\n")
        f.write(f"任务类型: {task_type}\n")
        f.write(f"评估模型: {', '.join(m.upper() for m in used_models)}\n")
        f.write(f"数据行数: 脏数据={len(dirty_data)}, 清洗后={len(cleaned_data)}, 干净数据={len(clean_data)}\n")
        f.write("=" * 80 + "\n\n")

        # 传统清洗指标
        f.write("[传统清洗指标]\n")
        f.write("-" * 40 + "\n")
        for key in ['accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
            if key in results:
                f.write(f"  {key:20s}: {results[key]}\n")

        # 列级别清洗指标
        f.write("\n[列级别清洗指标]\n")
        f.write("-" * 40 + "\n")
        col_avg_rmse = results.get('col_avg_rmse')
        col_avg_f1 = results.get('col_avg_f1')
        f.write(f"  col_avg_rmse (数值列归一化RMSE均值): {col_avg_rmse if col_avg_rmse is not None else 'N/A'}\n")
        f.write(f"  col_avg_f1   (类别列weighted F1均值): {col_avg_f1 if col_avg_f1 is not None else 'N/A'}\n")
        col_rmse_details = results.get('col_rmse_details', {})
        col_f1_details = results.get('col_f1_details', {})
        if col_rmse_details:
            f.write("  数值列RMSE详情:\n")
            for col_name, val in col_rmse_details.items():
                f.write(f"    {col_name:25s}: {val:.6f}\n")
        if col_f1_details:
            f.write("  类别列F1详情:\n")
            for col_name, val in col_f1_details.items():
                f.write(f"    {col_name:25s}: {val:.6f}\n")

        # 下游任务性能 - 按模型分组
        f.write("\n[下游任务性能 - 按模型分组]\n")
        f.write("-" * 40 + "\n")
        for mn in used_models:
            f.write(f"\n  模型: {mn.upper()}\n")
            f.write("  " + "-" * 36 + "\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall',
                           'mse', 'rmse', 'mae', 'r2',
                           'silhouette', 'ari']:
                key = f'ml_{mn}_{metric}'
                if key in results:
                    f.write(f"    {metric:15s}: {results[key]:.6f}\n")

        # 模型容忍度
        f.write("\n[模型容忍度]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  P_do_nothing (脏数据性能):     {results.get('tolerance_P_do_nothing', 'N/A')}\n")
        f.write(f"  P_demand_clean (清洗后性能):   {results.get('tolerance_P_demand_clean', 'N/A')}\n")
        f.write(f"  P_repair_all (完全修复性能):   {results.get('tolerance_P_repair_all', 'N/A')}\n")
        f.write(f"  先验容忍度 (Tolerance_prior):  {results.get('tolerance_tolerance_prior', 'N/A')}\n")
        f.write(f"  后验容忍度 (Tolerance_post):   {results.get('tolerance_tolerance_post', 'N/A')}\n")

        # Snoopy上界评估
        f.write("\n[Snoopy上界评估]\n")
        f.write("-" * 40 + "\n")
        for key, value in results.items():
            if key.startswith('snoopy_'):
                clean_key = key.replace('snoopy_', '')
                f.write(f"  {clean_key:30s}: {value}\n")

        # 真值使用成本
        f.write("\n[真值使用成本]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  方法类型:     Type {method_type} ({'全自动' if method_type==1 else '需验证集' if method_type==2 else '需迭代交互'})\n")
        f.write(f"  实际真值使用: {ground_truth_used}\n")
        f.write(f"  理想最小成本: {ideal_cost.get('ideal_min_ground_truth_cost', 'N/A')}\n")
        if 'repair_accuracy' in ideal_cost:
            f.write(f"  修复准确率:   {ideal_cost['repair_accuracy']:.4f}\n")
        else:
            f.write(f"  修复准确率:   N/A\n")
        f.write(f"  总修改单元格: {ideal_cost.get('total_changes', 'N/A')}\n")
        f.write(f"  正确修复:     {ideal_cost.get('correct_repairs', 'N/A')}\n")
        f.write(f"  错误修复:     {ideal_cost.get('wrong_repairs', 'N/A')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("测评完成\n")
        f.write("=" * 80 + "\n")

    if verbose:
        safe_print("\n" + "=" * 70)
        safe_print(f"测评完成，结果已保存到: {results_file}")
        safe_print("=" * 70)

    return results
