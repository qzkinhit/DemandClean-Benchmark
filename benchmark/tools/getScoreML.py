"""
getScoreML.py - Unified data cleaning evaluation module

The core evaluation program for Clean4MLBaseline. All run_*_base.py scripts should
invoke this module to perform evaluation.

Metrics included:
1. Traditional cleaning metrics: precision, recall, F1, EDR, hybrid distance, R-EDR (from getScore.py)
2. Downstream task performance: classification (Accuracy, F1), regression (MSE, R2),
   clustering (Silhouette, ARI)
3. Model tolerance: prior tolerance (Tolerance_prior) and posterior tolerance (Tolerance_post)
4. Snoopy metric: embedding-based data quality upper-bound evaluation
5. Ground-truth usage cost: automation level (Type 1/2/3)

Usage:
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

# Add utils directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def safe_print(msg: str) -> None:
    """Fault-tolerant print: avoids crashes when stdout is redirected by a TeeLogger and the file has been closed."""
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
    Preprocess data for machine learning.

    To guarantee consistency, all encoders and scalers should be fit on the dirty data
    and then transform the cleaned/clean data. Controlled via fit=True/False.

    Args:
        data: Input DataFrame
        label_column: Label column name
        label_encoders: Dictionary of already-fit LabelEncoders (required when fit=False)
        scaler: Already-fit StandardScaler (required when fit=False)
        label_le: Already-fit LabelEncoder for the label (provided when fit=False)
        fit: True = fit encoders on this data; False = transform using existing encoders

    Returns:
        (X_scaled, y, label_encoders, scaler, label_le)
    """
    # Separate features and label, excluding index/ID and other non-feature columns
    non_feature_cols = {label_column, 'index', 'id'}
    X = data.drop(columns=[c for c in non_feature_cols if c in data.columns]).copy()
    y = data[label_column].copy()

    if label_encoders is None:
        label_encoders = {}

    # Encode categorical features
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

    # Handle non-numeric values in numeric columns (dirty data may contain 'empty', '33..', etc.)
    for col in X.columns:
        if col not in label_encoders:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Handle missing values
    X = X.fillna(X.mean())
    X = X.fillna(0)  # Safe fallback

    # Standardize
    if fit:
        scaler = StandardScaler()
        scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Encode the label
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
    """Return a classifier."""
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
    """Return a regressor."""
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
    """Evaluate a classification task."""
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
    """Evaluate a regression task."""
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
    """Evaluate a clustering task."""
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
    """Return a clusterer."""
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
    Evaluate downstream task performance.

    Args:
        cleaned_data: Data after cleaning
        clean_data: Clean data (ground truth)
        label_column: Label column name
        task_type: Task type ('classification', 'regression', 'clustering')
        models: List of models to evaluate
        test_size: Test-set ratio
        index_column: Index column name (used for row alignment)

    Returns:
        Dictionary of evaluation results
    """
    if models is None:
        models = ['rf', 'lr']

    results = {}

    # Handle the case where row counts differ (e.g., DeleteAll baseline)
    if len(cleaned_data) != len(clean_data):
        safe_print(f"Warning: cleaned row count ({len(cleaned_data)}) does not match clean row count ({len(clean_data)})")
        safe_print("Evaluation will use the cleaned data's own labels (row alignment)")

        # Try to align by index
        if index_column in cleaned_data.columns and index_column in clean_data.columns:
            cleaned_indexed = cleaned_data.set_index(index_column)
            clean_indexed = clean_data.set_index(index_column)
            common_indices = cleaned_indexed.index.intersection(clean_indexed.index)

            if len(common_indices) > 0:
                cleaned_aligned = cleaned_indexed.loc[common_indices].reset_index()
                clean_aligned = clean_indexed.loc[common_indices].reset_index()
                safe_print(f"Aligned by index, evaluating on {len(common_indices)} rows")
            else:
                # No common index: use the cleaned data itself
                cleaned_aligned = cleaned_data
                clean_aligned = cleaned_data
                safe_print("Unable to align; using cleaned data's own labels")
        else:
            # No index column: use the cleaned data itself
            cleaned_aligned = cleaned_data
            clean_aligned = cleaned_data
            safe_print("No index column; using cleaned data's own labels")
    else:
        cleaned_aligned = cleaned_data
        clean_aligned = clean_data

    # Preprocess
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_aligned, label_column)

    # Use labels from the aligned clean data
    _, y_clean = preprocess_for_ml(clean_aligned, label_column)

    # Minimum row check: train_test_split requires at least 5 rows
    min_rows = max(5, int(1 / test_size) + 1)  # Ensure at least 1 row after test_size split
    if len(X_cleaned) < min_rows:
        safe_print(f"[Skip] Cleaned data has only {len(X_cleaned)} rows, fewer than {min_rows}; cannot evaluate downstream task")
        return results

    if task_type == 'clustering':
        # Clustering task - KMeans only (AgglomerativeClustering O(n^2~n^3) is infeasible on large datasets)
        n_clusters = len(np.unique(y_clean))
        clustering_models = models if models else ['kmeans']
        n_rows = len(X_cleaned)
        sil_sample_size = min(n_rows, 10000)  # Sample silhouette O(n^2) for speed

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
                safe_print(f"{model_name.upper()} clustering failed: {e}")
    else:
        # Classification or regression task
        # Train on y_cleaned (labels repaired by the agent), test on y_clean (clean labels)
        # so the effect of label repair can be reflected in downstream task performance
        indices = np.arange(len(X_cleaned))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        X_train, X_test = X_cleaned[train_idx], X_cleaned[test_idx]
        y_train = y_cleaned[train_idx]   # Training labels: agent-repaired
        y_test = y_clean[test_idx]       # Test labels: clean ground truth

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
    Compute model noise tolerance.

    Prior tolerance (Tolerance_prior):
        Tolerance_prior(M) = (1/|E|) * sum (P_TolerClean(M,er) / P_do_nothing(M,er))

    Posterior tolerance (Tolerance_post):
        Tolerance_post(M) = (1/|E|) * sum (P_DemandClean(M,er) - P_do_nothing(M,er)) /
                                        (P_repair_all(M,er) - P_do_nothing(M,er))

    Args:
        dirty_data: Dirty data
        cleaned_data: Data after cleaning (with the current method)
        clean_data: Clean data (fully repaired)
        label_column: Label column name
        task_type: Task type
        model_name: Model name

    Returns:
        Dictionary of tolerance metrics
    """
    # Preprocess each version of the data
    X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column)
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column)
    X_clean, y_clean = preprocess_for_ml(clean_data, label_column)

    # Minimum row check: cleaned rows need to match clean by index for correct evaluation
    if len(X_cleaned) < 5:
        return {
            'P_do_nothing': 0.0,
            'P_demand_clean': 0.0,
            'P_repair_all': 0.0,
            'tolerance_prior': 0.0,
            'tolerance_post': 0.0,
        }

    # If cleaned rows != clean rows, use cleaned data's own train/test split
    use_cleaned_split = (len(X_cleaned) != len(X_clean))

    # Split data
    test_size = 0.2
    n_samples = len(X_clean)
    test_indices = np.random.RandomState(42).choice(
        n_samples, size=int(n_samples * test_size), replace=False
    )
    train_indices = np.array([i for i in range(n_samples) if i not in test_indices])

    def get_performance(X, y_train_labels, X_test, y_test):
        """Return model performance.

        Args:
            X: Training feature matrix (sliced with train_indices)
            y_train_labels: Training labels (sliced with train_indices)
            X_test: Test features
            y_test: Test labels (clean ground truth)
        """
        if task_type == 'clustering':
            # Clustering: fit on all data and return silhouette_score (sampled for large datasets)
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
            return r2_score(y_test, y_pred)  # R2 in (-inf, 1]; higher is better

    # Compute performance under each scenario
    # P_do_nothing: train on dirty features + dirty labels, test on clean data
    P_do_nothing = get_performance(X_dirty, y_dirty, X_clean[test_indices], y_clean[test_indices])

    # P_DemandClean: train on cleaned data
    if use_cleaned_split:
        # cleaned row count differs from clean (e.g., agent deleted some rows), cannot use the same indices
        # Fallback: split on cleaned itself; train with y_cleaned, test with y_cleaned as well
        # Note: this branch cannot test with y_clean because row correspondence has been lost
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
                    # Clustering: fit on all cleaned data and return silhouette_score (sampled for large datasets)
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
        # Train on cleaned features + cleaned labels, test on clean data
        P_demand_clean = get_performance(X_cleaned, y_cleaned, X_clean[test_indices], y_clean[test_indices])

    # P_repair_all: train on fully clean data
    P_repair_all = get_performance(X_clean, y_clean, X_clean[test_indices], y_clean[test_indices])

    # Compute prior tolerance
    if P_do_nothing != 0:
        tolerance_prior = P_demand_clean / P_do_nothing
    else:
        tolerance_prior = 0

    # Compute posterior tolerance
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

    safe_print(f"\nTolerance computation results:")
    safe_print(f"  P_do_nothing (dirty-data performance): {P_do_nothing:.4f}")
    safe_print(f"  P_demand_clean (post-cleaning performance): {P_demand_clean:.4f}")
    safe_print(f"  P_repair_all (fully-repaired performance): {P_repair_all:.4f}")
    safe_print(f"  Prior tolerance (Tolerance_prior): {tolerance_prior:.4f}")
    safe_print(f"  Posterior tolerance (Tolerance_post): {tolerance_post:.4f}")

    return results


def calculate_tolerance_multi_error_rates(dirty_data: pd.DataFrame,
                                           clean_data: pd.DataFrame,
                                           cleaned_data: pd.DataFrame,
                                           label_column: str,
                                           task_type: str = 'classification',
                                           model_name: str = 'rf',
                                           error_rates: List[float] = None) -> Dict:
    """
    Compute average tolerance across multiple error rates.

    Args:
        dirty_data: Dirty data
        clean_data: Clean data
        cleaned_data: Data after cleaning
        label_column: Label column name
        task_type: Task type
        model_name: Model name
        error_rates: List of error rates

    Returns:
        Tolerance results across multiple error rates
    """
    if error_rates is None:
        error_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Simplified handling here: uses the result of a single error rate.
    # In practice, one can combine an error injector to generate data at different error rates.
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
    Compute the ground-truth usage cost.

    Ground-truth usage types:
    - Type 1: Fully automatic execution, no human involvement required (cost = 0)
    - Type 2: Requires a small validation-set ground truth to assess cleaning effectiveness
      (cost = validation_samples)
    - Type 3: User iteratively cleans dirty samples selected by the model one by one
      (cost = iterations * batch_size)

    Args:
        method_type: Method type (1, 2, 3)
        total_samples: Total number of samples
        labeled_samples: Number of labeled samples
        validation_samples: Number of validation-set samples
        iterations: Number of iterations (used for Type 3)

    Returns:
        Cost dictionary
    """
    if method_type == 1:
        # Type 1 also uses labeled_samples; the caller is responsible for passing the correct value.
        # Oracle mode is "fully automatic" but actually consumes ground truth.
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
    Comprehensive evaluation function.

    Includes:
    1. Downstream task performance
    2. Model tolerance
    3. Ground-truth usage cost

    Args:
        dirty_data: Dirty data
        cleaned_data: Data after cleaning
        clean_data: Clean data
        label_column: Label column name
        task_type: Task type
        models: List of models
        method_type: Method type
        ground_truth_used: Number of ground-truth samples used

    Returns:
        Comprehensive evaluation results
    """
    if models is None:
        models = ['rf', 'lr']

    safe_print("="*60)
    safe_print("Comprehensive evaluation started")
    safe_print("="*60)

    # 1. Downstream task performance evaluation
    safe_print("\n1. Downstream task performance evaluation")
    safe_print("-"*40)
    ml_results = evaluate_downstream_task(
        cleaned_data, clean_data, label_column, task_type, models
    )

    # 2. Model tolerance evaluation
    safe_print("\n2. Model tolerance evaluation")
    safe_print("-"*40)
    tolerance_results = calculate_tolerance(
        dirty_data, cleaned_data, clean_data, label_column, task_type
    )

    # 3. Ground-truth usage cost
    safe_print("\n3. Ground-truth usage cost")
    safe_print("-"*40)
    cost_results = calculate_ground_truth_cost(
        method_type=method_type,
        total_samples=len(clean_data),
        labeled_samples=ground_truth_used
    )
    safe_print(f"  Ground-truth usage type: Type {method_type}")
    safe_print(f"  Ground-truth usage count: {cost_results['ground_truth_cost']}")
    safe_print(f"  Ground-truth usage ratio: {cost_results['cost_ratio']:.2%}")

    # Aggregate results
    results = {
        'task_type': task_type,
        **ml_results,
        **tolerance_results,
        **cost_results
    }

    safe_print("\n" + "="*60)
    safe_print("Comprehensive evaluation finished")
    safe_print("="*60)

    return results


# Test function
def test_evaluation():
    """Test the evaluation function."""
    # Create test data
    np.random.seed(42)
    n_samples = 1000

    # Clean data
    clean_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.choice([0, 1], n_samples)
    })

    # Dirty data (add noise)
    dirty_data = clean_data.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    dirty_data.loc[noise_idx, 'feature1'] = np.nan
    dirty_data.loc[noise_idx[:50], 'feature2'] *= 10  # Outliers

    # Cleaned data (partially repaired)
    cleaned_data = dirty_data.copy()
    cleaned_data['feature1'].fillna(cleaned_data['feature1'].mean(), inplace=True)

    # Run evaluation
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

    safe_print("\nFinal results:")
    for key, value in results.items():
        safe_print(f"  {key}: {value}")


# =============================================================================
# Snoopy evaluation function - embedding-based data quality upper-bound evaluation
# =============================================================================

def evaluate_snoopy_upper_bound(dirty_data: pd.DataFrame,
                                 cleaned_data: pd.DataFrame,
                                 clean_data: pd.DataFrame,
                                 label_column: str,
                                 task_type: str = 'classification') -> Dict:
    """
    Use Snoopy to evaluate the data-quality upper bound before and after cleaning.

    Snoopy evaluates the data-quality upper bound via embeddings, to judge whether
    cleaning improved the upper bound.

    Args:
        dirty_data: Dirty data
        cleaned_data: Data after cleaning
        clean_data: Clean data (ground truth)
        label_column: Label column name
        task_type: Task type

    Returns:
        Snoopy evaluation results
    """
    results = {
        'snoopy_available': False,
        'upper_bound_dirty': 0.0,
        'upper_bound_cleaned': 0.0,
        'upper_bound_clean': 0.0,
        'upper_bound_improvement': 0.0
    }

    try:
        # Attempt to import snoopy
        # Note: tools/snoopy/snoopy/ is the actual package. We must insert tools/snoopy at
        # the front of sys.path; otherwise Python finds tools/snoopy/__init__.py first (an empty file).
        _snoopy_parent = os.path.join(_current_dir, 'snoopy')
        if _snoopy_parent in sys.path:
            sys.path.remove(_snoopy_parent)
        sys.path.insert(0, _snoopy_parent)
        from snoopy.pipeline import run as snoopy_run

        # Preprocess data
        X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column)
        X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column)
        X_clean, y_clean = preprocess_for_ml(clean_data, label_column)

        # Guard: if cleaned data is too small for cross_val, skip
        if len(X_cleaned) < 5:
            safe_print(f"[Snoopy] Cleaned data has only {len(X_cleaned)} rows, fewer than 5; skipping upper-bound evaluation")
            return results

        # Evaluate the upper bound of each dataset
        results['snoopy_available'] = True
        safe_print("Snoopy module available; evaluating data-quality upper bound...")

        # Simplified upper-bound estimate (based on model cross-validation)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task_type == 'clustering':
            # Clustering: use silhouette_score as the upper-bound indicator (sampled for large datasets)
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
            # dirty: dirty features + dirty labels; cleaned: cleaned features + cleaned labels; clean: clean data
            results['upper_bound_dirty'] = np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5))
            results['upper_bound_cleaned'] = np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned))))
            results['upper_bound_clean'] = np.mean(cross_val_score(model, X_clean, y_clean, cv=5))
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            results['upper_bound_dirty'] = -np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5, scoring='neg_mean_squared_error'))
            results['upper_bound_cleaned'] = -np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned)), scoring='neg_mean_squared_error'))
            results['upper_bound_clean'] = -np.mean(cross_val_score(model, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error'))

        # Compute upper-bound improvement
        if results['upper_bound_dirty'] != 0:
            results['upper_bound_improvement'] = (
                (results['upper_bound_cleaned'] - results['upper_bound_dirty']) /
                (results['upper_bound_clean'] - results['upper_bound_dirty'])
                if results['upper_bound_clean'] != results['upper_bound_dirty'] else 0
            )

    except ImportError:
        safe_print("Snoopy module unavailable; skipping upper-bound evaluation")
    except Exception as e:
        safe_print(f"Snoopy evaluation error: {e}")

    return results


# =============================================================================
# Ideal minimum ground-truth cost calculation
# =============================================================================

def calculate_ideal_min_ground_truth_cost(dirty_data: pd.DataFrame,
                                           cleaned_data: pd.DataFrame,
                                           clean_data: pd.DataFrame,
                                           index_attribute: str = 'index') -> Dict:
    """
    Compute the ideal minimum ground-truth cost.

    Ideal minimum cost = among the differences between cleaned and dirty data, the
    number of cells that were actually repaired correctly, i.e., only cells where
    ground truth was truly used are counted.

    When the number of cleaned rows differs from dirty/clean (e.g., the agent deleted
    some rows), we only compare the rows retained in the cleaned data.

    Args:
        dirty_data: Dirty data
        cleaned_data: Data after cleaning
        clean_data: Clean data (ground truth)
        index_attribute: Index column name

    Returns:
        Information about the ideal minimum ground-truth cost
    """
    dirty = dirty_data.copy()
    cleaned = cleaned_data.copy()
    clean = clean_data.copy()

    # Exclude index column
    cols_to_compare = [c for c in dirty.columns if c != index_attribute]

    # Align by index: only compare rows retained in cleaned
    if index_attribute in dirty.columns and index_attribute in cleaned.columns:
        dirty = dirty.set_index(index_attribute)
        cleaned = cleaned.set_index(index_attribute)
        clean = clean.set_index(index_attribute)
        # Intersection of all three index sets
        common_idx = dirty.index.intersection(cleaned.index).intersection(clean.index)
        dirty = dirty.loc[common_idx].reset_index()
        cleaned = cleaned.loc[common_idx].reset_index()
        clean = clean.loc[common_idx].reset_index()
    else:
        # Without an index column, use the shortest length
        min_len = min(len(dirty), len(cleaned), len(clean))
        dirty = dirty.iloc[:min_len].reset_index(drop=True)
        cleaned = cleaned.iloc[:min_len].reset_index(drop=True)
        clean = clean.iloc[:min_len].reset_index(drop=True)

    # Compare cell by cell
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

            # Check whether a modification occurred
            if dirty_val != cleaned_val:
                changes_count += 1
                # Check whether the modification is correct (matches the ground truth)
                if cleaned_val == clean_val:
                    correct_repairs += 1
                else:
                    wrong_repairs += 1

    # Additional statistic: number of deleted rows
    deleted_rows = len(dirty_data) - len(cleaned_data)

    results = {
        'total_changes': changes_count,
        'correct_repairs': correct_repairs,
        'wrong_repairs': wrong_repairs,
        'deleted_rows': max(0, deleted_rows),
        'ideal_min_ground_truth_cost': correct_repairs,  # Ideally only this many ground-truth samples are needed
        'repair_accuracy': correct_repairs / changes_count if changes_count > 0 else 0
    }

    return results


# =============================================================================
# Unified evaluation function - run_all_evaluation
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
    Unified data cleaning evaluation function.

    All run_*_base.py scripts should invoke this function for standardized evaluation.

    Args:
        dirty_path: Dirty data path
        cleaned_path: Cleaned data path
        clean_path: Clean data path (ground truth)
        output_path: Result output path
        task_name: Task name
        label_column: Label column name (optional, used for downstream task evaluation)
        task_type: Task type ('classification', 'regression', 'clustering')
        models: List of evaluation models
        method_type: Cleaning method type (1 = fully automatic, 2 = requires validation set,
            3 = requires iterative interaction)
        ground_truth_used: Number of ground-truth samples actually used (provided by the
            cleaning method)
        index_attribute: Index column name
        mse_attributes: List of attributes for which to compute MSE
        verbose: Whether to print detailed information

    Returns:
        Dictionary of complete evaluation results
    """
    if models is None:
        models = ['rf', 'lr']

    if mse_attributes is None:
        mse_attributes = []

    # Load data
    dirty_data = pd.read_csv(dirty_path)
    cleaned_data = pd.read_csv(cleaned_path)
    clean_data = pd.read_csv(clean_path)

    # Get the list of attributes
    attributes = clean_data.columns.tolist()

    # Guard against 0-row cleaned data: return empty results immediately
    if len(cleaned_data) == 0:
        if verbose:
            safe_print("[Warning] Cleaned data is empty (0 rows); all metrics set to 0")
        results = {
            'task_name': task_name,
            'task_type': task_type,
            'total_records': len(clean_data),
            'total_cells': len(clean_data) * len(attributes),
            'cleaned_rows': 0,
            'note': 'cleaned_data is empty (0 rows), all metrics set to 0',
        }
        # Write a placeholder result file
        os.makedirs(output_path, exist_ok=True)
        eval_file = os.path.join(output_path, f'{task_name}_evaluation_results.txt')
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write(f"[Warning] Cleaned data is empty (0 rows); no metrics can be computed\n")
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
        safe_print(f"Clean4MLBaseline unified evaluation - {task_name}")
        safe_print("=" * 70)

    # ==========================================================================
    # 1. Traditional cleaning metrics (from getScore.py)
    # ==========================================================================
    if verbose:
        safe_print("\n[1/5] Traditional cleaning metrics")
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
            safe_print("Warning: getScore module unavailable; skipping traditional cleaning metrics")
    except Exception as e:
        if verbose:
            import traceback
            safe_print(f"Error computing traditional cleaning metrics: {type(e).__name__}: {e}")
            safe_print(traceback.format_exc())

    # ==========================================================================
    # 2. Downstream task performance
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[2/5] Downstream task performance ({task_type})")
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
            safe_print("\n[2/5] Downstream task performance - skipped (no label column specified)")

    # ==========================================================================
    # 3. Model tolerance
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[3/5] Model tolerance")
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
            safe_print("\n[3/5] Model tolerance - skipped (no label column specified)")

    # ==========================================================================
    # 4. Snoopy upper-bound evaluation
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[4/5] Snoopy upper-bound evaluation")
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
            safe_print(f"  Dirty-data upper bound: {snoopy_results['upper_bound_dirty']:.4f}")
            safe_print(f"  Cleaned upper bound: {snoopy_results['upper_bound_cleaned']:.4f}")
            safe_print(f"  Clean-data upper bound: {snoopy_results['upper_bound_clean']:.4f}")
            safe_print(f"  Upper-bound improvement ratio: {snoopy_results['upper_bound_improvement']:.4f}")
    else:
        if verbose:
            safe_print("\n[4/5] Snoopy upper-bound evaluation - skipped (no label column specified)")

    # ==========================================================================
    # 5. Ground-truth usage cost
    # ==========================================================================
    if verbose:
        safe_print(f"\n[5/5] Ground-truth usage cost")
        safe_print("-" * 50)

    # Compute the ideal minimum ground-truth cost
    ideal_cost = calculate_ideal_min_ground_truth_cost(
        dirty_data=dirty_data,
        cleaned_data=cleaned_data,
        clean_data=clean_data,
        index_attribute=index_attribute
    )
    results.update({f'ideal_{k}': v for k, v in ideal_cost.items()})

    # Actual ground-truth cost (provided by the cleaning method)
    cost_results = calculate_ground_truth_cost(
        method_type=method_type,
        total_samples=len(clean_data),
        labeled_samples=ground_truth_used
    )
    results.update(cost_results)

    if verbose:
        method_desc = 'fully automatic' if method_type == 1 else ('requires validation set' if method_type == 2 else 'requires iterative interaction')
        safe_print(f"  Method type: Type {method_type} ({method_desc})")
        safe_print(f"  Actual ground-truth used: {ground_truth_used}")
        safe_print(f"  Ideal minimum cost: {ideal_cost['ideal_min_ground_truth_cost']}")
        safe_print(f"  Total modified cells: {ideal_cost['total_changes']}")
        safe_print(f"  Correct repairs: {ideal_cost['correct_repairs']}")
        safe_print(f"  Incorrect repairs: {ideal_cost['wrong_repairs']}")

    # ==========================================================================
    # Save results
    # ==========================================================================
    results_file = os.path.join(output_path, f"{task_name}_report.txt")
    os.makedirs(output_path, exist_ok=True)

    # Extract the list of models used
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
        f.write("Clean4MLBaseline unified evaluation report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Task name: {task_name}\n")
        f.write(f"Task type: {task_type}\n")
        f.write(f"Evaluation models: {', '.join(m.upper() for m in used_models)}\n")
        f.write(f"Row counts: dirty={len(dirty_data)}, cleaned={len(cleaned_data)}, clean={len(clean_data)}\n")
        f.write("=" * 80 + "\n\n")

        # Traditional cleaning metrics
        f.write("[Traditional cleaning metrics]\n")
        f.write("-" * 40 + "\n")
        for key in ['accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
            if key in results:
                f.write(f"  {key:20s}: {results[key]}\n")

        # Column-level cleaning metrics
        f.write("\n[Column-level cleaning metrics]\n")
        f.write("-" * 40 + "\n")
        col_avg_rmse = results.get('col_avg_rmse')
        col_avg_f1 = results.get('col_avg_f1')
        f.write(f"  col_avg_rmse (mean normalized RMSE over numeric columns): {col_avg_rmse if col_avg_rmse is not None else 'N/A'}\n")
        f.write(f"  col_avg_f1   (mean weighted F1 over categorical columns): {col_avg_f1 if col_avg_f1 is not None else 'N/A'}\n")
        col_rmse_details = results.get('col_rmse_details', {})
        col_f1_details = results.get('col_f1_details', {})
        if col_rmse_details:
            f.write("  Numeric column RMSE details:\n")
            for col_name, val in col_rmse_details.items():
                f.write(f"    {col_name:25s}: {val:.6f}\n")
        if col_f1_details:
            f.write("  Categorical column F1 details:\n")
            for col_name, val in col_f1_details.items():
                f.write(f"    {col_name:25s}: {val:.6f}\n")

        # Downstream task performance - grouped by model
        f.write("\n[Downstream task performance - grouped by model]\n")
        f.write("-" * 40 + "\n")
        for mn in used_models:
            f.write(f"\n  Model: {mn.upper()}\n")
            f.write("  " + "-" * 36 + "\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall',
                           'mse', 'rmse', 'mae', 'r2',
                           'silhouette', 'ari']:
                key = f'ml_{mn}_{metric}'
                if key in results:
                    f.write(f"    {metric:15s}: {results[key]:.6f}\n")

        # Model tolerance
        f.write("\n[Model tolerance]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  P_do_nothing (dirty-data performance):     {results.get('tolerance_P_do_nothing', 'N/A')}\n")
        f.write(f"  P_demand_clean (post-cleaning performance):   {results.get('tolerance_P_demand_clean', 'N/A')}\n")
        f.write(f"  P_repair_all (fully-repaired performance):   {results.get('tolerance_P_repair_all', 'N/A')}\n")
        f.write(f"  Prior tolerance (Tolerance_prior):  {results.get('tolerance_tolerance_prior', 'N/A')}\n")
        f.write(f"  Posterior tolerance (Tolerance_post):   {results.get('tolerance_tolerance_post', 'N/A')}\n")

        # Snoopy upper-bound evaluation
        f.write("\n[Snoopy upper-bound evaluation]\n")
        f.write("-" * 40 + "\n")
        for key, value in results.items():
            if key.startswith('snoopy_'):
                clean_key = key.replace('snoopy_', '')
                f.write(f"  {clean_key:30s}: {value}\n")

        # Ground-truth usage cost
        f.write("\n[Ground-truth usage cost]\n")
        f.write("-" * 40 + "\n")
        method_desc = 'fully automatic' if method_type == 1 else ('requires validation set' if method_type == 2 else 'requires iterative interaction')
        f.write(f"  Method type:     Type {method_type} ({method_desc})\n")
        f.write(f"  Actual ground-truth used: {ground_truth_used}\n")
        f.write(f"  Ideal minimum cost: {ideal_cost.get('ideal_min_ground_truth_cost', 'N/A')}\n")
        if 'repair_accuracy' in ideal_cost:
            f.write(f"  Repair accuracy:   {ideal_cost['repair_accuracy']:.4f}\n")
        else:
            f.write(f"  Repair accuracy:   N/A\n")
        f.write(f"  Total modified cells: {ideal_cost.get('total_changes', 'N/A')}\n")
        f.write(f"  Correct repairs:     {ideal_cost.get('correct_repairs', 'N/A')}\n")
        f.write(f"  Incorrect repairs:     {ideal_cost.get('wrong_repairs', 'N/A')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Evaluation complete\n")
        f.write("=" * 80 + "\n")

    if verbose:
        safe_print("\n" + "=" * 70)
        safe_print(f"Evaluation complete; results saved to: {results_file}")
        safe_print("=" * 70)

    return results
