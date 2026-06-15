"""
getScoreML.py - unified data-cleaning evaluation module

Core DemandClean-Benchmark evaluator; all run_*_base.py scripts should call this module.

Metrics included:
1. Traditional cleaning metrics: precision, recall, F1, EDR, hybrid distance, R-EDR (from getScore.py)
2. Downstream task performance: classification (Accuracy, F1), regression (MSE, R2), clustering (Silhouette, ARI)
3. Model tolerance: prior tolerance (Tolerance_prior) and posterior tolerance (Tolerance_post)
4. Snoopy metric: embedding-based data quality upper-bound evaluation
5. Ground-truth cost: automation level (Type 1/2/3)

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
    """Fault-tolerant print: doesn't crash when stdout is redirected by TeeLogger and the file is closed."""
    try:
        print(msg)
    except (ValueError, IOError, OSError):
        if sys.__stdout__ is not None:
            sys.__stdout__.write(str(msg) + '\n')
            sys.__stdout__.flush()


def preprocess_for_ml(data: pd.DataFrame, label_column: str,
                      shared_encoders: dict = None,
                      shared_scaler: StandardScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for machine learning.

    Args:
        data: input DataFrame
        label_column: label column name
        shared_encoders: shared {col_name: LabelEncoder} (transform if provided, else fit_transform)
        shared_scaler: shared StandardScaler (transform if provided, else fit_transform)

    Returns:
        Feature matrix X and label vector y
    """
    # Separate features and label, excluding index/ID-style non-feature columns
    non_feature_cols = {label_column, 'index', 'id'}
    X = data.drop(columns=[c for c in non_feature_cols if c in data.columns])
    y = data[label_column]

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        if shared_encoders and col in shared_encoders:
            le = shared_encoders[col]
            # transform (unseen categories map to nearest known)
            X[col] = X[col].astype(str).map(
                lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_
                else 0
            )
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.mean())

    # Standardize
    if shared_scaler is not None:
        X_scaled = shared_scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Encode label (if classification)
    if y.dtype == 'object':
        if shared_encoders and label_column in shared_encoders:
            le = shared_encoders[label_column]
            y = y.astype(str).map(
                lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_
                else 0
            )
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)

    return X_scaled, np.array(y)


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
                             index_column: str = 'index',
                             X_cleaned_encoded: np.ndarray = None,
                             y_cleaned_encoded: np.ndarray = None,
                             X_clean_encoded: np.ndarray = None,
                             y_clean_encoded: np.ndarray = None) -> Dict:
    """
    Evaluate downstream task performance.

    Args:
        cleaned_data: cleaned data
        clean_data: clean ground-truth data
        label_column: label column name
        task_type: task type ('classification', 'regression', 'clustering')
        models: list of models to evaluate
        test_size: test set fraction
        index_column: index column name (for row alignment)

    Returns:
        Results dict
    """
    if models is None:
        models = ['rf', 'lr']

    results = {}

    # Encoded-data shortcut: skip CSV -> preprocess_for_ml to avoid roundtrip precision loss
    if X_cleaned_encoded is not None and X_clean_encoded is not None:
        safe_print("  [downstream] using encoded version (skipping CSV preprocess)")
        X_cleaned = X_cleaned_encoded.copy()
        y_cleaned = y_cleaned_encoded.copy()
        X_clean_test = X_clean_encoded.copy()
        y_clean = y_clean_encoded.copy()

        # NaN-safe handling (agent may leave no_action so NaNs remain)
        # Drop rows where y is NaN
        y_valid_cleaned = ~np.isnan(y_cleaned)
        X_cleaned = X_cleaned[y_valid_cleaned]
        y_cleaned = y_cleaned[y_valid_cleaned]
        y_valid_clean = ~np.isnan(y_clean)
        X_clean_test = X_clean_test[y_valid_clean]
        y_clean = y_clean[y_valid_clean]
        # Fill remaining NaN in X with column means
        for arr in (X_cleaned, X_clean_test):
            col_means = np.nanmean(arr, axis=0)
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                for j in range(arr.shape[1]):
                    arr[nan_mask[:, j], j] = col_means[j]

        # Minimum row check
        min_rows = max(5, int(1 / test_size) + 1)
        if len(X_cleaned) < min_rows:
            safe_print(f"[skip] cleaned data has only {len(X_cleaned)} rows, below {min_rows}; skipping downstream eval")
            return results

        if task_type == 'clustering':
            n_clusters = len(np.unique(y_clean))
            clustering_models = models if models else ['kmeans']
            n_rows = len(X_cleaned)
            sil_sample_size = min(n_rows, 10000)

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
            # Classification or regression
            # If X_clean_test row count differs from X_cleaned (Oracle test set),
            # use train-on-all / test-on-test (matches Step 5).
            if len(X_clean_test) != len(X_cleaned):
                safe_print(f"  Oracle mode: train={len(X_cleaned)}, test={len(X_clean_test)}")
                X_train = X_cleaned
                y_train = y_cleaned
                X_test = X_clean_test
                y_test = y_clean
            else:
                # Same-source data: internal train/test split
                indices = np.arange(len(X_cleaned))
                train_idx, test_idx = train_test_split(
                    indices, test_size=test_size, random_state=42
                )
                X_train, X_test = X_cleaned[train_idx], X_cleaned[test_idx]
                y_train = y_cleaned[train_idx]
                y_test = y_clean[test_idx]

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

    # Handle row-count mismatch (e.g. DeleteAll baseline)
    if len(cleaned_data) != len(clean_data):
        safe_print(f"warning: cleaned rows ({len(cleaned_data)}) != clean rows ({len(clean_data)})")
        safe_print("using cleaned data itself for evaluation (row alignment)")

        # Try aligning via index
        if index_column in cleaned_data.columns and index_column in clean_data.columns:
            cleaned_indexed = cleaned_data.set_index(index_column)
            clean_indexed = clean_data.set_index(index_column)
            common_indices = cleaned_indexed.index.intersection(clean_indexed.index)

            if len(common_indices) > 0:
                cleaned_aligned = cleaned_indexed.loc[common_indices].reset_index()
                clean_aligned = clean_indexed.loc[common_indices].reset_index()
                safe_print(f"aligned via index, using {len(common_indices)} rows")
            else:
                # No common indices; use cleaned data itself
                cleaned_aligned = cleaned_data
                clean_aligned = cleaned_data
                safe_print("cannot align; using cleaned data's own labels")
        else:
            # No index column; use cleaned data itself
            cleaned_aligned = cleaned_data
            clean_aligned = cleaned_data
            safe_print("no index column; using cleaned data's own labels")
    else:
        cleaned_aligned = cleaned_data
        clean_aligned = clean_data

    # Unified encoding: fit LE/SS on combined data to keep label maps consistent across sides.
    # Avoids mis-alignment from fitting LabelEncoder separately on cleaned and clean.
    combined = pd.concat([cleaned_aligned, clean_aligned], ignore_index=True)
    non_feature_cols = {label_column, 'index', 'id'}
    combined_features = combined.drop(columns=[c for c in non_feature_cols if c in combined.columns])

    # Fit shared LabelEncoder on combined data
    shared_encoders = {}
    for col in combined_features.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        le.fit(combined_features[col].astype(str))
        shared_encoders[col] = le
    # LE for the label column
    if combined[label_column].dtype == 'object':
        le_label = LabelEncoder()
        le_label.fit(combined[label_column].astype(str))
        shared_encoders[label_column] = le_label

    # Fit shared StandardScaler on combined data
    combined_encoded = combined_features.copy()
    for col in combined_encoded.select_dtypes(include=['object']).columns:
        if col in shared_encoders:
            le = shared_encoders[col]
            combined_encoded[col] = combined_encoded[col].astype(str).map(
                lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else 0
            )
    combined_encoded = combined_encoded.fillna(combined_encoded.mean())
    shared_scaler = StandardScaler()
    shared_scaler.fit(combined_encoded)

    # Transform cleaned and clean separately using the shared encoders
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_aligned, label_column,
                                              shared_encoders=shared_encoders,
                                              shared_scaler=shared_scaler)
    # Use aligned clean-data labels
    _, y_clean = preprocess_for_ml(clean_aligned, label_column,
                                    shared_encoders=shared_encoders,
                                    shared_scaler=shared_scaler)

    # Minimum row check: train_test_split needs at least 5
    min_rows = max(5, int(1 / test_size) + 1)  # ensure test_size yields at least 1 row
    if len(X_cleaned) < min_rows:
        safe_print(f"[skip] cleaned data has only {len(X_cleaned)} rows, below {min_rows}; skipping downstream eval")
        return results

    if task_type == 'clustering':
        # Clustering task - KMeans only (AgglomerativeClustering O(n^2~n^3) is too costly)
        n_clusters = len(np.unique(y_clean))
        clustering_models = models if models else ['kmeans']
        n_rows = len(X_cleaned)
        sil_sample_size = min(n_rows, 10000)  # silhouette O(n^2); sample to speed up

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
        # Classification or regression
        # Train on y_cleaned (agent-repaired labels); test on y_clean (clean labels)
        # So the effect of label repair shows up in downstream task performance.
        indices = np.arange(len(X_cleaned))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        X_train, X_test = X_cleaned[train_idx], X_cleaned[test_idx]
        y_train = y_cleaned[train_idx]   # training labels: agent-repaired
        y_test = y_clean[test_idx]       # test labels: clean ground truth

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
                        model_name: str = 'rf',
                        X_dirty_encoded: np.ndarray = None,
                        y_dirty_encoded: np.ndarray = None,
                        X_cleaned_encoded: np.ndarray = None,
                        y_cleaned_encoded: np.ndarray = None,
                        X_clean_encoded: np.ndarray = None,
                        y_clean_encoded: np.ndarray = None,
                        X_clean_test_encoded: np.ndarray = None,
                        y_clean_test_encoded: np.ndarray = None) -> Dict:
    """
    Compute model noise tolerance.

    Prior tolerance (Tolerance_prior):
        Tolerance_prior(M) = (1/|E|) * sum_er (P_TolerClean(M,er) / P_do_nothing(M,er))

    Posterior tolerance (Tolerance_post):
        Tolerance_post(M) = (1/|E|) * sum_er (P_DemandClean(M,er) - P_do_nothing(M,er)) /
                                         (P_repair_all(M,er) - P_do_nothing(M,er))

    Args:
        dirty_data: dirty data
        cleaned_data: cleaned data (by the current method)
        clean_data: clean data (fully repaired)
        label_column: label column name
        task_type: task type
        model_name: model name

    Returns:
        Tolerance-metric dict
    """
    # Encoded-data shortcut: skip CSV -> preprocess_for_ml
    if (X_dirty_encoded is not None and X_cleaned_encoded is not None
            and X_clean_encoded is not None):
        safe_print("  [tolerance] using encoded version (skipping CSV preprocess)")
        X_dirty = X_dirty_encoded.copy()
        y_dirty = y_dirty_encoded.copy()
        X_cleaned = X_cleaned_encoded.copy()
        y_cleaned = y_cleaned_encoded.copy()
        X_clean = X_clean_encoded.copy()
        y_clean = y_clean_encoded.copy()

        # NaN-safe handling
        def _nan_safe(X, y):
            """Drop rows with NaN in y; fill NaN in X with column means."""
            y_valid = ~np.isnan(y)
            X, y = X[y_valid].copy(), y[y_valid].copy()
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            if nan_mask.any():
                for j in range(X.shape[1]):
                    m = col_means[j]
                    if np.isnan(m):
                        m = 0.0
                    X[nan_mask[:, j], j] = m
            return X, y

        X_dirty, y_dirty = _nan_safe(X_dirty, y_dirty)
        X_cleaned, y_cleaned = _nan_safe(X_cleaned, y_cleaned)
        X_clean, y_clean = _nan_safe(X_clean, y_clean)

        if len(X_cleaned) < 5:
            return {
                'P_do_nothing': 0.0, 'P_demand_clean': 0.0,
                'P_repair_all': 0.0, 'tolerance_prior': 0.0,
                'tolerance_post': 0.0,
            }

        # Check for an Oracle test set
        oracle_test = False
        if X_clean_test_encoded is not None and y_clean_test_encoded is not None:
            X_test_eval = X_clean_test_encoded.copy()
            y_test_eval = y_clean_test_encoded.copy()
            X_test_eval, y_test_eval = _nan_safe(X_test_eval, y_test_eval)
            if len(X_test_eval) > 0:
                oracle_test = True

        def _fit_predict(X_train, y_train, X_test, y_test):
            """Train and predict; return the performance metric."""
            if task_type == 'clustering':
                from sklearn.cluster import KMeans as _KMeans
                n_clusters = len(np.unique(y_test))
                km = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                y_pred = km.fit_predict(X_train)
                try:
                    return silhouette_score(X_train, y_pred,
                                            sample_size=min(len(X_train), 10000),
                                            random_state=42)
                except Exception:
                    return 0.0
            elif task_type == 'classification':
                model = get_classifier(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return accuracy_score(y_test, y_pred)
            else:
                model = get_regressor(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return r2_score(y_test, y_pred)

        if oracle_test:
            # Oracle mode: train on ALL, test on oracle test set (matches Step 5)
            safe_print(f"  Oracle mode: train dirty={len(X_dirty)}, "
                       f"cleaned={len(X_cleaned)}, clean={len(X_clean)}, "
                       f"test={len(X_test_eval)}")
            P_do_nothing = _fit_predict(X_dirty, y_dirty, X_test_eval, y_test_eval)
            P_demand_clean = _fit_predict(X_cleaned, y_cleaned, X_test_eval, y_test_eval)
            P_repair_all = _fit_predict(X_clean, y_clean, X_test_eval, y_test_eval)
        else:
            # Non-Oracle mode: derive test set via train_test_split on X_clean
            test_size = 0.2
            n_samples = len(X_clean)
            test_indices = np.random.RandomState(42).choice(
                n_samples, size=int(n_samples * test_size), replace=False
            )
            train_indices = np.array([i for i in range(n_samples) if i not in set(test_indices)])

            X_test_eval = X_clean[test_indices]
            y_test_eval = y_clean[test_indices]

            # Split dirty and cleaned at the same ratio (using their own data)
            def _split_and_eval(X, y):
                n = len(X)
                t_idx = np.random.RandomState(42).choice(
                    n, size=int(n * test_size), replace=False
                )
                tr_idx = np.array([i for i in range(n) if i not in set(t_idx)])
                return _fit_predict(X[tr_idx], y[tr_idx], X_test_eval, y_test_eval)

            P_do_nothing = _split_and_eval(X_dirty, y_dirty)
            P_demand_clean = _split_and_eval(X_cleaned, y_cleaned)
            P_repair_all = _fit_predict(X_clean[train_indices], y_clean[train_indices],
                                        X_test_eval, y_test_eval)

        tolerance_prior = P_demand_clean / P_do_nothing if P_do_nothing != 0 else 0
        tolerance_post = ((P_demand_clean - P_do_nothing) / (P_repair_all - P_do_nothing)
                          if P_repair_all != P_do_nothing else 0)

        results = {
            'P_do_nothing': P_do_nothing,
            'P_demand_clean': P_demand_clean,
            'P_repair_all': P_repair_all,
            'tolerance_prior': tolerance_prior,
            'tolerance_post': tolerance_post,
        }

        # Note: P_do_nothing/P_demand_clean/P_repair_all are INTERNAL quantities for the
        # tolerance formula only. The do-nothing / repair-all DOWNSTREAM performance to cite
        # is the one produced by the donothing / repairall BASELINE runs, not these P_ values.
        safe_print(f"\nTolerance results:")
        safe_print(f"  Prior tolerance (Tolerance_prior): {tolerance_prior:.4f}")
        safe_print(f"  Posterior tolerance (Tolerance_post): {tolerance_post:.4f}")

        return results

    # Unified encoding: all three datasets share the same LE/SS
    combined = pd.concat([dirty_data, cleaned_data, clean_data], ignore_index=True)
    non_feature_cols = {label_column, 'index', 'id'}
    combined_features = combined.drop(columns=[c for c in non_feature_cols if c in combined.columns])

    shared_encoders = {}
    for col in combined_features.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        le.fit(combined_features[col].astype(str))
        shared_encoders[col] = le
    if combined[label_column].dtype == 'object':
        le_label = LabelEncoder()
        le_label.fit(combined[label_column].astype(str))
        shared_encoders[label_column] = le_label

    combined_encoded = combined_features.copy()
    for col in combined_encoded.select_dtypes(include=['object']).columns:
        if col in shared_encoders:
            le = shared_encoders[col]
            combined_encoded[col] = combined_encoded[col].astype(str).map(
                lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else 0
            )
    combined_encoded = combined_encoded.fillna(combined_encoded.mean())
    shared_scaler = StandardScaler()
    shared_scaler.fit(combined_encoded)

    X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column,
                                          shared_encoders=shared_encoders,
                                          shared_scaler=shared_scaler)
    X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column,
                                              shared_encoders=shared_encoders,
                                              shared_scaler=shared_scaler)
    X_clean, y_clean = preprocess_for_ml(clean_data, label_column,
                                          shared_encoders=shared_encoders,
                                          shared_scaler=shared_scaler)

    # Minimum row check: cleaned rows must align with clean for correct evaluation
    if len(X_cleaned) < 5:
        return {
            'P_do_nothing': 0.0,
            'P_demand_clean': 0.0,
            'P_repair_all': 0.0,
            'tolerance_prior': 0.0,
            'tolerance_post': 0.0,
        }

    # When cleaned rows != clean rows, do train/test split on cleaned itself
    use_cleaned_split = (len(X_cleaned) != len(X_clean))

    # Split data
    test_size = 0.2
    n_samples = len(X_clean)
    test_indices = np.random.RandomState(42).choice(
        n_samples, size=int(n_samples * test_size), replace=False
    )
    train_indices = np.array([i for i in range(n_samples) if i not in test_indices])

    def get_performance(X, y_train_labels, X_test, y_test):
        """Get model performance.

        Args:
            X: training feature matrix (sliced by train_indices)
            y_train_labels: training labels (sliced by train_indices)
            X_test: test features
            y_test: test labels (clean ground truth)
        """
        if task_type == 'clustering':
            # Clustering: fit on all data, return silhouette_score (sample for large data)
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

    # Compute performance per scenario
    # P_do_nothing: train on dirty features + dirty labels; test on clean data
    P_do_nothing = get_performance(X_dirty, y_dirty, X_clean[test_indices], y_clean[test_indices])

    # P_DemandClean: train on cleaned data
    if use_cleaned_split:
        # cleaned rows differ from clean (agent deleted rows); cannot share indices.
        # Fallback: split cleaned itself, train on y_cleaned, test on y_cleaned.
        # Note: this branch cannot test against y_clean since row correspondence is lost.
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
                    # Clustering: fit on all cleaned data, return silhouette_score (sample for large data)
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
        # Cleaned features + cleaned labels train; clean data test
        P_demand_clean = get_performance(X_cleaned, y_cleaned, X_clean[test_indices], y_clean[test_indices])

    # P_repair_all: train on fully clean data
    P_repair_all = get_performance(X_clean, y_clean, X_clean[test_indices], y_clean[test_indices])

    # Prior tolerance
    if P_do_nothing != 0:
        tolerance_prior = P_demand_clean / P_do_nothing
    else:
        tolerance_prior = 0

    # Posterior tolerance
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

    # P_* are INTERNAL tolerance-formula quantities; cite donothing/repairall BASELINE perf instead.
    safe_print(f"\nTolerance results:")
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
        dirty_data: dirty data
        clean_data: clean data
        cleaned_data: cleaned data
        label_column: label column name
        task_type: task type
        model_name: model name
        error_rates: list of error rates

    Returns:
        Tolerance results across multiple error rates
    """
    if error_rates is None:
        error_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # Simplified: use a single error-rate result.
    # In practice, combine with an error injector to generate data at various rates.
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
    - Type 1: fully automatic, no human involvement (cost = 0)
    - Type 2: uses a small validation-set ground truth to evaluate cleaning (cost = validation_samples)
    - Type 3: user iteratively cleans dirty samples selected by the model (cost = iterations * batch_size)

    Args:
        method_type: method type (1, 2, 3)
        total_samples: total samples
        labeled_samples: number of labeled samples
        validation_samples: validation-set size
        iterations: iteration count (for Type 3)

    Returns:
        Cost dict
    """
    if method_type == 1:
        # Type 1 still uses labeled_samples; caller is responsible for the correct value.
        # Oracle mode is "fully automatic" but in practice uses ground truth.
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
    Comprehensive evaluation.

    Includes:
    1. Downstream task performance
    2. Model tolerance
    3. Ground-truth cost

    Args:
        dirty_data: dirty data
        cleaned_data: cleaned data
        clean_data: clean data
        label_column: label column name
        task_type: task type
        models: model list
        method_type: method type
        ground_truth_used: ground-truth samples used

    Returns:
        Comprehensive results
    """
    if models is None:
        models = ['rf', 'lr']

    safe_print("="*60)
    safe_print("Comprehensive evaluation start")
    safe_print("="*60)

    # 1. Downstream task performance
    safe_print("\n1. Downstream task performance")
    safe_print("-"*40)
    ml_results = evaluate_downstream_task(
        cleaned_data, clean_data, label_column, task_type, models
    )

    # 2. Model tolerance
    safe_print("\n2. Model tolerance")
    safe_print("-"*40)
    tolerance_results = calculate_tolerance(
        dirty_data, cleaned_data, clean_data, label_column, task_type
    )

    # 3. Ground-truth cost
    safe_print("\n3. Ground-truth cost")
    safe_print("-"*40)
    cost_results = calculate_ground_truth_cost(
        method_type=method_type,
        total_samples=len(clean_data),
        labeled_samples=ground_truth_used
    )
    safe_print(f"  Ground-truth usage type: Type {method_type}")
    safe_print(f"  Ground-truth used:       {cost_results['ground_truth_cost']}")
    safe_print(f"  Ground-truth ratio:      {cost_results['cost_ratio']:.2%}")

    # Aggregate results
    results = {
        'task_type': task_type,
        **ml_results,
        **tolerance_results,
        **cost_results
    }

    safe_print("\n" + "="*60)
    safe_print("Comprehensive evaluation done")
    safe_print("="*60)

    return results


# Test helper
def test_evaluation():
    """Test the evaluation functions."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Clean data
    clean_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.choice([0, 1], n_samples)
    })

    # Dirty data (with injected noise)
    dirty_data = clean_data.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    dirty_data.loc[noise_idx, 'feature1'] = np.nan
    dirty_data.loc[noise_idx[:50], 'feature2'] *= 10  # outliers

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
# Snoopy evaluation - embedding-based data quality upper-bound evaluation
# =============================================================================

def evaluate_snoopy_upper_bound(dirty_data: pd.DataFrame,
                                 cleaned_data: pd.DataFrame,
                                 clean_data: pd.DataFrame,
                                 label_column: str,
                                 task_type: str = 'classification',
                                 X_dirty_encoded: np.ndarray = None,
                                 y_dirty_encoded: np.ndarray = None,
                                 X_cleaned_encoded: np.ndarray = None,
                                 y_cleaned_encoded: np.ndarray = None,
                                 X_clean_encoded: np.ndarray = None,
                                 y_clean_encoded: np.ndarray = None,
                                 X_clean_test_encoded: np.ndarray = None,
                                 y_clean_test_encoded: np.ndarray = None) -> Dict:
    """
    Use Snoopy to evaluate data quality upper bound before/after cleaning.

    Snoopy uses embeddings to estimate the data-quality upper bound and tell whether cleaning improves it.

    Args:
        dirty_data: dirty data
        cleaned_data: cleaned data
        clean_data: clean data (ground truth)
        label_column: label column name
        task_type: task type

    Returns:
        Snoopy results
    """
    results = {
        'snoopy_available': False,
        'upper_bound_dirty': 0.0,
        'upper_bound_cleaned': 0.0,
        'upper_bound_clean': 0.0,
        'upper_bound_improvement': 0.0
    }

    try:
        # Encoded-data shortcut: skip CSV -> preprocess_for_ml
        if (X_dirty_encoded is not None and X_cleaned_encoded is not None
                and X_clean_encoded is not None):
            safe_print("  [Snoopy] using encoded version (skipping CSV preprocess)")
            X_dirty = X_dirty_encoded.copy()
            y_dirty = y_dirty_encoded.copy()
            X_cleaned = X_cleaned_encoded.copy()
            y_cleaned = y_cleaned_encoded.copy()
            X_clean = X_clean_encoded.copy()
            y_clean = y_clean_encoded.copy()

            # NaN-safe handling
            def _nan_safe_snoopy(X, y):
                y_valid = ~np.isnan(y)
                X, y = X[y_valid].copy(), y[y_valid].copy()
                col_means = np.nanmean(X, axis=0)
                nan_mask = np.isnan(X)
                if nan_mask.any():
                    for j in range(X.shape[1]):
                        m = col_means[j]
                        if np.isnan(m):
                            m = 0.0
                        X[nan_mask[:, j], j] = m
                return X, y

            X_dirty, y_dirty = _nan_safe_snoopy(X_dirty, y_dirty)
            X_cleaned, y_cleaned = _nan_safe_snoopy(X_cleaned, y_cleaned)
            X_clean, y_clean = _nan_safe_snoopy(X_clean, y_clean)

            if len(X_cleaned) < 5:
                safe_print(f"[Snoopy] cleaned data has only {len(X_cleaned)} rows (<5); skipping upper-bound eval")
                return results

            results['snoopy_available'] = True
            safe_print("Evaluating data-quality upper bound on encoded data...")

            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if task_type == 'clustering':
                from sklearn.cluster import KMeans as _KMeans
                def _clustering_upper_bound(X, y):
                    n_clusters = len(np.unique(y))
                    km = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    y_pred = km.fit_predict(X)
                    try:
                        return silhouette_score(X, y_pred, sample_size=min(len(X), 10000), random_state=42)
                    except Exception:
                        return 0.0
                results['upper_bound_dirty'] = _clustering_upper_bound(X_dirty, y_dirty)
                results['upper_bound_cleaned'] = _clustering_upper_bound(X_cleaned, y_cleaned)
                results['upper_bound_clean'] = _clustering_upper_bound(X_clean, y_clean)
            elif task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                results['upper_bound_dirty'] = np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5))
                results['upper_bound_cleaned'] = np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned))))
                results['upper_bound_clean'] = np.mean(cross_val_score(model, X_clean, y_clean, cv=5))
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                results['upper_bound_dirty'] = -np.mean(cross_val_score(model, X_dirty, y_dirty, cv=5, scoring='neg_mean_squared_error'))
                results['upper_bound_cleaned'] = -np.mean(cross_val_score(model, X_cleaned, y_cleaned, cv=min(5, len(X_cleaned)), scoring='neg_mean_squared_error'))
                results['upper_bound_clean'] = -np.mean(cross_val_score(model, X_clean, y_clean, cv=5, scoring='neg_mean_squared_error'))

            if results['upper_bound_dirty'] != 0:
                results['upper_bound_improvement'] = (
                    (results['upper_bound_cleaned'] - results['upper_bound_dirty']) /
                    (results['upper_bound_clean'] - results['upper_bound_dirty'])
                    if results['upper_bound_clean'] != results['upper_bound_dirty'] else 0
                )

            return results

        # Try to import snoopy.
        # Note: tools/snoopy/snoopy/ is the real package; we insert tools/snoopy at the
        # front of sys.path, otherwise Python finds tools/snoopy/__init__.py (empty) first.
        _snoopy_parent = os.path.join(_current_dir, 'snoopy')
        if _snoopy_parent in sys.path:
            sys.path.remove(_snoopy_parent)
        sys.path.insert(0, _snoopy_parent)
        from snoopy.pipeline import run as snoopy_run

        # Preprocess data (unified encoding)
        combined = pd.concat([dirty_data, cleaned_data, clean_data], ignore_index=True)
        non_feature_cols = {label_column, 'index', 'id'}
        combined_features = combined.drop(columns=[c for c in non_feature_cols if c in combined.columns])

        shared_encoders = {}
        for col in combined_features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            le.fit(combined_features[col].astype(str))
            shared_encoders[col] = le
        if combined[label_column].dtype == 'object':
            le_label = LabelEncoder()
            le_label.fit(combined[label_column].astype(str))
            shared_encoders[label_column] = le_label

        combined_encoded = combined_features.copy()
        for col in combined_encoded.select_dtypes(include=['object']).columns:
            if col in shared_encoders:
                le = shared_encoders[col]
                combined_encoded[col] = combined_encoded[col].astype(str).map(
                    lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else 0
                )
        combined_encoded = combined_encoded.fillna(combined_encoded.mean())
        shared_scaler = StandardScaler()
        shared_scaler.fit(combined_encoded)

        X_dirty, y_dirty = preprocess_for_ml(dirty_data, label_column,
                                              shared_encoders=shared_encoders,
                                              shared_scaler=shared_scaler)
        X_cleaned, y_cleaned = preprocess_for_ml(cleaned_data, label_column,
                                                  shared_encoders=shared_encoders,
                                                  shared_scaler=shared_scaler)
        X_clean, y_clean = preprocess_for_ml(clean_data, label_column,
                                              shared_encoders=shared_encoders,
                                              shared_scaler=shared_scaler)

        # Guard: if cleaned data is too small for cross_val
        if len(X_cleaned) < 5:
            safe_print(f"[Snoopy] cleaned data has only {len(X_cleaned)} rows (<5); skipping upper-bound eval")
            return results

        # Evaluate upper bound for each dataset
        results['snoopy_available'] = True
        safe_print("Snoopy module available; evaluating data-quality upper bound...")

        # Simplified upper-bound estimate (based on model cross-validation)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if task_type == 'clustering':
            # Clustering: use silhouette_score as the upper-bound metric (sample for large data)
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
        safe_print("Snoopy module unavailable; skipping upper-bound eval")
    except Exception as e:
        safe_print(f"Snoopy evaluation error: {e}")

    return results


# =============================================================================
# Ideal minimum ground-truth cost
# =============================================================================

def calculate_ideal_min_ground_truth_cost(dirty_data: pd.DataFrame,
                                           cleaned_data: pd.DataFrame,
                                           clean_data: pd.DataFrame,
                                           index_attribute: str = 'index') -> Dict:
    """
    Compute the ideal minimum ground-truth cost.

    Ideal minimum = number of cells in the cleaned-vs-dirty diff that are actually correct repairs.
    Only cells that truly use ground truth are counted.

    When cleaned rows differ from dirty/clean (e.g. agent deleted some rows),
    only the rows kept in cleaned are compared.

    Args:
        dirty_data: dirty data
        cleaned_data: cleaned data
        clean_data: clean data (ground truth)
        index_attribute: index column name

    Returns:
        Ideal minimum ground-truth cost info
    """
    dirty = dirty_data.copy()
    cleaned = cleaned_data.copy()
    clean = clean_data.copy()

    # Exclude the index column
    cols_to_compare = [c for c in dirty.columns if c != index_attribute]

    # Align by index: compare only rows kept in cleaned
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
        # No index column; take the shortest length
        min_len = min(len(dirty), len(cleaned), len(clean))
        dirty = dirty.iloc[:min_len].reset_index(drop=True)
        cleaned = cleaned.iloc[:min_len].reset_index(drop=True)
        clean = clean.iloc[:min_len].reset_index(drop=True)

    # Per-cell comparison
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

            # Check whether the cell was modified
            if dirty_val != cleaned_val:
                changes_count += 1
                # Check whether the modification matches the ground truth
                if cleaned_val == clean_val:
                    correct_repairs += 1
                else:
                    wrong_repairs += 1

    # Additional stat: number of deleted rows
    deleted_rows = len(dirty_data) - len(cleaned_data)

    results = {
        'total_changes': changes_count,
        'correct_repairs': correct_repairs,
        'wrong_repairs': wrong_repairs,
        'deleted_rows': max(0, deleted_rows),
        'ideal_min_ground_truth_cost': correct_repairs,  # ideally, only these ground-truth values are needed
        'repair_accuracy': correct_repairs / changes_count if changes_count > 0 else 0
    }

    return results


# =============================================================================
# Unified evaluation entry point - run_all_evaluation
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
                       verbose: bool = True,
                       cleaned_encoded_path: str = None,
                       encoded_arrays: dict = None) -> Dict:
    """
    Unified data-cleaning evaluation function.

    All run_*_base.py scripts should call this function for standardized evaluation.

    Args:
        dirty_path: dirty CSV path
        cleaned_path: cleaned CSV path
        clean_path: clean CSV path (ground truth)
        output_path: directory for output files
        task_name: task name
        label_column: label column name (optional; used for downstream evaluation)
        task_type: task type ('classification', 'regression', 'clustering')
        models: list of evaluation models
        method_type: cleaning method type (1=auto, 2=needs validation set, 3=iterative)
        ground_truth_used: number of ground-truth values actually used (supplied by the cleaning method)
        index_attribute: index column name
        mse_attributes: attributes for MSE computation
        verbose: whether to print details

    Returns:
        Full evaluation results dict
    """
    if models is None:
        models = ['rf', 'lr']

    if mse_attributes is None:
        mse_attributes = []

    # Load data
    dirty_data = pd.read_csv(dirty_path)
    cleaned_data = pd.read_csv(cleaned_path)
    clean_data = pd.read_csv(clean_path)

    # Prefer encoded version (avoids CSV roundtrip precision loss)
    enc_cleaned = None  # (X_cleaned_encoded, y_cleaned_encoded)
    enc_arrays = None   # {'X_dirty', 'y_dirty', 'X_clean', 'y_clean'}
    if cleaned_encoded_path and os.path.exists(cleaned_encoded_path):
        try:
            npz = np.load(cleaned_encoded_path, allow_pickle=True)
            enc_cleaned = (npz['X_result'], npz['y_result'])
            if verbose:
                safe_print(f"  using encoded version: {cleaned_encoded_path}")
        except Exception as e:
            if verbose:
                safe_print(f"  [warn] encoded-version load failed; falling back to CSV: {e}")
            enc_cleaned = None
    if encoded_arrays is not None and enc_cleaned is not None:
        enc_arrays = encoded_arrays

    # Attribute list
    attributes = clean_data.columns.tolist()

    # Guard for empty cleaned data: return empty result
    if len(cleaned_data) == 0:
        if verbose:
            safe_print("[warn] cleaned data is empty (0 rows); all metrics set to 0")
        results = {
            'task_name': task_name,
            'task_type': task_type,
            'total_records': len(clean_data),
            'total_cells': len(clean_data) * len(attributes),
            'cleaned_rows': 0,
            'note': 'cleaned_data is empty (0 rows), all metrics set to 0',
        }
        # Write placeholder result file
        os.makedirs(output_path, exist_ok=True)
        eval_file = os.path.join(output_path, f'{task_name}_evaluation_results.txt')
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write(f"[warn] cleaned data is empty (0 rows); cannot compute any metrics\n")
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
        safe_print(f"DemandClean-Benchmark unified evaluation - {task_name}")
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
            mse_attributes=mse_attributes
        )
        results.update(traditional_results)
    except ImportError:
        if verbose:
            safe_print("warn: getScore module unavailable; skipping traditional cleaning metrics")
    except Exception as e:
        if verbose:
            import traceback
            safe_print(f"traditional cleaning metrics failed: {type(e).__name__}: {e}")
            safe_print(traceback.format_exc())

    # ==========================================================================
    # 2. Downstream task performance
    # ==========================================================================
    if label_column and label_column in clean_data.columns:
        if verbose:
            safe_print(f"\n[2/5] Downstream task performance ({task_type})")
            safe_print("-" * 50)

        # Choose test set: prefer oracle test (matches Step 5)
        _X_clean_test = enc_arrays.get('X_clean_test') if enc_arrays else None
        _y_clean_test = enc_arrays.get('y_clean_test') if enc_arrays else None
        if _X_clean_test is None:
            _X_clean_test = enc_arrays['X_clean'] if enc_arrays else None
            _y_clean_test = enc_arrays['y_clean'] if enc_arrays else None

        ml_results = evaluate_downstream_task(
            cleaned_data=cleaned_data,
            clean_data=clean_data,
            label_column=label_column,
            task_type=task_type,
            models=models,
            index_column=index_attribute,
            X_cleaned_encoded=enc_cleaned[0] if enc_cleaned else None,
            y_cleaned_encoded=enc_cleaned[1] if enc_cleaned else None,
            X_clean_encoded=_X_clean_test,
            y_clean_encoded=_y_clean_test,
        )
        results.update({f'ml_{k}': v for k, v in ml_results.items()})
    else:
        if verbose:
            safe_print("\n[2/5] Downstream task performance - skipped (no label column)")

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
            task_type=task_type,
            X_dirty_encoded=enc_arrays['X_dirty'] if enc_arrays else None,
            y_dirty_encoded=enc_arrays['y_dirty'] if enc_arrays else None,
            X_cleaned_encoded=enc_cleaned[0] if enc_cleaned else None,
            y_cleaned_encoded=enc_cleaned[1] if enc_cleaned else None,
            X_clean_encoded=enc_arrays['X_clean'] if enc_arrays else None,
            y_clean_encoded=enc_arrays['y_clean'] if enc_arrays else None,
            X_clean_test_encoded=_X_clean_test,
            y_clean_test_encoded=_y_clean_test,
        )
        results.update({f'tolerance_{k}': v for k, v in tolerance_results.items()})
    else:
        if verbose:
            safe_print("\n[3/5] Model tolerance - skipped (no label column)")

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
            task_type=task_type,
            X_dirty_encoded=enc_arrays['X_dirty'] if enc_arrays else None,
            y_dirty_encoded=enc_arrays['y_dirty'] if enc_arrays else None,
            X_cleaned_encoded=enc_cleaned[0] if enc_cleaned else None,
            y_cleaned_encoded=enc_cleaned[1] if enc_cleaned else None,
            X_clean_encoded=enc_arrays['X_clean'] if enc_arrays else None,
            y_clean_encoded=enc_arrays['y_clean'] if enc_arrays else None,
            X_clean_test_encoded=_X_clean_test,
            y_clean_test_encoded=_y_clean_test,
        )
        results.update({f'snoopy_{k}': v for k, v in snoopy_results.items()})

        if verbose and snoopy_results.get('snoopy_available'):
            safe_print(f"  dirty upper bound:    {snoopy_results['upper_bound_dirty']:.4f}")
            safe_print(f"  cleaned upper bound:  {snoopy_results['upper_bound_cleaned']:.4f}")
            safe_print(f"  clean upper bound:    {snoopy_results['upper_bound_clean']:.4f}")
            safe_print(f"  upper-bound improvement ratio: {snoopy_results['upper_bound_improvement']:.4f}")
    else:
        if verbose:
            safe_print("\n[4/5] Snoopy upper-bound evaluation - skipped (no label column)")

    # ==========================================================================
    # 5. Ground-truth cost
    # ==========================================================================
    if verbose:
        safe_print(f"\n[5/5] Ground-truth cost")
        safe_print("-" * 50)

    # Compute ideal minimum ground-truth cost
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
        safe_print(f"  Method type: Type {method_type} ({'auto' if method_type==1 else 'needs validation set' if method_type==2 else 'iterative'})")
        safe_print(f"  Actual ground-truth used: {ground_truth_used}")
        safe_print(f"  Ideal minimum cost:       {ideal_cost['ideal_min_ground_truth_cost']}")
        safe_print(f"  Total changed cells:      {ideal_cost['total_changes']}")
        safe_print(f"  Correct repairs:          {ideal_cost['correct_repairs']}")
        safe_print(f"  Wrong repairs:            {ideal_cost['wrong_repairs']}")

    # ==========================================================================
    # Save results
    # ==========================================================================
    results_file = os.path.join(output_path, f"{task_name}_evaluation_results.txt")
    os.makedirs(output_path, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"DemandClean-Benchmark unified evaluation report\n")
        f.write(f"Task: {task_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("[Traditional cleaning metrics]\n")
        for key in ['precision', 'accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
            if key in results:
                f.write(f"  {key}: {results[key]}\n")

        f.write("\n[Downstream task performance]\n")
        for key, value in results.items():
            if key.startswith('ml_'):
                f.write(f"  {key}: {value}\n")

        f.write("\n[Model tolerance]\n")
        for key, value in results.items():
            if key.startswith('tolerance_'):
                f.write(f"  {key}: {value}\n")

        f.write("\n[Snoopy upper bound]\n")
        for key, value in results.items():
            if key.startswith('snoopy_'):
                f.write(f"  {key}: {value}\n")

        f.write("\n[Ground-truth cost]\n")
        f.write(f"  Method type: Type {method_type}\n")
        f.write(f"  Actual ground-truth used: {ground_truth_used}\n")
        f.write(f"  Ideal minimum cost: {ideal_cost['ideal_min_ground_truth_cost']}\n")
        f.write(f"  Repair accuracy:    {ideal_cost['repair_accuracy']:.4f}\n")

    if verbose:
        safe_print("\n" + "=" * 70)
        safe_print(f"Evaluation done; results saved: {results_file}")
        safe_print("=" * 70)

    return results
