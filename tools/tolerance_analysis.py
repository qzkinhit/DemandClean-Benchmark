"""
Tolerance analysis
==================

Compute a downstream model's tolerance to data quality, tau_M = Acc(dirty) / Acc(clean),
and the relative recovery rate rho_{M,S} of each cleaning strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------------
# Helper: fetch a sklearn model by name
# ---------------------------------------------------------------------------

def _get_model(name: str, task: str):
    """Return a sklearn model instance for the given name."""
    if task == 'classification':
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        pool = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
        }
    else:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        pool = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
        }
    return pool.get(name, list(pool.values())[0])


def _score(model, X_train, y_train, X_test, y_test, task: str) -> float:
    """Fit and return a performance score (higher is better)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if task == 'classification':
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred)  # negative MSE


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_model_tolerance(X_clean: np.ndarray,
                            y_clean: np.ndarray,
                            X_dirty: np.ndarray,
                            y_dirty: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            task: str = 'classification',
                            models: List[str] = None) -> Dict[str, float]:
    """
    Compute each downstream model's tolerance to data quality.

    tau_M = Performance(dirty) / Performance(clean)

    Args:
        X_clean, y_clean: clean training data
        X_dirty, y_dirty: dirty training data
        X_test, y_test: test data (with clean labels)
        task: 'classification' or 'regression'
        models: list of model names

    Returns:
        {model_name: tolerance} dict.
    """
    if models is None:
        models = ['rf', 'lr']

    result = {}
    for name in models:
        m_clean = _get_model(name, task)
        m_dirty = _get_model(name, task)

        perf_clean = _score(m_clean, X_clean, y_clean, X_test, y_test, task)
        perf_dirty = _score(m_dirty, X_dirty, y_dirty, X_test, y_test, task)

        tau = perf_dirty / perf_clean if perf_clean != 0 else 0.0
        result[name] = tau
        print(f"  {name}: Perf(clean)={perf_clean:.4f}, Perf(dirty)={perf_dirty:.4f}, tau={tau:.4f}")

    return result


def compute_strategy_tolerance(cleaned_results: Dict[str, np.ndarray],
                               y_train: np.ndarray,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               ideal_perf: Dict[str, float],
                               dirty_perf: Dict[str, float],
                               task: str = 'classification',
                               models: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute the relative recovery rate of each cleaning strategy across models.

    rho_{M,S} = (Perf(cleaned) - Perf(dirty)) / (Perf(clean) - Perf(dirty))

    Args:
        cleaned_results: {strategy_name: X_cleaned} cleaned features per strategy
        y_train: training labels
        X_test, y_test: test data
        ideal_perf: {model_name: perf_on_clean} ideal performance on clean data
        dirty_perf: {model_name: perf_on_dirty} performance on dirty data
        task: task type
        models: model list

    Returns:
        {strategy: {model: recovery_rate}} nested dict.
    """
    if models is None:
        models = ['rf', 'lr']

    result = {}
    for strategy, X_cleaned in cleaned_results.items():
        row = {}
        for name in models:
            m = _get_model(name, task)
            perf = _score(m, X_cleaned, y_train, X_test, y_test, task)
            gap = ideal_perf[name] - dirty_perf[name]
            rho = (perf - dirty_perf[name]) / gap if gap != 0 else 0.0
            row[name] = rho
        result[strategy] = row
    return result


def plot_tolerance_heatmap(strategy_tolerance: Dict[str, Dict[str, float]],
                           save_path: str = None) -> None:
    """
    Plot the strategy x model recovery-rate heatmap.

    Args:
        strategy_tolerance: the return value of compute_strategy_tolerance
        save_path: save path (plt.show when omitted)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn unavailable; skipping plot")
        return

    df = pd.DataFrame(strategy_tolerance).T
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.5), max(4, len(df) * 0.6)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Strategy x Model Recovery Rate (rho)')
    ax.set_ylabel('Strategy')
    ax.set_xlabel('Model')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Heatmap saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
