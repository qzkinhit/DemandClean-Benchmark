"""
容忍度分析模块
==============

计算下游模型对数据质量的容忍度 τ_M = Acc(dirty) / Acc(clean),
以及清洗策略的相对恢复率 ρ_{M,S}。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# ---------------------------------------------------------------------------
# 辅助: 获取 sklearn 模型
# ---------------------------------------------------------------------------

def _get_model(name: str, task: str):
    """根据名称获取 sklearn 模型实例"""
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
    """训练并返回性能分数 (越大越好)"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if task == 'classification':
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred)  # 负MSE


# ---------------------------------------------------------------------------
# 核心函数
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
    计算各下游模型对数据质量的容忍度

    τ_M = Performance(dirty) / Performance(clean)

    Args:
        X_clean, y_clean: 干净训练数据
        X_dirty, y_dirty: 脏训练数据
        X_test, y_test: 测试数据 (使用干净标签)
        task: 'classification' 或 'regression'
        models: 模型名称列表

    Returns:
        {model_name: tolerance} 字典
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
        print(f"  {name}: Perf(clean)={perf_clean:.4f}, Perf(dirty)={perf_dirty:.4f}, τ={tau:.4f}")

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
    计算各清洗策略在不同下游模型上的相对恢复率

    ρ_{M,S} = (Perf(cleaned) - Perf(dirty)) / (Perf(clean) - Perf(dirty))

    Args:
        cleaned_results: {strategy_name: X_cleaned} 各策略清洗后的特征
        y_train: 训练标签
        X_test, y_test: 测试数据
        ideal_perf: {model_name: perf_on_clean} 干净数据的理想性能
        dirty_perf: {model_name: perf_on_dirty} 脏数据性能
        task: 任务类型
        models: 模型列表

    Returns:
        {strategy: {model: recovery_rate}} 嵌套字典
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
    绘制策略×模型容忍度热力图

    Args:
        strategy_tolerance: compute_strategy_tolerance 的返回值
        save_path: 保存路径 (不指定则 plt.show)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn 不可用，跳过绘图")
        return

    df = pd.DataFrame(strategy_tolerance).T
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.5), max(4, len(df) * 0.6)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Strategy × Model Recovery Rate (ρ)')
    ax.set_ylabel('Strategy')
    ax.set_xlabel('Model')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"热力图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()
