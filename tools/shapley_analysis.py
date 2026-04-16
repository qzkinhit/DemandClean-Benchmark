"""
Shapley Value 分析模块
======================

将 DQN Agent 的 4 种操作视为博弈玩家,
通过 Shapley Value 量化每种操作对下游性能的边际贡献。

玩家: {no_action, repair_value, delete, replace_nearby}
联盟值 v(S) = 限制 Agent 只能使用 S 中的操作后, 下游模型准确率。
"""

import math
import itertools
import random as _random
from typing import Dict, List, Callable, Optional

import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error


# 4 种操作名称
OPERATIONS = ['no_action', 'repair_value', 'delete', 'replace_nearby']


def _default_coalition_value(allowed_actions: set,
                             agent,
                             X_dirty: np.ndarray,
                             y: np.ndarray,
                             error_list: list,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             eval_model,
                             task: str) -> float:
    """
    计算联盟值: 限制 Agent 只能使用 allowed_actions 中的操作

    1. 对每个 error_cell 调用 agent.act()
    2. 如果动作不在 allowed_actions 中, 强制替换为 no_action
    3. 执行清洗, 训练下游模型, 返回分数
    """
    X_work = X_dirty.copy()

    action_map = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}

    for row_idx, col_idx, error_type, clean_val in error_list:
        if row_idx >= len(X_work):
            continue

        # 构造简化 state (使用 dirty 值本身)
        state = np.zeros(8)
        state[0] = X_work[row_idx, col_idx] if col_idx < X_work.shape[1] else 0
        state[1] = float(error_type)

        # 获取 Agent 推荐动作
        if hasattr(agent, 'act_stage1'):
            final_action, _, _ = agent.act(state, training=False)
        else:
            final_action = agent.act(state, training=False)

        action_name = action_map.get(final_action, 'no_action')

        # 限制操作集
        if action_name not in allowed_actions:
            final_action = 0  # 退化为 no_action

        # 执行操作
        if final_action == 1 and clean_val is not None:
            X_work[row_idx, col_idx] = clean_val
        elif final_action == 2:
            X_work[row_idx, col_idx] = np.nan
        elif final_action == 3:
            # 用列均值近似
            col_vals = X_work[:, col_idx]
            valid = col_vals[~np.isnan(col_vals)]
            X_work[row_idx, col_idx] = np.mean(valid) if len(valid) > 0 else 0

    # 处理 NaN (delete 产生的)
    col_means = np.nanmean(X_work, axis=0)
    for j in range(X_work.shape[1]):
        mask = np.isnan(X_work[:, j])
        X_work[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    # 训练并评估
    eval_model.fit(X_work, y)
    y_pred = eval_model.predict(X_test)

    if task == 'classification':
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred)


# ---------------------------------------------------------------------------
# 精确 Shapley (4! = 24 排列)
# ---------------------------------------------------------------------------

def compute_exact_shapley(agent,
                          X_dirty: np.ndarray,
                          y: np.ndarray,
                          error_list: list,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          eval_model=None,
                          task: str = 'classification',
                          value_fn: Callable = None) -> Dict[str, float]:
    """
    精确 Shapley Value (4! = 24 排列全枚举)

    Args:
        agent: DQN Agent (需要有 act 方法)
        X_dirty: 脏数据特征矩阵
        y: 标签
        error_list: [(row, col, error_type, clean_val), ...]
        X_test, y_test: 测试数据
        eval_model: sklearn 模型实例 (默认 RF)
        task: 'classification' 或 'regression'
        value_fn: 自定义联盟值函数 (可选)

    Returns:
        {operation_name: shapley_value}
    """
    if eval_model is None:
        from sklearn.ensemble import RandomForestClassifier
        eval_model = RandomForestClassifier(n_estimators=50, random_state=42)

    if value_fn is None:
        def value_fn(allowed):
            return _default_coalition_value(
                allowed, agent, X_dirty, y, error_list,
                X_test, y_test, eval_model, task
            )

    n = len(OPERATIONS)
    shapley = {op: 0.0 for op in OPERATIONS}

    # 枚举所有排列
    for perm in itertools.permutations(range(n)):
        coalition = set()
        prev_val = value_fn(coalition)

        for idx in perm:
            coalition.add(OPERATIONS[idx])
            curr_val = value_fn(coalition)
            shapley[OPERATIONS[idx]] += (curr_val - prev_val)
            prev_val = curr_val

    # 平均
    n_perms = math.factorial(n)
    for op in OPERATIONS:
        shapley[op] /= n_perms

    return shapley


# ---------------------------------------------------------------------------
# 蒙特卡洛 Shapley (大规模近似)
# ---------------------------------------------------------------------------

def compute_shapley_values(agent,
                           X_dirty: np.ndarray,
                           y: np.ndarray,
                           error_list: list,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           n_samples: int = 200,
                           eval_model=None,
                           task: str = 'classification',
                           value_fn: Callable = None,
                           seed: int = 42) -> Dict[str, float]:
    """
    蒙特卡洛 Shapley Value (采样近似)

    Args:
        n_samples: 采样排列数
        其余参数同 compute_exact_shapley

    Returns:
        {operation_name: shapley_value}
    """
    if eval_model is None:
        from sklearn.ensemble import RandomForestClassifier
        eval_model = RandomForestClassifier(n_estimators=50, random_state=42)

    if value_fn is None:
        def value_fn(allowed):
            return _default_coalition_value(
                allowed, agent, X_dirty, y, error_list,
                X_test, y_test, eval_model, task
            )

    n = len(OPERATIONS)
    shapley = {op: 0.0 for op in OPERATIONS}
    rng = _random.Random(seed)

    for _ in range(n_samples):
        perm = list(range(n))
        rng.shuffle(perm)

        coalition = set()
        prev_val = value_fn(coalition)

        for idx in perm:
            coalition.add(OPERATIONS[idx])
            curr_val = value_fn(coalition)
            shapley[OPERATIONS[idx]] += (curr_val - prev_val)
            prev_val = curr_val

    for op in OPERATIONS:
        shapley[op] /= n_samples

    return shapley


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_shapley_bar(shapley_values: Dict[str, float],
                     agent_name: str = 'Agent',
                     save_path: str = None) -> None:
    """单 Agent 的 Shapley 柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    ops = list(shapley_values.keys())
    vals = [shapley_values[op] for op in ops]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in vals]
    ax.barh(ops, vals, color=colors)
    ax.set_xlabel('Shapley Value')
    ax.set_title(f'Shapley Values - {agent_name}')
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Shapley 柱状图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_shapley_comparison(all_shapley: Dict[str, Dict[str, float]],
                            save_path: str = None) -> None:
    """多 Agent 对比 Shapley 柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    agents = list(all_shapley.keys())
    ops = OPERATIONS
    n_agents = len(agents)
    n_ops = len(ops)

    x = np.arange(n_ops)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, agent_name in enumerate(agents):
        vals = [all_shapley[agent_name].get(op, 0) for op in ops]
        ax.bar(x + i * width, vals, width, label=agent_name)

    ax.set_xlabel('Operation')
    ax.set_ylabel('Shapley Value')
    ax.set_title('Shapley Values Comparison')
    ax.set_xticks(x + width * (n_agents - 1) / 2)
    ax.set_xticklabels(ops)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Shapley 对比图已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()
