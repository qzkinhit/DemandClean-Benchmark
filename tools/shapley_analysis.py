"""
Shapley Value analysis module
=============================

Treat the 4 DQN-Agent actions as game-theoretic players and use Shapley
values to quantify each action's marginal contribution to downstream
performance.

Players: {no_action, repair_value, delete, replace_nearby}
Coalition value v(S) = downstream model accuracy obtained when the agent
is restricted to only using actions in S.
"""

import math
import itertools
import random as _random
from typing import Dict, List, Callable, Optional

import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error


# Names of the 4 actions
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
    Compute the coalition value: restrict the agent to actions in
    ``allowed_actions``.

    1. For each error cell, call ``agent.act()``.
    2. If the selected action is not in ``allowed_actions``, force it to
       no_action.
    3. Run cleaning, train the downstream model, and return the score.
    """
    X_work = X_dirty.copy()

    action_map = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}

    for row_idx, col_idx, error_type, clean_val in error_list:
        if row_idx >= len(X_work):
            continue

        # Build a simplified state (use the dirty value itself)
        state = np.zeros(8)
        state[0] = X_work[row_idx, col_idx] if col_idx < X_work.shape[1] else 0
        state[1] = float(error_type)

        # Query the agent for a recommended action
        if hasattr(agent, 'act_stage1'):
            final_action, _, _ = agent.act(state, training=False)
        else:
            final_action = agent.act(state, training=False)

        action_name = action_map.get(final_action, 'no_action')

        # Restrict the action set
        if action_name not in allowed_actions:
            final_action = 0  # fall back to no_action

        # Execute the action
        if final_action == 1 and clean_val is not None:
            X_work[row_idx, col_idx] = clean_val
        elif final_action == 2:
            X_work[row_idx, col_idx] = np.nan
        elif final_action == 3:
            # Approximate with the column mean
            col_vals = X_work[:, col_idx]
            valid = col_vals[~np.isnan(col_vals)]
            X_work[row_idx, col_idx] = np.mean(valid) if len(valid) > 0 else 0

    # Handle NaNs introduced by delete
    col_means = np.nanmean(X_work, axis=0)
    for j in range(X_work.shape[1]):
        mask = np.isnan(X_work[:, j])
        X_work[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    # Train and evaluate
    eval_model.fit(X_work, y)
    y_pred = eval_model.predict(X_test)

    if task == 'classification':
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred)


# ---------------------------------------------------------------------------
# Exact Shapley (4! = 24 permutations)
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
    Exact Shapley values by full enumeration of all 4! = 24 permutations.

    Args:
        agent: DQN Agent (must expose an ``act`` method)
        X_dirty: dirty feature matrix
        y: labels
        error_list: [(row, col, error_type, clean_val), ...]
        X_test, y_test: test data
        eval_model: sklearn model instance (default: RF)
        task: 'classification' or 'regression'
        value_fn: optional custom coalition-value function

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

    # Enumerate all permutations
    for perm in itertools.permutations(range(n)):
        coalition = set()
        prev_val = value_fn(coalition)

        for idx in perm:
            coalition.add(OPERATIONS[idx])
            curr_val = value_fn(coalition)
            shapley[OPERATIONS[idx]] += (curr_val - prev_val)
            prev_val = curr_val

    # Average
    n_perms = math.factorial(n)
    for op in OPERATIONS:
        shapley[op] /= n_perms

    return shapley


# ---------------------------------------------------------------------------
# Monte Carlo Shapley (large-scale approximation)
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
    Monte Carlo approximation of Shapley values via permutation sampling.

    Args:
        n_samples: number of sampled permutations
        remaining arguments are the same as ``compute_exact_shapley``

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
# Visualization
# ---------------------------------------------------------------------------

def plot_shapley_bar(shapley_values: Dict[str, float],
                     agent_name: str = 'Agent',
                     save_path: str = None) -> None:
    """Per-agent Shapley bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
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
        print(f"Shapley bar chart saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_shapley_comparison(all_shapley: Dict[str, Dict[str, float]],
                            save_path: str = None) -> None:
    """Multi-agent Shapley comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
    except ImportError:
        print("matplotlib not available; skipping plot")
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
        print(f"Shapley comparison chart saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
