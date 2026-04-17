#!/usr/bin/env python3
"""
Search-space visualization experiment
=====================================

Validates the search-space theory for DemandClean cleaning strategies:
maps the performance distribution of all possible strategies onto the
Authenticity-Diversity plane.

Method:
1. Four extreme points: NoFix, FullFix, DeleteFix, RelaxFix
2. Random sampling: sweep all action probability combinations (no_action, repair, delete, replace_nearby)
3. DQN inference: locate DemandFix within the search space
4. Heatmap: show performance (Accuracy) across the search space

Theoretical search space (crescent):

    Diversity ^
              |
     FullFix  *---------------*  DeleteFix (diversity lower bound)
              |\\  search space
              | \\  (crescent)
              |  \\            /
              |   * DemandFix /
              |    \\        /
              |     \\      /
    NoFix *---+------*-----+---> Authenticity
              |  RelaxFix

Rewritten from search_space_experiment.py; the main change swaps the old
TensorFlow DQN for the released demandclean/ API, dropping the TF dependency.
All visualization functions are kept; logic and style are unchanged, only
save paths were updated.
"""

import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import time
import copy

# alphashape / shapely (search-space boundary)
try:
    import alphashape
    from shapely.geometry import Point
    HAS_ALPHASHAPE = True
except ImportError:
    HAS_ALPHASHAPE = False
    print("[WARN] alphashape/shapely not installed; some plots will skip alpha-shape clipping")
    print("       pip install alphashape shapely")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')
import sys
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

# Experiment parameters
N_RANDOM_SAMPLES = 1000    # random sample count
N_DQN_RUNS = 1             # DQN inference runs
SEMANTIC_RATE = 0.15        # semantic error injection rate
SYNTACTIC_RATE = 0.20       # syntactic error injection rate

RUN_EXPERIMENT = False      # True=run experiment, False=reuse existing results and only replot
LOAD_MODEL = True           # True=load trained model, False=retrain DQN

# Extreme strategy names (to distinguish extreme points from random samples)
EXTREME_NAMES = {'NoFix', 'FullFix', 'DeleteFix', 'RelaxFix'}


def _is_random(name):
    """Return True if the entry is a random sample (not extreme, not DQN)."""
    return name not in EXTREME_NAMES and not name.startswith('DemandFix')

# Import DemandClean API
from demandclean.api.demand_clean import DemandClean

np.random.seed(2024)


# =============================================================================
# 1. Data preparation
# =============================================================================
def load_and_preprocess():
    """Load the Beer data."""
    clean_path = os.path.join(DATA_DIR, 'clean.csv')
    df = pd.read_csv(clean_path)
    df = df[df['abv'].notna() & df['ibu'].notna()].reset_index(drop=True)
    df['is_ipa'] = df['style'].apply(
        lambda s: 1 if pd.notna(s) and 'ipa' in str(s).lower() else 0)
    print(f"  Loaded: {len(df)} rows, IPA={df['is_ipa'].sum()}")
    return df


def inject_errors(X_clean, y, semantic_rate=0.15, syntactic_rate=0.20, seed=2024):
    """
    Inject two error types (IBU column only).

    - Semantic error: replace with an existing in-column value (prefer near values).
    - Syntactic error: add Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    X_dirty = X_clean.copy()
    n = len(X_clean)
    error_list = []  # [(idx, col, type, true_value, dirty_value), ...]
    selected_rows = set()

    all_ibu = X_clean[:, 1]

    # Semantic errors
    n_semantic = int(n * semantic_rate)
    available = [i for i in range(n) if i not in selected_rows]
    if n_semantic > 0 and available:
        chosen = rng.choice(available, min(n_semantic, len(available)), replace=False)
        for idx in chosen:
            true_val = X_clean[idx, 1]
            candidates = all_ibu[all_ibu != true_val]
            if len(candidates) == 0:
                continue
            distances = np.abs(candidates - true_val)
            threshold = np.percentile(distances, 50)
            near = candidates[distances <= threshold]
            dirty_val = rng.choice(near) if len(near) > 0 else rng.choice(candidates)
            X_dirty[idx, 1] = dirty_val
            error_list.append((idx, 1, 'semantic', true_val, dirty_val))
            selected_rows.add(idx)

    # Syntactic errors
    n_syntactic = int(n * syntactic_rate)
    available = [i for i in range(n) if i not in selected_rows]
    if n_syntactic > 0 and available:
        chosen = rng.choice(available, min(n_syntactic, len(available)), replace=False)
        for idx in chosen:
            true_val = X_clean[idx, 1]
            noise = rng.normal(0, 25)
            dirty_val = max(0, true_val + noise)
            X_dirty[idx, 1] = dirty_val
            error_list.append((idx, 1, 'syntactic', true_val, dirty_val))
            selected_rows.add(idx)

    print(f"  Injected errors: semantic={sum(1 for e in error_list if e[2]=='semantic')}, "
          f"syntactic={sum(1 for e in error_list if e[2]=='syntactic')}")
    return X_dirty, error_list


# =============================================================================
# 2. Strategy execution
# =============================================================================
def apply_strategy(X_dirty, X_clean, y, error_list, actions, knn_imputer):
    """
    Apply a cleaning strategy.

    Actions:
      0 = no_action   (keep as-is)
      1 = repair_value (repair with ground truth)
      2 = delete       (drop the entire row)
      3 = replace_nearby (semantic error = truth; syntactic error = KNN estimate)
    """
    X_result = X_dirty.copy()
    keep_mask = np.ones(len(X_dirty), dtype=bool)

    # KNN precomputation
    X_knn = knn_imputer.transform(X_dirty)

    for i, (idx, col, err_type, true_val, dirty_val) in enumerate(error_list):
        action = actions[i]
        if action == 0:  # no_action
            pass
        elif action == 1:  # repair_value (ground truth)
            X_result[idx, col] = true_val
        elif action == 2:  # delete
            keep_mask[idx] = False
        elif action == 3:  # replace_nearby
            if err_type == 'semantic':
                X_result[idx, col] = true_val
            else:
                X_result[idx, col] = X_knn[idx, col]

    X_result = X_result[keep_mask]
    y_result = y[keep_mask]

    # Fill remaining NaN
    col_means = np.nanmean(X_result, axis=0) if len(X_result) > 0 else np.zeros(X_dirty.shape[1])
    for c in range(X_result.shape[1]):
        nan_mask = np.isnan(X_result[:, c])
        if nan_mask.any():
            X_result[nan_mask, c] = col_means[c]

    return X_result, y_result, keep_mask


# =============================================================================
# 3. Metric computation
# =============================================================================
def compute_veracity(X_result, X_clean, keep_mask, col=1, tolerance=0.01):
    """Veracity = correct values / total values."""
    if len(X_result) == 0:
        return 0.0
    X_clean_kept = X_clean[keep_mask]
    n = min(len(X_result), len(X_clean_kept))
    correct = 0
    for i in range(n):
        if abs(X_result[i, col] - X_clean_kept[i, col]) < tolerance:
            correct += 1
    return correct / n if n > 0 else 0.0


def compute_diversity(X_result, X_clean, keep_mask, col=1):
    """Diversity = sample retention rate x variance retention rate."""
    n_total = len(X_clean)
    n_kept = len(X_result)
    if n_kept <= 1 or n_total == 0:
        return 0.0

    sample_ret = n_kept / n_total

    X_clean_kept = X_clean[keep_mask]
    if len(X_clean_kept) <= 1:
        return 0.0

    result_var = np.var(X_result[:, col])
    clean_kept_var = np.var(X_clean_kept[:, col])

    if clean_kept_var > 1e-10:
        var_ret = np.clip(result_var / clean_kept_var, 0, 1.5)
    else:
        var_ret = 1.0

    return sample_ret * var_ret


def evaluate_performance(X_train, y_train, X_test, y_test):
    """Linear-SVM classification accuracy."""
    if len(X_train) < 5 or len(np.unique(y_train)) < 2:
        return 0.0
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


# =============================================================================
# 4. Sampling and experiment
# =============================================================================
def run_single_strategy(X_dirty, X_clean, y, error_list, actions, X_test, y_test,
                        knn_imputer, name='random'):
    """Run a single strategy and return a result dict."""
    X_result, y_result, keep_mask = apply_strategy(
        X_dirty, X_clean, y, error_list, actions, knn_imputer)

    if len(X_result) < 5:
        return None

    veracity = compute_veracity(X_result, X_clean, keep_mask)
    diversity = compute_diversity(X_result, X_clean, keep_mask)
    accuracy = evaluate_performance(X_result, y_result, X_test, y_test)

    # Tally action distribution
    unique, counts = np.unique(actions, return_counts=True)
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for u, c in zip(unique, counts):
        action_counts[int(u)] = int(c)

    return {
        'name': name,
        'veracity': veracity,
        'diversity': diversity,
        'accuracy': accuracy,
        'action_counts': action_counts,
        'n_samples': len(X_result),
    }


def generate_random_actions(error_list):
    """Uniform random actions."""
    return np.random.randint(0, 4, size=len(error_list))


def generate_extreme_actions(error_list, strategy):
    """Generate actions for an extreme strategy."""
    n = len(error_list)
    if strategy == 'NoFix':
        return np.zeros(n, dtype=int)
    elif strategy == 'FullFix':
        return np.ones(n, dtype=int)
    elif strategy == 'DeleteFix':
        return np.full(n, 2, dtype=int)
    elif strategy == 'RelaxFix':
        return np.full(n, 3, dtype=int)
    else:
        raise ValueError(f"unknown strategy: {strategy}")


def generate_uniform_bias_configs(step=0.01):
    """
    Generate all action-probability combinations summing to 1.

    Format: [p_noaction, p_repair, p_delete, p_nearby]
    Constraint: p_nearby (replace_nearby) in [0.90, 1.00]
    """
    configs = []
    values = np.arange(0, 1.0 + step / 2, step)
    values = np.round(values, 4)

    for p1 in values:           # no_action
        for p2 in values:       # repair_value
            for p3 in values:   # delete
                p4 = round(1.0 - p1 - p2 - p3, 4)
                if p4 < -0.001 or p4 > 1.001:
                    continue
                p4 = np.clip(p4, 0, 1)
                configs.append([p1, p2, p3, p4])

    print(f"  Probability combinations: {len(configs)}")
    return configs


def generate_biased_random_actions(error_list, bias):
    """Generate actions according to a probability bias."""
    n = len(error_list)
    bias = np.array(bias)
    bias = bias / bias.sum()  # normalize
    return np.random.choice([0, 1, 2, 3], size=n, p=bias)


def sample_search_space(X_dirty, X_clean, y, error_list, X_test, y_test,
                        n_samples=1000, save_interval=200, run_id=0):
    """
    Sample the search space.

    1. Run 4 extreme strategies
    2. Sweep probability combinations for biased random sampling
    """
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(X_dirty)

    results = []

    # Extreme strategies
    print("  Running extreme strategies...", flush=True)
    for strategy in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
        actions = generate_extreme_actions(error_list, strategy)
        r = run_single_strategy(X_dirty, X_clean, y, error_list, actions,
                                X_test, y_test, knn_imputer, name=strategy)
        if r:
            results.append(r)
            print(f"    {strategy}: V={r['veracity']:.4f}, D={r['diversity']:.4f}, "
                  f"A={r['accuracy']:.4f}")

    # Biased sampling
    print("  Generating probability combinations...", flush=True)
    configs = generate_uniform_bias_configs(step=0.01)
    n_per_config = max(1, n_samples // len(configs))
    total = len(configs) * n_per_config
    print(f"  Sampling: {len(configs)} combos x {n_per_config} runs = {total} strategies", flush=True)

    count = 0
    for config in tqdm(configs, desc='  sampling'):
        for _ in range(n_per_config):
            actions = generate_biased_random_actions(error_list, config)
            r = run_single_strategy(X_dirty, X_clean, y, error_list, actions,
                                    X_test, y_test, knn_imputer, name='Random')
            if r:
                results.append(r)
            count += 1

            if count % save_interval == 0:
                _save_intermediate_results(results, run_id)

    return results


def _save_intermediate_results(results, run_id):
    """Save intermediate results to CSV."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, 'search_space_results.csv')

    rows = []
    for r in results:
        rows.append({
            'name': r['name'],
            'veracity': r['veracity'],
            'diversity': r['diversity'],
            'accuracy': r['accuracy'],
            'n_samples': r['n_samples'],
            'run_id': run_id,
            'no_action': r['action_counts'].get(0, 0),
            'repair_value': r['action_counts'].get(1, 0),
            'delete': r['action_counts'].get(2, 0),
            'replace_nearby': r['action_counts'].get(3, 0),
        })

    new_df = pd.DataFrame(rows)

    # Incremental save: keep old data, append new Random, refresh extreme points
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        # Keep old Random records
        old_random = old_df[~old_df['name'].isin(EXTREME_NAMES)]
        # New extreme points + new Random
        new_extreme = new_df[new_df['name'].isin(EXTREME_NAMES)]
        new_random = new_df[~new_df['name'].isin(EXTREME_NAMES)]
        combined = pd.concat([new_extreme, old_random, new_random], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)


# =============================================================================
# 5. DQN training and inference (via the released DemandClean API)
# =============================================================================
def run_dqn_training_and_inference(X_dirty_train, X_clean_train, y_train,
                                   X_dirty_test, X_clean_test, y_test,
                                   error_list_test, X_test_split, y_test_split,
                                   knn_imputer, n_runs=1, load_model=True):
    """
    Train and run DQN inference via the released DemandClean API.

    Replaces the original TensorFlow implementation:
    - old: from demand_clean_dqn import DemandCleanDQNAgent, DemandCleanEnv
    - new: from demandclean.api.demand_clean import DemandClean

    Phase 1: train or load model
    Phase 2: multi-run inference, recording (veracity, diversity, accuracy, action_counts) per run
    """
    model_path = os.path.join(MODEL_DIR, 'search_space_dqn.pt')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Build DataFrame for the DemandClean API
    dirty_csv_path = os.path.join(RESULT_DIR, 'dirty_data.csv')
    clean_csv_path = os.path.join(DATA_DIR, 'clean.csv')

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single',
        detector_mode='oracle',
        inference_mode='single_phase',
        n_episodes=500,
        min_truth_budget=10,
        max_truth_budget=200,
        column_names=['abv', 'ibu'],
        label_col='is_ipa',
        save_path=RESULT_DIR,
        dirty_csv_path=dirty_csv_path,
        clean_csv_path=clean_csv_path,
        csv_columns=['abv', 'ibu', 'is_ipa'],
    )

    # Phase 1: train or load
    if load_model and os.path.exists(model_path):
        print(f"  [DQN] loading model: {model_path}", flush=True)
        dc.load(model_path)
    else:
        print(f"  [DQN] training model (episodes=500)...", flush=True)
        dc.fit(X_dirty_train, y_train, X_clean=X_clean_train, y_clean=y_train)
        dc.save(model_path)
        print(f"  [DQN] model saved: {model_path}", flush=True)

    # Phase 2: multi-run inference
    dqn_results = []
    for run in range(n_runs):
        print(f"  [DQN] inference run {run+1}/{n_runs}...", flush=True)

        X_result, y_result, stats = dc.clean(
            X_dirty_test, y_test, X_clean_test)

        keep_mask = stats.get('keep_mask', np.ones(len(X_dirty_test), dtype=bool))
        action_counts = stats.get('action_counts', {0: 0, 1: 0, 2: 0, 3: 0})

        veracity = compute_veracity(X_result, X_clean_test, keep_mask)
        diversity = compute_diversity(X_result, X_clean_test, keep_mask)
        accuracy = evaluate_performance(X_result, y_result, X_test_split, y_test_split)

        result = {
            'name': 'DemandFix_DQN',
            'veracity': veracity,
            'diversity': diversity,
            'accuracy': accuracy,
            'action_counts': action_counts,
            'n_samples': len(X_result),
        }
        dqn_results.append(result)
        print(f"    V={veracity:.4f}, D={diversity:.4f}, A={accuracy:.4f}, "
              f"actions={action_counts}")

    return dqn_results


# =============================================================================
# 6. Visualization (original logic preserved; only save paths updated)
# =============================================================================
def _compute_alpha_shape(x_arr, y_arr, alpha_val=2.0, max_points=5000):
    """Compute alpha-shape boundary (auto-subsample for large datasets)."""
    if not HAS_ALPHASHAPE:
        return None
    try:
        points = np.column_stack([x_arr, y_arr])
        # Subsample (alphashape doesn't need all points to define the boundary)
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        return alphashape.alphashape(points, alpha=alpha_val)
    except Exception as e:
        print(f"  [WARN] alpha shape failed: {e}")
        return None


MAX_SCATTER_POINTS = 30000  # max scatter points (440K -> 30K; visually unchanged)


def _subsample_results(random_results, max_n=MAX_SCATTER_POINTS):
    """Subsample random-strategy results to speed up visualization."""
    if len(random_results) <= max_n:
        return random_results
    idx = np.random.choice(len(random_results), max_n, replace=False)
    return [random_results[i] for i in idx]


def _mask_outside_shape(shape, XX, YY, AA):
    """Set grid cells outside the alpha shape to NaN (vectorized via shapely.vectorized)."""
    if shape is None:
        return AA
    try:
        from shapely import vectorized as sv
        mask = sv.contains(shape, XX, YY)
        AA = AA.copy()
        AA[~mask] = np.nan
        return AA
    except ImportError:
        # Fallback to prepared geometry
        try:
            from shapely.prepared import prep
            prepared_shape = prep(shape)
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    if not prepared_shape.contains(Point(XX[i, j], YY[i, j])):
                        AA[i, j] = np.nan
        except Exception:
            pass
        return AA


def set_academic_style():
    """Academic paper figure style."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })


def plot_search_space_scatter(results, dqn_results, output_path):
    """
    Search-space scatter (two subplots).

    Left: search-space distribution + extreme points + DemandFix
    Right: performance heatmap scatter
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # Subsample to speed up plotting (440K -> 30K)
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    a_rand = [r['accuracy'] for r in plot_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: search space ---
    ax1.scatter(v_rand, d_rand, c='lightgray', s=3, alpha=0.4, label='Random strategies')

    # DQN results
    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        ax1.scatter(v_dqn, d_dqn, c='orange', s=60, marker='*',
                    zorder=5, label='DemandFix (DQN)')

    # Extreme points
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax1.scatter(r['veracity'], r['diversity'], c=color, s=150,
                        marker=extreme_markers[name], zorder=10, edgecolors='black',
                        linewidths=1.5, label=name)

    # Boundary lines
    boundary_order = ['NoFix', 'RelaxFix', 'FullFix', 'DeleteFix', 'NoFix']
    for i in range(len(boundary_order) - 1):
        n1, n2 = boundary_order[i], boundary_order[i + 1]
        if n1 in extreme_map and n2 in extreme_map:
            ax1.plot([extreme_map[n1]['veracity'], extreme_map[n2]['veracity']],
                     [extreme_map[n1]['diversity'], extreme_map[n2]['diversity']],
                     'k--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Authenticity (Veracity)')
    ax1.set_ylabel('Diversity')
    ax1.set_title('Search Space Distribution')
    ax1.legend(loc='upper left', fontsize=9)

    # --- Right: performance heatmap ---
    if len(a_rand) > 0:
        sc = ax2.scatter(v_rand, d_rand, c=a_rand, cmap='RdYlBu_r', s=5, alpha=0.6,
                         norm=PowerNorm(gamma=0.5, vmin=min(a_rand), vmax=max(a_rand)))
        plt.colorbar(sc, ax=ax2, label='Classification Accuracy')

    # Extreme points (outline)
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax2.scatter(r['veracity'], r['diversity'], facecolors='none',
                        edgecolors=color, s=150, marker=extreme_markers[name],
                        zorder=10, linewidths=2, label=name)

    # DQN
    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        ax2.scatter(v_dqn, d_dqn, c='gold', s=200, marker='*',
                    zorder=10, edgecolors='black', linewidths=1, label='DemandFix')

    ax2.set_xlabel('Authenticity (Veracity)')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Performance Heatmap')
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_search_space_grid_heatmap(results, output_path, dqn_results=None, grid_size=80):
    """
    Grid heatmap.

    Uses alphashape to clip grid cells outside the crescent region.
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    v = np.array([r['veracity'] for r in random_results])
    d = np.array([r['diversity'] for r in random_results])
    a = np.array([r['accuracy'] for r in random_results])

    if len(v) < 10:
        print(f"  [skip] too few data points: {len(v)}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Grid
    v_grid = np.linspace(v.min(), v.max(), grid_size)
    d_grid = np.linspace(d.min(), d.max(), grid_size)
    VV, DD = np.meshgrid(v_grid, d_grid)

    # Interpolation
    AA = griddata((v, d), a, (VV, DD), method='linear')

    # Alpha-shape clipping
    shape = _compute_alpha_shape(v, d)
    AA = _mask_outside_shape(shape, VV, DD, AA)

    # Plot
    mesh = ax.pcolormesh(VV, DD, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # Contours
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(VV, DD, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Extreme points
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax.scatter(r['veracity'], r['diversity'], c=color, s=200,
                       marker=extreme_markers[name], zorder=10, edgecolors='black',
                       linewidths=2, label=name)

    # DQN
    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        ax.scatter(v_dqn, d_dqn, c='gold', s=300, marker='*',
                   zorder=10, edgecolors='black', linewidths=1.5, label='DemandFix')

    ax.set_xlabel('Authenticity (Veracity)')
    ax.set_ylabel('Diversity')
    ax.set_title('Search Space Performance Heatmap')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_combined_figure(results, dqn_results, output_path):
    """
    Combined figure (three subplots).

    1: search-space boundary
    2: performance heatmap
    3: DemandFix distribution
    """
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # Subsample for faster plotting
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    a_rand = [r['accuracy'] for r in plot_results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}

    # --- Plot 1: search-space boundary ---
    ax1.scatter(v_rand, d_rand, c='lightgray', s=2, alpha=0.3)

    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax1.scatter(r['veracity'], r['diversity'], c=color, s=150,
                        marker=extreme_markers[name], zorder=10,
                        edgecolors='black', linewidths=1.5, label=name)

    boundary_order = ['NoFix', 'RelaxFix', 'FullFix', 'DeleteFix', 'NoFix']
    for i in range(len(boundary_order) - 1):
        n1, n2 = boundary_order[i], boundary_order[i + 1]
        if n1 in extreme_map and n2 in extreme_map:
            ax1.plot([extreme_map[n1]['veracity'], extreme_map[n2]['veracity']],
                     [extreme_map[n1]['diversity'], extreme_map[n2]['diversity']],
                     'k--', alpha=0.4, linewidth=1.5)

    ax1.set_xlabel('Authenticity')
    ax1.set_ylabel('Diversity')
    ax1.set_title('(a) Search Space Boundary')
    ax1.legend(fontsize=9)

    # --- Plot 2: performance heatmap ---
    if len(a_rand) > 0:
        # Percentile normalization (vectorized)
        a_arr = np.array(a_rand)
        sorted_a = np.sort(a_arr)
        pct = np.searchsorted(sorted_a, a_arr) / len(a_arr)
        sc = ax2.scatter(v_rand, d_rand, c=pct, cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=ax2, label='Performance Percentile')

    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax2.scatter(r['veracity'], r['diversity'], facecolors='none',
                        edgecolors=color, s=150, marker=extreme_markers[name],
                        zorder=10, linewidths=2, label=name)

    ax2.set_xlabel('Authenticity')
    ax2.set_ylabel('Diversity')
    ax2.set_title('(b) Performance Distribution')
    ax2.legend(fontsize=9)

    # --- Plot 3: DemandFix distribution ---
    ax3.scatter(v_rand, d_rand, c='lightgray', s=2, alpha=0.2)

    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        a_dqn = [r['accuracy'] for r in dqn_results]

        sc3 = ax3.scatter(v_dqn, d_dqn, c=a_dqn, cmap='Oranges', s=80,
                          edgecolors='black', linewidths=0.5, zorder=5)
        plt.colorbar(sc3, ax=ax3, label='DQN Accuracy')

        # Mean
        mean_v = np.mean(v_dqn)
        mean_d = np.mean(d_dqn)
        ax3.scatter(mean_v, mean_d, c='red', s=200, marker='*',
                    zorder=10, edgecolors='black', linewidths=1.5,
                    label=f'Mean ({mean_v:.3f}, {mean_d:.3f})')

    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax3.scatter(r['veracity'], r['diversity'], facecolors='none',
                        edgecolors=color, s=150, marker=extreme_markers[name],
                        zorder=10, linewidths=2)

    ax3.set_xlabel('Authenticity')
    ax3.set_ylabel('Diversity')
    ax3.set_title('(c) DemandFix Distribution')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_action_pair_heatmap(results, action_x, action_y, output_path,
                             dqn_results=None, grid_size=50):
    """
    Action-combination heatmap.

    X/Y axes are two action ratios; color is performance.
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    action_names = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
    action_short = {0: 'noact', 1: 'repair', 2: 'delete', 3: 'nearby'}

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    if not random_results:
        return

    # Compute ratios
    x_ratios = []
    y_ratios = []
    accuracies = []
    for r in random_results:
        total = sum(r['action_counts'].values())
        if total == 0:
            continue
        x_ratios.append(r['action_counts'].get(action_x, 0) / total)
        y_ratios.append(r['action_counts'].get(action_y, 0) / total)
        accuracies.append(r['accuracy'])

    x_arr = np.array(x_ratios)
    y_arr = np.array(y_ratios)
    a_arr = np.array(accuracies)

    if len(x_arr) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Grid interpolation
    x_grid = np.linspace(x_arr.min(), x_arr.max(), grid_size)
    y_grid = np.linspace(y_arr.min(), y_arr.max(), grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    AA = griddata((x_arr, y_arr), a_arr, (XX, YY), method='linear')

    # Alpha-shape clipping
    shape = _compute_alpha_shape(x_arr, y_arr)
    AA = _mask_outside_shape(shape, XX, YY, AA)

    mesh = ax.pcolormesh(XX, YY, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # Contours
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(XX, YY, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Extreme points
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            total = sum(r['action_counts'].values())
            if total > 0:
                ex = r['action_counts'].get(action_x, 0) / total
                ey = r['action_counts'].get(action_y, 0) / total
                ax.scatter(ex, ey, c=color, s=200, marker=extreme_markers[name],
                           zorder=10, edgecolors='black', linewidths=2, label=name)

    # DQN
    if dqn_results:
        for r in dqn_results:
            total = sum(r['action_counts'].values())
            if total > 0:
                dx = r['action_counts'].get(action_x, 0) / total
                dy = r['action_counts'].get(action_y, 0) / total
                ax.scatter(dx, dy, c='gold', s=300, marker='*',
                           zorder=10, edgecolors='black', linewidths=1.5)

    ax.set_xlabel(f'{action_names[action_x]} ratio')
    ax.set_ylabel(f'{action_names[action_y]} ratio')
    ax.set_title(f'{action_short[action_x]} vs {action_short[action_y]}')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_all_action_pair_heatmaps(results, output_dir, dqn_results=None):
    """Generate heatmaps for all pairs of actions (C(4,2)=6)."""
    os.makedirs(output_dir, exist_ok=True)
    action_short = {0: 'noact', 1: 'repair', 2: 'delete', 3: 'nearby'}

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for ax, ay in pairs:
        filename = f'{action_short[ax]}_vs_{action_short[ay]}.png'
        output_path = os.path.join(output_dir, filename)
        plot_action_pair_heatmap(results, ax, ay, output_path, dqn_results)


def plot_repair_vs_nonrepair_heatmap(results, output_path, dqn_results=None, grid_size=50):
    """
    Repair vs Non-repair heatmap.

    X = repair_value ratio
    Y = (delete + replace_nearby) ratio
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    if not random_results:
        return

    repair_ratios = []
    nonrepair_ratios = []
    accuracies = []
    for r in random_results:
        total = sum(r['action_counts'].values())
        if total == 0:
            continue
        repair = r['action_counts'].get(1, 0) / total
        nonrepair = (r['action_counts'].get(2, 0) + r['action_counts'].get(3, 0)) / total
        repair_ratios.append(repair)
        nonrepair_ratios.append(nonrepair)
        accuracies.append(r['accuracy'])

    x = np.array(repair_ratios)
    y = np.array(nonrepair_ratios)
    a = np.array(accuracies)

    if len(x) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Grid interpolation
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    AA = griddata((x, y), a, (XX, YY), method='linear')

    # Alpha-shape clipping
    shape = _compute_alpha_shape(x, y)
    AA = _mask_outside_shape(shape, XX, YY, AA)

    mesh = ax.pcolormesh(XX, YY, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # Contours
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(XX, YY, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Extreme points
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            total = sum(r['action_counts'].values())
            if total > 0:
                ex = r['action_counts'].get(1, 0) / total
                ey = (r['action_counts'].get(2, 0) + r['action_counts'].get(3, 0)) / total
                ax.scatter(ex, ey, c=color, s=200, marker=extreme_markers[name],
                           zorder=10, edgecolors='black', linewidths=2, label=name)

    # DQN
    if dqn_results:
        for r in dqn_results:
            total = sum(r['action_counts'].values())
            if total > 0:
                dx = r['action_counts'].get(1, 0) / total
                dy = (r['action_counts'].get(2, 0) + r['action_counts'].get(3, 0)) / total
                ax.scatter(dx, dy, c='gold', s=300, marker='*',
                           zorder=10, edgecolors='black', linewidths=1.5, label='DemandFix')

    ax.set_xlabel('repair_value ratio')
    ax.set_ylabel('(delete + replace_nearby) ratio')
    ax.set_title('Repair vs Non-repair Actions')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_cost_heatmap(results, output_path, dqn_results=None, grid_size=80):
    """
    Cost heatmap.

    X = Authenticity, Y = Diversity, color = repair_value count (repair cost).
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    v = np.array([r['veracity'] for r in random_results])
    d = np.array([r['diversity'] for r in random_results])
    costs = np.array([r['action_counts'].get(1, 0) for r in random_results])

    if len(v) < 10:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Grid interpolation
    v_grid = np.linspace(v.min(), v.max(), grid_size)
    d_grid = np.linspace(d.min(), d.max(), grid_size)
    VV, DD = np.meshgrid(v_grid, d_grid)
    CC = griddata((v, d), costs, (VV, DD), method='linear')

    # Alpha-shape clipping
    shape = _compute_alpha_shape(v, d)
    CC = _mask_outside_shape(shape, VV, DD, CC)

    mesh = ax.pcolormesh(VV, DD, CC, cmap='YlOrRd', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Truth Value Cost (repair_value count)')

    # Contours (power-law spaced)
    valid_costs = CC[~np.isnan(CC)]
    if len(valid_costs) > 0:
        c_min, c_max = valid_costs.min(), valid_costs.max()
        if c_max > c_min:
            n_levels = 8
            levels = c_min + (c_max - c_min) * np.power(
                np.linspace(0, 1, n_levels), 2)
            ax.contour(VV, DD, CC, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Extreme points
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax.scatter(r['veracity'], r['diversity'], c=color, s=200,
                       marker=extreme_markers[name], zorder=10, edgecolors='black',
                       linewidths=2, label=name)

    # DQN
    if dqn_results:
        for r in dqn_results:
            ax.scatter(r['veracity'], r['diversity'], c='gold', s=300,
                       marker='*', zorder=10, edgecolors='black',
                       linewidths=1.5, label='DemandFix')

    ax.set_xlabel('Authenticity (Veracity)')
    ax.set_ylabel('Diversity')
    ax.set_title('Truth Value Cost in Search Space')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


def plot_cost_scatter(results, output_path, dqn_results=None):
    """
    Cost scatter (two subplots).

    Left: search-space distribution
    Right: scatter colored by cost
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # Subsample for faster plotting
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    costs = [r['action_counts'].get(1, 0) for r in plot_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}

    # --- Left: distribution ---
    ax1.scatter(v_rand, d_rand, c='lightgray', s=3, alpha=0.4)
    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        ax1.scatter(v_dqn, d_dqn, c='orange', s=60, marker='*', zorder=5, label='DemandFix')

    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax1.scatter(r['veracity'], r['diversity'], c=color, s=150,
                        marker=extreme_markers[name], zorder=10, edgecolors='black',
                        linewidths=1.5, label=name)

    ax1.set_xlabel('Authenticity (Veracity)')
    ax1.set_ylabel('Diversity')
    ax1.set_title('Search Space Distribution')
    ax1.legend(loc='upper left', fontsize=9)

    # --- Right: cost heatmap scatter ---
    if len(costs) > 0:
        sc = ax2.scatter(v_rand, d_rand, c=costs, cmap='YlOrRd', s=5, alpha=0.6,
                         norm=PowerNorm(gamma=0.5))
        plt.colorbar(sc, ax=ax2, label='Truth Value Cost')

    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax2.scatter(r['veracity'], r['diversity'], facecolors='none',
                        edgecolors=color, s=150, marker=extreme_markers[name],
                        zorder=10, linewidths=2, label=name)

    if dqn_results:
        for r in dqn_results:
            ax2.scatter(r['veracity'], r['diversity'], c='gold', s=200,
                        marker='*', zorder=10, edgecolors='black',
                        linewidths=1, label='DemandFix')

    ax2.set_xlabel('Authenticity (Veracity)')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Cost Distribution')
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [fig] {os.path.basename(output_path)}")


# =============================================================================
# 7. Main entry point
# =============================================================================
def main():
    print("=" * 70)
    print("Search-space visualization experiment")
    print("=" * 70)
    print(f"  RUN_EXPERIMENT={RUN_EXPERIMENT}, LOAD_MODEL={LOAD_MODEL}")
    print(f"  N_RANDOM_SAMPLES={N_RANDOM_SAMPLES}, N_DQN_RUNS={N_DQN_RUNS}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # ------------------------------------------------------------------
    # [1] Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading data")
    clean_df = load_and_preprocess()
    X_clean = clean_df[['abv', 'ibu']].values.astype(float)
    y = clean_df['is_ipa'].values

    # Standardize
    scaler = StandardScaler()
    X_clean_scaled = scaler.fit_transform(X_clean)

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(X_clean_scaled)), test_size=0.3,
        random_state=42, stratify=y)
    X_test, y_test = X_clean_scaled[test_idx], y[test_idx]

    # ------------------------------------------------------------------
    # [2] Inject test-time errors
    # ------------------------------------------------------------------
    print("\n[2] Injecting test-time errors (seed=2024)")
    X_dirty_test_scaled, error_list_test = inject_errors(
        X_clean_scaled, y, SEMANTIC_RATE, SYNTACTIC_RATE, seed=2024)

    # Save dirty data CSV
    dirty_csv_path = os.path.join(RESULT_DIR, 'dirty_data.csv')
    dirty_df = pd.DataFrame({
        'abv': X_dirty_test_scaled[:, 0],
        'ibu': X_dirty_test_scaled[:, 1],
        'is_ipa': y,
        'has_error': [1 if any(e[0] == i for e in error_list_test) else 0
                      for i in range(len(y))],
        'error_type': [''] * len(y),
        'true_ibu': X_clean_scaled[:, 1],
    })
    for e in error_list_test:
        dirty_df.loc[e[0], 'error_type'] = e[2]
    dirty_df.to_csv(dirty_csv_path, index=False)
    print(f"  Dirty data saved: {dirty_csv_path}")

    results = None
    dqn_results = None
    extreme_points = None

    if RUN_EXPERIMENT:
        # ------------------------------------------------------------------
        # [3] Random-sample the strategy space
        # ------------------------------------------------------------------
        print("\n[3] Random sampling of strategy space")
        results = sample_search_space(
            X_dirty_test_scaled, X_clean_scaled, y, error_list_test,
            X_test, y_test, n_samples=N_RANDOM_SAMPLES)

        # ------------------------------------------------------------------
        # [4] DQN training and inference
        # ------------------------------------------------------------------
        print("\n[4] DQN training and inference")
        # Inject training errors (different seed)
        X_dirty_train_scaled, error_list_train = inject_errors(
            X_clean_scaled, y, SEMANTIC_RATE, SYNTACTIC_RATE, seed=42)

        knn_imputer = KNNImputer(n_neighbors=5)
        knn_imputer.fit(X_dirty_test_scaled)

        dqn_results = run_dqn_training_and_inference(
            X_dirty_train_scaled, X_clean_scaled, y,
            X_dirty_test_scaled, X_clean_scaled, y,
            error_list_test, X_test, y_test,
            knn_imputer, n_runs=N_DQN_RUNS, load_model=LOAD_MODEL)

        # ------------------------------------------------------------------
        # [5] Save results
        # ------------------------------------------------------------------
        print("\n[5] Saving results")

        # Save sampling results CSV
        csv_path = os.path.join(RESULT_DIR, 'search_space_results.csv')
        rows = []
        for r in results:
            rows.append({
                'name': r['name'],
                'veracity': r['veracity'],
                'diversity': r['diversity'],
                'accuracy': r['accuracy'],
                'n_samples': r['n_samples'],
                'run_id': 0,
                'no_action': r['action_counts'].get(0, 0),
                'repair_value': r['action_counts'].get(1, 0),
                'delete': r['action_counts'].get(2, 0),
                'replace_nearby': r['action_counts'].get(3, 0),
            })

        # Incremental save
        new_df = pd.DataFrame(rows)
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            old_random = old_df[~old_df['name'].isin(EXTREME_NAMES)]
            new_extreme = new_df[new_df['name'].isin(EXTREME_NAMES)]
            new_random = new_df[~new_df['name'].isin(EXTREME_NAMES)]
            combined_df = pd.concat([new_extreme, old_random, new_random], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(csv_path, index=False)
        print(f"  Sampling results: {csv_path} ({len(combined_df)} rows)")

        # Rebuild results from combined_df (for plotting)
        results = []
        for _, row in combined_df.iterrows():
            results.append({
                'name': row['name'],
                'veracity': row['veracity'],
                'diversity': row['diversity'],
                'accuracy': row['accuracy'],
                'n_samples': int(row['n_samples']),
                'action_counts': {
                    0: int(row['no_action']),
                    1: int(row['repair_value']),
                    2: int(row['delete']),
                    3: int(row['replace_nearby']),
                },
            })

        # Save DQN results
        dqn_csv_path = os.path.join(RESULT_DIR, 'dqn_distribution.csv')
        dqn_rows = []
        for r in dqn_results:
            dqn_rows.append({
                'name': r['name'],
                'veracity': r['veracity'],
                'diversity': r['diversity'],
                'accuracy': r['accuracy'],
                'n_samples': r['n_samples'],
                'no_action': r['action_counts'].get(0, 0),
                'repair_value': r['action_counts'].get(1, 0),
                'delete': r['action_counts'].get(2, 0),
                'replace_nearby': r['action_counts'].get(3, 0),
            })
        pd.DataFrame(dqn_rows).to_csv(dqn_csv_path, index=False)
        print(f"  DQN results: {dqn_csv_path}")

        # Save extreme points
        extreme_points = {}
        for r in results:
            if r['name'] in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
                extreme_points[r['name']] = {
                    'veracity': r['veracity'],
                    'diversity': r['diversity'],
                    'accuracy': r['accuracy'],
                    'action_counts': r['action_counts'],
                }
        ep_path = os.path.join(RESULT_DIR, 'extreme_points.json')
        with open(ep_path, 'w') as f:
            json.dump(extreme_points, f, indent=2)
        print(f"  Extreme points: {ep_path}")

    else:
        # ------------------------------------------------------------------
        # [3] Load existing results
        # ------------------------------------------------------------------
        print("\n[3] Loading existing results")

        csv_path = os.path.join(RESULT_DIR, 'search_space_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results = []
            for _, row in df.iterrows():
                results.append({
                    'name': row['name'],
                    'veracity': row['veracity'],
                    'diversity': row['diversity'],
                    'accuracy': row['accuracy'],
                    'n_samples': int(row['n_samples']),
                    'action_counts': {
                        0: int(row['no_action']),
                        1: int(row['repair_value']),
                        2: int(row['delete']),
                        3: int(row['replace_nearby']),
                    },
                })
            print(f"  Sampling results: {len(results)} rows")
        else:
            print(f"  [ERROR] {csv_path} not found; set RUN_EXPERIMENT=True to run the experiment")
            return None, None

        dqn_csv_path = os.path.join(RESULT_DIR, 'dqn_distribution.csv')
        if os.path.exists(dqn_csv_path):
            dqn_df = pd.read_csv(dqn_csv_path)
            dqn_results = []
            for _, row in dqn_df.iterrows():
                dqn_results.append({
                    'name': row['name'],
                    'veracity': row['veracity'],
                    'diversity': row['diversity'],
                    'accuracy': row['accuracy'],
                    'n_samples': int(row['n_samples']),
                    'action_counts': {
                        0: int(row['no_action']),
                        1: int(row['repair_value']),
                        2: int(row['delete']),
                        3: int(row['replace_nearby']),
                    },
                })
            print(f"  DQN results: {len(dqn_results)} rows")
        else:
            print(f"  [WARN] {dqn_csv_path} not found; DQN points will not be shown")
            dqn_results = []

        ep_path = os.path.join(RESULT_DIR, 'extreme_points.json')
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                extreme_points = json.load(f)
            print(f"  Extreme points: {list(extreme_points.keys())}")

    # ------------------------------------------------------------------
    # [6] Generate visualizations
    # ------------------------------------------------------------------
    print("\n[6] Generating visualizations")

    import time as _time

    _t0 = _time.time()
    plot_search_space_scatter(
        results, dqn_results,
        os.path.join(RESULT_DIR, 'search_space_scatter.png'))
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_search_space_grid_heatmap(
        results,
        os.path.join(RESULT_DIR, 'search_space_heatmap.png'),
        dqn_results=dqn_results)
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_combined_figure(
        results, dqn_results,
        os.path.join(RESULT_DIR, 'search_space_combined.png'))
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_cost_heatmap(
        results,
        os.path.join(RESULT_DIR, 'search_space_cost_heatmap.png'),
        dqn_results=dqn_results)
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_cost_scatter(
        results,
        os.path.join(RESULT_DIR, 'search_space_cost_scatter.png'),
        dqn_results=dqn_results)
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_repair_vs_nonrepair_heatmap(
        results,
        os.path.join(RESULT_DIR, 'repair_vs_nonrepair.png'),
        dqn_results=dqn_results)
    print(f"  elapsed: {_time.time()-_t0:.1f}s")

    # Action-pair heatmaps (optional; disabled by default to save time)
    # plot_all_action_pair_heatmaps(
    #     results,
    #     os.path.join(RESULT_DIR, 'action_pairs'),
    #     dqn_results=dqn_results)

    # ------------------------------------------------------------------
    # [7] Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Extreme strategies
    print(f"\n{'Strategy':<15} {'Veracity':>10} {'Diversity':>10} {'Accuracy':>10}")
    print("-" * 50)
    for r in results:
        if r['name'] in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
            print(f"{r['name']:<15} {r['veracity']:>10.4f} {r['diversity']:>10.4f} "
                  f"{r['accuracy']:>10.4f}")

    # DemandFix DQN
    if dqn_results:
        mean_v = np.mean([r['veracity'] for r in dqn_results])
        mean_d = np.mean([r['diversity'] for r in dqn_results])
        mean_a = np.mean([r['accuracy'] for r in dqn_results])
        print(f"\nDemandFix (DQN) mean: V={mean_v:.4f}, D={mean_d:.4f}, A={mean_a:.4f}")

    print(f"\nTotal elapsed: {time.time() - total_start:.2f}s")
    return results, dqn_results


if __name__ == '__main__':
    results, dqn_results = main()
