#!/usr/bin/env python3
"""
Ablation study on the Beer (IPA) dataset
========================================

Rewritten from the original real_beers_experiment_with_detector.py. The key
change is to replace the legacy 12 DQN strategy functions with the release
``demandclean/`` API, removing the TensorFlow dependency.

Data split: 60/20/20 (train/val/test). The Scaler is fit only on the dirty
training set.

Strategies:
- Baseline strategies: NoFix, DeleteAll, DeleteFix, ReplaceAll, FullFix, DemandFix
- DQN strategies (via the DemandClean API):
  - Supervised (oracle): DQN_Single, DQN_TwoStage
  - Semi-supervised (oracle): SemiSup_Single, SemiSup_TwoStage,
                               SemiSup_Dueling_Single, SemiSup_Dueling_TwoStage
  - Fully unsupervised (auto):  FullUnsup_Single, FullUnsup_TwoStage,
                                 FullUnsup_Single_2P, FullUnsup_TwoStage_2P,
                                 FullUnsup_Dueling_Single, FullUnsup_Dueling_TwoStage
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets/beers')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

# Import DemandClean API
from demandclean.api.demand_clean import DemandClean
from demandclean.core.environments.value_estimation import ValueEstimator
from demandclean.config import DemandCleanConfig

np.random.seed(2024)

# Truth-value budget configuration
MIN_TRUTH_BUDGET = 10
MAX_TRUTH_BUDGET = 300  # Comparable to DemandFix; falls back to replace_nearby when exceeded


# =============================================================================
# 1. Error injector (original logic preserved)
# =============================================================================
class ErrorInjector:
    """Error injector that injects three error categories."""

    def __init__(self, df):
        self.df = df.copy()
        self.df['ABV_dirty'] = df['abv'].copy()
        self.df['IBU_dirty'] = df['ibu'].copy()
        self.errors = {'missing': [], 'semantic': [], 'syntactic': []}
        self.used_rows = set()

        # Partition into boundary / non-boundary regions
        boundary_mask = (df['ibu'] >= 35) & (df['ibu'] <= 65)
        self.boundary_idx = df[boundary_mask].index.tolist()
        self.non_boundary_idx = df[~boundary_mask].index.tolist()
        self.all_ibu_values = df['ibu'].values

    def inject_semantic(self, rate=0.15):
        """Inject semantic errors (replace with another valid value)."""
        n_boundary = int(len(self.boundary_idx) * rate * 1.5)
        n_non_boundary = int(len(self.non_boundary_idx) * rate * 0.3)

        # Boundary region
        available = [i for i in self.boundary_idx if i not in self.used_rows]
        if n_boundary > 0 and available:
            chosen = np.random.choice(available, min(n_boundary, len(available)), replace=False)
            for idx in chosen:
                original = self.df.loc[idx, 'ibu']
                candidates = self.all_ibu_values[self.all_ibu_values != original]
                distances = np.abs(candidates - original)
                near = candidates[distances < np.percentile(distances, 30)]
                new_val = np.random.choice(near) if len(near) > 0 else np.random.choice(candidates)
                self.df.loc[idx, 'IBU_dirty'] = new_val
                self.errors['semantic'].append((idx, 'IBU', original, new_val))
                self.used_rows.add(idx)

        # Non-boundary region
        available = [i for i in self.non_boundary_idx if i not in self.used_rows]
        if n_non_boundary > 0 and available:
            chosen = np.random.choice(available, min(n_non_boundary, len(available)), replace=False)
            for idx in chosen:
                original = self.df.loc[idx, 'ibu']
                candidates = self.all_ibu_values[self.all_ibu_values != original]
                new_val = np.random.choice(candidates) if len(candidates) > 0 else original
                self.df.loc[idx, 'IBU_dirty'] = new_val
                self.errors['semantic'].append((idx, 'IBU', original, new_val))
                self.used_rows.add(idx)

        return self

    def inject_syntactic(self, rate=0.25):
        """Inject syntactic errors (additive noise)."""
        n_boundary = int(len(self.boundary_idx) * rate * 2.0)
        n_non_boundary = int(len(self.non_boundary_idx) * rate * 0.5)

        # Boundary region (large noise)
        available = [i for i in self.boundary_idx if i not in self.used_rows]
        if n_boundary > 0 and available:
            chosen = np.random.choice(available, min(n_boundary, len(available)), replace=False)
            for idx in chosen:
                original = self.df.loc[idx, 'ibu']
                direction = 1 if np.random.random() > 0.5 else -1
                noise = direction * abs(np.random.normal(50, 12))
                self.df.loc[idx, 'IBU_dirty'] = max(0, original + noise)
                self.errors['syntactic'].append((idx, 'IBU', original, noise))
                self.used_rows.add(idx)

        # Non-boundary region (moderate noise)
        available = [i for i in self.non_boundary_idx if i not in self.used_rows]
        if n_non_boundary > 0 and available:
            chosen = np.random.choice(available, min(n_non_boundary, len(available)), replace=False)
            for idx in chosen:
                original = self.df.loc[idx, 'ibu']
                noise = np.random.normal(0, 30)
                self.df.loc[idx, 'IBU_dirty'] = max(0, original + noise)
                self.errors['syntactic'].append((idx, 'IBU', original, noise))
                self.used_rows.add(idx)

        return self

    def inject_missing(self, rate=0.05):
        """Inject missing values."""
        available = [i for i in self.non_boundary_idx if i not in self.used_rows]
        n_missing = int(len(available) * rate)

        if n_missing > 0 and available:
            chosen = np.random.choice(available, min(n_missing, len(available)), replace=False)
            for idx in chosen:
                original = self.df.loc[idx, 'ibu']
                self.df.loc[idx, 'IBU_dirty'] = np.nan
                self.errors['missing'].append((idx, 'IBU', original))
                self.used_rows.add(idx)

        return self

    def get_result(self):
        """Return the injection result."""
        return self.df, self.errors

    def print_stats(self):
        """Print injection statistics."""
        sem_boundary = sum(1 for e in self.errors['semantic'] if e[0] in self.boundary_idx)
        syn_boundary = sum(1 for e in self.errors['syntactic'] if e[0] in self.boundary_idx)
        print(f"\nError injection:")
        print(f"  Semantic errors: {len(self.errors['semantic'])} (boundary: {sem_boundary})")
        print(f"  Syntactic errors: {len(self.errors['syntactic'])} (boundary: {syn_boundary})")
        print(f"  Missing values: {len(self.errors['missing'])}")


# =============================================================================
# 2. Baseline cleaning strategies
# =============================================================================
def strategy_nofix(X, y, detected, X_clean):
    """NoFix: only drop rows with missing values."""
    missing_rows = set(e[0] for e in detected['missing'])
    keep = np.array([i not in missing_rows for i in range(len(X))])
    X_out = X[keep].copy()
    # Fill any remaining NaNs
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(missing_rows), keep


def strategy_deletefix(X, y, detected, X_clean):
    """DeleteFix: delete all detected error rows (with a 20% minimum retention guard)."""
    error_rows = set()
    for key in detected:
        for e in detected[key]:
            error_rows.add(e[0])

    # 20% minimum retention guard
    min_keep = max(int(len(X) * 0.2), 10)
    n_keep = len(X) - len(error_rows)
    if n_keep < min_keep:
        n_to_delete = max(len(X) - min_keep, 0)
        error_list = sorted(error_rows)
        original_error_count = len(error_rows)
        error_rows = set(error_list[:n_to_delete])
        print(f"    [DeleteFix] Retention guard triggered: deleted {len(error_rows)}/{len(X)} rows "
              f"(original candidate count was {original_error_count})")

    keep = np.array([i not in error_rows for i in range(len(X))])
    X_out = X[keep].copy()
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(error_rows), keep


def strategy_replaceall(X, y, detected, X_clean):
    """ReplaceAll: fill all errors using ValueEstimator (multi-dimensional KNN k=5 weighted mean)."""
    print("    [ReplaceAll] ValueEstimator estimation...", flush=True)
    config = DemandCleanConfig(column_names=['abv', 'ibu'], save_path=RESULT_DIR)
    estimator = ValueEstimator(config)
    print(f"    [ReplaceAll] {estimator.summary()}", flush=True)

    X_out = X.copy()
    col_means = np.nanmean(X_out, axis=0)
    deleted_rows = set()
    n_replaced = 0

    for key in detected:
        for e in detected[key]:
            idx = e[0]
            col = e[1] if isinstance(e[1], int) else 1
            estimated = estimator.estimate_feature_value(
                X_out, idx, col, deleted_rows, col_means)
            X_out[idx, col] = estimated
            n_replaced += 1

    # Fill any remaining NaNs (safety)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = col_means[col]

    print(f"    [ReplaceAll] Replaced {n_replaced} erroneous values", flush=True)
    return X_out, y, 0, 0, None


def strategy_fullfix(X, y, detected, X_clean):
    """FullFix: repair every error with the ground-truth value."""
    X_out = X.copy()
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]

    cost = 0
    for key in detected:
        for e in detected[key]:
            idx = e[0]
            if idx < len(X_clean):
                X_out[idx, 1] = X_clean[idx, 1]
                cost += 1
    return X_out, y, cost, 0, None


def strategy_demandfix(X, y, detected, X_clean):
    """DemandFix: use truth values in the boundary region; use KNN / deletion elsewhere."""
    print("    [DemandFix] start...", flush=True)
    X_out = X.copy()

    # Use ValueEstimator for imputation (replacing a plain KNNImputer)
    print("    [DemandFix] ValueEstimator estimation...", flush=True)
    config = DemandCleanConfig(column_names=['abv', 'ibu'], save_path=RESULT_DIR)
    ve = ValueEstimator(config)
    col_means = np.nanmean(X_out, axis=0)
    deleted_rows_set = set()

    # Pre-compute the full VE-imputed matrix used for filling
    X_imp = X_out.copy()
    all_error_indices = set()
    for key in detected:
        for e in detected[key]:
            all_error_indices.add(e[0])
            idx = e[0]
            col = e[1] if isinstance(e[1], int) else 1
            X_imp[idx, col] = ve.estimate_feature_value(
                X_out, idx, col, deleted_rows_set, col_means)
    # Fill NaNs
    for c in range(X_imp.shape[1]):
        X_imp[np.isnan(X_imp[:, c]), c] = col_means[c]
    print("    [DemandFix] estimation done", flush=True)

    # Train a reference model to identify the decision boundary
    error_rows = all_error_indices
    clean_mask = np.array([i not in error_rows for i in range(len(X))])
    if clean_mask.sum() > 10:
        ref_clf = SVC(kernel='linear')
        ref_clf.fit(X_imp[clean_mask], y[clean_mask])
        distances = np.abs(ref_clf.decision_function(X_imp))
        threshold = np.percentile(distances, 40)
    else:
        distances = np.zeros(len(X))
        threshold = 0

    cost = 0
    to_delete = []

    # Semantic errors: always use truth values
    for e in detected['semantic']:
        X_out[e[0], 1] = X_clean[e[0], 1]
        cost += 1

    # Missing and syntactic errors: truth value on boundary; VE / deletion elsewhere
    for e in detected['missing']:
        if distances[e[0]] < threshold:
            X_out[e[0], 1] = X_clean[e[0], 1]
            cost += 1
        else:
            X_out[e[0], 1] = X_imp[e[0], 1]

    for e in detected['syntactic']:
        if distances[e[0]] < threshold:
            X_out[e[0], 1] = X_clean[e[0], 1]
            cost += 1
        else:
            to_delete.append(e[0])

    if to_delete:
        keep = np.array([i not in set(to_delete) for i in range(len(X_out))])
        return X_out[keep], y[keep], cost, len(to_delete), keep
    return X_out, y, cost, 0, None


def strategy_deleteall(X, y, detected, X_clean):
    """DeleteAll: drop every row containing any NaN."""
    nan_rows = set(np.where(np.isnan(X).any(axis=1))[0])
    keep = np.array([i not in nan_rows for i in range(len(X))])
    X_out = X[keep].copy()
    if len(X_out) == 0:
        return X_out, y[keep], 0, len(nan_rows), keep
    # Safety fill (there should be no remaining NaNs in principle)
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(nan_rows), keep


# =============================================================================
# 3. DQN strategies (wrapped uniformly through the DemandClean API)
# =============================================================================

# DQN strategy parameter table
DQN_STRATEGY_TABLE = {
    'DQN_Single': {
        'detector_mode': 'oracle',
        'agent_type': 'single',
        'inference_mode': 'single_phase',
        'model_name': 'demandfix_dqn.pt',
    },
    'DQN_TwoStage': {
        'detector_mode': 'oracle',
        'agent_type': 'two_stage',
        'inference_mode': 'single_phase',
        'model_name': 'two_stage_dqn.pt',
    },
    'SemiSup_Single': {
        'detector_mode': 'oracle',
        'agent_type': 'single',
        'inference_mode': 'single_phase',
        'model_name': 'semi_supervised_single.pt',
    },
    'SemiSup_TwoStage': {
        'detector_mode': 'oracle',
        'agent_type': 'two_stage',
        'inference_mode': 'single_phase',
        'model_name': 'semi_supervised_dqn.pt',
    },
    'FullUnsup_Single': {
        'detector_mode': 'auto',
        'agent_type': 'single',
        'inference_mode': 'single_phase',
        'model_name': 'full_unsupervised_single.pt',
    },
    'FullUnsup_TwoStage': {
        'detector_mode': 'auto',
        'agent_type': 'two_stage',
        'inference_mode': 'single_phase',
        'model_name': 'full_unsupervised_dqn.pt',
    },
    'FullUnsup_Single_2P': {
        'detector_mode': 'auto',
        'agent_type': 'single',
        'inference_mode': 'two_phase',
        'model_name': 'full_unsupervised_single_two_phase.pt',
    },
    'FullUnsup_TwoStage_2P': {
        'detector_mode': 'auto',
        'agent_type': 'two_stage',
        'inference_mode': 'two_phase',
        'model_name': 'full_unsupervised_dqn_two_phase.pt',
    },
    'SemiSup_Dueling_Single': {
        'detector_mode': 'oracle',
        'agent_type': 'dueling_single',
        'inference_mode': 'single_phase',
        'model_name': 'semi_supervised_dueling_single.pt',
    },
    'SemiSup_Dueling_TwoStage': {
        'detector_mode': 'oracle',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase',
        'model_name': 'semi_supervised_dueling.pt',
    },
    'FullUnsup_Dueling_Single': {
        'detector_mode': 'auto',
        'agent_type': 'dueling_single',
        'inference_mode': 'single_phase',
        'model_name': 'full_unsupervised_dueling_single.pt',
    },
    'FullUnsup_Dueling_TwoStage': {
        'detector_mode': 'auto',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase',
        'model_name': 'full_unsupervised_dueling.pt',
    },
}


def run_dqn_strategy(strategy_name, X_dirty, y, X_clean,
                     n_episodes=400, load_model=False,
                     X_clean_val=None, y_clean_val=None,
                     dirty_csv_path=None, clean_csv_path=None,
                     pre_detected=None):
    """
    Unified DQN strategy entry point: create a DemandClean instance, train/load, then run inference.

    Args:
        strategy_name: strategy name from DQN_STRATEGY_TABLE
        X_dirty: dirty data (N x 2, scaled, training subset)
        y: labels (N,)
        X_clean: clean data (N x 2, scaled, training subset)
        n_episodes: number of training episodes
        load_model: whether to skip training and load a saved model
        X_clean_val: clean validation features (20% validation split, reward signal in oracle mode)
        y_clean_val: clean validation labels
        dirty_csv_path: CSV path for the 60% dirty subset (required by RAHA in auto mode)
        clean_csv_path: CSV path for the 60% clean subset

    Returns:
        (X_result, y_result, truth_cost, deleted_count, keep_mask)
    """
    params = DQN_STRATEGY_TABLE[strategy_name]
    model_path = os.path.join(MODEL_DIR, params['model_name'])

    # Use the provided 60% subset CSV paths, falling back to the full-dataset paths
    if dirty_csv_path is None:
        dirty_csv_path = os.path.join(DATA_DIR, 'dirty.csv')
    if clean_csv_path is None:
        clean_csv_path = os.path.join(DATA_DIR, 'clean.csv')

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type=params['agent_type'],
        detector_mode=params['detector_mode'],
        inference_mode=params['inference_mode'],
        n_episodes=n_episodes,
        min_truth_budget=MIN_TRUTH_BUDGET,
        max_truth_budget=MAX_TRUTH_BUDGET,
        max_repair_ratio=0.8,         # Give DQN ample repair budget (matching DemandFix)
        column_names=['abv', 'ibu'],
        label_col='is_ipa',
        save_path=RESULT_DIR,
        dirty_csv_path=dirty_csv_path,
        clean_csv_path=clean_csv_path,
        csv_columns=['abv', 'ibu', 'is_ipa'],
        reward_eval_interval=10,    # Evaluate once every 10 steps: shaping guidance + eval calibration
        repair_lambda=0.005,          # Small cost penalty: DQN is cost-aware but not repair-averse
    )

    if load_model and os.path.exists(model_path):
        # Load the saved model
        print(f"    Loading model: {model_path}", flush=True)
        dc.load(model_path)
    else:
        # Train a new model (passing the validation set)
        print(f"    Training model (episodes={n_episodes})...", flush=True)
        dc.fit(X_dirty, y, X_clean=X_clean, y_clean=y,
               X_clean_val=X_clean_val, y_clean_val=y_clean_val)
        # Save the model
        os.makedirs(MODEL_DIR, exist_ok=True)
        dc.save(model_path)
        print(f"    Model saved to: {model_path}", flush=True)

    # Inference
    if params['inference_mode'] == 'two_phase':
        # Two-phase inference: plan, then execute
        print(f"    Two-phase inference: plan...", flush=True)
        plan = dc.plan(X_dirty, y, X_clean=X_clean)

        # Collect positions that need truth values and build the true_values dict
        plan_positions = dc.get_plan_positions()
        true_values = {}
        for (idx, col) in plan_positions:
            if idx < len(X_clean):
                true_values[(idx, col)] = X_clean[idx, col]

        print(f"    Two-phase inference: execute ({len(true_values)} truth values required)...", flush=True)
        X_result, y_result, keep_mask = dc.execute(X_dirty, true_values, y_dirty=y)

        truth_cost = len(true_values)
        deleted_count = int(np.sum(~keep_mask)) if keep_mask is not None else 0
    else:
        # Single-phase inference
        print(f"    Single-phase inference...", flush=True)
        X_result, y_result, stats = dc.clean(X_dirty, y, X_clean,
                                              pre_detected=pre_detected)

        keep_mask = stats.get('keep_mask', None)
        truth_cost = stats.get('truth_cost', 0)
        deleted_count = stats.get('deleted_count', 0)

    return X_result, y_result, truth_cost, deleted_count, keep_mask


# =============================================================================
# 4. Evaluation metrics (original logic preserved)
# =============================================================================
def compute_metrics(X_result, X_clean, X_dirty, keep_mask=None):
    """
    Compute auth (authenticity) and div (diversity).

    - auth = correct values / total values (comparing X_result vs X_clean)
    - div = sample_retention * noise_retention
    """
    n = len(X_result)
    n_total = len(X_clean)
    if n == 0:
        return 0, 0

    # Authenticity
    if keep_mask is not None:
        X_clean_kept = X_clean[keep_mask]
        correct = sum(1 for i in range(min(n, len(X_clean_kept)))
                      if abs(X_result[i, 1] - X_clean_kept[i, 1]) < 0.01)
    else:
        correct = sum(1 for i in range(n) if abs(X_result[i, 1] - X_clean[i, 1]) < 0.01)
    auth = correct / n

    # Diversity
    sample_ret = n / n_total
    X_dirty_valid = X_dirty[~np.isnan(X_dirty).any(axis=1)]
    if len(X_result) > 1 and len(X_dirty_valid) > 1:
        result_var = np.var(X_result[:, 1])
        clean_var = np.var(X_clean[:, 1])
        dirty_var = np.var(X_dirty_valid[:, 1])
        if dirty_var > clean_var + 1e-6:
            noise_ret = np.clip((result_var - clean_var) / (dirty_var - clean_var), 0, 1)
        else:
            noise_ret = 1.0
        div = sample_ret * noise_ret
    else:
        div = 0

    return auth, div


# =============================================================================
# 5. Visualization (original logic preserved)
# =============================================================================
def plot_boundary(clf, ideal_clf, X_train, y_train, X_test, y_test,
                  name, cost, deleted, auth, div, path, xlim, ylim):
    """SVM decision boundary visualization (misclassified points highlighted with red circles)."""
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),
                          np.linspace(ylim[0], ylim[1], 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Zi = ideal_clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    # Training data scatter
    for label, marker, color, lbl in [(0, 'o', 'C0', 'Non-IPA (train)'),
                                       (1, 'x', 'C1', 'IPA (train)')]:
        mask = y_train == label
        plt.scatter(X_train[mask, 0], X_train[mask, 1],
                    c=color, marker=marker, s=25, alpha=0.7, label=lbl)

    # Highlight misclassified points with red circles
    mis_mask = y_test != y_pred
    if mis_mask.sum() > 0:
        plt.scatter(X_test[mis_mask, 0], X_test[mis_mask, 1],
                    facecolors='none', edgecolors='red', s=80, linewidths=2,
                    label=f'Misclassified ({mis_mask.sum()})')

    # Decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors=['purple'], linewidths=2.5)
    plt.contour(xx, yy, Zi, levels=[0], colors=['green'], linestyles=['--'], linewidths=2)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('ABV', fontsize=11)
    plt.ylabel('IBU', fontsize=11)
    plt.title(f'{name}', fontsize=12)

    summary = f'Acc: {acc:.3f} | Cost: {cost} | Auth: {auth:.2f} | Div: {div:.2f}'
    plt.text(0.02, 0.02, summary, transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return acc


# =============================================================================
# 6. Main
# =============================================================================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Beer (IPA) dataset ablation study')
    parser.add_argument('--load_model', action='store_true',
                        help='load saved models (skip training and infer only)')
    parser.add_argument('--n_episodes', type=int, default=400,
                        help='DQN training episodes (default 400)')
    parser.add_argument('--strategies', type=str, default=None,
                        help='comma-separated strategy names to run, e.g. '
                             '"NoFix,FullFix,DQN_Single". Omit to run all.')
    args = parser.parse_args()

    print("=" * 70)
    print("Beer (IPA) dataset ablation study")
    print("=" * 70)
    print(f"  load_model={args.load_model}, n_episodes={args.n_episodes}")
    if args.strategies:
        print(f"  strategies: {args.strategies}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[1] Load data")
    clean_csv_path = os.path.join(DATA_DIR, 'clean.csv')
    clean_df = pd.read_csv(clean_csv_path)
    valid_mask = clean_df['abv'].notna() & clean_df['ibu'].notna()
    clean_df = clean_df[valid_mask].reset_index(drop=True)
    clean_df['is_ipa'] = clean_df['style'].apply(
        lambda s: 1 if pd.notna(s) and 'ipa' in str(s).lower() else 0)
    print(f"  Data: {len(clean_df)} rows, IPA: {clean_df['is_ipa'].sum()}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Inject errors (full dataset)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[2] Inject errors")
    injector = ErrorInjector(clean_df)
    injector.inject_semantic(0.25).inject_syntactic(0.35).inject_missing(0.05)
    dirty_df, errors = injector.get_result()
    injector.print_stats()

    # Save the full dirty CSV
    full_dirty_csv_path = os.path.join(DATA_DIR, 'dirty.csv')
    dirty_export = dirty_df[['ABV_dirty', 'IBU_dirty']].copy()
    dirty_export.columns = ['abv', 'ibu']
    dirty_export['is_ipa'] = clean_df['is_ipa'].values
    dirty_export.to_csv(full_dirty_csv_path, index=False)
    print(f"  Full dirty data saved: {full_dirty_csv_path}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Feature prep + 60/20/20 split + standardization
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[3] Feature prep + 60/20/20 split + standardization")

    X_clean_full = clean_df[['abv', 'ibu']].values.astype(float)
    X_dirty_full = dirty_df[['ABV_dirty', 'IBU_dirty']].values.astype(float)
    y_full = clean_df['is_ipa'].values
    print(f"  Full feature shape: {X_clean_full.shape}")

    # --- 60/20/20 split (seed=42, stratify=y) ---
    all_idx = np.arange(len(X_clean_full))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.4, random_state=42, stratify=y_full)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=y_full[temp_idx])

    print(f"  60/20/20 split: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")

    # --- Fit scaler on the dirty training set ---
    X_dirty_train_raw = X_dirty_full[train_idx]
    X_dirty_train_filled = X_dirty_train_raw.copy()
    train_col_means = np.nanmean(X_dirty_train_filled, axis=0)
    for col in range(X_dirty_train_filled.shape[1]):
        nan_mask = np.isnan(X_dirty_train_filled[:, col])
        X_dirty_train_filled[nan_mask, col] = train_col_means[col]

    scaler = StandardScaler()
    scaler.fit(X_dirty_train_filled)
    print(f"  Scaler fit done (on {len(train_idx)} dirty training rows)")
    print(f"    mean={scaler.mean_}, scale={scaler.scale_}")

    # --- Helper: standardize a subset while keeping NaN positions ---
    def scale_subset(X_raw):
        """Standardize a subset while preserving NaN positions."""
        X_filled = X_raw.copy()
        nan_mask = np.isnan(X_filled)
        cm = np.nanmean(X_filled, axis=0)
        for c in range(X_filled.shape[1]):
            nan_c = np.isnan(X_filled[:, c])
            fill_val = cm[c] if not np.isnan(cm[c]) else 0.0
            X_filled[nan_c, c] = fill_val
        X_scaled = scaler.transform(X_filled)
        X_scaled[nan_mask] = np.nan
        return X_scaled

    # --- Standardize each subset ---
    X_dirty_train = scale_subset(X_dirty_full[train_idx])
    X_clean_train = scaler.transform(X_clean_full[train_idx])
    X_clean_val = scaler.transform(X_clean_full[val_idx])
    X_clean_test = scaler.transform(X_clean_full[test_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    print(f"  X_dirty_train={X_dirty_train.shape}, X_clean_train={X_clean_train.shape}")
    print(f"  X_clean_val={X_clean_val.shape}, X_clean_test={X_clean_test.shape}")

    # --- Save the 60% subset CSV (required by RAHA in auto mode) ---
    dirty_train_df = dirty_df.iloc[train_idx].reset_index(drop=True)
    dirty_train_export = dirty_train_df[['ABV_dirty', 'IBU_dirty']].copy()
    dirty_train_export.columns = ['abv', 'ibu']
    dirty_train_export['is_ipa'] = clean_df.iloc[train_idx]['is_ipa'].values
    dirty_train_csv_path = os.path.join(RESULT_DIR, 'dirty_train_60pct.csv')
    dirty_train_export.to_csv(dirty_train_csv_path, index=False)

    clean_train_df = clean_df.iloc[train_idx].reset_index(drop=True)
    clean_train_export = clean_train_df[['abv', 'ibu']].copy()
    clean_train_export['is_ipa'] = clean_train_df['is_ipa'].values
    clean_train_csv_path = os.path.join(RESULT_DIR, 'clean_train_60pct.csv')
    clean_train_export.to_csv(clean_train_csv_path, index=False)
    print(f"  60% subset CSV: {dirty_train_csv_path}")

    print(f"  Elapsed: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Build detected dict (remap indices to the training subset)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[4] Build detection results (from injection log; remapped to training subset)")

    # Full-index -> training-subset local index
    global_to_local = {int(g): l for l, g in enumerate(train_idx)}

    detected_for_baseline = {'missing': [], 'semantic': [], 'syntactic': []}

    total_global = sum(len(errors[k]) for k in errors)
    for e in errors['missing']:
        if e[0] in global_to_local:
            local_idx = global_to_local[e[0]]
            detected_for_baseline['missing'].append((local_idx, 1, e[2]))
    for e in errors['semantic']:
        if e[0] in global_to_local:
            local_idx = global_to_local[e[0]]
            detected_for_baseline['semantic'].append((local_idx, 1, e[2], e[3]))
    for e in errors['syntactic']:
        if e[0] in global_to_local:
            local_idx = global_to_local[e[0]]
            detected_for_baseline['syntactic'].append((local_idx, 1, e[2], e[3]))

    total_train = sum(len(v) for v in detected_for_baseline.values())
    print(f"  Full errors: {total_global}, training-set errors: {total_train}")
    print(f"    missing: {len(detected_for_baseline['missing'])}")
    print(f"    semantic: {len(detected_for_baseline['semantic'])}")
    print(f"    syntactic: {len(detected_for_baseline['syntactic'])}")
    print(f"  Elapsed: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Train the ideal classifier (60% clean training set)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[5] Train ideal classifier (60% clean training set)")
    ideal_clf = SVC(kernel='linear')
    ideal_clf.fit(X_clean_train, y_train)
    print("  Ideal classifier trained", flush=True)

    # xlim/ylim: use the full clean range (needed for visualization)
    X_clean_full_scaled = scaler.transform(X_clean_full)
    xlim = (X_clean_full_scaled[:, 0].min() - 0.5, X_clean_full_scaled[:, 0].max() + 0.5)
    ylim = (X_clean_full_scaled[:, 1].min() - 0.5, X_clean_full_scaled[:, 1].max() + 0.5)
    print(f"  Elapsed: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 6. Build the strategy dict and run
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[6] Apply cleaning strategies", flush=True)

    # Baselines (all operate on the 60% training subset)
    strategies = {
        'NoFix': lambda: strategy_nofix(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
        'DeleteAll': lambda: strategy_deleteall(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
        'DeleteFix': lambda: strategy_deletefix(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
        'ReplaceAll': lambda: strategy_replaceall(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
        'FullFix': lambda: strategy_fullfix(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
        'DemandFix': lambda: strategy_demandfix(
            X_dirty_train, y_train, detected_for_baseline, X_clean_train),
    }

    # DQN strategies (via the DemandClean API; pass validation set and 60% CSV paths)
    for dqn_name in DQN_STRATEGY_TABLE:
        def _make_dqn_func(name):
            def _run():
                return run_dqn_strategy(
                    name, X_dirty_train, y_train, X_clean_train,
                    n_episodes=args.n_episodes, load_model=args.load_model,
                    X_clean_val=X_clean_val, y_clean_val=y_val,
                    dirty_csv_path=dirty_train_csv_path,
                    clean_csv_path=clean_train_csv_path,
                    pre_detected=detected_for_baseline)
            return _run
        strategies[dqn_name] = _make_dqn_func(dqn_name)

    # Filter to selected strategies
    if args.strategies:
        selected = [s.strip() for s in args.strategies.split(',')]
        strategies = {k: v for k, v in strategies.items() if k in selected}
        print(f"  Selected {len(strategies)} strategies: {list(strategies.keys())}")

    # ------------------------------------------------------------------
    # 7. Run each strategy (evaluate on the 20% test set)
    # ------------------------------------------------------------------
    results = {}
    for name, func in strategies.items():
        print(f"\n--- {name} ---", flush=True)
        t_strategy = time.time()

        try:
            X_train_result, y_train_result, cost, deleted, keep_mask = func()
        except Exception as e:
            print(f"  {name}: failed - {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Strategy function completed", flush=True)

        if len(X_train_result) < 10 or len(np.unique(y_train_result)) < 2:
            print(f"  {name}: too few samples, skipping")
            continue

        # Compute auth/div on the training subset
        auth, div = compute_metrics(
            X_train_result, X_clean_train, X_dirty_train, keep_mask)

        # Train SVM on the cleaned training set
        clf = SVC(kernel='linear')
        clf.fit(X_train_result, y_train_result)

        # Evaluate on the 20% test set
        save_path = os.path.join(RESULT_DIR, f'{name}.png')
        acc = plot_boundary(clf, ideal_clf, X_train_result, y_train_result,
                           X_clean_test, y_test,
                           name, cost, deleted, auth, div,
                           save_path, xlim, ylim)

        results[name] = {'acc': acc, 'cost': cost, 'auth': auth, 'div': div}
        print(f"  Acc={acc:.4f}, Cost={cost}, Auth={auth:.2f}, Div={div:.2f} "
              f"(elapsed {time.time() - t_strategy:.2f}s)")

    print(f"  Total strategy time: {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Strategy':<30} {'Acc':>8} {'Cost':>6} {'Auth':>8} {'Div':>8}")
    print("-" * 68)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        print(f"{name:<30} {r['acc']:>8.4f} {r['cost']:>6} {r['auth']:>8.2f} {r['div']:>8.2f}")

    print("\nTarget checks:")
    if 'DemandFix' in results and 'FullFix' in results:
        if results['DemandFix']['acc'] >= results['FullFix']['acc']:
            print("  [OK] DemandFix >= FullFix")
        else:
            diff = results['FullFix']['acc'] - results['DemandFix']['acc']
            print(f"  [WARN] DemandFix < FullFix (gap {diff:.4f})")
        print(f"  Cost: DemandFix({results['DemandFix']['cost']}) "
              f"vs FullFix({results['FullFix']['cost']})")

    if 'DeleteFix' in results:
        print(f"  DeleteFix: deleted {results['DeleteFix'].get('cost', 0)} rows")
    if 'ReplaceAll' in results:
        print(f"  ReplaceAll: imputed via ValueEstimator")

    print(f"\nTotal elapsed: {time.time() - total_start:.2f}s")


if __name__ == '__main__':
    main()
