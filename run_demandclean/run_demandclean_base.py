"""
DemandClean unified runner
==========================

Supports 8 version combinations (2 detectors x 2 agents x 2 inference modes):
  v1: oracle + dueling_two_stage + single_phase
  v2: oracle + dueling_two_stage + two_phase
  v3: oracle + single_stage      + single_phase
  v4: oracle + single_stage      + two_phase
  v5: auto   + dueling_two_stage + single_phase
  v6: auto   + dueling_two_stage + two_phase
  v7: auto   + single_stage      + single_phase
  v8: auto   + single_stage      + two_phase

Full pipeline per version:
  train -> inference -> 5 baseline evaluations -> tolerance eval -> detector accuracy -> Shapley analysis -> training curves

Usage:
    python run_demandclean_base.py --dataset beers --n_episodes 300
    python run_demandclean_base.py --dataset beers --n_episodes 300 --all_datasets
"""

import sys
import os
import json
import warnings
import time
import traceback
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Edit-distance safe encoding
try:
    from demandclean.utils.edit_distance import find_nearest_known as _find_nearest_known
except ImportError:
    _find_nearest_known = None


# =========================================================================
# TeeLogger: mirrors output to terminal and log file
# =========================================================================
class TeeLogger:
    """Write stdout/stderr to both terminal and log file (fault-tolerant)."""

    def __init__(self, log_path: str, stream=None):
        self.terminal = stream or sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # line buffered

    @property
    def _file_ok(self):
        """Check whether log_file is still writable."""
        try:
            return self.log_file is not None and not self.log_file.closed
        except Exception:
            return False

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            # terminal is gone too; fall back to original stdout
            if sys.__stdout__ is not None and not sys.__stdout__.closed:
                sys.__stdout__.write(message)
        if self._file_ok:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (ValueError, IOError, OSError):
                pass  # silent failure; degrade to terminal-only output

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        if self._file_ok:
            try:
                self.log_file.flush()
            except (ValueError, IOError, OSError):
                pass

    def close(self):
        if self._file_ok:
            try:
                self.log_file.close()
            except (ValueError, IOError, OSError):
                pass

    def fileno(self):
        return self.terminal.fileno()

# =========================================================================
# Project root
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# DEBUG: print first few sys.path entries (main process only)
if __name__ == "__main__":
    print(f"[DEBUG] _PROJECT_ROOT={_PROJECT_ROOT}", file=sys.stderr)
try:
    import tools as _t
    pass  # silent check
except Exception:
    pass

# =========================================================================
# Core imports
# =========================================================================
from demandclean.api import DemandClean
from demandclean.config import (
    DemandCleanConfig, TaskType, ModelType,
    AgentType, DetectorMode, InferenceMode,
)
from demandclean.detectors import OracleDetector
from demandclean.utils.model_io import ModelIO
from demandclean.tools.tolerance_analysis import (
    compute_model_tolerance,
    evaluate_detector_accuracy,
    compare_versions,
)
from demandclean.tools.shapley_analysis import run_full_shapley_analysis

# Lazy import: avoid failing when multiprocessing spawn reruns the main script.
# RAHA uses multiprocessing, so child processes reload this module.
try:
    from tools.rules_parser import get_horizon_fds
except (ModuleNotFoundError, ImportError):
    # Not needed inside child processes.
    get_horizon_fds = None


# ============================================================================
# Version switches (8 total)
# ============================================================================
ENABLE_ORACLE_DUELING_SINGLE_PHASE = False  # v1
ENABLE_ORACLE_DUELING_TWO_PHASE   = False  # v2
ENABLE_ORACLE_PLAIN_SINGLE_PHASE  = False  # v3
ENABLE_ORACLE_PLAIN_TWO_PHASE     = False  # v4
ENABLE_AUTO_DUELING_SINGLE_PHASE  = True   # v5 (default)
ENABLE_AUTO_DUELING_TWO_PHASE     = False  # v6
ENABLE_AUTO_PLAIN_SINGLE_PHASE    = False  # v7
ENABLE_AUTO_PLAIN_TWO_PHASE       = False  # v8


# ============================================================================
# VERSION_CONFIGS: per-version configuration dict
# ============================================================================
VERSION_CONFIGS = {
    'v1_oracle_dueling_single': {
        'version_name': 'v1_oracle_dueling_single',
        'detector_mode': 'oracle',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase',
        'enabled': ENABLE_ORACLE_DUELING_SINGLE_PHASE,
    },
    'v2_oracle_dueling_two': {
        'version_name': 'v2_oracle_dueling_two',
        'detector_mode': 'oracle',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'two_phase',
        'enabled': ENABLE_ORACLE_DUELING_TWO_PHASE,
    },
    'v3_oracle_plain_single': {
        'version_name': 'v3_oracle_plain_single',
        'detector_mode': 'oracle',
        'agent_type': 'single_stage',
        'inference_mode': 'single_phase',
        'enabled': ENABLE_ORACLE_PLAIN_SINGLE_PHASE,
    },
    'v4_oracle_plain_two': {
        'version_name': 'v4_oracle_plain_two',
        'detector_mode': 'oracle',
        'agent_type': 'single_stage',
        'inference_mode': 'two_phase',
        'enabled': ENABLE_ORACLE_PLAIN_TWO_PHASE,
    },
    'v5_auto_dueling_single': {
        'version_name': 'v5_auto_dueling_single',
        'detector_mode': 'auto',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase',
        'enabled': ENABLE_AUTO_DUELING_SINGLE_PHASE,
    },
    'v6_auto_dueling_two': {
        'version_name': 'v6_auto_dueling_two',
        'detector_mode': 'auto',
        'agent_type': 'dueling_two_stage',
        'inference_mode': 'two_phase',
        'enabled': ENABLE_AUTO_DUELING_TWO_PHASE,
    },
    'v7_auto_plain_single': {
        'version_name': 'v7_auto_plain_single',
        'detector_mode': 'auto',
        'agent_type': 'single_stage',
        'inference_mode': 'single_phase',
        'enabled': ENABLE_AUTO_PLAIN_SINGLE_PHASE,
    },
    'v8_auto_plain_two': {
        'version_name': 'v8_auto_plain_two',
        'detector_mode': 'auto',
        'agent_type': 'single_stage',
        'inference_mode': 'two_phase',
        'enabled': ENABLE_AUTO_PLAIN_TWO_PHASE,
    },
}


# ============================================================================
# Dataset configuration
# ============================================================================
DATASETS = {
    'beers': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'style',
        # beer_name/brewery_name/city/state are text; ounces/abv contain 'empty' but are numeric
        'categorical_cols': {'beer_name', 'brewery_name', 'city', 'state'},
        'protected_cols': {'brewery_name'},  # FD LHS: brewery_name -> city, brewery_name -> state
    },
    'adult': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'income',
        'categorical_cols': {'workclass', 'education', 'marital_status', 'occupation',
                             'relationship', 'race', 'gender', 'native_country'},
        'protected_cols': set(),
    },
    'bike': {
        'task_type': 'regression',
        'model_type': 'random_forest',
        'label_col': 'cnt',
        'categorical_cols': set(),  # all numeric columns
        'protected_cols': set(),
    },
    'breast_cancer': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'class',
        'categorical_cols': set(),  # all INT [1,10] numeric columns
        'protected_cols': set(),
    },
    'har': {
        'task_type': 'clustering',
        'model_type': 'kmeans',
        'label_col': 'gt',
        'categorical_cols': set(),  # x/y/z all FLOAT
        'protected_cols': set(),
    },
    'mercedes': {
        'task_type': 'regression',
        'model_type': 'ridge',
        'label_col': 'y',
        'categorical_cols': {'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'},
        'protected_cols': set(),
    },
    'nasa': {
        'task_type': 'regression',
        'model_type': 'ridge',
        'label_col': 'sound_pressure_level',
        'categorical_cols': set(),  # frequency/angle/chord_length/velocity are numeric
        'protected_cols': set(),
    },
    'smartfactory': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'labels',
        'categorical_cols': set(),  # all INT sensor readings
        'protected_cols': set(),
    },
    'soilmoisture': {
        'task_type': 'regression',
        'model_type': 'ridge',
        'label_col': 'soil_moisture',
        'categorical_cols': set(),  # all FLOAT spectral bands
        'protected_cols': set(),
    },
}

# Short model names for evaluation (classification / regression / clustering, two each)
EVAL_MODELS_CLASSIFICATION = ['rf', 'lr']
EVAL_MODELS_REGRESSION = ['rf', 'ridge']
EVAL_MODELS_CLUSTERING = ['kmeans']  # AgglomerativeClustering O(n^2~n^3) too costly on large datasets; KMeans only

# Lightweight model kwargs for reward evaluation during training (by model_type).
# Each step's reward eval runs fit+evaluate and must stay fast; baseline evaluation uses independent standard models.
REWARD_MODEL_KWARGS = {
    'random_forest': {'n_estimators': 10, 'max_depth': 8},
    'xgboost':       {'n_estimators': 10, 'max_depth': 3},
    'xgboost_reg':   {'n_estimators': 10, 'max_depth': 3},
    'ridge':         {},  # linear models already lightweight
    'linear':        {},  # linear models already lightweight
    'svm':           {},  # linear kernel SVM is fine on small datasets
    'kmeans':        {'n_init': 3},  # fewer KMeans initializations
}


# ============================================================================
# Data preprocessing
# ============================================================================
def preprocess_data(dataset_name: str, dirty_df: pd.DataFrame = None, clean_df: pd.DataFrame = None):
    """
    Load and preprocess the specified dataset.

    Steps:
      1. Read dirty_with_index.csv and clean_with_index.csv (or use supplied DataFrames)
      2. Replace "empty"/"Empty"/"EMPTY" with NaN
      3. Fit LabelEncoder on dirty data only (not on clean)
      4. Fit StandardScaler on dirty data only
      5. Auto-derive and parse FD rules

    Args:
        dataset_name: dataset name (e.g. 'beers')
        dirty_df: optional dirty-data DataFrame (skip file read if provided)
        clean_df: optional clean-data DataFrame (used for encoding; None skips clean encoding)

    Returns:
        (X_dirty_scaled, y_dirty, X_clean_scaled_or_None, y_clean_or_None,
         column_names, fd_rules, rules_path,
         dirty_csv_path, clean_csv_path, csv_columns,
         scaler, label_encoders, categorical_cols,
         dirty_df, clean_df_or_None)
    """
    ds_cfg = DATASETS[dataset_name]
    label_col = ds_cfg['label_col']

    # --- Paths ---
    data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(data_dir, 'clean_with_index.csv')
    rules_path = os.path.join(data_dir, 'rules.txt')

    # --- Load (or use supplied DataFrame) ---
    read_clean_from_file = (dirty_df is None and clean_df is None)  # default: read both from files
    if dirty_df is None:
        dirty_df = pd.read_csv(dirty_path)
    if clean_df is None and read_clean_from_file:
        clean_df = pd.read_csv(clean_path)
    # Ensure dirty_df has clean column names
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    if clean_df is not None:
        clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]

    # --- Replace empty with NaN ---
    dirty_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)
    if clean_df is not None:
        clean_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    # --- Determine feature columns ---
    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # --- Identify categorical vs numeric columns ---
    config_cat_cols = ds_cfg.get('categorical_cols')
    if config_cat_cols is not None:
        categorical_cols = set(config_cat_cols) & set(feature_cols)
    else:
        categorical_cols = set()
        for col in feature_cols:
            vals = dirty_df[col].dropna()
            vals = vals[~vals.astype(str).str.strip().isin(
                ['?', '', 'N/A', 'NA', 'nan', 'NaN', 'null', 'None', '-', 'empty', 'Empty', 'EMPTY']
            )]
            if len(vals) == 0:
                categorical_cols.add(col)
                continue
            try:
                pd.to_numeric(vals, errors='raise')
            except (ValueError, TypeError):
                categorical_cols.add(col)

    # --- Encoding helpers ---
    # LabelEncoder is fit on dirty data only (no dependency on clean)
    label_encoders = {}

    def encode_df(df, feature_cols, label_col, fit_le=False):
        """Encode features and label of a DataFrame, returning (X, y).

        Args:
            df: DataFrame to encode
            feature_cols: list of feature column names
            label_col: label column name
            fit_le: whether to fit LabelEncoder (set True only for dirty data)
        """
        X_df = df[feature_cols].copy()
        y_series = df[label_col].copy()

        for col in feature_cols:
            if col in categorical_cols:
                if col not in label_encoders and fit_le:
                    le = LabelEncoder()
                    # Fit LE on dirty data only
                    all_vals = dirty_df[col].dropna().astype(str).unique()
                    le.fit(all_vals)
                    label_encoders[col] = le

                nan_mask = X_df[col].isna()
                non_nan_values = X_df.loc[~nan_mask, col].astype(str)
                le = label_encoders[col]

                if _find_nearest_known is not None:
                    known_set = set(le.classes_)
                    known_list = list(le.classes_)

                    def _safe_encode(val, _le=le, _ks=known_set, _kl=known_list):
                        if val in _ks:
                            return _le.transform([val])[0]
                        nearest = _find_nearest_known(val, _kl, threshold=0.3)
                        if nearest is not None:
                            return _le.transform([nearest])[0]
                        return np.nan  # unreachable when LE is dirty-fit

                    X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_encode)
                else:
                    X_df.loc[~nan_mask, col] = le.transform(non_nan_values)
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # Encode label
        label_is_categorical = False
        if fit_le:
            combined_labels = dirty_df[label_col].dropna()
        else:
            combined_labels = df[label_col].dropna()
        try:
            pd.to_numeric(combined_labels, errors='raise')
        except (ValueError, TypeError):
            label_is_categorical = True

        if label_is_categorical:
            if label_col not in label_encoders and fit_le:
                le = LabelEncoder()
                all_labels = dirty_df[label_col].dropna().astype(str).unique()
                le.fit(all_labels)
                label_encoders[label_col] = le
            if label_col in label_encoders:
                nan_mask_y = y_series.isna()
                y_encoded = y_series.copy()
                le_label = label_encoders[label_col]
                known_set_y = set(le_label.classes_)

                def _safe_encode_label(val, _le=le_label, _ks=known_set_y):
                    val_str = str(val)
                    if val_str in _ks:
                        return _le.transform([val_str])[0]
                    if _find_nearest_known is not None:
                        nearest = _find_nearest_known(val_str, list(_le.classes_), threshold=0.3)
                        if nearest is not None:
                            return _le.transform([nearest])[0]
                    return np.nan

                y_encoded.loc[~nan_mask_y] = y_series.loc[~nan_mask_y].astype(str).map(_safe_encode_label)
                y_series = pd.to_numeric(y_encoded, errors='coerce')

        X = X_df.values.astype(float)
        y = y_series.values.astype(float)
        return X, y

    # Encode dirty (also fits LE)
    X_dirty, y_dirty = encode_df(dirty_df, feature_cols, label_col, fit_le=True)

    # Encode clean (using already-fit LE)
    X_clean_scaled = None
    y_clean = None
    if clean_df is not None:
        X_clean, y_clean = encode_df(clean_df, feature_cols, label_col, fit_le=False)

    # --- Standardize (fit on dirty only) ---
    scaler = StandardScaler()
    X_dirty_for_fit = X_dirty.copy()
    col_means = np.nanmean(X_dirty_for_fit, axis=0)
    for j in range(X_dirty_for_fit.shape[1]):
        mask = np.isnan(X_dirty_for_fit[:, j])
        X_dirty_for_fit[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    scaler.fit(X_dirty_for_fit)

    # Standardize dirty (preserving NaN positions)
    def scale_with_nan(X, scaler):
        X_out = X.copy()
        col_m = np.nanmean(X_out, axis=0)
        for j in range(X_out.shape[1]):
            nan_mask = np.isnan(X_out[:, j])
            X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
        X_out = scaler.transform(X_out)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X_out[nan_mask, j] = np.nan
        return X_out

    X_dirty_scaled = scale_with_nan(X_dirty, scaler)

    # Standardize clean (if available)
    if clean_df is not None:
        X_clean_for_scale = X_clean.copy()
        col_means_c = np.nanmean(X_clean_for_scale, axis=0)
        for j in range(X_clean_for_scale.shape[1]):
            mask = np.isnan(X_clean_for_scale[:, j])
            X_clean_for_scale[mask, j] = col_means_c[j] if not np.isnan(col_means_c[j]) else 0.0
        X_clean_scaled = scaler.transform(X_clean_for_scale)

    # --- Parse FD rules ---
    fd_rules = []
    if os.path.exists(rules_path) and get_horizon_fds is not None:
        try:
            fd_rules = get_horizon_fds(rules_path)
        except Exception as e:
            print(f"  [warn] FD rule parse failed: {e}")

    print(f"  Dataset: {dataset_name}")
    print(f"  Dirty:   {X_dirty_scaled.shape}")
    if X_clean_scaled is not None:
        print(f"  Clean:   {X_clean_scaled.shape}")
    print(f"  Features: {len(feature_cols)}, label: {label_col}")
    print(f"  Categorical cols: {categorical_cols if categorical_cols else '(none)'}")
    print(f"  LE fit: dirty only ({len(dirty_df)} rows)")
    print(f"  SS fit: dirty only ({len(dirty_df)} rows)")
    print(f"  FD rules: {len(fd_rules)}")

    csv_columns = list(dirty_df.columns)

    return (
        X_dirty_scaled, y_dirty,
        X_clean_scaled, y_clean,
        feature_cols, fd_rules, rules_path,
        dirty_path, clean_path, csv_columns,
        scaler, label_encoders, categorical_cols,
        dirty_df, clean_df,
    )


# ============================================================================
# encode_subset: encode a val/test subset using pre-fit LE/SS
# ============================================================================
def encode_subset(df, feature_cols, label_col, label_encoders, scaler,
                  categorical_cols, dataset_name=None):
    """Encode a val/test subset with already-fit LE/SS; return (X_scaled, y).

    Args:
        df: DataFrame to encode
        feature_cols: feature column names
        label_col: label column name
        label_encoders: already-fit {col_name: LabelEncoder}
        scaler: already-fit StandardScaler
        categorical_cols: set of categorical column names
        dataset_name: optional; used for logs

    Returns:
        (X_scaled, y) — standardized feature matrix and label array
    """
    df = df.copy()
    df.columns = [c.strip().strip('\ufeff') for c in df.columns]
    df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    X_df = df[feature_cols].copy()
    y_series = df[label_col].copy()

    for col in feature_cols:
        if col in categorical_cols and col in label_encoders:
            le = label_encoders[col]
            nan_mask = X_df[col].isna()
            non_nan_values = X_df.loc[~nan_mask, col].astype(str)

            if _find_nearest_known is not None:
                known_set = set(le.classes_)
                known_list = list(le.classes_)

                def _safe_enc(val, _le=le, _ks=known_set, _kl=known_list):
                    if val in _ks:
                        return _le.transform([val])[0]
                    nearest = _find_nearest_known(val, _kl, threshold=0.3)
                    if nearest is not None:
                        return _le.transform([nearest])[0]
                    return np.nan

                X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_enc)
            else:
                known_set = set(le.classes_)
                encoded = []
                for v in non_nan_values:
                    if v in known_set:
                        encoded.append(le.transform([v])[0])
                    else:
                        encoded.append(np.nan)
                X_df.loc[~nan_mask, col] = encoded
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    # Encode label
    if label_col in label_encoders:
        le_label = label_encoders[label_col]
        nan_mask_y = y_series.isna()
        y_encoded = y_series.copy()
        known_set_y = set(le_label.classes_)

        def _safe_enc_label(val, _le=le_label, _ks=known_set_y):
            val_str = str(val)
            if val_str in _ks:
                return _le.transform([val_str])[0]
            if _find_nearest_known is not None:
                nearest = _find_nearest_known(val_str, list(_le.classes_), threshold=0.3)
                if nearest is not None:
                    return _le.transform([nearest])[0]
            return np.nan

        y_encoded.loc[~nan_mask_y] = y_series.loc[~nan_mask_y].astype(str).map(_safe_enc_label)
        y_series = pd.to_numeric(y_encoded, errors='coerce')
    else:
        y_series = pd.to_numeric(y_series, errors='coerce')

    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    # Standardize (NaN-safe)
    X_out = X.copy()
    col_m = np.nanmean(X_out, axis=0)
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
    X_scaled = scaler.transform(X_out)

    return X_scaled, y


# ============================================================================
# Visualization: training curves (4 subplots)
# ============================================================================
def plot_training_curves(history: dict, save_dir: str, version_name: str,
                         resume_episode: int = 0):
    """
    Plot six training-process subplots (3x2):
      Score, Reward, Epsilon, Action Count, Action Ratio, Q-Value Uncertainty

    Args:
        history: training history dict
        save_dir: output directory
        version_name: version name
        resume_episode: resume start episode (0 = no resume)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f'{version_name} - Training Curves', fontsize=14)

    episodes = history.get('episode', list(range(1, len(history.get('score', [])) + 1)))

    def _split_old_new(data, eps_list, resume_ep):
        """Split data into pre-/post-resume segments."""
        if resume_ep <= 0 or not eps_list:
            return [], [], data, eps_list
        old_data, old_eps, new_data, new_eps = [], [], [], []
        for e, d in zip(eps_list, data):
            if e < resume_ep:
                old_data.append(d)
                old_eps.append(e)
            else:
                new_data.append(d)
                new_eps.append(e)
        return old_data, old_eps, new_data, new_eps

    def _plot_with_resume(ax, episodes_list, data, color, resume_ep, label=None):
        """Resume-aware plotting: old data gray dashed, new data colored solid."""
        old_d, old_e, new_d, new_e = _split_old_new(data, episodes_list, resume_ep)
        if old_d:
            ax.plot(old_e, old_d, alpha=0.25, color='gray', linestyle='--')
        if new_d:
            ax.plot(new_e, new_d, alpha=0.3, color=color, label=label)
        # Resume marker line
        if resume_ep > 0 and old_d and new_d:
            ax.axvline(x=resume_ep, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # --- Score ---
    ax = axes[0, 0]
    scores = history.get('score', [])
    if scores:
        _plot_with_resume(ax, episodes[:len(scores)], scores, 'blue', resume_episode)
        window = max(1, len(scores) // 20)
        if window > 1:
            avg = np.convolve(scores, np.ones(window) / window, mode='valid')
            ax.plot(episodes[window - 1:len(scores)], avg, color='blue', linewidth=2, label=f'MA({window})')
            ax.legend(fontsize=8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Score')
    ax.grid(True, alpha=0.3)

    # --- Reward ---
    ax = axes[0, 1]
    rewards = history.get('reward', [])
    if rewards:
        _plot_with_resume(ax, episodes[:len(rewards)], rewards, 'green', resume_episode)
        window = max(1, len(rewards) // 20)
        if window > 1:
            avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(episodes[window - 1:len(rewards)], avg, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Cumulative Reward')
    ax.grid(True, alpha=0.3)

    # --- Epsilon ---
    ax = axes[1, 0]
    eps = history.get('epsilon', [])
    if eps:
        _plot_with_resume(ax, episodes[:len(eps)], eps, 'red', resume_episode)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.grid(True, alpha=0.3)

    # --- Action Count ---
    ax = axes[1, 1]
    for key, color in [('no_action', 'gray'), ('repair_value', 'blue'),
                        ('delete', 'red'), ('replace_nearby', 'orange')]:
        vals = history.get(key, [])
        if vals:
            ax.plot(episodes[:len(vals)], vals, label=key, alpha=0.6, color=color)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Action Distribution (Count)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Action Ratio (stacked area chart) ---
    ax = axes[2, 0]
    action_keys = ['no_action', 'repair_value', 'delete', 'replace_nearby']
    action_colors = ['gray', 'blue', 'red', 'orange']
    action_arrays = []
    for key in action_keys:
        vals = history.get(key, [])
        action_arrays.append(np.array(vals) if vals else np.zeros(len(episodes)))
    # Compute ratios
    total = np.sum(action_arrays, axis=0)
    total = np.where(total > 0, total, 1)  # avoid divide-by-zero
    ratios = [arr / total for arr in action_arrays]
    if len(ratios[0]) > 0:
        ax.stackplot(episodes[:len(ratios[0])], *ratios,
                     labels=action_keys, colors=action_colors, alpha=0.7)
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Ratio')
    ax.set_title('Action Distribution (Ratio)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- Q-Value Uncertainty ---
    ax = axes[2, 1]
    q_std = history.get('q_std', [])
    if q_std and any(v > 0 for v in q_std):
        _plot_with_resume(ax, episodes[:len(q_std)], q_std, 'purple', resume_episode, label='Q-Std')
        # Action probability entropy
        action_probs = history.get('action_probs', [])
        if action_probs:
            entropies = []
            for probs in action_probs:
                if isinstance(probs, (list, tuple)) and len(probs) > 0:
                    p = np.array(probs)
                    p = np.clip(p, 1e-10, 1.0)
                    entropies.append(-np.sum(p * np.log(p)))
                else:
                    entropies.append(0.0)
            if entropies:
                ax2 = ax.twinx()
                ax2.plot(episodes[:len(entropies)], entropies,
                         alpha=0.5, color='darkcyan', label='Entropy')
                ax2.set_ylabel('Action Entropy', fontsize=8)
                ax2.legend(fontsize=7, loc='upper left')
        ax.legend(fontsize=7, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No Q-value data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-Value Std')
    ax.set_title('Q-Value Uncertainty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{version_name}_training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [viz] training curves saved: {path}")


# ============================================================================
# Visualization: scatter plots (PCA for classification, actual-vs-predicted for regression)
# ============================================================================
def compute_auth_div(X_result, X_clean, X_dirty):
    """
    Compute Authenticity and Diversity.

    Ref: real_beers_experiment_with_detector.py:692-722

    Args:
        X_result: data after policy processing (row count may be < X_clean)
        X_clean: clean data (full)
        X_dirty: dirty data (full)

    Returns:
        (auth, div) — authenticity [0,1] and diversity [0,1]
    """
    n = len(X_result)
    n_total = len(X_clean)
    if n == 0:
        return 0.0, 0.0

    # Authenticity: fraction of rows matching clean data exactly
    compare_len = min(n, n_total)
    correct = 0
    for i in range(compare_len):
        if np.allclose(X_result[i], X_clean[i], atol=0.01, equal_nan=False):
            correct += 1
    auth = correct / n if n > 0 else 0.0

    # Diversity: sample retention rate x noise retention rate
    sample_ret = n / n_total
    X_dirty_valid = X_dirty[~np.isnan(X_dirty).any(axis=1)]
    if len(X_result) > 1 and len(X_dirty_valid) > 1:
        result_var = np.mean(np.var(X_result, axis=0))
        clean_var = np.mean(np.var(X_clean, axis=0))
        dirty_var = np.mean(np.var(X_dirty_valid, axis=0))
        if dirty_var > clean_var + 1e-6:
            noise_ret = np.clip((result_var - clean_var) / (dirty_var - clean_var), 0, 1)
        else:
            noise_ret = 1.0
        div = sample_ret * noise_ret
    else:
        div = 0.0

    return round(auth, 4), round(div, 4)


def _safe_nan(X):
    """NaN-safe: fill NaN with column means."""
    X_c = X.copy()
    col_means = np.nanmean(X_c, axis=0)
    for j in range(X_c.shape[1]):
        mask = np.isnan(X_c[:, j])
        X_c[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    return X_c



# ============================================================================
# 5 baseline evaluations + visualization
# ============================================================================
def _knn_label_estimate(X, y, idx, deleted_rows, task_type, k=5):
    """KNN label estimate (simplified from CleaningEnv._get_majority_label)."""
    X_filled = X.copy()
    col_means = np.nanmean(X_filled, axis=0)
    for j in range(X_filled.shape[1]):
        mask = np.isnan(X_filled[:, j])
        X_filled[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    target = X_filled[idx]
    distances = np.linalg.norm(X_filled - target, axis=1)
    distances[idx] = np.inf
    for d_idx in deleted_rows:
        if d_idx < len(distances):
            distances[d_idx] = np.inf

    k = min(k, int((distances < np.inf).sum()))
    if k == 0:
        return y[idx]

    nearest = np.argsort(distances)[:k]
    labels = y[nearest]
    valid = labels[~np.isnan(labels)]
    if len(valid) == 0:
        return y[idx]

    if task_type == 'regression':
        dists = distances[nearest][~np.isnan(labels)]
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum()
        return float(np.average(valid, weights=weights))

    unique, counts = np.unique(valid, return_counts=True)
    return unique[np.argmax(counts)]


def evaluate_and_visualize(
    X_dirty, y_dirty,
    X_clean, y_clean,
    X_result, y_result,
    task_type: str,
    save_dir: str,
    version_name: str,
    detected_errors: dict = None,
    gt_cost: int = 0,
    oracle_test_X: np.ndarray = None,
    oracle_test_y: np.ndarray = None,
    value_estimator=None,
):
    """
    Evaluate 6 baselines and plot a comparison figure:
      NoFix, DeleteAll, DeleteFix, ReplaceAll, DemandClean, FullFix

    Args:
        gt_cost: ground-truth cost used by DemandClean (ground_truth_used)

    Returns:
        (baseline_results, vis_time) — metric dict + visualization time (s)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    def safe_prepare(X, y, name):
        """NaN-safe: drop rows with NaN in X or y; fill remaining NaN with column means."""
        X_c = X.copy()
        y_c = y.copy() if len(y) == len(X) else y[:len(X)].copy()
        # Align lengths
        min_len = min(len(X_c), len(y_c))
        X_c = X_c[:min_len]
        y_c = y_c[:min_len]
        # Drop NaN in y
        y_valid = ~np.isnan(y_c)
        X_c = X_c[y_valid]
        y_c = y_c[y_valid]
        # Drop all-NaN rows in X
        x_valid = ~np.isnan(X_c).all(axis=1)
        X_c = X_c[x_valid]
        y_c = y_c[x_valid]
        # Fill remaining NaN with column means
        col_means = np.nanmean(X_c, axis=0)
        for j in range(X_c.shape[1]):
            mask = np.isnan(X_c[:, j])
            X_c[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
        return X_c, y_c

    # --- Prepare data for each baseline ---

    # NoFix: use dirty data directly
    X_nofix, y_nofix = safe_prepare(X_dirty, y_dirty, 'NoFix')

    # FullFix: use clean data directly
    X_fullfix, y_fullfix = safe_prepare(X_clean, y_clean, 'FullFix')

    # DeleteAll: drop rows containing any NaN
    nan_mask = ~np.isnan(X_dirty).any(axis=1)
    X_delete = X_dirty[nan_mask].copy()
    y_delete = y_dirty[nan_mask].copy()
    X_delete, y_delete = safe_prepare(X_delete, y_delete, 'DeleteAll')

    # ReplaceAll: replace every detected error cell with VE-estimated value
    X_replace = X_dirty.copy()
    y_replace = y_dirty.copy()
    if value_estimator is not None and detected_errors is not None:
        t_ve = time.time()
        col_means = np.nanmean(X_replace, axis=0)
        deleted_rows = set()
        ve_count = 0
        for err_type_name in ('missing', 'semantic', 'syntactic', 'label_noise'):
            for item in detected_errors.get(err_type_name, []):
                if isinstance(item, (list, tuple)):
                    idx_err = int(item[0])
                    col_err = int(item[1]) if len(item) > 1 else -1
                elif isinstance(item, dict):
                    idx_err = int(item['idx'])
                    col_err = int(item.get('col', -1))
                else:
                    continue
                if col_err == -1:
                    # Label error: KNN majority vote
                    y_replace[idx_err] = _knn_label_estimate(
                        X_replace, y_replace, idx_err, deleted_rows, task_type)
                else:
                    if 0 <= idx_err < X_replace.shape[0] and 0 <= col_err < X_replace.shape[1]:
                        X_replace[idx_err, col_err] = value_estimator.estimate_feature_value(
                            X_replace, idx_err, col_err, deleted_rows, col_means)
                ve_count += 1
        ve_time = time.time() - t_ve
        print(f"  ReplaceAll VE estimation: {ve_count} errors, {ve_time:.1f}s")
    X_replace, y_replace = safe_prepare(X_replace, y_replace, 'ReplaceAll')

    # DemandClean: result from our system
    X_dc, y_dc = safe_prepare(X_result, y_result, 'DemandClean')

    # DeleteFix: drop rows flagged by the detector (keep at least 20% of data)
    X_dfix, y_dfix = None, None
    if detected_errors is not None:
        # Collect indices of all rows detected as having errors
        error_row_set = set()
        for err_type in ('missing', 'semantic', 'syntactic', 'label_noise'):
            for item in detected_errors.get(err_type, []):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    error_row_set.add(int(item[0]))
                elif isinstance(item, dict) and 'idx' in item:
                    error_row_set.add(int(item['idx']))

        n_total = len(X_dirty)
        min_keep = max(10, int(n_total * 0.2))  # keep at least 20%

        if len(error_row_set) <= n_total - min_keep:
            # Drop every detected error row
            keep_mask = np.array([i not in error_row_set for i in range(n_total)])
        else:
            # Too many to drop; prioritize label noise and missing values (higher confidence)
            from collections import Counter
            row_error_count = Counter()
            for err_type in ('label_noise', 'missing', 'syntactic', 'semantic'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        row_error_count[int(item[0])] += 1
                    elif isinstance(item, dict) and 'idx' in item:
                        row_error_count[int(item['idx'])] += 1
            # Sort rows by descending error count; take first (n_total - min_keep)
            max_delete = n_total - min_keep
            sorted_rows = sorted(row_error_count.keys(),
                                 key=lambda r: row_error_count[r], reverse=True)
            delete_set = set(sorted_rows[:max_delete])
            keep_mask = np.array([i not in delete_set for i in range(n_total)])

        X_dfix_raw = X_dirty[keep_mask].copy()
        y_dfix_raw = y_dirty[keep_mask].copy()
        X_dfix, y_dfix = safe_prepare(X_dfix_raw, y_dfix_raw, 'DeleteFix')
        print(f"  DeleteFix: dropped {n_total - keep_mask.sum()} rows, kept {keep_mask.sum()} rows")

    datasets = {
        'NoFix': (X_nofix, y_nofix),
        'DeleteAll': (X_delete, y_delete),
    }
    if X_dfix is not None and len(X_dfix) >= 10:
        datasets['DeleteFix'] = (X_dfix, y_dfix)
    datasets['ReplaceAll'] = (X_replace, y_replace)
    datasets['DemandClean'] = (X_dc, y_dc)
    datasets['FullFix'] = (X_fullfix, y_fullfix)

    # --- Unified test set (fixed split from clean data) ---
    # Oracle mode: use externally supplied test set; default: 80/20 split from clean data
    if oracle_test_X is not None and oracle_test_y is not None:
        X_test_common, y_test_common = safe_prepare(
            oracle_test_X, oracle_test_y, 'OracleTest')
        print(f"  Unified test set (Oracle): {len(X_test_common)} rows")
        # In Oracle mode every baseline trains on all passed-in data (no internal split)
        n_total_rows = len(X_clean)
        train_indices = np.arange(n_total_rows)
        test_indices = None  # unused
    else:
        # All baselines share the same test set for fair comparison
        n_total_rows = len(X_clean)
        all_indices = np.arange(n_total_rows)
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        # Common test set (clean data)
        X_test_common_raw = X_clean[test_indices].copy()
        y_test_common_raw = y_clean[test_indices].copy()
        X_test_common, y_test_common = safe_prepare(
            X_test_common_raw, y_test_common_raw, 'CommonTest')
        print(f"  Unified test set: {len(X_test_common)} rows (fixed split from {n_total_rows} clean rows)")

    # Select evaluation models
    if task_type == 'classification':
        eval_models = EVAL_MODELS_CLASSIFICATION
    elif task_type == 'clustering':
        eval_models = EVAL_MODELS_CLUSTERING
    else:
        eval_models = EVAL_MODELS_REGRESSION

    results_all = {}

    print(f"\n{'='*60}")
    print(f"Baseline comparison - {version_name}")
    print(f"{'='*60}")

    for ds_name, (X_ds, y_ds) in datasets.items():
        t_bl = time.time()
        if len(X_ds) < 10:
            print(f"  {ds_name}: too few rows ({len(X_ds)}), skipped")
            results_all[ds_name] = {}
            continue

        ds_results = {'n_samples': len(X_ds), 'n_test': len(y_test_common)}

        try:
            # Build training data: Oracle mode uses full input (already split externally); otherwise use train_indices subset
            if oracle_test_X is not None:
                # Oracle mode: inputs are already the train subset
                X_train_raw = X_ds
                y_train_raw = y_ds
            elif len(X_ds) == n_total_rows:
                X_train_raw = X_ds[train_indices]
                y_train_raw = y_ds[train_indices]
            else:
                # DeleteAll/DeleteFix have different row counts; train on all
                X_train_raw = X_ds
                y_train_raw = y_ds

            X_train, y_train = safe_prepare(X_train_raw, y_train_raw, ds_name)

            if len(X_train) < 10:
                print(f"  {ds_name}: too few training rows ({len(X_train)}), skipped")
                results_all[ds_name] = {}
                continue

            # Data already went through LE+SS in preprocess_data; use as-is.
            # No second StandardScaler (avoids double standardization drifting from Step 9).
            X_train_scaled = X_train
            X_test_scaled = X_test_common

            if task_type == 'clustering':
                # Clustering: fit_predict on all data, no train/test split
                from sklearn.cluster import KMeans as _KMeans
                from sklearn.metrics import silhouette_score as _sil_score, adjusted_rand_score as _ari_score
                # Clustering uses full (already standardized) data without reprocessing
                X_clust_prepared, _ = safe_prepare(X_ds, y_ds, ds_name + '_clust')
                n_clusters = len(np.unique(y_ds[~np.isnan(y_ds)]))
                n_rows = len(X_clust_prepared)
                sil_sample_size = min(n_rows, 10000)
                clusterers = {
                    'kmeans': _KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                }
                y_aligned = y_ds[:len(X_clust_prepared)]
                y_aligned = y_aligned[~np.isnan(y_aligned)]
                for model_name in eval_models:
                    try:
                        clust = clusterers.get(model_name)
                        if clust is None:
                            continue
                        y_pred = clust.fit_predict(X_clust_prepared)
                        sil = _sil_score(X_clust_prepared, y_pred, sample_size=sil_sample_size, random_state=42)
                        ari = _ari_score(y_aligned[:len(y_pred)], y_pred)
                        ds_results[f'{model_name}_silhouette'] = round(sil, 4)
                        ds_results[f'{model_name}_ari'] = round(ari, 4)
                    except Exception as e:
                        ds_results[f'{model_name}_error'] = str(e)
            elif task_type == 'classification':
                from demandclean.tools.tolerance_analysis import get_classifier
                for model_name in eval_models:
                    try:
                        model = get_classifier(model_name)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        acc = accuracy_score(y_test_common, y_pred)
                        f1 = f1_score(y_test_common, y_pred, average='weighted', zero_division=0)
                        ds_results[f'{model_name}_accuracy'] = round(acc, 4)
                        ds_results[f'{model_name}_f1'] = round(f1, 4)
                    except Exception as e:
                        ds_results[f'{model_name}_error'] = str(e)
            else:
                from demandclean.tools.tolerance_analysis import get_regressor
                for model_name in eval_models:
                    try:
                        model = get_regressor(model_name)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        mse = mean_squared_error(y_test_common, y_pred)
                        r2 = r2_score(y_test_common, y_pred)
                        ds_results[f'{model_name}_mse'] = round(mse, 4)
                        ds_results[f'{model_name}_r2'] = round(r2, 4)
                    except Exception as e:
                        ds_results[f'{model_name}_error'] = str(e)

            # (visualization is done once after the evaluation loop)

        except Exception as e:
            ds_results['error'] = str(e)

        results_all[ds_name] = ds_results

        # Print main metrics (with elapsed time)
        bl_time = time.time() - t_bl
        metric_str = ' | '.join(
            f'{k}={v}' for k, v in ds_results.items()
            if not k.startswith('n_') and 'error' not in k
        )
        print(f"  {ds_name:15s}: {metric_str} | {bl_time:.1f}s")

    # --- Compute Auth/Div/Cost and attach to each baseline's results ---
    n_total = len(X_clean)
    # Cost per baseline
    n_detected = 0
    if detected_errors is not None:
        err_set = set()
        for err_type in ('missing', 'semantic', 'syntactic', 'label_noise'):
            for item in detected_errors.get(err_type, []):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    err_set.add(int(item[0]))
                elif isinstance(item, dict) and 'idx' in item:
                    err_set.add(int(item['idx']))
        n_detected = len(err_set)

    baseline_costs = {
        'NoFix': 0,
        'DeleteAll': 0,
        'DeleteFix': n_detected,
        'ReplaceAll': 0,
        'DemandClean': gt_cost,
        'FullFix': n_total,
    }

    for ds_name, (X_ds, y_ds) in datasets.items():
        if ds_name not in results_all or len(X_ds) < 10:
            continue
        try:
            auth, div = compute_auth_div(X_ds, X_clean, X_dirty)
            results_all[ds_name]['auth'] = auth
            results_all[ds_name]['div'] = div
            results_all[ds_name]['cost'] = baseline_costs.get(ds_name, 0)
        except Exception:
            results_all[ds_name]['auth'] = 0.0
            results_all[ds_name]['div'] = 0.0
            results_all[ds_name]['cost'] = baseline_costs.get(ds_name, 0)

    vis_time = 0.0

    return results_all, vis_time


# ============================================================================
# Baseline comparison bar chart
# ============================================================================
def plot_baseline_comparison(results_all: dict, save_dir: str,
                              version_name: str, task_type: str,
                              primary_model: str):
    """
    Bar chart comparing the main metric across baselines.

    Args:
        results_all: return value of evaluate_and_visualize
        save_dir: output directory
        version_name: version name
        task_type: task type
        primary_model: short model name (e.g. 'rf')
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if task_type == 'classification':
        metric_key = f'{primary_model}_accuracy'
        ylabel = 'Accuracy'
    else:
        metric_key = f'{primary_model}_r2'
        ylabel = 'R2 Score'

    names = []
    values = []
    color_map = {
        'NoFix': '#999999',
        'DeleteAll': '#ff7f7f',
        'DeleteFix': '#ff4444',
        'ReplaceAll': '#ffcc66',
        'DemandClean': '#66b3ff',
        'FullFix': '#99ff99',
    }
    colors = []

    for name in ['NoFix', 'DeleteAll', 'DeleteFix', 'ReplaceAll', 'DemandClean', 'FullFix']:
        if name in results_all and metric_key in results_all[name]:
            names.append(name)
            values.append(results_all[name][metric_key])
            colors.append(color_map.get(name, '#cccccc'))

    if not names:
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{version_name} - Baseline Comparison ({primary_model.upper()})', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{version_name}_baseline_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [viz] baseline comparison saved: {path}")


# ============================================================================
# Inverse transform: numpy -> original CSV format
# ============================================================================
def _inverse_transform_to_csv(
    X_result: np.ndarray,
    y_result: np.ndarray,
    feature_cols: list,
    label_col: str,
    scaler,
    label_encoders: dict,
    categorical_cols: set,
    dirty_df: pd.DataFrame,
    inference_mode: str = 'single_phase',
    keep_mask=None,
) -> pd.DataFrame:
    """
    Convert DemandClean's standardized numpy arrays back to a DataFrame in original CSV format.

    Strategy:
      1. Start from original dirty_df (keeping index column and raw formatting)
      2. Filter rows by keep_mask
      3. scaler.inverse_transform on standardized X_result
      4. LabelEncoder.inverse_transform on encoded categorical columns
      5. Inverse-transform the label column as well

    Args:
        X_result: cleaned standardized feature matrix (n_kept, n_features)
        y_result: cleaned label vector (n_kept,)
        feature_cols: feature column names
        label_col: label column name
        scaler: StandardScaler
        label_encoders: {col_name: LabelEncoder}
        categorical_cols: set of categorical column names
        dirty_df: original dirty DataFrame (with index column)
        inference_mode: inference mode
        keep_mask: boolean array marking which original rows are kept

    Returns:
        Cleaned DataFrame in original format (columns match dirty_df, includes index)
    """
    n_result = len(X_result)

    # Edge case: agent deleted all rows -> return empty DataFrame preserving dirty_df columns
    if n_result == 0:
        return pd.DataFrame(columns=dirty_df.columns)

    # Step 1: Inverse standardize
    X_inv = scaler.inverse_transform(X_result)

    # Step 2: Build the inverse-transformed DataFrame
    result_df = pd.DataFrame(X_inv, columns=feature_cols)

    # Step 3: Inverse-encode categorical columns
    if label_encoders and categorical_cols:
        for col in feature_cols:
            if col in categorical_cols and col in label_encoders:
                le = label_encoders[col]
                col_vals = result_df[col].values
                n_classes = len(le.classes_)

                if n_classes > 0 and scaler is not None:
                    # Use scaler info to compute the standardized value of each valid integer code
                    col_idx = feature_cols.index(col)
                    mean_val = scaler.mean_[col_idx]
                    scale_val = scaler.scale_[col_idx]
                    # Valid codes [0, 1, ..., n_classes-1] in standardized space
                    valid_scaled = np.array([(i - mean_val) / scale_val for i in range(n_classes)])

                    # For each value pick the nearest valid code in standardized space
                    col_rounded = np.zeros(len(col_vals), dtype=int)
                    for i, v in enumerate(col_vals):
                        nearest_idx = np.argmin(np.abs(valid_scaled - v))
                        col_rounded[i] = nearest_idx
                else:
                    # fallback: plain round + clip
                    col_rounded = np.round(col_vals).astype(int)
                    col_rounded = np.clip(col_rounded, 0, n_classes - 1)

                try:
                    result_df[col] = le.inverse_transform(col_rounded)
                except Exception:
                    # If inverse-transform fails, try edit-distance safe encoding
                    if _find_nearest_known is not None:
                        safe_vals = []
                        known = list(le.classes_)
                        for v in col_vals:
                            nearest = _find_nearest_known(str(v), known)
                            safe_vals.append(nearest if nearest else known[0])
                        result_df[col] = safe_vals
                    # Otherwise keep the numeric values

    # Step 4: Inverse-encode the label column
    y_inv = y_result[:n_result].copy()
    if label_col in label_encoders:
        le = label_encoders[label_col]
        n_classes = len(le.classes_)

        if n_classes > 0 and scaler is not None and label_col in feature_cols:
            # Label column may have been scaled (if part of features); nearest-neighbor match
            col_idx = feature_cols.index(label_col)
            mean_val = scaler.mean_[col_idx]
            scale_val = scaler.scale_[col_idx]
            valid_scaled = np.array([(i - mean_val) / scale_val for i in range(n_classes)])
            y_rounded = np.zeros(len(y_inv), dtype=int)
            for i, v in enumerate(y_inv):
                nearest_idx = np.argmin(np.abs(valid_scaled - v))
                y_rounded[i] = nearest_idx
        else:
            # Label column is usually not scaled; plain round + clip
            y_rounded = np.round(y_inv).astype(int)
            y_rounded = np.clip(y_rounded, 0, n_classes - 1)

        try:
            y_inv = le.inverse_transform(y_rounded)
        except Exception:
            pass
    result_df[label_col] = y_inv

    # Step 5: Restore the index column
    if 'index' in dirty_df.columns:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            # keep_mask matches rows in original dirty_df
            kept_indices = dirty_df.loc[keep_mask, 'index'].values[:n_result]
        else:
            # Default: take the first n_result indices
            kept_indices = dirty_df['index'].values[:n_result]
        result_df.insert(0, 'index', kept_indices)

    # Step 5.5: Restore non-feature columns dropped by preprocess_data (e.g. 'id').
    #   Columns present in dirty_df but not in feature_cols / label_col / 'index'
    #   are copied back from the corresponding rows of dirty_df.
    extra_cols = [c for c in dirty_df.columns
                  if c not in result_df.columns and c not in ('index',)]
    if extra_cols:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            src = dirty_df.loc[keep_mask].iloc[:n_result]
        else:
            src = dirty_df.iloc[:n_result]
        for col in extra_cols:
            result_df[col] = src[col].values

    # Step 6: Match column order to dirty_df
    target_cols = [c for c in dirty_df.columns if c in result_df.columns]
    result_df = result_df[target_cols]

    return result_df


# ============================================================================
# run_version(): run the full pipeline for a single version
# ============================================================================
def run_version(
    dataset_name: str,
    version_cfg: dict,
    X_dirty: np.ndarray,
    y_dirty: np.ndarray,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    column_names: list,
    fd_rules: list,
    rules_path: str,
    dirty_csv_path: str = '',
    clean_csv_path: str = '',
    csv_columns: list = None,
    n_episodes: int = 300,
    verbose: bool = True,
    data_scaler=None,
    label_encoders: dict = None,
    categorical_cols: set = None,
    dirty_df: pd.DataFrame = None,
    clean_df: pd.DataFrame = None,
    inject_kwargs: dict = None,
    resume_mode: str = 'auto',
    apply_raha_truth: bool = True,
    count_raha_cost: bool = True,
    visualize_only: bool = False,
    oracle_split: dict = None,
    min_repair_ratio: float = None,
    max_repair_ratio: float = None,
    repair_sensitivity: float = None,
    max_truth_budget: int = None,
    repair_lambda: float = None,
    reward_model_type: str = None,
    disable_raha: bool = False,
    delete_shaping_reward: float = None,
    keep_rate_weight: float = None,
    inference_only: bool = False,
    output_suffix: str = '',
) -> dict:
    """
    Run the full pipeline for a single version.

    Pipeline:
      1. Create DemandClean instance
      2. Train (dc.fit)
      3. Inference (dc.clean or dc.plan/execute)
      4. Evaluate 5 baselines
      5. Model tolerance (compute_model_tolerance)
      6. Detector accuracy (evaluate_detector_accuracy)
      7. Three-dimensional Shapley analysis (run_full_shapley_analysis)
      8. Training-curve visualization
      9. Save outputs to results/demandclean/{dataset}/{version_name}/

    Args:
        dataset_name: dataset name
        version_cfg: version config dict
        X_dirty, y_dirty: dirty data
        X_clean, y_clean: clean data
        column_names: feature column names
        fd_rules: FD rules list
        rules_path: rules file path
        n_episodes: number of training episodes
        verbose: whether to print detailed info

    Returns:
        run result dict for this version
    """
    version_name = version_cfg['version_name']
    # Output-dir suffix (e.g. NGT variant uses _ngt to avoid overwriting the base version)
    output_version_name = version_name + output_suffix if output_suffix else version_name
    detector_mode = version_cfg['detector_mode']
    agent_type = version_cfg['agent_type']
    inference_mode = version_cfg['inference_mode']

    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']
    model_type = ds_cfg['model_type']

    # --- Create output directories (categorized subdirs) ---
    save_dir = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name, output_version_name)
    dir_model = os.path.join(save_dir, 'model')
    dir_data = os.path.join(save_dir, 'data')
    dir_eval = os.path.join(save_dir, 'evaluation')
    dir_vis = os.path.join(save_dir, 'visualization')
    dir_report = os.path.join(save_dir, 'report')
    for d in [save_dir, dir_model, dir_data, dir_eval, dir_vis, dir_report]:
        os.makedirs(d, exist_ok=True)

    start_time = time.time()
    step_times = {}  # per-step timing

    print("\n" + "=" * 70)
    print(f"Version: {version_name}")
    print(f"Dataset: {dataset_name} | task: {task_type} | model: {model_type}")
    print(f"Detector: {detector_mode} | Agent: {agent_type} | inference: {inference_mode}")
    print(f"Episodes: {n_episodes} | output dir: {save_dir}")
    print("=" * 70)

    # Select evaluation models
    if task_type == 'classification':
        eval_models = EVAL_MODELS_CLASSIFICATION
    elif task_type == 'clustering':
        eval_models = EVAL_MODELS_CLUSTERING
    else:
        eval_models = EVAL_MODELS_REGRESSION

    report = {
        'version_name': version_name,
        'dataset': dataset_name,
        'task_type': task_type,
        'model_type': model_type,
        'detector_mode': detector_mode,
        'agent_type': agent_type,
        'inference_mode': inference_mode,
        'n_episodes': n_episodes,
        'data_shape': list(X_dirty.shape),
    }

    # ================================================================
    # visualize_only mode: skip training/inference; load cleaned results from disk
    # ================================================================
    history_path = os.path.join(dir_report, f'{version_name}_history.json')
    prev_history = None

    if visualize_only:
        print(f"\n[visualize_only] skipping train/inference, loading existing results...")

        # Load cleaned data
        cleaned_csv = os.path.join(dir_data, f'{version_name}_cleaned.csv')
        if not os.path.exists(cleaned_csv):
            print(f"  [error] cleaned result not found: {cleaned_csv}; cannot visualize only")
            return report

        cleaned_df = pd.read_csv(cleaned_csv)
        X_result = cleaned_df[column_names].values
        y_result = cleaned_df[ds_cfg['label_col']].values if ds_cfg['label_col'] in cleaned_df.columns else y_dirty[:len(cleaned_df)]
        print(f"  loaded cleaned result: {cleaned_csv} ({len(X_result)} rows)")

        # Load training history
        history = None
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    history = json.load(f)
                print(f"  loaded training history: {len(history.get('score', []))} episodes")
            except Exception as e:
                print(f"  [warn] failed to load training history: {e}")

        # Load previous report
        report_path = os.path.join(dir_report, f'{version_name}_report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    old_report = json.load(f)
                # Preserve training-related fields from the old report
                for key in ['action_counts', 'ground_truth_used', 'repair_log',
                            'detector_accuracy', 'raha_cost']:
                    if key in old_report:
                        report[key] = old_report[key]
                print(f"  loaded prior report: {report_path}")
            except Exception as e:
                print(f"  [warn] failed to load prior report: {e}")

        # Skip dc instantiation; set detected_errors and dc to None
        dc = None
        detected_errors = None
        output_csv_original = os.path.join(dir_data, f'{version_name}_cleaned_original.csv')
        if not os.path.exists(output_csv_original):
            output_csv_original = None

        # Try loading encoded version if saved from a previous run
        output_npz = os.path.join(dir_data, f'{version_name}_cleaned_encoded.npz')
        if not os.path.exists(output_npz):
            output_npz = None

        # Restore locals needed by downstream steps from the prior report
        ground_truth_used = report.get('ground_truth_used', 0)
        detector_accuracy = report.get('detector_accuracy', {})

    # ================================================================
    # Step 1-4: train, infer, save (skipped in visualize_only mode)
    # ================================================================
    if not visualize_only:

        # ================================================================
        # 1. Create DemandClean instance
        # ================================================================
        print("\n[Step 1] creating DemandClean instance...")
        t_step = time.time()

        # Lightweight reward-evaluation model params (baseline evaluation unaffected).
        # reward_model_type can override the model type used for reward evaluation only.
        effective_reward_model_type = reward_model_type or model_type
        reward_model_kwargs = REWARD_MODEL_KWARGS.get(effective_reward_model_type, {})

        # Build extra config kwargs
        extra_kwargs = {}
        if min_repair_ratio is not None:
            extra_kwargs['min_repair_ratio'] = min_repair_ratio
        if max_repair_ratio is not None:
            extra_kwargs['max_repair_ratio'] = max_repair_ratio
        if repair_sensitivity is not None:
            extra_kwargs['repair_sensitivity'] = repair_sensitivity
        if max_truth_budget is not None:
            extra_kwargs['max_truth_budget'] = max_truth_budget
        if repair_lambda is not None:
            extra_kwargs['repair_lambda'] = repair_lambda
        if disable_raha:
            extra_kwargs['disable_raha'] = True

        # CLI override of reward params
        if delete_shaping_reward is not None:
            extra_kwargs['delete_shaping_reward'] = delete_shaping_reward
        if keep_rate_weight is not None:
            extra_kwargs['keep_rate_weight'] = keep_rate_weight

        # Regression tasks auto-enable tuned reward params (classification/clustering use defaults)
        if task_type == 'regression':
            extra_kwargs.setdefault('delete_shaping_reward', -0.05)
            extra_kwargs.setdefault('keep_rate_weight', 1.0)
            extra_kwargs.setdefault('regression_log_normalize', True)
        if oracle_split is not None:
            extra_kwargs['use_clean_validation'] = True

        dc = DemandClean(
            task_type=task_type,
            model_type=effective_reward_model_type,
            agent_type=agent_type,
            detector_mode=detector_mode,
            inference_mode=inference_mode,
            n_episodes=n_episodes,
            rules_path=rules_path,
            fd_rules=fd_rules,
            column_names=column_names,
            dirty_csv_path=dirty_csv_path,
            clean_csv_path=clean_csv_path,
            csv_columns=csv_columns,
            label_col=ds_cfg['label_col'],
            save_path=save_dir,
            apply_raha_truth=apply_raha_truth,
            count_raha_cost=count_raha_cost,
            # Reward evaluation config
            model_kwargs=reward_model_kwargs,
            eval_sample_ratio=1.0,
            # Encoding tools (used by ErrorInjector when injecting in CSV space)
            encoding_label_encoders=label_encoders,
            encoding_scaler=data_scaler,
            encoding_categorical_cols=categorical_cols,
            encoding_dirty_df=dirty_df,
            encoding_clean_df=clean_df,
            **(inject_kwargs or {}),
            **extra_kwargs,
        )
        step_times['step1_init'] = round(time.time() - t_step, 2)
        print(f"  [time] Step 1 (init): {step_times['step1_init']:.2f}s")

        # ================================================================
        # 2. Train (supports resume; inference_only skips training)
        # ================================================================
        model_path = os.path.join(dir_model, f'{version_name}_agent.pt')
        history_path = os.path.join(dir_report, f'{version_name}_history.json')

        t_step = time.time()

        if inference_only:
            # inference_only: skip training, load existing model
            if not ModelIO.agent_model_exists(model_path):
                raise FileNotFoundError(
                    f"inference_only mode but model not found: {model_path}")
            dc.load(model_path)
            print(f"\n[Step 2] training skipped (inference_only); loaded model: {model_path}")
            step_times['step2_train'] = 0.0
        else:
            resume_from = None
            prev_history = None

            if resume_mode == 'auto' and ModelIO.agent_model_exists(model_path):
                # Auto-detected existing model; resume
                resume_from = model_path
                print(f"\n[Step 2] resuming (existing model detected: {model_path})...")
                if os.path.exists(history_path):
                    try:
                        with open(history_path) as f:
                            prev_history = json.load(f)
                        prev_eps = len(prev_history.get('score', []))
                        print(f"  existing history: {prev_eps} episodes")
                    except Exception as e:
                        print(f"  [warn] failed to load history: {e}")
                        prev_history = None
                print(f"  adding {n_episodes} episodes")
            else:
                print(f"\n[Step 2] training ({n_episodes} episodes)...")

            dc.fit(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean,
                   n_episodes=n_episodes, verbose=verbose,
                   resume_from=resume_from, prev_history=prev_history,
                   X_clean_val=oracle_split['X_clean_val'] if oracle_split else None,
                   y_clean_val=oracle_split['y_clean_val'] if oracle_split else None)

            # Save model
            try:
                dc.save(model_path)
                print(f"  model saved: {model_path}")
            except Exception as e:
                print(f"  [warn] model save failed: {e}")

            # Save training history
            history = dc.get_training_history()
            if history and history.get('score'):
                history_file = os.path.join(dir_report, f'{version_name}_history.json')
                try:
                    # action_probs is list[list]; needs special handling
                    serializable_history = {}
                    for k, vals in history.items():
                        if k == 'action_probs':
                            serializable_history[k] = vals  # already serializable
                        else:
                            serializable_history[k] = [float(v) for v in vals]
                    with open(history_file, 'w') as f:
                        json.dump(serializable_history, f, indent=2)
                    print(f"  history saved: {history_file}")
                except Exception as e:
                    print(f"  [warn] history save failed: {e}")

            step_times['step2_train'] = round(time.time() - t_step, 2)
            print(f"  [time] Step 2 (train): {step_times['step2_train']:.2f}s")

        # ================================================================
        # 3. Inference
        # ================================================================
        print(f"\n[Step 3] inference (mode: {inference_mode})...")
        t_step = time.time()
        ground_truth_used = 0

        if inference_mode == 'single_phase':
            X_result, y_result, stats = dc.clean(
                X_dirty, y_dirty, X_clean, y_clean=y_clean, verbose=verbose
            )
            ground_truth_used = stats.get('truth_cost', 0)
            report['action_counts'] = stats.get('action_counts', {})
            report['keep_mask'] = stats.get('keep_mask', None)
            print(f"  ground truth used: {ground_truth_used}")
            print(f"  action counts: {stats.get('action_counts', {})}")

        elif inference_mode == 'two_phase':
            # Phase 1: produce repair plan
            repair_plan = dc.plan(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean, verbose=verbose)
            print(f"  repair plan: needs {len(repair_plan)} ground-truth values")

            # Prepare ground truth (supports feature columns and label column col=-1)
            true_values = {}
            for item in repair_plan:
                idx, col = item['idx'], item['col']
                if col == -1:
                    # Label noise: fetch ground truth from y_clean
                    if idx < len(y_clean):
                        true_values[(idx, col)] = y_clean[idx]
                elif idx < X_clean.shape[0] and col < X_clean.shape[1]:
                    true_values[(idx, col)] = X_clean[idx, col]

            # Phase 2: execute repair
            X_result, y_result, keep_mask = dc.execute(
                X_dirty, true_values, verbose=verbose, y_dirty=y_dirty
            )
            ground_truth_used = len(true_values)
            report['keep_mask'] = keep_mask
            print(f"  ground truth used: {ground_truth_used}")

        report['ground_truth_used'] = ground_truth_used
        report['cleaned_shape'] = list(X_result.shape)

        # Save cleaned data (standardized version + original-format version)
        output_csv = os.path.join(dir_data, f'{version_name}_cleaned.csv')
        output_csv_original = os.path.join(dir_data, f'{version_name}_cleaned_original.csv')

        try:
            # Standardized version (for internal analysis)
            cleaned_df = pd.DataFrame(X_result, columns=column_names)
            cleaned_df[ds_cfg['label_col']] = y_result[:len(cleaned_df)]
            cleaned_df.to_csv(output_csv, index=False)
            print(f"  cleaned data saved: {output_csv}")
        except Exception as e:
            print(f"  [warn] cleaned data save failed: {e}")

        # Produce original-format CSV (consumed by getScoreML unified evaluation)
        try:
            cleaned_original_df = _inverse_transform_to_csv(
                X_result=X_result,
                y_result=y_result,
                feature_cols=column_names,
                label_col=ds_cfg['label_col'],
                scaler=data_scaler,
                label_encoders=label_encoders,
                categorical_cols=categorical_cols,
                dirty_df=dirty_df,
                inference_mode=inference_mode,
                keep_mask=report.get('keep_mask', None),
            )
            cleaned_original_df.to_csv(output_csv_original, index=False)
            print(f"  cleaned (original format) saved: {output_csv_original}")
        except Exception as e:
            print(f"  [warn] original-format CSV generation failed: {e}")
            traceback.print_exc()
            output_csv_original = None

        # Save encoded version (used directly by Step 9 to avoid CSV roundtrip precision loss)
        output_npz = os.path.join(dir_data, f'{version_name}_cleaned_encoded.npz')
        try:
            np.savez(output_npz,
                     X_result=X_result,
                     y_result=y_result,
                     column_names=np.array(column_names),
                     label_col=np.array([ds_cfg['label_col']]))
            print(f"  encoded version saved: {output_npz}")
        except Exception as e:
            print(f"  [warn] encoded save failed: {e}")
            output_npz = None

        step_times['step3_infer'] = round(time.time() - t_step, 2)
        print(f"  [time] Step 3 (infer): {step_times['step3_infer']:.2f}s")

        # ================================================================
        # 4. Detector accuracy (moved before Baseline eval; needed by DeleteFix)
        # ================================================================
        print(f"\n[Step 4] detector accuracy evaluation...")
        t_step = time.time()
        detected_errors = None
        try:
            detected_errors = dc.detect_errors(X_dirty, X_clean, y_dirty=y_dirty, y_clean=y_clean, verbose=verbose)
            detector_accuracy = evaluate_detector_accuracy(
                detected_errors, X_dirty, X_clean, verbose=verbose,
                y_dirty=y_dirty, y_clean=y_clean,
            )
            report['detector_accuracy'] = detector_accuracy
        except Exception as e:
            print(f"  [error] detector accuracy evaluation failed: {e}")
            traceback.print_exc()
            detector_accuracy = {}

        step_times['step4_detect_eval'] = round(time.time() - t_step, 2)
        print(f"  [time] Step 4 (detector eval): {step_times['step4_detect_eval']:.2f}s")

    # ================================================================
    # 5. 6 baseline evaluations + decision-boundary comparison chart
    # ================================================================
    total_vis_time = 0.0  # cumulative visualization time (later deducted from elapsed_time)

    print(f"\n[Step 5] baseline evaluation and visualization...")
    t_step = time.time()
    try:
        # Oracle mode: pass train subset and test set explicitly
        if oracle_split is not None:
            eval_X_dirty = oracle_split['X_dirty_train']
            eval_y_dirty = oracle_split['y_dirty_train']
            eval_X_clean = oracle_split['X_clean_train']
            eval_y_clean = oracle_split['y_clean_train']
            eval_test_X = oracle_split['X_clean_test']
            eval_test_y = oracle_split['y_clean_test']
        else:
            eval_X_dirty = X_dirty
            eval_y_dirty = y_dirty
            eval_X_clean = X_clean
            eval_y_clean = y_clean
            eval_test_X = None
            eval_test_y = None

        # Construct ValueEstimator for the ReplaceAll baseline
        ve_for_baseline = None
        if dc is not None:
            try:
                from demandclean.core.environments.value_estimation import ValueEstimator
                ve_for_baseline = ValueEstimator(dc.config)
                print(f"  ReplaceAll VE: {ve_for_baseline.summary()}")
            except Exception as e:
                print(f"  [warn] ValueEstimator construction failed; ReplaceAll falls back to mean fill: {e}")

        baseline_results, step5_vis_time = evaluate_and_visualize(
            eval_X_dirty, eval_y_dirty, eval_X_clean, eval_y_clean,
            X_result, y_result,
            task_type, dir_vis, version_name,
            detected_errors=detected_errors,
            gt_cost=ground_truth_used,
            oracle_test_X=eval_test_X,
            oracle_test_y=eval_test_y,
            value_estimator=ve_for_baseline,
        )
        report['baseline_results'] = baseline_results
        total_vis_time += step5_vis_time

        # Baseline comparison bar chart (also counted as visualization time)
        vis_start = time.time()
        plot_baseline_comparison(
            baseline_results, dir_vis, version_name,
            task_type, eval_models[0],
        )
        total_vis_time += time.time() - vis_start
    except Exception as e:
        print(f"  [error] baseline evaluation failed: {e}")
        traceback.print_exc()
        baseline_results = {}

    step_times['step5_baseline'] = round(time.time() - t_step, 2)
    print(f"  [time] Step 5 (baseline eval): {step_times['step5_baseline']:.2f}s")

    # ================================================================
    # 6. Model tolerance (compute_model_tolerance)
    # ================================================================
    print(f"\n[Step 6] model tolerance evaluation...")
    t_step = time.time()
    try:
        tolerance_results = compute_model_tolerance(
            X_dirty=X_dirty,
            y_dirty=y_dirty,
            X_clean=X_clean,
            y_clean=y_clean,
            X_result=X_result,
            y_result=y_result,
            task_type=task_type,
            models_list=eval_models,
            save_dir=dir_vis,
            task_name=version_name,
        )
        report['tolerance_results'] = tolerance_results
    except Exception as e:
        print(f"  [error] tolerance evaluation failed: {e}")
        traceback.print_exc()
        tolerance_results = {}

    step_times['step6_tolerance'] = round(time.time() - t_step, 2)
    print(f"  [time] Step 6 (tolerance): {step_times['step6_tolerance']:.2f}s")

    # ================================================================
    # 7. Three-dimensional Shapley analysis (run_full_shapley_analysis)
    # ================================================================
    print(f"\n[Step 7] three-dimensional Shapley analysis...")
    vis_start = time.time()
    try:
        # Build error_list (converted from detection results)
        error_list = []
        if detected_errors is not None:
            for err_type_name, err_type_id in [('missing', 0), ('semantic', 1), ('syntactic', 2), ('label_noise', 3)]:
                for item in detected_errors.get(err_type_name, []):
                    if isinstance(item, (list, tuple)):
                        error_entry = {
                            'idx': item[0],
                            'col': item[1] if len(item) > 1 else -1,
                            'type': err_type_id,
                            'repair_value': item[2] if len(item) > 2 else 0.0,
                        }
                        error_list.append(error_entry)

        if error_list and dc is not None and dc.agent is not None:
            shapley_dir = os.path.join(dir_eval, 'shapley')
            shapley_results = run_full_shapley_analysis(
                agent=dc.agent,
                X_dirty=X_dirty,
                y=y_dirty,
                X_clean=X_clean,
                error_list=error_list,
                config=dc.config,
                output_dir=shapley_dir,
                column_names=column_names,
                verbose=verbose,
            )
            report['shapley_results'] = shapley_results
        else:
            print("  [skip] no error list available or agent not ready")
    except Exception as e:
        print(f"  [error] Shapley analysis failed: {e}")
        traceback.print_exc()
    _step7_elapsed = time.time() - vis_start
    total_vis_time += _step7_elapsed
    step_times['step7_shapley'] = round(_step7_elapsed, 2)
    print(f"  [time] Step 7 (Shapley): {step_times['step7_shapley']:.2f}s")

    # ================================================================
    # 8. Training-curve visualization
    # ================================================================
    print(f"\n[Step 8] training-curve visualization...")
    vis_start = time.time()
    try:
        # In inference_only mode, try to load training history from file
        if inference_only:
            history = None
            prev_history = None
            history_path = os.path.join(dir_report, f'{output_version_name}_history.json')
            if os.path.exists(history_path):
                try:
                    with open(history_path) as f:
                        history = json.load(f)
                except Exception:
                    pass
        if history and history.get('score'):
            # Compute resume start for the visualization marker
            resume_ep = len(prev_history.get('score', [])) if prev_history else 0
            plot_training_curves(history, dir_vis, version_name,
                                 resume_episode=resume_ep)
    except Exception as e:
        print(f"  [error] training-curve plot failed: {e}")
        traceback.print_exc()
    _step8_elapsed = time.time() - vis_start
    total_vis_time += _step8_elapsed
    step_times['step8_vis'] = round(_step8_elapsed, 2)
    print(f"  [time] Step 8 (training curves): {step_times['step8_vis']:.2f}s")

    # ================================================================
    # 9. getScoreML unified evaluation (matches run_baran_raha output format)
    # ================================================================
    print(f"\n[Step 9] Clean4MLBaseline unified evaluation...")
    t_step = time.time()
    getscoreml_results = {}
    try:
        # Ensure tools dir is on sys.path
        _tools_dir = os.path.join(_PROJECT_ROOT, 'tools')
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)

        from tools.getScoreML import run_all_evaluation

        if output_csv_original and os.path.exists(output_csv_original):
            # Pick the model list (matches raha_baran)
            if task_type == 'classification':
                ml_models = ['rf', 'lr', 'svm', 'knn', 'dt', 'gb']
            elif task_type == 'clustering':
                ml_models = ['kmeans', 'agglomerative', 'spectral']
            else:
                ml_models = ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb']

            # DemandClean method type:
            #   oracle + single_phase = Type 1 (fully automatic but uses all ground truth)
            #   oracle + two_phase    = Type 3 (iterative, user supplies ground truth)
            #   auto   + single_phase = Type 1 (fully automatic)
            #   auto   + two_phase    = Type 2 (needs validation set, partial ground truth)
            if detector_mode == 'oracle':
                method_type = 3 if inference_mode == 'two_phase' else 1
            else:
                method_type = 2 if inference_mode == 'two_phase' else 1

            # Compute mse_attributes (numeric columns)
            mse_attrs = [c for c in column_names if c not in (categorical_cols or set())]

            getscoreml_results = run_all_evaluation(
                dirty_path=dirty_csv_path,
                cleaned_path=output_csv_original,
                clean_path=clean_csv_path,
                cleaned_encoded_path=output_npz if output_npz and os.path.exists(output_npz) else None,
                encoded_arrays={
                    'X_dirty': X_dirty, 'y_dirty': y_dirty,
                    'X_clean': X_clean, 'y_clean': y_clean,
                    'X_clean_test': oracle_split['X_clean_test'] if oracle_split else None,
                    'y_clean_test': oracle_split['y_clean_test'] if oracle_split else None,
                } if X_clean is not None else None,
                output_path=dir_eval,
                task_name=version_name,
                label_column=ds_cfg['label_col'],
                task_type=task_type,
                models=ml_models,
                method_type=method_type,
                ground_truth_used=ground_truth_used,
                index_attribute='index',
                mse_attributes=mse_attrs,
                verbose=verbose,
            )

            # Ensure no None values: replace None with 0 or "N/A"
            for k, v in getscoreml_results.items():
                if v is None:
                    getscoreml_results[k] = 0.0

            report['getscoreml_results'] = getscoreml_results
            print(f"  unified evaluation done; results saved: {save_dir}/{version_name}_evaluation_results.txt")
        else:
            print(f"  [skip] original-format CSV unavailable; skip unified evaluation")
    except ImportError as e:
        print(f"  [warn] cannot import getScoreML module: {e}")
    except Exception as e:
        try:
            print(f"  [error] unified evaluation failed: {e}")
            traceback.print_exc()
        except (ValueError, IOError):
            # stdout may be closed by a child process
            sys.stderr.write(f"  [error] unified evaluation failed: {e}\n")
            traceback.print_exc(file=sys.stderr)

    step_times['step9_ml_eval'] = round(time.time() - t_step, 2)
    print(f"  [time] Step 9 (unified eval): {step_times['step9_ml_eval']:.2f}s")

    # ================================================================
    # 10. Ground-truth cost summary
    # ================================================================
    print(f"\n[Step 10] ground-truth cost summary...")
    t_step = time.time()
    try:
        # Fetch RAHA detection cost
        raha_cost_info = {}
        if dc is not None and hasattr(dc, 'detector') and hasattr(dc.detector, 'raha_cost_info'):
            raha_cost_info = dc.detector.raha_cost_info or {}
        elif 'raha_cost' in report:
            # visualize_only mode restores from old report
            raha_cost_info = report['raha_cost']

        raha_detection_cost = raha_cost_info.get('raha_total_cost', 0)
        if detector_mode == 'oracle':
            raha_detection_cost = 0  # no RAHA cost in oracle mode

        # Agent repair cost = ground_truth_used
        agent_repair_cost = ground_truth_used

        # Total cost
        total_cost = raha_detection_cost + agent_repair_cost if count_raha_cost else agent_repair_cost
        total_data_cells = X_dirty.shape[0] * X_dirty.shape[1]
        cost_ratio = total_cost / total_data_cells if total_data_cells > 0 else 0.0

        report['raha_cost'] = raha_cost_info
        report['ground_truth_cost_summary'] = {
            'raha_detection_cost': raha_detection_cost,
            'agent_repair_cost': agent_repair_cost,
            'total_cost': total_cost,
            'total_data_cells': total_data_cells,
            'cost_ratio': round(cost_ratio, 6),
            'apply_raha_truth': apply_raha_truth,
            'count_raha_cost': count_raha_cost,
        }

        print(f"  RAHA detection cost: {raha_detection_cost}")
        print(f"  Agent repair cost:   {agent_repair_cost}")
        print(f"  Total cost: {total_cost} / {total_data_cells} cells = {cost_ratio:.4%}")
    except Exception as e:
        print(f"  [warn] cost summary failed: {e}")
        traceback.print_exc()

    step_times['step10_cost'] = round(time.time() - t_step, 2)
    print(f"  [time] Step 10 (cost summary): {step_times['step10_cost']:.2f}s")

    # ================================================================
    # Save full report
    # ================================================================
    elapsed_time = time.time() - start_time - total_vis_time

    # Timing summary
    print(f"\n{'-'*50}")
    print(f"  Step timing summary")
    print(f"{'-'*50}")
    step_labels = {
        'step1_init': 'init',
        'step2_train': 'train',
        'step3_infer': 'infer',
        'step4_detect_eval': 'detector eval',
        'step5_baseline': 'baseline eval',
        'step6_tolerance': 'tolerance',
        'step7_shapley': 'Shapley',
        'step8_vis': 'training curves',
        'step9_ml_eval': 'unified eval',
        'step10_cost': 'cost summary',
    }
    for key, label in step_labels.items():
        t = step_times.get(key, 0)
        print(f"  Step {key.split('_')[0].replace('step','')} {label:<16s}| {t:>8.2f}s")
    print(f"{'-'*50}")
    print(f"  total (incl. viz)          | {time.time() - start_time:>8.2f}s")
    print(f"  visualization time         | {total_vis_time:>8.2f}s")
    print(f"  net (excl. viz)            | {elapsed_time:>8.2f}s")
    print(f"{'-'*50}")

    report['elapsed_time'] = round(elapsed_time, 2)
    report['vis_time'] = round(total_vis_time, 2)
    report['step_times'] = step_times

    report_file = os.path.join(dir_report, f'{version_name}_report.json')
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  [out] report saved: {report_file}")
    except Exception as e:
        print(f"  [warn] report save failed: {e}")

    # Text summary
    summary_file = os.path.join(dir_report, f'{version_name}_summary.txt')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"DemandClean result summary - {version_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Detector: {detector_mode}\n")
            f.write(f"Agent: {agent_type}\n")
            f.write(f"Inference mode: {inference_mode}\n")
            f.write(f"Task: {task_type} ({model_type})\n")
            f.write(f"Episodes: {n_episodes}\n")
            f.write(f"Elapsed: {elapsed_time:.2f}s\n")
            f.write(f"Visualization time: {total_vis_time:.2f}s\n")
            f.write(f"Shape: {X_dirty.shape} -> {X_result.shape}\n")
            f.write(f"Ground-truth cost: {ground_truth_used}\n\n")

            # Step timing
            if step_times:
                f.write("Step timing:\n")
                f.write("-" * 60 + "\n")
                _step_labels = {
                    'step1_init': 'init',
                    'step2_train': 'train',
                    'step3_infer': 'infer',
                    'step4_detect_eval': 'detector eval',
                    'step5_baseline': 'baseline eval',
                    'step6_tolerance': 'tolerance',
                    'step7_shapley': 'Shapley',
                    'step8_vis': 'training curves',
                    'step9_ml_eval': 'unified eval',
                    'step10_cost': 'cost summary',
                }
                for _key, _label in _step_labels.items():
                    _t = step_times.get(_key, 0)
                    f.write(f"  Step {_key.split('_')[0].replace('step','')} {_label}: {_t:.2f}s\n")
                f.write("\n")

            f.write("Baseline comparison:\n")
            f.write("-" * 60 + "\n")
            for name, res in baseline_results.items():
                f.write(f"  {name}: {res}\n")

            if tolerance_results:
                f.write("\nModel tolerance:\n")
                f.write("-" * 60 + "\n")
                for name, res in tolerance_results.items():
                    f.write(f"  {name}: {res}\n")

            if detector_accuracy:
                f.write("\nDetector accuracy:\n")
                f.write("-" * 60 + "\n")
                for name, res in detector_accuracy.items():
                    f.write(f"  {name}: {res}\n")

            if getscoreml_results:
                f.write("\nClean4MLBaseline unified evaluation:\n")
                f.write("-" * 60 + "\n")
                for key in ['accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
                    if key in getscoreml_results:
                        f.write(f"  {key}: {getscoreml_results[key]}\n")
                f.write("\n  Downstream task performance:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('ml_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  Tolerance:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('tolerance_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  Snoopy upper bound:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('snoopy_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  Ground-truth cost:\n")
                f.write(f"    method_type: {getscoreml_results.get('method_type', 'N/A')}\n")
                f.write(f"    ground_truth_cost: {getscoreml_results.get('ground_truth_cost', 'N/A')}\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('ideal_'):
                        f.write(f"    {key}: {value}\n")

            # Shapley analysis results
            shapley = report.get('shapley_results', {})
            if shapley and 'report_text' in shapley:
                f.write("\n" + "=" * 60 + "\n")
                f.write("Shapley contribution analysis:\n")
                f.write("=" * 60 + "\n")
                f.write(shapley['report_text'])
                f.write("\n")
            elif shapley:
                # Fallback: emit raw data when report_text is missing
                f.write("\nShapley analysis:\n")
                f.write("-" * 60 + "\n")
                for dim_key in ['action_shapley', 'feature_shapley', 'error_type_shapley']:
                    dim_data = shapley.get(dim_key, {})
                    if dim_data:
                        f.write(f"  {dim_key}:\n")
                        for name, val in sorted(dim_data.items(), key=lambda x: -x[1]):
                            f.write(f"    {name}: {val:+.6f}\n")

            # Ground-truth cost summary
            cost_summary = report.get('ground_truth_cost_summary', {})
            if cost_summary:
                f.write("\n" + "=" * 60 + "\n")
                f.write("Ground-truth cost summary:\n")
                f.write("=" * 60 + "\n")
                f.write(f"  RAHA detection cost: {cost_summary.get('raha_detection_cost', 0)}\n")
                f.write(f"  Agent repair cost:   {cost_summary.get('agent_repair_cost', 0)}\n")
                f.write(f"  Total cost:          {cost_summary.get('total_cost', 0)}\n")
                f.write(f"  Total data cells:    {cost_summary.get('total_data_cells', 0)}\n")
                f.write(f"  Cost ratio:          {cost_summary.get('cost_ratio', 0):.4%}\n")
                f.write(f"  apply_raha_truth:    {cost_summary.get('apply_raha_truth', True)}\n")
                f.write(f"  count_raha_cost:     {cost_summary.get('count_raha_cost', True)}\n")
    except Exception as e:
        print(f"  [warn] summary save failed: {e}")

    print(f"\n  Version {version_name} done! elapsed: {elapsed_time:.2f}s")
    print(f"  Output dir: {save_dir}")

    return report


# ============================================================================
# Cross-version comparison
# ============================================================================
def cross_version_comparison(
    dataset_name: str,
    all_reports: dict,
):
    """
    After all versions finish, produce cross-version comparison tables and charts.

    Args:
        dataset_name: dataset name
        all_reports: {version_name: report_dict}
    """
    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']

    save_dir = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Cross-version comparison - {dataset_name}")
    print("=" * 70)

    # --- Collect tolerance results for compare_versions ---
    tolerance_all = {}
    for version_name, report in all_reports.items():
        tol = report.get('tolerance_results', {})
        if tol:
            tolerance_all[version_name] = tol

    if tolerance_all:
        try:
            comparison_df = compare_versions(
                tolerance_all,
                save_dir=save_dir,
                task_name=dataset_name,
            )
            print(f"  cross-version comparison saved to: {save_dir}")
        except Exception as e:
            print(f"  [error] cross-version comparison failed: {e}")
            traceback.print_exc()

    # --- Build summary table ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        summary_rows = []
        for version_name, report in all_reports.items():
            row = {
                'version': version_name,
                'detector': report.get('detector_mode', ''),
                'agent': report.get('agent_type', ''),
                'inference': report.get('inference_mode', ''),
                'truth_cost': report.get('ground_truth_used', 0),
                'elapsed_time': report.get('elapsed_time', 0),
            }

            # Ground-truth cost summary
            cost_sum = report.get('ground_truth_cost_summary', {})
            row['raha_cost'] = cost_sum.get('raha_detection_cost', 0)
            row['total_cost'] = cost_sum.get('total_cost', 0)
            row['cost_ratio'] = cost_sum.get('cost_ratio', 0.0)

            # Extract DemandClean baseline metrics for the primary model
            baseline = report.get('baseline_results', {}).get('DemandClean', {})
            if task_type == 'classification':
                primary_model = EVAL_MODELS_CLASSIFICATION[0]
                row['dc_accuracy'] = baseline.get(f'{primary_model}_accuracy', None)
                row['dc_f1'] = baseline.get(f'{primary_model}_f1', None)
            elif task_type == 'clustering':
                primary_model = EVAL_MODELS_CLUSTERING[0]
                row['dc_silhouette'] = baseline.get(f'{primary_model}_silhouette', None)
                row['dc_ari'] = baseline.get(f'{primary_model}_ari', None)
            else:
                primary_model = EVAL_MODELS_REGRESSION[0]
                row['dc_r2'] = baseline.get(f'{primary_model}_r2', None)
                row['dc_mse'] = baseline.get(f'{primary_model}_mse', None)

            # Extract tolerance
            tol = report.get('tolerance_results', {})
            first_model_tol = next(iter(tol.values()), {}) if tol else {}
            row['tol_prior'] = first_model_tol.get('tolerance_prior', None)
            row['tol_post'] = first_model_tol.get('tolerance_post', None)

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(save_dir, f'{dataset_name}_all_versions_summary.csv')
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  summary table saved: {csv_path}")

        # Print the table
        print("\n" + summary_df.to_string(index=False))

        # --- Summary bar chart ---
        if len(summary_df) > 1:
            if task_type == 'classification' and 'dc_accuracy' in summary_df.columns:
                metric_col = 'dc_accuracy'
                ylabel = 'Accuracy'
            elif 'dc_r2' in summary_df.columns:
                metric_col = 'dc_r2'
                ylabel = 'R2 Score'
            else:
                metric_col = None

            if metric_col and summary_df[metric_col].notna().any():
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(summary_df))
                vals = summary_df[metric_col].fillna(0).values
                colors_list = plt.cm.Set2(np.linspace(0, 1, len(summary_df)))
                bars = ax.bar(x, vals, color=colors_list)

                for bar, val in zip(bars, vals):
                    if val != 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

                ax.set_xticks(x)
                ax.set_xticklabels(summary_df['version'].values, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(ylabel)
                ax.set_title(f'{dataset_name} - All Versions DemandClean Performance')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()

                fig_path = os.path.join(save_dir, f'{dataset_name}_all_versions_comparison.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  cross-version plot saved: {fig_path}")

    except Exception as e:
        print(f"  [error] summary analysis failed: {e}")
        traceback.print_exc()


# ============================================================================
# main()
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='DemandClean unified runner - supports 8 version combos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: v5 + all datasets
    python run_demandclean_base.py --n_episodes 300

    # Single dataset
    python run_demandclean_base.py --dataset beers --n_episodes 300

    # Specific versions (short names v1-v8 or full names)
    python run_demandclean_base.py --dataset beers --versions v3
    python run_demandclean_base.py --dataset beers --versions v1,v3,v5
    python run_demandclean_base.py --dataset beers --versions v3_oracle_plain_single

    # Custom error-injection rates (format: min,max)
    python run_demandclean_base.py --dataset beers --missing_rate 0.05,0.1
    python run_demandclean_base.py --dataset beers --semantic_rate 0.1,0.2 --syntactic_rate 0.15,0.3

    # Resume / visualize only
    python run_demandclean_base.py --dataset beers --resume auto
    python run_demandclean_base.py --dataset beers --versions v6 --visualize_only

Versions:
    v1: oracle + dueling + single_phase    v5: auto + dueling + single_phase (default)
    v2: oracle + dueling + two_phase       v6: auto + dueling + two_phase
    v3: oracle + plain   + single_phase    v7: auto + plain   + single_phase
    v4: oracle + plain   + two_phase       v8: auto + plain   + two_phase
        """,
    )

    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=list(DATASETS.keys()),
        help='Dataset name (if omitted, run all datasets)',
    )
    parser.add_argument(
        '--n_episodes', type=int, default=300,
        help='Training episodes (default: 300)',
    )
    parser.add_argument(
        '--all_datasets', action='store_true',
        help='Explicitly run all datasets (this is the default when --dataset is omitted)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Verbose training output',
    )
    parser.add_argument(
        '--versions', type=str, default='v5',
        help='Comma-separated versions to run (e.g. v3,v5 or v1,v2,v3). '
             'Short names (v1-v8) or full names (v3_oracle_plain_single) are accepted. '
             'Default: use versions enabled by the ENABLE_ switches in code',
    )
    parser.add_argument(
        '--missing_rate', type=str, default='',
        help='Missing-value injection rate range, format: min,max (e.g. 0.02,0.08). Defaults to config values',
    )
    parser.add_argument(
        '--semantic_rate', type=str, default='',
        help='Semantic-error injection rate range, format: min,max (e.g. 0.05,0.15). Defaults to config values',
    )
    parser.add_argument(
        '--syntactic_rate', type=str, default='',
        help='Syntactic-error injection rate range, format: min,max (e.g. 0.1,0.25). Defaults to config values',
    )
    parser.add_argument(
        '--label_rate', type=str, default='',
        help='Label-error injection rate range, format: min,max (e.g. 0.0,0.05). Defaults to config values',
    )
    parser.add_argument(
        '--resume', type=str, choices=['auto', 'force_new'], default='force_new',
        help='Resume mode: auto=resume if an existing model is detected; force_new=always train from scratch (default: auto)',
    )
    parser.add_argument(
        '--apply_raha_truth', type=str, choices=['true', 'false'], default='true',
        help='Apply ground-truth values from RAHA-labeled rows during repair (default: true)',
    )
    parser.add_argument(
        '--count_raha_cost', type=str, choices=['true', 'false'], default='true',
        help='Count RAHA labeling cost in the total ground-truth cost (default: true)',
    )
    parser.add_argument(
        '--visualize_only', action='store_true',
        help='Regenerate visualization and evaluation from existing cleaned results; skip training and inference',
    )
    parser.add_argument(
        '--min_repair_ratio', type=float, default=None,
        help='Minimum repair ratio (fraction of VE-fill total); 0 = no limit (e.g. 0.05)',
    )
    parser.add_argument(
        '--max_repair_ratio', type=float, default=None,
        help='Maximum repair ratio; 1.0 = no limit (e.g. 0.80)',
    )
    parser.add_argument(
        '--repair_sensitivity', type=float, default=None,
        help='Performance sensitivity for dynamic tuning (default: 10.0)',
    )
    parser.add_argument(
        '--max_truth_budget', type=int, default=None,
        help='Maximum ground-truth budget (set to 0 for the NGT variant, which disables ground-truth repair)',
    )
    parser.add_argument(
        '--repair_lambda', type=float, default=None,
        help='Ground-truth repair cost coefficient (default: 0.03; regression tasks may use lower values like 0.005)',
    )
    parser.add_argument(
        '--reward_model_type', type=str, default=None,
        help='Override the model type used in reward evaluation (e.g. random_forest); does not affect baseline evaluation',
    )
    parser.add_argument(
        '--disable_raha', action='store_true',
        help='Disable RAHA detection; use rule-based detection only (for label-error-only datasets like adult)',
    )
    parser.add_argument(
        '--delete_shaping_reward', type=float, default=None,
        help='Shaping reward for the delete action (default: -0.02 for classification/clustering, -0.05 for regression)',
    )
    parser.add_argument(
        '--keep_rate_weight', type=float, default=None,
        help='Weight of keep_rate in final reward (default: 0.2 for classification/clustering, 1.0 for regression)',
    )
    parser.add_argument(
        '--oracle', action='store_true', default=True,
        help='Enable Oracle mode: three-way split (train/val/test) with clean validation set for reward (enabled by default)',
    )
    parser.add_argument(
        '--no-oracle', dest='oracle', action='store_false',
        help='Disable Oracle mode; use dirty-data-derived Clean Base for reward',
    )
    parser.add_argument(
        '--inference_only', action='store_true',
        help='Inference-only mode: skip training, load existing model, run inference + evaluation (useful when training finished but inference failed)',
    )
    parser.add_argument(
        '--output_suffix', type=str, default='',
        help='Output-dir suffix to avoid overwriting the base version (e.g. --output_suffix _ngt)',
    )

    args = parser.parse_args()

    # --- Parse error injection rates ---
    inject_kwargs = {}
    for rate_name in ['missing_rate', 'semantic_rate', 'syntactic_rate', 'label_rate']:
        rate_str = getattr(args, rate_name, '')
        if rate_str:
            try:
                parts = [float(x.strip()) for x in rate_str.split(',')]
                if len(parts) == 2:
                    inject_kwargs[f'{rate_name}_range'] = (parts[0], parts[1])
                elif len(parts) == 1:
                    inject_kwargs[f'{rate_name}_range'] = (parts[0], parts[0])
                else:
                    print(f"[warn] {rate_name} bad format: {rate_str}, ignored")
            except ValueError:
                print(f"[warn] {rate_name} bad format: {rate_str}, ignored")

    # Determine datasets to run (default: all)
    if args.dataset is not None:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(DATASETS.keys())

    # ======================================================================
    # Log redirection: tee output to both terminal and log file
    # Log file: logs/demandclean/{dataset}_{timestamp}.log
    # ======================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_tag = args.dataset if args.dataset else 'all'
    log_dir = os.path.join(_PROJECT_ROOT, 'logs', 'demandclean')
    version_tag = args.versions.replace(',', '_') if args.versions else 'default'
    log_filename = f"{dataset_tag}_{version_tag}_ep{args.n_episodes}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    tee_stdout = TeeLogger(log_path, stream=sys.stdout)
    tee_stderr = TeeLogger(log_path.replace('.log', '_stderr.log'), stream=sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    print(f"[log] full output will be saved to: {log_path}")

    # Select enabled versions
    if args.versions:
        # CLI-provided versions
        requested = [v.strip() for v in args.versions.split(',') if v.strip()]
        # Short-name (v1-v8) -> full name
        short_name_map = {
            f'v{i}': k for i, k in enumerate(VERSION_CONFIGS.keys(), 1)
        }
        enabled_versions = {}
        for req in requested:
            # Try full name first
            if req in VERSION_CONFIGS:
                cfg = VERSION_CONFIGS[req].copy()
                cfg['enabled'] = True
                enabled_versions[req] = cfg
            # Then short name
            elif req in short_name_map:
                full_name = short_name_map[req]
                cfg = VERSION_CONFIGS[full_name].copy()
                cfg['enabled'] = True
                enabled_versions[full_name] = cfg
            else:
                print(f"[warn] unknown version: {req}, ignored. "
                      f"Available: {list(short_name_map.keys())} or {list(VERSION_CONFIGS.keys())}")
    else:
        # Use ENABLE_ switches in code
        enabled_versions = {
            k: v for k, v in VERSION_CONFIGS.items() if v['enabled']
        }

    if not enabled_versions:
        print("[error] no versions enabled! Check version switch settings.")
        return

    print("=" * 70)
    print("DemandClean unified runner")
    print("=" * 70)
    print(f"Datasets: {datasets_to_run}")
    print(f"Enabled versions: {list(enabled_versions.keys())}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Verbose: {args.verbose}")
    if args.oracle:
        print(f"Oracle mode: ON (60/20/20 three-way split; disable with --no-oracle)")
    else:
        print(f"Oracle mode: OFF (using dirty-data Clean Base for reward)")
    if args.min_repair_ratio is not None or args.max_repair_ratio is not None:
        print(f"Repair-ratio control: min={args.min_repair_ratio}, max={args.max_repair_ratio}")
    if inject_kwargs:
        print(f"Custom injection rates: {inject_kwargs}")
    if args.inference_only:
        print(f"Inference-only mode: ON (skip training, load existing model)")
    print("=" * 70)

    total_start = time.time()

    # Iterate datasets
    for dataset_name in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*70}")

        ds_cfg = DATASETS[dataset_name]

        try:
            if args.oracle:
                # =============================================================
                # Strict 60/20/20 split: split raw CSV first, then encode
                # =============================================================
                from sklearn.model_selection import train_test_split as _tts

                # 1. Load raw CSV
                data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
                _dirty_path = os.path.join(data_dir, 'dirty_index.csv')
                _clean_path = os.path.join(data_dir, 'clean_index.csv')
                if not os.path.exists(_dirty_path):
                    _dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
                if not os.path.exists(_clean_path):
                    _clean_path = os.path.join(data_dir, 'clean_with_index.csv')
                raw_dirty_df = pd.read_csv(_dirty_path)
                raw_clean_df = pd.read_csv(_clean_path)

                # 2. 60/20/20 split (seed=42)
                n_total = len(raw_dirty_df)
                all_idx = np.arange(n_total)
                train_idx, temp_idx = _tts(all_idx, test_size=0.4, random_state=42)
                val_idx, test_idx = _tts(temp_idx, test_size=0.5, random_state=42)

                dirty_train_df = raw_dirty_df.iloc[train_idx].reset_index(drop=True)
                clean_train_df = raw_clean_df.iloc[train_idx].reset_index(drop=True)

                print(f"\n  60/20/20 split: train={len(train_idx)}, "
                      f"val={len(val_idx)}, test={len(test_idx)}")

                # 3. Write 60% subset CSV (for RAHA)
                save_base = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name)
                os.makedirs(save_base, exist_ok=True)
                dirty_train_csv_path = os.path.join(save_base, 'dirty_train_60pct.csv')
                clean_train_csv_path = os.path.join(save_base, 'clean_train_60pct.csv')
                dirty_train_df.to_csv(dirty_train_csv_path, index=False)
                clean_train_df.to_csv(clean_train_csv_path, index=False)
                print(f"  60% subset CSV: {dirty_train_csv_path}")

                # 4. preprocess_data only processes 60% dirty (LE/SS fit here only)
                (X_dirty, y_dirty,
                 _, _,
                 column_names, fd_rules, rules_path,
                 _orig_dirty_path, _orig_clean_path, csv_columns,
                 data_scaler, label_encoders, categorical_cols,
                 dirty_df, _) = preprocess_data(
                    dataset_name, dirty_df=dirty_train_df, clean_df=None)

                # Use 60% train subset CSV path (no longer the full-data path)
                dirty_csv_path = dirty_train_csv_path
                clean_csv_path = clean_train_csv_path

                # 5. Encode val/test/train clean data with the train LE/SS
                X_clean_train, y_clean_train = encode_subset(
                    clean_train_df, column_names, ds_cfg['label_col'],
                    label_encoders, data_scaler, categorical_cols)
                X_clean_val, y_clean_val = encode_subset(
                    raw_clean_df.iloc[val_idx].reset_index(drop=True),
                    column_names, ds_cfg['label_col'],
                    label_encoders, data_scaler, categorical_cols)
                X_clean_test, y_clean_test = encode_subset(
                    raw_clean_df.iloc[test_idx].reset_index(drop=True),
                    column_names, ds_cfg['label_col'],
                    label_encoders, data_scaler, categorical_cols)

                print(f"  encoding done: X_dirty_train={X_dirty.shape}, "
                      f"X_clean_train={X_clean_train.shape}, "
                      f"X_clean_val={X_clean_val.shape}, "
                      f"X_clean_test={X_clean_test.shape}")

                # Pass-through for run_version
                X_clean = X_clean_train
                y_clean = y_clean_train
                clean_df = clean_train_df

                oracle_split = {
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'test_idx': test_idx,
                    'X_dirty_train': X_dirty,
                    'y_dirty_train': y_dirty,
                    'X_clean_train': X_clean_train,
                    'y_clean_train': y_clean_train,
                    'X_clean_val': X_clean_val,
                    'y_clean_val': y_clean_val,
                    'X_clean_test': X_clean_test,
                    'y_clean_test': y_clean_test,
                }

            else:
                # =============================================================
                # Non-Oracle mode: full-data encoding (backward compatible)
                # =============================================================
                (X_dirty, y_dirty,
                 X_clean, y_clean,
                 column_names, fd_rules, rules_path,
                 dirty_csv_path, clean_csv_path, csv_columns,
                 data_scaler, label_encoders, categorical_cols,
                 dirty_df, clean_df) = preprocess_data(dataset_name)
                oracle_split = None

        except Exception as e:
            print(f"\n[error] preprocessing failed for {dataset_name}: {e}")
            traceback.print_exc()
            continue

        # --- Iterate enabled versions ---
        all_reports = {}

        for version_key, version_cfg in enabled_versions.items():
            try:
                report = run_version(
                    dataset_name=dataset_name,
                    version_cfg=version_cfg,
                    X_dirty=X_dirty,
                    y_dirty=y_dirty,
                    X_clean=X_clean,
                    y_clean=y_clean,
                    column_names=column_names,
                    fd_rules=fd_rules,
                    rules_path=rules_path,
                    dirty_csv_path=dirty_csv_path,
                    clean_csv_path=clean_csv_path,
                    csv_columns=csv_columns,
                    n_episodes=args.n_episodes,
                    verbose=args.verbose,
                    data_scaler=data_scaler,
                    label_encoders=label_encoders,
                    categorical_cols=categorical_cols,
                    dirty_df=dirty_df,
                    clean_df=clean_df,
                    inject_kwargs=inject_kwargs,
                    resume_mode=args.resume,
                    apply_raha_truth=(args.apply_raha_truth == 'true'),
                    count_raha_cost=(args.count_raha_cost == 'true'),
                    visualize_only=args.visualize_only,
                    oracle_split=oracle_split,
                    min_repair_ratio=args.min_repair_ratio,
                    max_repair_ratio=args.max_repair_ratio,
                    repair_sensitivity=args.repair_sensitivity,
                    max_truth_budget=args.max_truth_budget,
                    repair_lambda=args.repair_lambda,
                    reward_model_type=args.reward_model_type,
                    disable_raha=args.disable_raha,
                    delete_shaping_reward=args.delete_shaping_reward,
                    keep_rate_weight=args.keep_rate_weight,
                    inference_only=args.inference_only,
                    output_suffix=args.output_suffix,
                )
                all_reports[version_cfg['version_name']] = report

            except Exception as e:
                print(f"\n[error] version {version_key} failed, skipped:")
                print(f"  {e}")
                traceback.print_exc()
                continue

        # --- Cross-version comparison ---
        if len(all_reports) >= 2:
            try:
                cross_version_comparison(dataset_name, all_reports)
            except Exception as e:
                print(f"\n[error] cross-version comparison failed: {e}")
                traceback.print_exc()
        elif len(all_reports) == 1:
            print("\n  only 1 version finished; skip cross-version comparison")
        else:
            print("\n  no versions finished; skip cross-version comparison")

    # --- Global end ---
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All done! total: {total_elapsed:.2f}s")
    print(f"Results dir: {os.path.join(_PROJECT_ROOT, 'results', 'demandclean')}")
    print(f"Log file: {log_path}")
    print(f"{'='*70}")

    # Close TeeLogger, restore stdout/stderr
    sys.stdout = tee_stdout.terminal
    sys.stderr = tee_stderr.terminal
    tee_stdout.close()
    tee_stderr.close()
    print(f"[log] full log saved: {log_path}")


if __name__ == "__main__":
    main()
