"""
[Deprecated] Generate [STATISTICAL] sections for rules.txt across all 9 datasets
================================================================================

This script is deprecated and no longer used.

Reason: generate_col_stats fits a StandardScaler on the dirty data to compute
statistics, while the main pipeline (shared_preprocess) fits StandardScaler on
the clean data. That leaves the two encoding spaces inconsistent.

Replacement: AutoDetector now computes col_stats fully at runtime:
  1. Preferentially from actual data during fit(X_clean_subset).
  2. Falls back to X_dirty during detect().
All [STATISTICAL] sections have been removed from rules.txt.

If this needs to be regenerated, make sure the scaler is fit on the same basis
as the main pipeline (clean data).

Legacy usage (no longer recommended):
    python -m demandclean.tools.generate_col_stats
    python -m demandclean.tools.generate_col_stats --datasets beers adult
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Dataset config (matches run_demandclean_base.py)
DATASETS = {
    'beers':         {'label_col': 'style'},
    'adult':         {'label_col': 'income'},
    'bike':          {'label_col': 'cnt'},
    'breast_cancer': {'label_col': 'class'},
    'har':           {'label_col': 'gt'},
    'mercedes':      {'label_col': 'y'},
    'nasa':          {'label_col': 'sound_pressure_level'},
    'smartfactory':  {'label_col': 'labels'},
    'soilmoisture':  {'label_col': 'soil_moisture'},
}


def load_dirty_encoded(dataset_name: str):
    """Load dirty data and encode it into the numeric space
    (matches run_demandclean_base.py).

    Returns:
        (X_dirty, column_names)
    """
    label_col = DATASETS[dataset_name]['label_col']
    data_dir = os.path.join(PROJECT_ROOT, 'data', dataset_name)

    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(data_dir, 'clean_with_index.csv')

    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    # Normalize column names
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]
    dirty_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)
    clean_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # Identify categorical columns
    categorical_cols = set()
    for col in feature_cols:
        combined = pd.concat([dirty_df[col], clean_df[col]]).dropna()
        combined = combined[~combined.astype(str).str.strip().isin(['?', '', 'N/A'])]
        try:
            pd.to_numeric(combined, errors='raise')
        except (ValueError, TypeError):
            categorical_cols.add(col)

    # LabelEncoder (fit on the union of dirty + clean values)
    label_encoders = {}
    X_df = dirty_df[feature_cols].copy()
    for col in feature_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            all_vals = pd.concat([
                dirty_df[col].dropna().astype(str),
                clean_df[col].dropna().astype(str),
            ]).unique()
            le.fit(all_vals)
            label_encoders[col] = le
            nan_mask = X_df[col].isna()
            if not nan_mask.all():
                X_df.loc[~nan_mask, col] = le.transform(
                    X_df.loc[~nan_mask, col].astype(str)
                )
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    X_dirty_raw = X_df.values.astype(float)

    # StandardScaler (fit on non-NaN rows of dirty data; matches run_demandclean_base.py)
    nan_mask_row = np.isnan(X_dirty_raw).any(axis=1)
    X_for_fit = X_dirty_raw[~nan_mask_row]
    if len(X_for_fit) == 0:
        X_for_fit = X_dirty_raw.copy()
        col_means = np.nanmean(X_for_fit, axis=0)
        for c in range(X_for_fit.shape[1]):
            m = np.isnan(X_for_fit[:, c])
            if m.any():
                X_for_fit[m, c] = col_means[c] if not np.isnan(col_means[c]) else 0

    scaler = StandardScaler()
    scaler.fit(X_for_fit)

    # Scale (preserve NaNs)
    X_out = X_dirty_raw.copy()
    nan_mask = np.isnan(X_out)
    X_out[nan_mask] = 0
    X_scaled = scaler.transform(X_out)
    X_scaled[nan_mask] = np.nan

    return X_scaled, feature_cols


def compute_col_stats(X: np.ndarray) -> dict:
    """Compute per-column statistics in the encoded space
    (matches auto_detector._compute_col_stats exactly)."""
    col_stats = {}
    for col in range(X.shape[1]):
        valid = X[:, col][~np.isnan(X[:, col])]
        if len(valid) > 0:
            col_stats[col] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid) + 1e-6),
                'q1': float(np.percentile(valid, 25)),
                'q3': float(np.percentile(valid, 75)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'median': float(np.median(valid)),
            }
    return col_stats


def write_stats_to_rules(rules_path: str, column_names: list, col_stats: dict):
    """Append or replace the [STATISTICAL] section in rules.txt."""
    # Read existing content
    existing_lines = []
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()

    # Remove any existing [STATISTICAL] section
    new_lines = []
    in_statistical = False
    for line in existing_lines:
        stripped = line.strip()
        if stripped == '[STATISTICAL]':
            in_statistical = True
            continue
        if in_statistical:
            if stripped.startswith('[') and stripped.endswith(']'):
                in_statistical = False
                new_lines.append(line)
            continue
        new_lines.append(line)

    # Ensure trailing newline
    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    # Append the [STATISTICAL] section
    new_lines.append('\n[STATISTICAL]\n')
    new_lines.append('# Per-column stats (encoded space, computed from dirty data)\n')
    for col_idx, stats in sorted(col_stats.items()):
        if col_idx < len(column_names):
            col_name = column_names[col_idx]
            parts = [f"{k}={v:.6f}" for k, v in stats.items()]
            new_lines.append(f"COL_STATS: {col_name} | {' | '.join(parts)}\n")

    with open(rules_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(description='Generate [STATISTICAL] section for rules.txt')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    args = parser.parse_args()

    print("=" * 60)
    print("Generating [STATISTICAL] section for rules.txt")
    print("=" * 60)

    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"  [skip] unknown dataset: {ds}")
            continue

        try:
            rules_path = os.path.join(PROJECT_ROOT, 'data', ds, 'rules.txt')
            if not os.path.exists(rules_path):
                print(f"  [skip] {ds}: rules.txt not found")
                continue

            X_dirty, column_names = load_dirty_encoded(ds)
            col_stats = compute_col_stats(X_dirty)

            write_stats_to_rules(rules_path, column_names, col_stats)
            print(f"  {ds}: wrote {len(col_stats)} column stats to {rules_path}")

        except Exception as e:
            print(f"  [error] {ds}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == '__main__':
    main()
