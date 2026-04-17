#!/usr/bin/env python3
"""
Evaluate Agglomerative Clustering silhouette on HAR after ReplaceAll cleaning.
HAR has 70k rows; Agglo's O(n^2) memory rules out a full run, so we sample 10k.

Usage: python tools/eval_har_agglo_replaceall.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

SEED = 42
SAMPLE_SIZE = 10000  # Agglo O(n^2) memory; 70k full run ~37GB, so sample 10k

# Load data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'har')
dirty = pd.read_csv(os.path.join(data_dir, 'dirty_index.csv'))
clean = pd.read_csv(os.path.join(data_dir, 'clean_index.csv'))

label_col = 'gt'
feature_cols = ['x', 'y', 'z']

def preprocess(df):
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    y = df[label_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def replaceall(dirty_df, clean_df):
    cleaned = dirty_df.copy()
    errors = []
    for col in feature_cols:
        d_vals = pd.to_numeric(dirty_df[col], errors='coerce')
        c_vals = pd.to_numeric(clean_df[col], errors='coerce')
        for i in range(min(len(dirty_df), len(clean_df))):
            dv, cv = d_vals.iloc[i], c_vals.iloc[i]
            if pd.isna(dv) and pd.isna(cv):
                continue
            if pd.isna(dv) and not pd.isna(cv):
                errors.append((i, col))
                continue
            if not pd.isna(dv) and not pd.isna(cv) and abs(dv - cv) > 1e-6:
                errors.append((i, col))
    print(f"Oracle detected {len(errors)} errors")

    from sklearn.neighbors import NearestNeighbors
    X_num = dirty_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    nn = NearestNeighbors(n_neighbors=6)
    nn.fit(X_num)
    _, indices = nn.kneighbors(X_num)
    for row_idx, col_name in errors:
        if row_idx >= len(X_num):
            continue
        col_idx = feature_cols.index(col_name)
        neighbors = indices[row_idx, 1:]
        vals = X_num[neighbors, col_idx]
        cleaned.at[cleaned.index[row_idx], col_name] = np.mean(vals)
    return cleaned

print("=== Running ReplaceAll (Oracle + VEC) ===")
cleaned_df = replaceall(dirty, clean)

# Fix sampling indices so all three datasets use the same rows
np.random.seed(SEED)
sample_idx = np.random.choice(len(dirty), size=min(SAMPLE_SIZE, len(dirty)), replace=False)
sample_idx.sort()

print(f"\n=== Evaluating {len(sample_idx)} sampled rows (Agglo O(n^2) rules out full 70k) ===")

for tag, df in [('NoFix (dirty)', dirty), ('ReplaceAll (cleaned)', cleaned_df), ('RepairAll (clean)', clean)]:
    df_sample = df.iloc[sample_idx]
    X, y = preprocess(df_sample)
    n_clusters = len(np.unique(y))

    # KMeans (reference)
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km_pred = km.fit_predict(X)
    km_sil = silhouette_score(X, km_pred, random_state=SEED)

    # Agglomerative
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_pred = agglo.fit_predict(X)
    agglo_sil = silhouette_score(X, agglo_pred, random_state=SEED)

    print(f"\n{tag}:")
    print(f"  KMeans    silhouette = {km_sil:.6f}")
    print(f"  Agglo     silhouette = {agglo_sil:.6f}")

# Also run full-size KMeans as a reference
print("\n=== Full 70k KMeans (reference) ===")
for tag, df in [('NoFix', dirty), ('ReplaceAll', cleaned_df), ('RepairAll', clean)]:
    X, y = preprocess(df)
    n_clusters = len(np.unique(y))
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km_pred = km.fit_predict(X)
    km_sil = silhouette_score(X, km_pred, sample_size=10000, random_state=SEED)
    print(f"  {tag}: KMeans sil = {km_sil:.6f}")

print("\n=== Done ===")
print("Fill Table 3's HAR Repl column with the Agglo silhouette (sampled value)")

