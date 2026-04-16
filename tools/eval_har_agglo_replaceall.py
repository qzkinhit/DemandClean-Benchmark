#!/usr/bin/env python3
"""
评估 har 数据集在 ReplaceAll 清洗后的 Agglomerative Clustering silhouette。
har 有 70k 行，Agglo O(n²) 内存无法全量跑，采样 10k 行评估。

用法: python tools/eval_har_agglo_replaceall.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

SEED = 42
SAMPLE_SIZE = 10000  # Agglo O(n²) 内存, 70k 全量 ~37GB, 采样 10k

# 加载数据
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
    print(f"Oracle 检测到 {len(errors)} 个错误")

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

print("=== 执行 ReplaceAll (Oracle + VEC) ===")
cleaned_df = replaceall(dirty, clean)

# 固定采样索引, 保证三组数据用同样的行
np.random.seed(SEED)
sample_idx = np.random.choice(len(dirty), size=min(SAMPLE_SIZE, len(dirty)), replace=False)
sample_idx.sort()

print(f"\n=== 采样 {len(sample_idx)} 行评估 (Agglo O(n²) 无法全量 70k) ===")

for tag, df in [('NoFix (dirty)', dirty), ('ReplaceAll (cleaned)', cleaned_df), ('RepairAll (clean)', clean)]:
    df_sample = df.iloc[sample_idx]
    X, y = preprocess(df_sample)
    n_clusters = len(np.unique(y))

    # KMeans (对照)
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

# 同时跑全量 KMeans 作为对照
print("\n=== 全量 70k KMeans (对照) ===")
for tag, df in [('NoFix', dirty), ('ReplaceAll', cleaned_df), ('RepairAll', clean)]:
    X, y = preprocess(df)
    n_clusters = len(np.unique(y))
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km_pred = km.fit_predict(X)
    km_sil = silhouette_score(X, km_pred, sample_size=10000, random_state=SEED)
    print(f"  {tag}: KMeans sil = {km_sil:.6f}")

print("\n=== 完成 ===")
print("Table 3 har Repl 列应填入 Agglo silhouette (采样值)")

