"""
共享预处理模块
=============

从 run_demandclean_base.py 提取的公共预处理函数，供测试脚本复用。
确保 test_value_estimation / verify_injection / verify_detection
与主流程编码/逆转换逻辑完全一致。

注意: 本模块与 run_demandclean_base.py 中的同名函数保持**语义一致**。
      暂不修改 base.py 本身（避免影响已跑通的主流程），等测试脚本稳定后再统一。
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================================================================
# 项目根目录
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from demandclean.utils.edit_distance import find_nearest_known

# 延迟导入 FD 解析器（优先使用 demandclean 新解析器）
try:
    from demandclean.detectors.rule_parser import load_rules, extract_fd_pairs
    _use_new_parser = True
except (ModuleNotFoundError, ImportError):
    _use_new_parser = False

# 旧解析器作为 fallback
try:
    from tools.rules_parser import get_horizon_fds
except (ModuleNotFoundError, ImportError):
    get_horizon_fds = None


# ============================================================================
# 数据集配置 (与 run_demandclean_base.py 完全一致)
# ============================================================================
DATASETS = {
    'beers': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'style',
        'categorical_cols': {'beer_name', 'brewery_name', 'city', 'state'},
        'protected_cols': {'brewery_name'},  # FD LHS: brewery_name → city, brewery_name → state
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
        'categorical_cols': set(),
        'protected_cols': set(),
    },
    'breast_cancer': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'class',
        'categorical_cols': set(),
        'protected_cols': set(),
    },
    'har': {
        'task_type': 'clustering',
        'model_type': 'kmeans',
        'label_col': 'gt',
        'categorical_cols': set(),
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
        'categorical_cols': set(),
        'protected_cols': set(),
    },
    'smartfactory': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'labels',
        'categorical_cols': set(),
        'protected_cols': set(),
    },
    'soilmoisture': {
        'task_type': 'regression',
        'model_type': 'ridge',
        'label_col': 'soil_moisture',
        'categorical_cols': set(),
        'protected_cols': set(),
    },
}

# 评估模型简称
EVAL_MODELS_CLASSIFICATION = ['rf', 'lr']
EVAL_MODELS_REGRESSION = ['rf', 'ridge']
EVAL_MODELS_CLUSTERING = ['kmeans']


# ============================================================================
# scale_with_nan: NaN 安全的标准化
# ============================================================================
def scale_with_nan(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    对含 NaN 的矩阵做标准化，NaN 先用该列均值填充再 transform，
    完成后恢复原始 NaN 位置。

    填充值为 **X 自身** 的列均值（dirty 用 dirty 均值，clean 用 clean 均值），
    与 run_demandclean_base.py 一致。
    """
    X_out = X.copy()
    col_m = np.nanmean(X_out, axis=0)
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
    X_out = scaler.transform(X_out)
    # 恢复原始 NaN
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X_out[nan_mask, j] = np.nan
    return X_out


# ============================================================================
# preprocess_data: 与 run_demandclean_base.py 完全一致的编码流程
# ============================================================================
def preprocess_data(dataset_name: str):
    """
    读取并预处理指定数据集。

    步骤:
      1. 读取 dirty_index.csv 和 clean_index.csv
      2. 将 "empty"/"NaN"/"NULL" 等替换为 np.nan
      3. 分类列识别 (优先配置 → fallback 仅 dirty_df)
      4. LabelEncoder: fit(dirty ∪ clean)
      5. StandardScaler: 仅在 X_clean 上 fit
      6. scale_with_nan: dirty 用 dirty 自身列均值填充
      7. 不调用 normalize_dirty_format()

    Returns:
        tuple: (X_dirty, y_dirty, X_clean, y_clean, feature_cols, fd_rules,
                rules_path, dirty_csv_path, clean_csv_path, csv_columns,
                scaler, label_encoders, categorical_cols, dirty_df, clean_df)
    """
    ds_cfg = DATASETS[dataset_name]
    label_col = ds_cfg['label_col']

    # --- 路径 ---
    data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
    dirty_path = os.path.join(data_dir, 'dirty_index.csv')
    clean_path = os.path.join(data_dir, 'clean_index.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(data_dir, 'clean_with_index.csv')
    rules_path = os.path.join(data_dir, 'rules.txt')

    # --- 读取 ---
    dirty_df = pd.read_csv(dirty_path)
    clean_df = pd.read_csv(clean_path)

    # 处理 BOM 头和列名空白
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]

    # --- 替换 empty 为 NaN ---
    dirty_df.replace(
        ['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'],
        np.nan, inplace=True,
    )
    clean_df.replace(
        ['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'],
        np.nan, inplace=True,
    )

    # --- 确定特征列 ---
    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # --- 识别分类列 ---
    config_cat_cols = ds_cfg.get('categorical_cols')
    if config_cat_cols is not None:
        categorical_cols = set(config_cat_cols) & set(feature_cols)
    else:
        categorical_cols = set()
        for col in feature_cols:
            vals = dirty_df[col].dropna()
            vals = vals[~vals.astype(str).str.strip().isin(
                ['?', '', 'N/A', 'NA', 'nan', 'NaN', 'null', 'None', '-',
                 'empty', 'Empty', 'EMPTY']
            )]
            if len(vals) == 0:
                categorical_cols.add(col)
                continue
            try:
                pd.to_numeric(vals, errors='raise')
            except (ValueError, TypeError):
                categorical_cols.add(col)

    # --- 编码 ---
    label_encoders = {}

    def encode_df(df, feature_cols_, label_col_, fit_le=False):
        """编码 DataFrame 的特征和标签，返回 (X, y)"""
        X_df = df[feature_cols_].copy()
        y_series = df[label_col_].copy()

        for col in feature_cols_:
            if col in categorical_cols:
                if col not in label_encoders and fit_le:
                    le = LabelEncoder()
                    # 仅用 dirty 数据 fit LE
                    all_vals = dirty_df[col].dropna().astype(str).unique()
                    le.fit(all_vals)
                    label_encoders[col] = le

                nan_mask = X_df[col].isna()
                non_nan_values = X_df.loc[~nan_mask, col].astype(str)
                le = label_encoders[col]
                known_set = set(le.classes_)
                known_list = list(le.classes_)

                def _safe_encode_feature(val, _le=le, _ks=known_set, _kl=known_list):
                    if val in _ks:
                        return _le.transform([val])[0]
                    nearest = find_nearest_known(val, _kl, threshold=0.3)
                    if nearest is not None:
                        return _le.transform([nearest])[0]
                    return np.nan  # dirty-fit LE 下此行不会执行

                X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_encode_feature)
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # 编码标签
        label_is_categorical = False
        combined_labels = dirty_df[label_col_].dropna()
        try:
            pd.to_numeric(combined_labels, errors='raise')
        except (ValueError, TypeError):
            label_is_categorical = True

        if label_is_categorical:
            if label_col_ not in label_encoders and fit_le:
                le = LabelEncoder()
                all_labels = dirty_df[label_col_].dropna().astype(str).unique()
                le.fit(all_labels)
                label_encoders[label_col_] = le
            if label_col_ in label_encoders:
                nan_mask_y = y_series.isna()
                y_encoded = y_series.copy()
                y_encoded.loc[~nan_mask_y] = label_encoders[label_col_].transform(
                    y_series.loc[~nan_mask_y].astype(str)
                )
                y_series = pd.to_numeric(y_encoded, errors='coerce')

        X = X_df.values.astype(float)
        y = y_series.values.astype(float)
        return X, y

    X_dirty, y_dirty = encode_df(dirty_df, feature_cols, label_col, fit_le=True)
    X_clean, y_clean = encode_df(clean_df, feature_cols, label_col, fit_le=False)

    # --- 标准化（仅在 dirty 数据上 fit）---
    scaler = StandardScaler()
    X_dirty_for_fit = X_dirty.copy()
    col_means = np.nanmean(X_dirty_for_fit, axis=0)
    for j in range(X_dirty_for_fit.shape[1]):
        mask = np.isnan(X_dirty_for_fit[:, j])
        X_dirty_for_fit[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    scaler.fit(X_dirty_for_fit)

    X_dirty_scaled = scale_with_nan(X_dirty, scaler)
    # clean 也用同一个 scaler transform
    X_clean_for_scale = X_clean.copy()
    col_means_c = np.nanmean(X_clean_for_scale, axis=0)
    for j in range(X_clean_for_scale.shape[1]):
        mask = np.isnan(X_clean_for_scale[:, j])
        X_clean_for_scale[mask, j] = col_means_c[j] if not np.isnan(col_means_c[j]) else 0.0
    X_clean_scaled = scaler.transform(X_clean_for_scale)

    # --- 解析 FD 规则（优先使用新解析器）---
    fd_rules = []
    if os.path.exists(rules_path):
        if _use_new_parser:
            try:
                parsed = load_rules(rules_path)
                fd_rules = extract_fd_pairs(parsed)
            except Exception as e:
                print(f"  [警告] 新解析器解析FD规则失败: {e}")
        elif get_horizon_fds is not None:
            try:
                fd_rules = get_horizon_fds(rules_path)
            except Exception as e:
                print(f"  [警告] 旧解析器解析FD规则失败: {e}")

    print(f"  数据集: {dataset_name}")
    print(f"  脏数据: {X_dirty_scaled.shape}, 干净数据: {X_clean_scaled.shape}")
    print(f"  特征列: {len(feature_cols)}, 标签列: {label_col}")
    print(f"  分类列: {categorical_cols if categorical_cols else '(无)'}")
    print(f"  LE fit: 仅 dirty ({len(dirty_df)} 行)")
    print(f"  SS fit: 仅 dirty ({len(dirty_df)} 行)")
    print(f"  FD规则: {len(fd_rules)} 条")

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
# inverse_transform_to_csv: 与 run_demandclean_base.py 完全一致的逆转换
# ============================================================================
def inverse_transform_to_csv(
    X_result: np.ndarray,
    y_result: np.ndarray,
    feature_cols: list,
    label_col: str,
    scaler,
    label_encoders: dict,
    categorical_cols: set,
    dirty_df: pd.DataFrame,
    keep_mask=None,
) -> pd.DataFrame:
    """
    将 LE+SS 空间的 numpy 数组逆转换为 CSV 原始格式的 DataFrame。

    步骤:
      1. scaler.inverse_transform → LE 空间
      2. 分类列: round → clip → le.inverse_transform → 原始字符串
      3. 标签列: 同上
      4. 恢复 index 列和非特征列
      5. 列顺序对齐 dirty_df
    """
    n_result = len(X_result)

    if n_result == 0:
        return pd.DataFrame(columns=dirty_df.columns)

    # Step 1: 逆标准化
    X_inv = scaler.inverse_transform(X_result)

    # Step 2: 构建 DataFrame
    result_df = pd.DataFrame(X_inv, columns=feature_cols)

    # Step 3: 逆编码分类列
    if label_encoders and categorical_cols:
        for col in feature_cols:
            if col in categorical_cols and col in label_encoders:
                le = label_encoders[col]
                col_vals = result_df[col].values
                col_rounded = np.round(col_vals).astype(int)
                col_rounded = np.clip(col_rounded, 0, len(le.classes_) - 1)
                try:
                    result_df[col] = le.inverse_transform(col_rounded)
                except Exception:
                    pass

    # Step 4: 逆编码标签列
    y_inv = y_result[:n_result].copy()
    if label_col in label_encoders:
        le = label_encoders[label_col]
        y_rounded = np.round(y_inv).astype(int)
        y_rounded = np.clip(y_rounded, 0, len(le.classes_) - 1)
        try:
            y_inv = le.inverse_transform(y_rounded)
        except Exception:
            pass
    result_df[label_col] = y_inv

    # Step 5: 恢复 index 列
    if 'index' in dirty_df.columns:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            kept_indices = dirty_df.loc[keep_mask, 'index'].values[:n_result]
        else:
            kept_indices = dirty_df['index'].values[:n_result]
        result_df.insert(0, 'index', kept_indices)

    # Step 5.5: 恢复非特征列（如 'id'）
    extra_cols = [c for c in dirty_df.columns
                  if c not in result_df.columns and c not in ('index',)]
    if extra_cols:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            src = dirty_df.loc[keep_mask].iloc[:n_result]
        else:
            src = dirty_df.iloc[:n_result]
        for col in extra_cols:
            result_df[col] = src[col].values

    # Step 6: 列顺序对齐 dirty_df
    target_cols = [c for c in dirty_df.columns if c in result_df.columns]
    result_df = result_df[target_cols]

    return result_df


def get_project_root() -> str:
    """返回项目根目录路径"""
    return _PROJECT_ROOT
