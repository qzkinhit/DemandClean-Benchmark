"""
DemandClean 统一运行脚本
========================

支持 8 种版本组合（2 检测器 x 2 Agent x 2 推理模式）:
  v1: oracle + dueling_two_stage + single_phase
  v2: oracle + dueling_two_stage + two_phase
  v3: oracle + single_stage      + single_phase
  v4: oracle + single_stage      + two_phase
  v5: auto   + dueling_two_stage + single_phase
  v6: auto   + dueling_two_stage + two_phase
  v7: auto   + single_stage      + single_phase
  v8: auto   + single_stage      + two_phase

每个版本完整流程:
  训练 -> 推理 -> 5种Baseline评估 -> 容忍度评估 -> 检测器准确率 -> Shapley分析 -> 训练曲线

用法:
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

# 编辑距离安全编码
try:
    from demandclean.utils.edit_distance import find_nearest_known as _find_nearest_known
except ImportError:
    _find_nearest_known = None


# =========================================================================
# TeeLogger: 同时输出到终端和日志文件
# =========================================================================
class TeeLogger:
    """将 stdout/stderr 同时写入终端和日志文件（容错版）"""

    def __init__(self, log_path: str, stream=None):
        self.terminal = stream or sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # 行缓冲

    @property
    def _file_ok(self):
        """检查 log_file 是否仍然可写"""
        try:
            return self.log_file is not None and not self.log_file.closed
        except Exception:
            return False

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception:
            # terminal 也挂了，尝试原始 stdout
            if sys.__stdout__ is not None and not sys.__stdout__.closed:
                sys.__stdout__.write(message)
        if self._file_ok:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (ValueError, IOError, OSError):
                pass  # 静默失败，降级为仅终端输出

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
# 项目根目录
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# DEBUG: 打印 sys.path 前几项（仅主进程）
if __name__ == "__main__":
    print(f"[DEBUG] _PROJECT_ROOT={_PROJECT_ROOT}", file=sys.stderr)
try:
    import tools as _t
    pass  # 静默检查
except Exception:
    pass

# =========================================================================
# 关键导入
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

# 延迟导入: 避免 multiprocessing spawn 模式重执行主脚本时失败
# RAHA 使用 multiprocessing，子进程会重新加载主模块
try:
    from tools.rules_parser import get_horizon_fds
except (ModuleNotFoundError, ImportError):
    # 子进程中不需要此模块
    get_horizon_fds = None


# ============================================================================
# 版本开关（8 个）
# ============================================================================
ENABLE_ORACLE_DUELING_SINGLE_PHASE = False  # v1
ENABLE_ORACLE_DUELING_TWO_PHASE   = False  # v2
ENABLE_ORACLE_PLAIN_SINGLE_PHASE  = False  # v3
ENABLE_ORACLE_PLAIN_TWO_PHASE     = False  # v4
ENABLE_AUTO_DUELING_SINGLE_PHASE  = True   # v5 (默认版本)
ENABLE_AUTO_DUELING_TWO_PHASE     = False  # v6
ENABLE_AUTO_PLAIN_SINGLE_PHASE    = False  # v7
ENABLE_AUTO_PLAIN_TWO_PHASE       = False  # v8


# ============================================================================
# VERSION_CONFIGS: 每个版本的配置字典
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
# 数据集配置
# ============================================================================
DATASETS = {
    'beers': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'style',
        # beer_name/brewery_name/city/state 是文本；ounces/abv 虽含'empty'但本质数值
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
        'categorical_cols': set(),  # 全部数值列
        'protected_cols': set(),
    },
    'breast_cancer': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'class',
        'categorical_cols': set(),  # 全部 INT [1,10] 数值列
        'protected_cols': set(),
    },
    'har': {
        'task_type': 'clustering',
        'model_type': 'kmeans',
        'label_col': 'gt',
        'categorical_cols': set(),  # x/y/z 全部 FLOAT
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
        'categorical_cols': set(),  # frequency/angle/chord_length/velocity 本质数值
        'protected_cols': set(),
    },
    'smartfactory': {
        'task_type': 'classification',
        'model_type': 'random_forest',
        'label_col': 'labels',
        'categorical_cols': set(),  # 全部 INT 传感器数值
        'protected_cols': set(),
    },
    'soilmoisture': {
        'task_type': 'regression',
        'model_type': 'ridge',
        'label_col': 'soil_moisture',
        'categorical_cols': set(),  # 全部 FLOAT 光谱波段
        'protected_cols': set(),
    },
}

# 评估用的模型简称列表（分类 / 回归 / 聚类 各两个）
EVAL_MODELS_CLASSIFICATION = ['rf', 'lr']
EVAL_MODELS_REGRESSION = ['rf', 'ridge']
EVAL_MODELS_CLUSTERING = ['kmeans']  # AgglomerativeClustering O(n²~n³) 大数据集不可接受，仅用 KMeans

# 训练时 reward 评估用的轻量化模型参数（按 model_type 分配）
# 每步 reward 评估需要 fit+evaluate，必须保证速度；baseline 评估使用独立标准模型不受影响
REWARD_MODEL_KWARGS = {
    'random_forest': {'n_estimators': 10, 'max_depth': 8},
    'xgboost':       {'n_estimators': 10, 'max_depth': 3},
    'xgboost_reg':   {'n_estimators': 10, 'max_depth': 3},
    'ridge':         {},  # 线性模型已足够轻量
    'linear':        {},  # 线性模型已足够轻量
    'svm':           {},  # 线性核 SVM 在小数据集上可接受
    'kmeans':        {'n_init': 3},  # 减少 KMeans 初始化次数
}


# ============================================================================
# 数据预处理
# ============================================================================
def preprocess_data(dataset_name: str, dirty_df: pd.DataFrame = None, clean_df: pd.DataFrame = None):
    """
    读取并预处理指定数据集。

    步骤:
      1. 读取 dirty_with_index.csv 和 clean_with_index.csv（或使用传入的 DataFrame）
      2. 将 "empty"/"Empty"/"EMPTY" 替换为 NaN
      3. LabelEncoder 仅在 dirty 数据上 fit（不依赖 clean）
      4. StandardScaler 仅在 dirty 数据上 fit
      5. 自动推导并解析 FD 规则

    Args:
        dataset_name: 数据集名称 (如 'beers')
        dirty_df: 可选的脏数据 DataFrame（传入时不从文件读取）
        clean_df: 可选的干净数据 DataFrame（传入时用于编码；None 则不编码 clean）

    Returns:
        (X_dirty_scaled, y_dirty, X_clean_scaled_or_None, y_clean_or_None,
         column_names, fd_rules, rules_path,
         dirty_csv_path, clean_csv_path, csv_columns,
         scaler, label_encoders, categorical_cols,
         dirty_df, clean_df_or_None)
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

    # --- 读取（或使用传入的 DataFrame）---
    read_clean_from_file = (dirty_df is None and clean_df is None)  # 默认行为：都从文件读
    if dirty_df is None:
        dirty_df = pd.read_csv(dirty_path)
    if clean_df is None and read_clean_from_file:
        clean_df = pd.read_csv(clean_path)
    # 确保 dirty_df 有干净的列名
    dirty_df.columns = [c.strip().strip('\ufeff') for c in dirty_df.columns]
    if clean_df is not None:
        clean_df.columns = [c.strip().strip('\ufeff') for c in clean_df.columns]

    # --- 替换 empty 为 NaN ---
    dirty_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)
    if clean_df is not None:
        clean_df.replace(['empty', 'Empty', 'EMPTY', 'nan', 'NaN', 'NULL', 'null'], np.nan, inplace=True)

    # --- 确定特征列 ---
    drop_cols = [c for c in ['index', 'id', label_col] if c in dirty_df.columns]
    feature_cols = [c for c in dirty_df.columns if c not in drop_cols]

    # --- 识别分类列 vs 数值列 ---
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

    # --- 编码函数 ---
    # LabelEncoder 仅在 dirty 数据上 fit（不依赖 clean）
    label_encoders = {}

    def encode_df(df, feature_cols, label_col, fit_le=False):
        """编码 DataFrame 的特征和标签，返回 (X, y)

        Args:
            df: 待编码的 DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            fit_le: 是否 fit LabelEncoder（仅 dirty 数据设为 True）
        """
        X_df = df[feature_cols].copy()
        y_series = df[label_col].copy()

        for col in feature_cols:
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

                if _find_nearest_known is not None:
                    known_set = set(le.classes_)
                    known_list = list(le.classes_)

                    def _safe_encode(val, _le=le, _ks=known_set, _kl=known_list):
                        if val in _ks:
                            return _le.transform([val])[0]
                        nearest = _find_nearest_known(val, _kl, threshold=0.3)
                        if nearest is not None:
                            return _le.transform([nearest])[0]
                        return np.nan  # dirty-fit LE 下此行不会执行

                    X_df.loc[~nan_mask, col] = non_nan_values.map(_safe_encode)
                else:
                    X_df.loc[~nan_mask, col] = le.transform(non_nan_values)
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # 编码标签
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

    # 编码 dirty（同时 fit LE）
    X_dirty, y_dirty = encode_df(dirty_df, feature_cols, label_col, fit_le=True)

    # 编码 clean（使用已 fit 的 LE）
    X_clean_scaled = None
    y_clean = None
    if clean_df is not None:
        X_clean, y_clean = encode_df(clean_df, feature_cols, label_col, fit_le=False)

    # --- 标准化（仅在 dirty 数据上 fit）---
    scaler = StandardScaler()
    X_dirty_for_fit = X_dirty.copy()
    col_means = np.nanmean(X_dirty_for_fit, axis=0)
    for j in range(X_dirty_for_fit.shape[1]):
        mask = np.isnan(X_dirty_for_fit[:, j])
        X_dirty_for_fit[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    scaler.fit(X_dirty_for_fit)

    # 对 dirty 标准化（保留 NaN 位置）
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

    # 对 clean 标准化（如果有）
    if clean_df is not None:
        X_clean_for_scale = X_clean.copy()
        col_means_c = np.nanmean(X_clean_for_scale, axis=0)
        for j in range(X_clean_for_scale.shape[1]):
            mask = np.isnan(X_clean_for_scale[:, j])
            X_clean_for_scale[mask, j] = col_means_c[j] if not np.isnan(col_means_c[j]) else 0.0
        X_clean_scaled = scaler.transform(X_clean_for_scale)

    # --- 解析 FD 规则 ---
    fd_rules = []
    if os.path.exists(rules_path) and get_horizon_fds is not None:
        try:
            fd_rules = get_horizon_fds(rules_path)
        except Exception as e:
            print(f"  [警告] 解析FD规则失败: {e}")

    print(f"  数据集: {dataset_name}")
    print(f"  脏数据: {X_dirty_scaled.shape}")
    if X_clean_scaled is not None:
        print(f"  干净数据: {X_clean_scaled.shape}")
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
# encode_subset: 用已有的 LE/SS 编码一个数据子集（val/test）
# ============================================================================
def encode_subset(df, feature_cols, label_col, label_encoders, scaler,
                  categorical_cols, dataset_name=None):
    """用已有的 LE/SS 编码一个数据子集（val/test），返回 (X_scaled, y)

    Args:
        df: 待编码的 DataFrame
        feature_cols: 特征列名列表
        label_col: 标签列名
        label_encoders: 已 fit 的 {col_name: LabelEncoder}
        scaler: 已 fit 的 StandardScaler
        categorical_cols: 分类列名集合
        dataset_name: 可选，用于日志

    Returns:
        (X_scaled, y) — 标准化后的特征矩阵和标签数组
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

    # 编码标签
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

    # 标准化（NaN 安全）
    X_out = X.copy()
    col_m = np.nanmean(X_out, axis=0)
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = col_m[j] if not np.isnan(col_m[j]) else 0.0
    X_scaled = scaler.transform(X_out)

    return X_scaled, y


# ============================================================================
# 可视化: 训练曲线 (4子图)
# ============================================================================
def plot_training_curves(history: dict, save_dir: str, version_name: str,
                         resume_episode: int = 0):
    """
    绘制训练过程的 6 张子图 (3×2):
      Score, Reward, Epsilon, Action Count, Action Ratio, Q-Value Uncertainty

    Args:
        history: 训练历史字典
        save_dir: 保存目录
        version_name: 版本名称
        resume_episode: 续训起点 episode（0=无续训）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f'{version_name} - Training Curves', fontsize=14)

    episodes = history.get('episode', list(range(1, len(history.get('score', [])) + 1)))

    def _split_old_new(data, eps_list, resume_ep):
        """分割续训前后的数据"""
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
        """续训感知绘图：旧数据灰色虚线，新数据彩色实线"""
        old_d, old_e, new_d, new_e = _split_old_new(data, episodes_list, resume_ep)
        if old_d:
            ax.plot(old_e, old_d, alpha=0.25, color='gray', linestyle='--')
        if new_d:
            ax.plot(new_e, new_d, alpha=0.3, color=color, label=label)
        # 续训标记线
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

    # --- Action Ratio (堆叠面积图) ---
    ax = axes[2, 0]
    action_keys = ['no_action', 'repair_value', 'delete', 'replace_nearby']
    action_colors = ['gray', 'blue', 'red', 'orange']
    action_arrays = []
    for key in action_keys:
        vals = history.get(key, [])
        action_arrays.append(np.array(vals) if vals else np.zeros(len(episodes)))
    # 计算占比
    total = np.sum(action_arrays, axis=0)
    total = np.where(total > 0, total, 1)  # 避免除零
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
        # 动作概率熵
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
    print(f"  [可视化] 训练曲线已保存: {path}")


# ============================================================================
# 可视化: 散点图 (分类用PCA散点, 回归用实际vs预测)
# ============================================================================
def compute_auth_div(X_result, X_clean, X_dirty):
    """
    计算 Authenticity (真实性) 和 Diversity (多样性)

    参考: real_beers_experiment_with_detector.py:692-722

    Args:
        X_result: 策略处理后的数据 (可能行数 < X_clean)
        X_clean: 干净数据 (全量)
        X_dirty: 脏数据 (全量)

    Returns:
        (auth, div) — 真实性 [0,1] 和 多样性 [0,1]
    """
    n = len(X_result)
    n_total = len(X_clean)
    if n == 0:
        return 0.0, 0.0

    # 真实性: 修复后数据与干净数据的逐行一致比例
    compare_len = min(n, n_total)
    correct = 0
    for i in range(compare_len):
        if np.allclose(X_result[i], X_clean[i], atol=0.01, equal_nan=False):
            correct += 1
    auth = correct / n if n > 0 else 0.0

    # 多样性: 样本保留率 × 噪声保留率
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
    """NaN 安全处理: 用列均值填充 NaN"""
    X_c = X.copy()
    col_means = np.nanmean(X_c, axis=0)
    for j in range(X_c.shape[1]):
        mask = np.isnan(X_c[:, j])
        X_c[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    return X_c



# ============================================================================
# 5种Baseline评估 + 可视化
# ============================================================================
def _knn_label_estimate(X, y, idx, deleted_rows, task_type, k=5):
    """KNN 标签估计（从 CleaningEnv._get_majority_label 简化）"""
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
    评估 6 种 baseline 并生成决策边界对比图:
      NoFix, DeleteAll, DeleteFix, ReplaceAll, DemandClean, FullFix

    Args:
        gt_cost: DemandClean 使用的真值成本 (ground_truth_used)

    Returns:
        (baseline_results, vis_time) — 评估指标字典 + 可视化耗时(秒)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    def safe_prepare(X, y, name):
        """NaN 安全处理: 去掉含 NaN 的行（X 或 y），用列均值填充其余 NaN"""
        X_c = X.copy()
        y_c = y.copy() if len(y) == len(X) else y[:len(X)].copy()
        # 对齐长度
        min_len = min(len(X_c), len(y_c))
        X_c = X_c[:min_len]
        y_c = y_c[:min_len]
        # 去掉 y 中的 NaN
        y_valid = ~np.isnan(y_c)
        X_c = X_c[y_valid]
        y_c = y_c[y_valid]
        # 去掉 X 中全 NaN 的行
        x_valid = ~np.isnan(X_c).all(axis=1)
        X_c = X_c[x_valid]
        y_c = y_c[x_valid]
        # 用列均值填充剩余 NaN
        col_means = np.nanmean(X_c, axis=0)
        for j in range(X_c.shape[1]):
            mask = np.isnan(X_c[:, j])
            X_c[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
        return X_c, y_c

    # --- 准备各 baseline 的数据 ---

    # NoFix: 直接用脏数据
    X_nofix, y_nofix = safe_prepare(X_dirty, y_dirty, 'NoFix')

    # FullFix: 直接用干净数据
    X_fullfix, y_fullfix = safe_prepare(X_clean, y_clean, 'FullFix')

    # DeleteAll: 删除所有含 NaN 的行
    nan_mask = ~np.isnan(X_dirty).any(axis=1)
    X_delete = X_dirty[nan_mask].copy()
    y_delete = y_dirty[nan_mask].copy()
    X_delete, y_delete = safe_prepare(X_delete, y_delete, 'DeleteAll')

    # ReplaceAll: 用 VE 估值替换所有检出错误单元格
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
                    # 标签错误: KNN 多数投票
                    y_replace[idx_err] = _knn_label_estimate(
                        X_replace, y_replace, idx_err, deleted_rows, task_type)
                else:
                    if 0 <= idx_err < X_replace.shape[0] and 0 <= col_err < X_replace.shape[1]:
                        X_replace[idx_err, col_err] = value_estimator.estimate_feature_value(
                            X_replace, idx_err, col_err, deleted_rows, col_means)
                ve_count += 1
        ve_time = time.time() - t_ve
        print(f"  ReplaceAll VE 估值: {ve_count} 个错误, {ve_time:.1f}s")
    X_replace, y_replace = safe_prepare(X_replace, y_replace, 'ReplaceAll')

    # DemandClean: 系统清洗后的结果
    X_dc, y_dc = safe_prepare(X_result, y_result, 'DemandClean')

    # DeleteFix: 删除检测器标记为有错误的行（至少保留 20% 数据）
    X_dfix, y_dfix = None, None
    if detected_errors is not None:
        # 收集所有被检测到有错误的行索引
        error_row_set = set()
        for err_type in ('missing', 'semantic', 'syntactic', 'label_noise'):
            for item in detected_errors.get(err_type, []):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    error_row_set.add(int(item[0]))
                elif isinstance(item, dict) and 'idx' in item:
                    error_row_set.add(int(item['idx']))

        n_total = len(X_dirty)
        min_keep = max(10, int(n_total * 0.2))  # 至少保留 20%

        if len(error_row_set) <= n_total - min_keep:
            # 删除所有检测到的错误行
            keep_mask = np.array([i not in error_row_set for i in range(n_total)])
        else:
            # 删除过多，按类型优先级排序：优先删标签噪声和缺失值（高置信度）
            from collections import Counter
            row_error_count = Counter()
            for err_type in ('label_noise', 'missing', 'syntactic', 'semantic'):
                for item in detected_errors.get(err_type, []):
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        row_error_count[int(item[0])] += 1
                    elif isinstance(item, dict) and 'idx' in item:
                        row_error_count[int(item['idx'])] += 1
            # 按错误数量降序排列，取前 (n_total - min_keep) 个
            max_delete = n_total - min_keep
            sorted_rows = sorted(row_error_count.keys(),
                                 key=lambda r: row_error_count[r], reverse=True)
            delete_set = set(sorted_rows[:max_delete])
            keep_mask = np.array([i not in delete_set for i in range(n_total)])

        X_dfix_raw = X_dirty[keep_mask].copy()
        y_dfix_raw = y_dirty[keep_mask].copy()
        X_dfix, y_dfix = safe_prepare(X_dfix_raw, y_dfix_raw, 'DeleteFix')
        print(f"  DeleteFix: 删除 {n_total - keep_mask.sum()} 行, 保留 {keep_mask.sum()} 行")

    datasets = {
        'NoFix': (X_nofix, y_nofix),
        'DeleteAll': (X_delete, y_delete),
    }
    if X_dfix is not None and len(X_dfix) >= 10:
        datasets['DeleteFix'] = (X_dfix, y_dfix)
    datasets['ReplaceAll'] = (X_replace, y_replace)
    datasets['DemandClean'] = (X_dc, y_dc)
    datasets['FullFix'] = (X_fullfix, y_fullfix)

    # --- 统一测试集（从 clean 数据固定划分）---
    # Oracle 模式: 使用外部提供的测试集; 默认: 从 clean 数据 80/20 划分
    if oracle_test_X is not None and oracle_test_y is not None:
        X_test_common, y_test_common = safe_prepare(
            oracle_test_X, oracle_test_y, 'OracleTest')
        print(f"  统一测试集 (Oracle): {len(X_test_common)} 行")
        # Oracle 模式下所有 baseline 用全部传入数据训练（不再内部划分）
        n_total_rows = len(X_clean)
        train_indices = np.arange(n_total_rows)
        test_indices = None  # 不使用
    else:
        # 所有 baseline 共享同一个测试集，确保公平比较
        n_total_rows = len(X_clean)
        all_indices = np.arange(n_total_rows)
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        # 公共测试集（clean 数据）
        X_test_common_raw = X_clean[test_indices].copy()
        y_test_common_raw = y_clean[test_indices].copy()
        X_test_common, y_test_common = safe_prepare(
            X_test_common_raw, y_test_common_raw, 'CommonTest')
        print(f"  统一测试集: {len(X_test_common)} 行 (从 {n_total_rows} 行 clean 数据固定划分)")

    # 选择评估模型
    if task_type == 'classification':
        eval_models = EVAL_MODELS_CLASSIFICATION
    elif task_type == 'clustering':
        eval_models = EVAL_MODELS_CLUSTERING
    else:
        eval_models = EVAL_MODELS_REGRESSION

    results_all = {}

    print(f"\n{'='*60}")
    print(f"Baseline 对比评估 - {version_name}")
    print(f"{'='*60}")

    for ds_name, (X_ds, y_ds) in datasets.items():
        t_bl = time.time()
        if len(X_ds) < 10:
            print(f"  {ds_name}: 数据不足({len(X_ds)}行)，跳过")
            results_all[ds_name] = {}
            continue

        ds_results = {'n_samples': len(X_ds), 'n_test': len(y_test_common)}

        try:
            # 构建训练数据：Oracle 模式用全量（已在外部划分），否则用 train_indices 子集
            if oracle_test_X is not None:
                # Oracle 模式: 传入的数据已是 train 子集
                X_train_raw = X_ds
                y_train_raw = y_ds
            elif len(X_ds) == n_total_rows:
                X_train_raw = X_ds[train_indices]
                y_train_raw = y_ds[train_indices]
            else:
                # DeleteAll/DeleteFix 行数不同，用全部数据训练
                X_train_raw = X_ds
                y_train_raw = y_ds

            X_train, y_train = safe_prepare(X_train_raw, y_train_raw, ds_name)

            if len(X_train) < 10:
                print(f"  {ds_name}: 训练数据不足({len(X_train)}行)，跳过")
                results_all[ds_name] = {}
                continue

            # 数据已在 preprocess_data 中经过 LE+SS 标准化，直接使用
            # 不再二次 StandardScaler（避免双重标准化导致与 Step 9 不一致）
            X_train_scaled = X_train
            X_test_scaled = X_test_common

            if task_type == 'clustering':
                # 聚类评估：fit_predict 全量，不使用 train/test split
                from sklearn.cluster import KMeans as _KMeans
                from sklearn.metrics import silhouette_score as _sil_score, adjusted_rand_score as _ari_score
                # 聚类用原始全量数据（已标准化，不再二次处理）
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

            # (可视化在评估循环之后统一完成)

        except Exception as e:
            ds_results['error'] = str(e)

        results_all[ds_name] = ds_results

        # 打印主要指标（含耗时）
        bl_time = time.time() - t_bl
        metric_str = ' | '.join(
            f'{k}={v}' for k, v in ds_results.items()
            if not k.startswith('n_') and 'error' not in k
        )
        print(f"  {ds_name:15s}: {metric_str} | {bl_time:.1f}s")

    # --- 计算 Auth/Div/Cost 并写入各 baseline 的结果 ---
    n_total = len(X_clean)
    # 各 baseline 的 cost 定义
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
# Baseline 对比柱状图
# ============================================================================
def plot_baseline_comparison(results_all: dict, save_dir: str,
                              version_name: str, task_type: str,
                              primary_model: str):
    """
    柱状图对比各 Baseline 的主要指标

    Args:
        results_all: evaluate_and_visualize 的返回结果
        save_dir: 保存目录
        version_name: 版本名称
        task_type: 任务类型
        primary_model: 主要模型简称 (如 'rf')
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
    print(f"  [可视化] Baseline 对比图已保存: {path}")


# ============================================================================
# 逆转换: 将标准化后的 numpy 还原为原始 CSV 格式
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
    将 DemandClean 输出的标准化 numpy 数组逆转换为原始 CSV 格式的 DataFrame。

    策略:
      1. 以原始 dirty_df 为基础（保留 index 列和原始值格式）
      2. 根据 keep_mask 筛选保留的行
      3. 对标准化的 X_result 做 scaler.inverse_transform
      4. 对编码的分类列做 LabelEncoder.inverse_transform
      5. 对标签列也做逆转换

    Args:
        X_result: 清洗后的标准化特征矩阵 (n_kept, n_features)
        y_result: 清洗后的标签向量 (n_kept,)
        feature_cols: 特征列名列表
        label_col: 标签列名
        scaler: StandardScaler 对象
        label_encoders: {col_name: LabelEncoder} 字典
        categorical_cols: 分类列名集合
        dirty_df: 原始脏数据 DataFrame（含 index 列）
        inference_mode: 推理模式
        keep_mask: 布尔数组，标识原始行中哪些被保留

    Returns:
        原始格式的 cleaned DataFrame（与 dirty_df 同格式，含 index 列）
    """
    n_result = len(X_result)

    # 边界情况: agent 删除了所有行 → 返回空 DataFrame（保持列结构与 dirty_df 一致）
    if n_result == 0:
        return pd.DataFrame(columns=dirty_df.columns)

    # Step 1: 逆标准化
    X_inv = scaler.inverse_transform(X_result)

    # Step 2: 构建逆转换后的 DataFrame
    result_df = pd.DataFrame(X_inv, columns=feature_cols)

    # Step 3: 逆编码分类列
    if label_encoders and categorical_cols:
        for col in feature_cols:
            if col in categorical_cols and col in label_encoders:
                le = label_encoders[col]
                col_vals = result_df[col].values
                n_classes = len(le.classes_)

                if n_classes > 0 and scaler is not None:
                    # 使用 scaler 信息计算编码空间中每个合法整数值对应的标准化值
                    col_idx = feature_cols.index(col)
                    mean_val = scaler.mean_[col_idx]
                    scale_val = scaler.scale_[col_idx]
                    # 合法编码值 [0, 1, ..., n_classes-1] 在标准化空间中的值
                    valid_scaled = np.array([(i - mean_val) / scale_val for i in range(n_classes)])

                    # 每个值找标准化空间中最近的合法编码
                    col_rounded = np.zeros(len(col_vals), dtype=int)
                    for i, v in enumerate(col_vals):
                        nearest_idx = np.argmin(np.abs(valid_scaled - v))
                        col_rounded[i] = nearest_idx
                else:
                    # fallback: 简单 round + clip
                    col_rounded = np.round(col_vals).astype(int)
                    col_rounded = np.clip(col_rounded, 0, n_classes - 1)

                try:
                    result_df[col] = le.inverse_transform(col_rounded)
                except Exception:
                    # 如果逆转换失败，尝试编辑距离安全编码
                    if _find_nearest_known is not None:
                        safe_vals = []
                        known = list(le.classes_)
                        for v in col_vals:
                            nearest = _find_nearest_known(str(v), known)
                            safe_vals.append(nearest if nearest else known[0])
                        result_df[col] = safe_vals
                    # 否则保留数值

    # Step 4: 逆编码标签列
    y_inv = y_result[:n_result].copy()
    if label_col in label_encoders:
        le = label_encoders[label_col]
        n_classes = len(le.classes_)

        if n_classes > 0 and scaler is not None and label_col in feature_cols:
            # 标签列可能也经过 scaler（如果是特征的一部分），用最近邻匹配
            col_idx = feature_cols.index(label_col)
            mean_val = scaler.mean_[col_idx]
            scale_val = scaler.scale_[col_idx]
            valid_scaled = np.array([(i - mean_val) / scale_val for i in range(n_classes)])
            y_rounded = np.zeros(len(y_inv), dtype=int)
            for i, v in enumerate(y_inv):
                nearest_idx = np.argmin(np.abs(valid_scaled - v))
                y_rounded[i] = nearest_idx
        else:
            # 标签列通常不经过 scaler，简单 round + clip
            y_rounded = np.round(y_inv).astype(int)
            y_rounded = np.clip(y_rounded, 0, n_classes - 1)

        try:
            y_inv = le.inverse_transform(y_rounded)
        except Exception:
            pass
    result_df[label_col] = y_inv

    # Step 5: 恢复 index 列
    if 'index' in dirty_df.columns:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            # keep_mask 对应原始 dirty_df 的行
            kept_indices = dirty_df.loc[keep_mask, 'index'].values[:n_result]
        else:
            # 默认取前 n_result 行的 index
            kept_indices = dirty_df['index'].values[:n_result]
        result_df.insert(0, 'index', kept_indices)

    # Step 5.5: 恢复被 preprocess_data drop 掉的非特征列（如 'id'）
    #   dirty_df 中有但 feature_cols / label_col / 'index' 之外的列，
    #   从对应行的 dirty_df 中原样复制回来
    extra_cols = [c for c in dirty_df.columns
                  if c not in result_df.columns and c not in ('index',)]
    if extra_cols:
        if keep_mask is not None and len(keep_mask) == len(dirty_df):
            src = dirty_df.loc[keep_mask].iloc[:n_result]
        else:
            src = dirty_df.iloc[:n_result]
        for col in extra_cols:
            result_df[col] = src[col].values

    # Step 6: 确保列顺序与 dirty_df 一致
    target_cols = [c for c in dirty_df.columns if c in result_df.columns]
    result_df = result_df[target_cols]

    return result_df


# ============================================================================
# run_version(): 运行单个版本的完整流程
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
    运行单个版本的完整流程

    流程:
      1. 创建 DemandClean 实例
      2. 训练 (dc.fit)
      3. 推理 (dc.clean 或 dc.plan/execute)
      4. 评估 5 种 baseline
      5. 评估容忍度 (compute_model_tolerance)
      6. 评估检测器准确率 (evaluate_detector_accuracy)
      7. Shapley 三维度分析 (run_full_shapley_analysis)
      8. 训练曲线可视化
      9. 保存输出到 results/demandclean/{dataset}/{version_name}/

    Args:
        dataset_name: 数据集名称
        version_cfg: 版本配置字典
        X_dirty, y_dirty: 脏数据
        X_clean, y_clean: 干净数据
        column_names: 特征列名
        fd_rules: FD 规则列表
        rules_path: 规则文件路径
        n_episodes: 训练轮数
        verbose: 是否输出详细信息

    Returns:
        版本运行结果字典
    """
    version_name = version_cfg['version_name']
    # 输出目录后缀（如 NGT 变体用 _ngt 避免覆盖原始版本）
    output_version_name = version_name + output_suffix if output_suffix else version_name
    detector_mode = version_cfg['detector_mode']
    agent_type = version_cfg['agent_type']
    inference_mode = version_cfg['inference_mode']

    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']
    model_type = ds_cfg['model_type']

    # --- 创建输出目录（分类子目录） ---
    save_dir = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name, output_version_name)
    dir_model = os.path.join(save_dir, 'model')
    dir_data = os.path.join(save_dir, 'data')
    dir_eval = os.path.join(save_dir, 'evaluation')
    dir_vis = os.path.join(save_dir, 'visualization')
    dir_report = os.path.join(save_dir, 'report')
    for d in [save_dir, dir_model, dir_data, dir_eval, dir_vis, dir_report]:
        os.makedirs(d, exist_ok=True)

    start_time = time.time()
    step_times = {}  # 各步骤计时

    print("\n" + "=" * 70)
    print(f"版本: {version_name}")
    print(f"数据集: {dataset_name} | 任务: {task_type} | 模型: {model_type}")
    print(f"检测器: {detector_mode} | Agent: {agent_type} | 推理: {inference_mode}")
    print(f"训练轮数: {n_episodes} | 输出目录: {save_dir}")
    print("=" * 70)

    # 选择评估模型
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
    # visualize_only 模式: 跳过训练和推理，直接从文件加载清洗结果
    # ================================================================
    history_path = os.path.join(dir_report, f'{version_name}_history.json')
    prev_history = None

    if visualize_only:
        print(f"\n[visualize_only] 跳过训练和推理，从已有结果加载...")

        # 加载清洗后的数据
        cleaned_csv = os.path.join(dir_data, f'{version_name}_cleaned.csv')
        if not os.path.exists(cleaned_csv):
            print(f"  [错误] 找不到清洗结果: {cleaned_csv}，无法仅可视化")
            return report

        cleaned_df = pd.read_csv(cleaned_csv)
        X_result = cleaned_df[column_names].values
        y_result = cleaned_df[ds_cfg['label_col']].values if ds_cfg['label_col'] in cleaned_df.columns else y_dirty[:len(cleaned_df)]
        print(f"  已加载清洗结果: {cleaned_csv} ({len(X_result)} 行)")

        # 加载训练历史
        history = None
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    history = json.load(f)
                print(f"  已加载训练历史: {len(history.get('score', []))} episodes")
            except Exception as e:
                print(f"  [警告] 加载训练历史失败: {e}")

        # 加载之前的 report
        report_path = os.path.join(dir_report, f'{version_name}_report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    old_report = json.load(f)
                # 保留旧报告中的训练相关字段
                for key in ['action_counts', 'ground_truth_used', 'repair_log',
                            'detector_accuracy', 'raha_cost']:
                    if key in old_report:
                        report[key] = old_report[key]
                print(f"  已加载旧报告: {report_path}")
            except Exception as e:
                print(f"  [警告] 加载旧报告失败: {e}")

        # 跳过 dc 实例化，detected_errors 和 dc 置 None
        dc = None
        detected_errors = None
        output_csv_original = os.path.join(dir_data, f'{version_name}_cleaned_original.csv')
        if not os.path.exists(output_csv_original):
            output_csv_original = None

        # 尝试加载编码版本（若之前运行时已保存）
        output_npz = os.path.join(dir_data, f'{version_name}_cleaned_encoded.npz')
        if not os.path.exists(output_npz):
            output_npz = None

        # 从旧 report 恢复后续步骤需要的局部变量
        ground_truth_used = report.get('ground_truth_used', 0)
        detector_accuracy = report.get('detector_accuracy', {})

    # ================================================================
    # Step 1-4: 训练、推理、保存（visualize_only 模式跳过）
    # ================================================================
    if not visualize_only:

        # ================================================================
        # 1. 创建 DemandClean 实例
        # ================================================================
        print("\n[Step 1] 创建 DemandClean 实例...")
        t_step = time.time()

        # 轻量化模型参数（训练时 reward 评估用，baseline 评估不受影响）
        # reward_model_type 可覆盖 reward 评估用的模型类型（不影响 baseline 评估）
        effective_reward_model_type = reward_model_type or model_type
        reward_model_kwargs = REWARD_MODEL_KWARGS.get(effective_reward_model_type, {})

        # 构建额外配置参数
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

        # 命令行覆盖 reward 参数
        if delete_shaping_reward is not None:
            extra_kwargs['delete_shaping_reward'] = delete_shaping_reward
        if keep_rate_weight is not None:
            extra_kwargs['keep_rate_weight'] = keep_rate_weight

        # 回归任务自动启用优化 reward 参数（分类/聚类用默认值）
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
            # Reward 评估配置
            model_kwargs=reward_model_kwargs,
            eval_sample_ratio=1.0,
            # 编码工具（用于 ErrorInjector 在 CSV 空间注入）
            encoding_label_encoders=label_encoders,
            encoding_scaler=data_scaler,
            encoding_categorical_cols=categorical_cols,
            encoding_dirty_df=dirty_df,
            encoding_clean_df=clean_df,
            **(inject_kwargs or {}),
            **extra_kwargs,
        )
        step_times['step1_init'] = round(time.time() - t_step, 2)
        print(f"  [计时] Step 1 (初始化): {step_times['step1_init']:.2f}s")

        # ================================================================
        # 2. 训练（支持续训 / inference_only 跳过训练）
        # ================================================================
        model_path = os.path.join(dir_model, f'{version_name}_agent.pt')
        history_path = os.path.join(dir_report, f'{version_name}_history.json')

        t_step = time.time()

        if inference_only:
            # 仅推理模式：跳过训练，直接加载已有模型
            if not ModelIO.agent_model_exists(model_path):
                raise FileNotFoundError(
                    f"inference_only 模式但模型不存在: {model_path}")
            dc.load(model_path)
            print(f"\n[Step 2] 跳过训练 (inference_only)，已加载模型: {model_path}")
            step_times['step2_train'] = 0.0
        else:
            resume_from = None
            prev_history = None

            if resume_mode == 'auto' and ModelIO.agent_model_exists(model_path):
                # 自动检测已有模型，续训
                resume_from = model_path
                print(f"\n[Step 2] 续训 (检测到已有模型: {model_path})...")
                if os.path.exists(history_path):
                    try:
                        with open(history_path) as f:
                            prev_history = json.load(f)
                        prev_eps = len(prev_history.get('score', []))
                        print(f"  已有训练历史: {prev_eps} episodes")
                    except Exception as e:
                        print(f"  [警告] 加载训练历史失败: {e}")
                        prev_history = None
                print(f"  新增训练: {n_episodes} episodes")
            else:
                print(f"\n[Step 2] 训练 ({n_episodes} episodes)...")

            dc.fit(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean,
                   n_episodes=n_episodes, verbose=verbose,
                   resume_from=resume_from, prev_history=prev_history,
                   X_clean_val=oracle_split['X_clean_val'] if oracle_split else None,
                   y_clean_val=oracle_split['y_clean_val'] if oracle_split else None)

            # 保存模型
            try:
                dc.save(model_path)
                print(f"  模型已保存: {model_path}")
            except Exception as e:
                print(f"  [警告] 模型保存失败: {e}")

            # 保存训练历史
            history = dc.get_training_history()
            if history and history.get('score'):
                history_file = os.path.join(dir_report, f'{version_name}_history.json')
                try:
                    # action_probs 是 list[list]，需要特殊处理
                    serializable_history = {}
                    for k, vals in history.items():
                        if k == 'action_probs':
                            serializable_history[k] = vals  # 已经是可序列化的
                        else:
                            serializable_history[k] = [float(v) for v in vals]
                    with open(history_file, 'w') as f:
                        json.dump(serializable_history, f, indent=2)
                    print(f"  训练历史已保存: {history_file}")
                except Exception as e:
                    print(f"  [警告] 训练历史保存失败: {e}")

            step_times['step2_train'] = round(time.time() - t_step, 2)
            print(f"  [计时] Step 2 (训练): {step_times['step2_train']:.2f}s")

        # ================================================================
        # 3. 推理
        # ================================================================
        print(f"\n[Step 3] 推理 (模式: {inference_mode})...")
        t_step = time.time()
        ground_truth_used = 0

        if inference_mode == 'single_phase':
            X_result, y_result, stats = dc.clean(
                X_dirty, y_dirty, X_clean, y_clean=y_clean, verbose=verbose
            )
            ground_truth_used = stats.get('truth_cost', 0)
            report['action_counts'] = stats.get('action_counts', {})
            report['keep_mask'] = stats.get('keep_mask', None)
            print(f"  真值使用: {ground_truth_used}")
            print(f"  动作分布: {stats.get('action_counts', {})}")

        elif inference_mode == 'two_phase':
            # 第一阶段: 生成修复计划
            repair_plan = dc.plan(X_dirty, y_dirty, X_clean=X_clean, y_clean=y_clean, verbose=verbose)
            print(f"  修复计划: 需要 {len(repair_plan)} 个真值")

            # 准备真值（支持特征列和标签列 col=-1）
            true_values = {}
            for item in repair_plan:
                idx, col = item['idx'], item['col']
                if col == -1:
                    # 标签噪声：从 y_clean 获取真值
                    if idx < len(y_clean):
                        true_values[(idx, col)] = y_clean[idx]
                elif idx < X_clean.shape[0] and col < X_clean.shape[1]:
                    true_values[(idx, col)] = X_clean[idx, col]

            # 第二阶段: 执行修复
            X_result, y_result, keep_mask = dc.execute(
                X_dirty, true_values, verbose=verbose, y_dirty=y_dirty
            )
            ground_truth_used = len(true_values)
            report['keep_mask'] = keep_mask
            print(f"  真值使用: {ground_truth_used}")

        report['ground_truth_used'] = ground_truth_used
        report['cleaned_shape'] = list(X_result.shape)

        # 保存清洗后数据（标准化版本 + 原始格式版本）
        output_csv = os.path.join(dir_data, f'{version_name}_cleaned.csv')
        output_csv_original = os.path.join(dir_data, f'{version_name}_cleaned_original.csv')

        try:
            # 标准化版本（供内部分析用）
            cleaned_df = pd.DataFrame(X_result, columns=column_names)
            cleaned_df[ds_cfg['label_col']] = y_result[:len(cleaned_df)]
            cleaned_df.to_csv(output_csv, index=False)
            print(f"  清洗后数据已保存: {output_csv}")
        except Exception as e:
            print(f"  [警告] 清洗数据保存失败: {e}")

        # 生成原始格式 CSV（供 getScoreML 统一测评用）
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
            print(f"  原始格式清洗数据已保存: {output_csv_original}")
        except Exception as e:
            print(f"  [警告] 原始格式CSV生成失败: {e}")
            traceback.print_exc()
            output_csv_original = None

        # 保存编码版本（供 Step 9 直接使用，避免 CSV roundtrip 精度损失）
        output_npz = os.path.join(dir_data, f'{version_name}_cleaned_encoded.npz')
        try:
            np.savez(output_npz,
                     X_result=X_result,
                     y_result=y_result,
                     column_names=np.array(column_names),
                     label_col=np.array([ds_cfg['label_col']]))
            print(f"  编码版本已保存: {output_npz}")
        except Exception as e:
            print(f"  [警告] 编码版本保存失败: {e}")
            output_npz = None

        step_times['step3_infer'] = round(time.time() - t_step, 2)
        print(f"  [计时] Step 3 (推理): {step_times['step3_infer']:.2f}s")

        # ================================================================
        # 4. 检测器准确率评估 (提前到 Baseline 评估之前，供 DeleteFix 使用)
        # ================================================================
        print(f"\n[Step 4] 检测器准确率评估...")
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
            print(f"  [错误] 检测器准确率评估失败: {e}")
            traceback.print_exc()
            detector_accuracy = {}

        step_times['step4_detect_eval'] = round(time.time() - t_step, 2)
        print(f"  [计时] Step 4 (检测器评估): {step_times['step4_detect_eval']:.2f}s")

    # ================================================================
    # 5. 6种 Baseline 评估 + 决策边界对比图
    # ================================================================
    total_vis_time = 0.0  # 累计可视化耗时（最终从 elapsed_time 扣除）

    print(f"\n[Step 5] Baseline 评估与可视化...")
    t_step = time.time()
    try:
        # Oracle 模式: 传入训练子集和测试集
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

        # 构造 ValueEstimator 供 ReplaceAll baseline 使用
        ve_for_baseline = None
        if dc is not None:
            try:
                from demandclean.core.environments.value_estimation import ValueEstimator
                ve_for_baseline = ValueEstimator(dc.config)
                print(f"  ReplaceAll VE: {ve_for_baseline.summary()}")
            except Exception as e:
                print(f"  [警告] ValueEstimator 构造失败，ReplaceAll 降级为均值填充: {e}")

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

        # Baseline 对比柱状图（也计入可视化耗时）
        vis_start = time.time()
        plot_baseline_comparison(
            baseline_results, dir_vis, version_name,
            task_type, eval_models[0],
        )
        total_vis_time += time.time() - vis_start
    except Exception as e:
        print(f"  [错误] Baseline 评估失败: {e}")
        traceback.print_exc()
        baseline_results = {}

    step_times['step5_baseline'] = round(time.time() - t_step, 2)
    print(f"  [计时] Step 5 (Baseline评估): {step_times['step5_baseline']:.2f}s")

    # ================================================================
    # 6. 容忍度评估 (compute_model_tolerance)
    # ================================================================
    print(f"\n[Step 6] 模型容忍度评估...")
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
        print(f"  [错误] 容忍度评估失败: {e}")
        traceback.print_exc()
        tolerance_results = {}

    step_times['step6_tolerance'] = round(time.time() - t_step, 2)
    print(f"  [计时] Step 6 (容忍度评估): {step_times['step6_tolerance']:.2f}s")

    # ================================================================
    # 7. Shapley 三维度分析 (run_full_shapley_analysis)
    # ================================================================
    print(f"\n[Step 7] Shapley 三维度分析...")
    vis_start = time.time()
    try:
        # 构建 error_list 格式 (从检测结果转换)
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
            print("  [跳过] 无可用错误列表或 Agent 未就绪")
    except Exception as e:
        print(f"  [错误] Shapley 分析失败: {e}")
        traceback.print_exc()
    _step7_elapsed = time.time() - vis_start
    total_vis_time += _step7_elapsed
    step_times['step7_shapley'] = round(_step7_elapsed, 2)
    print(f"  [计时] Step 7 (Shapley分析): {step_times['step7_shapley']:.2f}s")

    # ================================================================
    # 8. 训练曲线可视化
    # ================================================================
    print(f"\n[Step 8] 训练曲线可视化...")
    vis_start = time.time()
    try:
        # inference_only 模式下尝试从文件加载训练历史
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
            # 计算续训起点用于可视化标记
            resume_ep = len(prev_history.get('score', [])) if prev_history else 0
            plot_training_curves(history, dir_vis, version_name,
                                 resume_episode=resume_ep)
    except Exception as e:
        print(f"  [错误] 训练曲线绘制失败: {e}")
        traceback.print_exc()
    _step8_elapsed = time.time() - vis_start
    total_vis_time += _step8_elapsed
    step_times['step8_vis'] = round(_step8_elapsed, 2)
    print(f"  [计时] Step 8 (训练曲线): {step_times['step8_vis']:.2f}s")

    # ================================================================
    # 9. getScoreML 统一测评 (与 run_baran_raha 输出格式一致)
    # ================================================================
    print(f"\n[Step 9] Clean4MLBaseline 统一测评...")
    t_step = time.time()
    getscoreml_results = {}
    try:
        # 确保 tools 目录在 sys.path 中
        _tools_dir = os.path.join(_PROJECT_ROOT, 'tools')
        if _tools_dir not in sys.path:
            sys.path.insert(0, _tools_dir)
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)

        from tools.getScoreML import run_all_evaluation

        if output_csv_original and os.path.exists(output_csv_original):
            # 选择合适的模型列表（与 raha_baran 一致）
            if task_type == 'classification':
                ml_models = ['rf', 'lr', 'svm', 'knn', 'dt', 'gb']
            elif task_type == 'clustering':
                ml_models = ['kmeans', 'agglomerative', 'spectral']
            else:
                ml_models = ['rf', 'lr', 'ridge', 'lasso', 'knn', 'gb']

            # DemandClean 的方法类型:
            #   oracle + single_phase = Type 1 (全自动，但使用了全部真值)
            #   oracle + two_phase = Type 3 (迭代交互，需要用户提供真值)
            #   auto + single_phase = Type 1 (全自动)
            #   auto + two_phase = Type 2 (需验证集，部分真值)
            if detector_mode == 'oracle':
                method_type = 3 if inference_mode == 'two_phase' else 1
            else:
                method_type = 2 if inference_mode == 'two_phase' else 1

            # 计算 mse_attributes（数值列）
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

            # 确保无 None 值：将所有 None 替换为 0 或 "N/A"
            for k, v in getscoreml_results.items():
                if v is None:
                    getscoreml_results[k] = 0.0

            report['getscoreml_results'] = getscoreml_results
            print(f"  统一测评完成，结果已保存到: {save_dir}/{version_name}_evaluation_results.txt")
        else:
            print(f"  [跳过] 原始格式 CSV 不可用，无法进行统一测评")
    except ImportError as e:
        print(f"  [警告] 无法导入 getScoreML 模块: {e}")
    except Exception as e:
        try:
            print(f"  [错误] 统一测评失败: {e}")
            traceback.print_exc()
        except (ValueError, IOError):
            # stdout 可能被子进程关闭
            sys.stderr.write(f"  [错误] 统一测评失败: {e}\n")
            traceback.print_exc(file=sys.stderr)

    step_times['step9_ml_eval'] = round(time.time() - t_step, 2)
    print(f"  [计时] Step 9 (统一测评): {step_times['step9_ml_eval']:.2f}s")

    # ================================================================
    # 10. 真值成本汇总
    # ================================================================
    print(f"\n[Step 10] 真值成本汇总...")
    t_step = time.time()
    try:
        # 获取 RAHA 检测成本
        raha_cost_info = {}
        if dc is not None and hasattr(dc, 'detector') and hasattr(dc.detector, 'raha_cost_info'):
            raha_cost_info = dc.detector.raha_cost_info or {}
        elif 'raha_cost' in report:
            # visualize_only 模式从旧 report 恢复
            raha_cost_info = report['raha_cost']

        raha_detection_cost = raha_cost_info.get('raha_total_cost', 0)
        if detector_mode == 'oracle':
            raha_detection_cost = 0  # Oracle 模式无 RAHA 成本

        # Agent 修复成本 = ground_truth_used
        agent_repair_cost = ground_truth_used

        # 总成本
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

        print(f"  RAHA 检测成本: {raha_detection_cost}")
        print(f"  Agent 修复成本: {agent_repair_cost}")
        print(f"  总成本: {total_cost} / {total_data_cells} cells = {cost_ratio:.4%}")
    except Exception as e:
        print(f"  [警告] 真值成本汇总失败: {e}")
        traceback.print_exc()

    step_times['step10_cost'] = round(time.time() - t_step, 2)
    print(f"  [计时] Step 10 (成本汇总): {step_times['step10_cost']:.2f}s")

    # ================================================================
    # 保存完整报告
    # ================================================================
    elapsed_time = time.time() - start_time - total_vis_time

    # 计时汇总
    print(f"\n{'─'*50}")
    print(f"  步骤计时汇总")
    print(f"{'─'*50}")
    step_labels = {
        'step1_init': '初始化',
        'step2_train': '训练',
        'step3_infer': '推理',
        'step4_detect_eval': '检测器评估',
        'step5_baseline': 'Baseline评估',
        'step6_tolerance': '容忍度评估',
        'step7_shapley': 'Shapley分析',
        'step8_vis': '训练曲线',
        'step9_ml_eval': '统一测评',
        'step10_cost': '成本汇总',
    }
    for key, label in step_labels.items():
        t = step_times.get(key, 0)
        print(f"  Step {key.split('_')[0].replace('step','')} {label:<12s}│ {t:>8.2f}s")
    print(f"{'─'*50}")
    print(f"  总计(含可视化)       │ {time.time() - start_time:>8.2f}s")
    print(f"  可视化耗时           │ {total_vis_time:>8.2f}s")
    print(f"  净耗时(扣可视化)     │ {elapsed_time:>8.2f}s")
    print(f"{'─'*50}")

    report['elapsed_time'] = round(elapsed_time, 2)
    report['vis_time'] = round(total_vis_time, 2)
    report['step_times'] = step_times

    report_file = os.path.join(dir_report, f'{version_name}_report.json')
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  [输出] 报告已保存: {report_file}")
    except Exception as e:
        print(f"  [警告] 报告保存失败: {e}")

    # 文本摘要
    summary_file = os.path.join(dir_report, f'{version_name}_summary.txt')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"DemandClean 结果摘要 - {version_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"检测器: {detector_mode}\n")
            f.write(f"Agent: {agent_type}\n")
            f.write(f"推理模式: {inference_mode}\n")
            f.write(f"任务: {task_type} ({model_type})\n")
            f.write(f"训练轮数: {n_episodes}\n")
            f.write(f"执行时间: {elapsed_time:.2f} 秒\n")
            f.write(f"可视化耗时: {total_vis_time:.2f} 秒\n")
            f.write(f"数据维度: {X_dirty.shape} -> {X_result.shape}\n")
            f.write(f"真值成本: {ground_truth_used}\n\n")

            # 步骤计时
            if step_times:
                f.write("步骤计时:\n")
                f.write("-" * 60 + "\n")
                _step_labels = {
                    'step1_init': '初始化',
                    'step2_train': '训练',
                    'step3_infer': '推理',
                    'step4_detect_eval': '检测器评估',
                    'step5_baseline': 'Baseline评估',
                    'step6_tolerance': '容忍度评估',
                    'step7_shapley': 'Shapley分析',
                    'step8_vis': '训练曲线',
                    'step9_ml_eval': '统一测评',
                    'step10_cost': '成本汇总',
                }
                for _key, _label in _step_labels.items():
                    _t = step_times.get(_key, 0)
                    f.write(f"  Step {_key.split('_')[0].replace('step','')} {_label}: {_t:.2f}s\n")
                f.write("\n")

            f.write("Baseline 对比:\n")
            f.write("-" * 60 + "\n")
            for name, res in baseline_results.items():
                f.write(f"  {name}: {res}\n")

            if tolerance_results:
                f.write("\n模型容忍度:\n")
                f.write("-" * 60 + "\n")
                for name, res in tolerance_results.items():
                    f.write(f"  {name}: {res}\n")

            if detector_accuracy:
                f.write("\n检测器准确率:\n")
                f.write("-" * 60 + "\n")
                for name, res in detector_accuracy.items():
                    f.write(f"  {name}: {res}\n")

            if getscoreml_results:
                f.write("\nClean4MLBaseline 统一测评:\n")
                f.write("-" * 60 + "\n")
                for key in ['accuracy', 'recall', 'f1_score', 'edr', 'hybrid_distance', 'r_edr']:
                    if key in getscoreml_results:
                        f.write(f"  {key}: {getscoreml_results[key]}\n")
                f.write("\n  下游任务性能:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('ml_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  容忍度:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('tolerance_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  Snoopy 上界:\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('snoopy_'):
                        f.write(f"    {key}: {value}\n")
                f.write("\n  真值成本:\n")
                f.write(f"    method_type: {getscoreml_results.get('method_type', 'N/A')}\n")
                f.write(f"    ground_truth_cost: {getscoreml_results.get('ground_truth_cost', 'N/A')}\n")
                for key, value in getscoreml_results.items():
                    if key.startswith('ideal_'):
                        f.write(f"    {key}: {value}\n")

            # Shapley 分析结果
            shapley = report.get('shapley_results', {})
            if shapley and 'report_text' in shapley:
                f.write("\n" + "=" * 60 + "\n")
                f.write("Shapley 贡献分析:\n")
                f.write("=" * 60 + "\n")
                f.write(shapley['report_text'])
                f.write("\n")
            elif shapley:
                # 兜底：没有 report_text 时输出裸数据
                f.write("\nShapley 分析:\n")
                f.write("-" * 60 + "\n")
                for dim_key in ['action_shapley', 'feature_shapley', 'error_type_shapley']:
                    dim_data = shapley.get(dim_key, {})
                    if dim_data:
                        f.write(f"  {dim_key}:\n")
                        for name, val in sorted(dim_data.items(), key=lambda x: -x[1]):
                            f.write(f"    {name}: {val:+.6f}\n")

            # 真值成本汇总
            cost_summary = report.get('ground_truth_cost_summary', {})
            if cost_summary:
                f.write("\n" + "=" * 60 + "\n")
                f.write("真值使用成本汇总:\n")
                f.write("=" * 60 + "\n")
                f.write(f"  RAHA 检测成本: {cost_summary.get('raha_detection_cost', 0)}\n")
                f.write(f"  Agent 修复成本: {cost_summary.get('agent_repair_cost', 0)}\n")
                f.write(f"  总成本: {cost_summary.get('total_cost', 0)}\n")
                f.write(f"  总数据单元: {cost_summary.get('total_data_cells', 0)}\n")
                f.write(f"  成本率: {cost_summary.get('cost_ratio', 0):.4%}\n")
                f.write(f"  apply_raha_truth: {cost_summary.get('apply_raha_truth', True)}\n")
                f.write(f"  count_raha_cost: {cost_summary.get('count_raha_cost', True)}\n")
    except Exception as e:
        print(f"  [警告] 摘要保存失败: {e}")

    print(f"\n  版本 {version_name} 完成! 耗时: {elapsed_time:.2f}s")
    print(f"  输出目录: {save_dir}")

    return report


# ============================================================================
# 跨版本对比分析
# ============================================================================
def cross_version_comparison(
    dataset_name: str,
    all_reports: dict,
):
    """
    所有版本完成后，生成跨版本对比表格和图表

    Args:
        dataset_name: 数据集名称
        all_reports: {version_name: report_dict}
    """
    ds_cfg = DATASETS[dataset_name]
    task_type = ds_cfg['task_type']

    save_dir = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"跨版本对比分析 - {dataset_name}")
    print("=" * 70)

    # --- 收集容忍度结果用于 compare_versions ---
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
            print(f"  跨版本对比已保存到: {save_dir}")
        except Exception as e:
            print(f"  [错误] 跨版本对比失败: {e}")
            traceback.print_exc()

    # --- 生成汇总表格 ---
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

            # 真值成本汇总
            cost_sum = report.get('ground_truth_cost_summary', {})
            row['raha_cost'] = cost_sum.get('raha_detection_cost', 0)
            row['total_cost'] = cost_sum.get('total_cost', 0)
            row['cost_ratio'] = cost_sum.get('cost_ratio', 0.0)

            # 提取主模型的 DemandClean baseline 指标
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

            # 提取容忍度
            tol = report.get('tolerance_results', {})
            first_model_tol = next(iter(tol.values()), {}) if tol else {}
            row['tol_prior'] = first_model_tol.get('tolerance_prior', None)
            row['tol_post'] = first_model_tol.get('tolerance_post', None)

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(save_dir, f'{dataset_name}_all_versions_summary.csv')
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  汇总表格已保存: {csv_path}")

        # 打印表格
        print("\n" + summary_df.to_string(index=False))

        # --- 汇总柱状图 ---
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
                print(f"  跨版本对比图已保存: {fig_path}")

    except Exception as e:
        print(f"  [错误] 汇总分析失败: {e}")
        traceback.print_exc()


# ============================================================================
# main() 函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='DemandClean 统一运行脚本 - 支持 8 种版本组合',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 默认: v5 + 全部数据集
    python run_demandclean_base.py --n_episodes 300

    # 指定单个数据集
    python run_demandclean_base.py --dataset beers --n_episodes 300

    # 指定版本 (支持短名 v1-v8 或全名)
    python run_demandclean_base.py --dataset beers --versions v3
    python run_demandclean_base.py --dataset beers --versions v1,v3,v5
    python run_demandclean_base.py --dataset beers --versions v3_oracle_plain_single

    # 自定义错误注入率 (格式: min,max)
    python run_demandclean_base.py --dataset beers --missing_rate 0.05,0.1
    python run_demandclean_base.py --dataset beers --semantic_rate 0.1,0.2 --syntactic_rate 0.15,0.3

    # 续训/仅可视化
    python run_demandclean_base.py --dataset beers --resume auto
    python run_demandclean_base.py --dataset beers --versions v6 --visualize_only

版本说明:
    v1: oracle + dueling + single_phase    v5: auto + dueling + single_phase (默认)
    v2: oracle + dueling + two_phase       v6: auto + dueling + two_phase
    v3: oracle + plain   + single_phase    v7: auto + plain   + single_phase
    v4: oracle + plain   + two_phase       v8: auto + plain   + two_phase
        """,
    )

    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=list(DATASETS.keys()),
        help='数据集名称 (不指定则运行全部数据集)',
    )
    parser.add_argument(
        '--n_episodes', type=int, default=300,
        help='训练轮数 (默认: 300)',
    )
    parser.add_argument(
        '--all_datasets', action='store_true',
        help='显式指定运行全部数据集 (不指定 --dataset 时即为默认行为)',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='输出详细训练信息',
    )
    parser.add_argument(
        '--versions', type=str, default='v5',
        help='指定运行版本，逗号分隔 (如 v3,v5 或 v1,v2,v3)。'
             '支持短名(v1-v8)和全名(v3_oracle_plain_single)。'
             '默认使用代码中ENABLE_开关控制的版本',
    )
    parser.add_argument(
        '--missing_rate', type=str, default='',
        help='缺失值注入率范围，格式: min,max (如 0.02,0.08)。默认使用 config 默认值',
    )
    parser.add_argument(
        '--semantic_rate', type=str, default='',
        help='语义错误注入率范围，格式: min,max (如 0.05,0.15)。默认使用 config 默认值',
    )
    parser.add_argument(
        '--syntactic_rate', type=str, default='',
        help='句法错误注入率范围，格式: min,max (如 0.1,0.25)。默认使用 config 默认值',
    )
    parser.add_argument(
        '--label_rate', type=str, default='',
        help='标签错误注入率范围，格式: min,max (如 0.0,0.05)。默认使用 config 默认值',
    )
    parser.add_argument(
        '--resume', type=str, choices=['auto', 'force_new'], default='force_new',
        help='续训模式: auto=检测到已有模型则续训, force_new=强制从头训练 (默认: auto)',
    )
    parser.add_argument(
        '--apply_raha_truth', type=str, choices=['true', 'false'], default='true',
        help='是否将 RAHA 标注行的真值应用到数据修复 (默认: true)',
    )
    parser.add_argument(
        '--count_raha_cost', type=str, choices=['true', 'false'], default='true',
        help='是否将 RAHA 的标注成本计入真值总成本 (默认: true)',
    )
    parser.add_argument(
        '--visualize_only', action='store_true',
        help='仅从已有清洗结果重新生成可视化和评估，跳过训练和推理',
    )
    parser.add_argument(
        '--min_repair_ratio', type=float, default=None,
        help='最低修复比例(占 VE-fill 总数), 0=不限制 (如 0.05)',
    )
    parser.add_argument(
        '--max_repair_ratio', type=float, default=None,
        help='最高修复比例, 1.0=不限制 (如 0.80)',
    )
    parser.add_argument(
        '--repair_sensitivity', type=float, default=None,
        help='动态调节的性能敏感度 (默认: 10.0)',
    )
    parser.add_argument(
        '--max_truth_budget', type=int, default=None,
        help='最大真值预算 (设为0实现NGT变体，即禁用真值修复)',
    )
    parser.add_argument(
        '--repair_lambda', type=float, default=None,
        help='真值修复成本系数 (默认: 0.03, 回归任务可调低如 0.005)',
    )
    parser.add_argument(
        '--reward_model_type', type=str, default=None,
        help='覆盖 reward 评估用的模型类型 (如 random_forest)，不影响 baseline 评估',
    )
    parser.add_argument(
        '--disable_raha', action='store_true',
        help='禁用 RAHA 检测，仅使用规则检测 (适用于纯标签错误数据集如 adult)',
    )
    parser.add_argument(
        '--delete_shaping_reward', type=float, default=None,
        help='delete 动作的 shaping reward (默认: 分类/聚类 -0.02, 回归 -0.05)',
    )
    parser.add_argument(
        '--keep_rate_weight', type=float, default=None,
        help='final reward 中 keep_rate 权重 (默认: 分类/聚类 0.2, 回归 1.0)',
    )
    parser.add_argument(
        '--oracle', action='store_true', default=True,
        help='启用 Oracle 模式: 三路划分(train/val/test), 用干净验证集做 reward (默认启用)',
    )
    parser.add_argument(
        '--no-oracle', dest='oracle', action='store_false',
        help='关闭 Oracle 模式, 使用脏数据构建的 Clean Base 做 reward',
    )
    parser.add_argument(
        '--inference_only', action='store_true',
        help='仅推理模式: 跳过训练，加载已有模型直接推理+评估 (适用于训练完成但推理失败的情况)',
    )
    parser.add_argument(
        '--output_suffix', type=str, default='',
        help='输出目录后缀，避免覆盖原始版本 (如 --output_suffix _ngt)',
    )

    args = parser.parse_args()

    # --- 解析错误注入率 ---
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
                    print(f"[警告] {rate_name} 格式错误: {rate_str}，已忽略")
            except ValueError:
                print(f"[警告] {rate_name} 格式错误: {rate_str}，已忽略")

    # 确定要运行的数据集（默认: 全部）
    if args.dataset is not None:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(DATASETS.keys())

    # ======================================================================
    # 日志重定向: 同时输出到终端和日志文件
    # 日志文件: logs/demandclean/{dataset}_{timestamp}.log
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

    print(f"[日志] 完整输出将保存到: {log_path}")

    # 筛选启用的版本
    if args.versions:
        # 用户通过命令行指定版本
        requested = [v.strip() for v in args.versions.split(',') if v.strip()]
        # 短名(v1-v8)到全名的映射
        short_name_map = {
            f'v{i}': k for i, k in enumerate(VERSION_CONFIGS.keys(), 1)
        }
        enabled_versions = {}
        for req in requested:
            # 先尝试完整名
            if req in VERSION_CONFIGS:
                cfg = VERSION_CONFIGS[req].copy()
                cfg['enabled'] = True
                enabled_versions[req] = cfg
            # 再尝试短名
            elif req in short_name_map:
                full_name = short_name_map[req]
                cfg = VERSION_CONFIGS[full_name].copy()
                cfg['enabled'] = True
                enabled_versions[full_name] = cfg
            else:
                print(f"[警告] 未知版本: {req}，已忽略。"
                      f"可用: {list(short_name_map.keys())} 或 {list(VERSION_CONFIGS.keys())}")
    else:
        # 使用代码中 ENABLE_ 开关
        enabled_versions = {
            k: v for k, v in VERSION_CONFIGS.items() if v['enabled']
        }

    if not enabled_versions:
        print("[错误] 没有启用任何版本! 请检查版本开关设置。")
        return

    print("=" * 70)
    print("DemandClean 统一运行脚本")
    print("=" * 70)
    print(f"数据集: {datasets_to_run}")
    print(f"启用版本: {list(enabled_versions.keys())}")
    print(f"训练轮数: {args.n_episodes}")
    print(f"详细输出: {args.verbose}")
    if args.oracle:
        print(f"Oracle 模式: 启用 (三路划分 60/20/20, --no-oracle 关闭)")
    else:
        print(f"Oracle 模式: 关闭 (使用脏数据 Clean Base 做 reward)")
    if args.min_repair_ratio is not None or args.max_repair_ratio is not None:
        print(f"修复率控制: min={args.min_repair_ratio}, max={args.max_repair_ratio}")
    if inject_kwargs:
        print(f"自定义注入率: {inject_kwargs}")
    if args.inference_only:
        print(f"仅推理模式: 启用 (跳过训练，直接加载已有模型)")
    print("=" * 70)

    total_start = time.time()

    # 遍历数据集
    for dataset_name in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# 数据集: {dataset_name}")
        print(f"{'#'*70}")

        ds_cfg = DATASETS[dataset_name]

        try:
            if args.oracle:
                # =============================================================
                # 严格 60/20/20 划分: 先划分原始 CSV，再编码
                # =============================================================
                from sklearn.model_selection import train_test_split as _tts

                # 1. 读取原始 CSV
                data_dir = os.path.join(_PROJECT_ROOT, 'data', dataset_name)
                _dirty_path = os.path.join(data_dir, 'dirty_index.csv')
                _clean_path = os.path.join(data_dir, 'clean_index.csv')
                if not os.path.exists(_dirty_path):
                    _dirty_path = os.path.join(data_dir, 'dirty_with_index.csv')
                if not os.path.exists(_clean_path):
                    _clean_path = os.path.join(data_dir, 'clean_with_index.csv')
                raw_dirty_df = pd.read_csv(_dirty_path)
                raw_clean_df = pd.read_csv(_clean_path)

                # 2. 60/20/20 划分 (seed=42)
                n_total = len(raw_dirty_df)
                all_idx = np.arange(n_total)
                train_idx, temp_idx = _tts(all_idx, test_size=0.4, random_state=42)
                val_idx, test_idx = _tts(temp_idx, test_size=0.5, random_state=42)

                dirty_train_df = raw_dirty_df.iloc[train_idx].reset_index(drop=True)
                clean_train_df = raw_clean_df.iloc[train_idx].reset_index(drop=True)

                print(f"\n  60/20/20 划分: train={len(train_idx)}, "
                      f"val={len(val_idx)}, test={len(test_idx)}")

                # 3. 生成 60% 子集 CSV (for RAHA)
                save_base = os.path.join(_PROJECT_ROOT, 'results', 'demandclean', dataset_name)
                os.makedirs(save_base, exist_ok=True)
                dirty_train_csv_path = os.path.join(save_base, 'dirty_train_60pct.csv')
                clean_train_csv_path = os.path.join(save_base, 'clean_train_60pct.csv')
                dirty_train_df.to_csv(dirty_train_csv_path, index=False)
                clean_train_df.to_csv(clean_train_csv_path, index=False)
                print(f"  60% 子集 CSV: {dirty_train_csv_path}")

                # 4. preprocess_data 只处理 60% dirty（LE/SS 仅在此 fit）
                (X_dirty, y_dirty,
                 _, _,
                 column_names, fd_rules, rules_path,
                 _orig_dirty_path, _orig_clean_path, csv_columns,
                 data_scaler, label_encoders, categorical_cols,
                 dirty_df, _) = preprocess_data(
                    dataset_name, dirty_df=dirty_train_df, clean_df=None)

                # 用 train 60% 子集 CSV 路径（不再用全量路径）
                dirty_csv_path = dirty_train_csv_path
                clean_csv_path = clean_train_csv_path

                # 5. 用 train LE/SS 编码 val/test/train 的 clean 数据
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

                print(f"  编码完成: X_dirty_train={X_dirty.shape}, "
                      f"X_clean_train={X_clean_train.shape}, "
                      f"X_clean_val={X_clean_val.shape}, "
                      f"X_clean_test={X_clean_test.shape}")

                # 用于 run_version 的参数
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
                # 非 Oracle 模式: 全量编码（向后兼容）
                # =============================================================
                (X_dirty, y_dirty,
                 X_clean, y_clean,
                 column_names, fd_rules, rules_path,
                 dirty_csv_path, clean_csv_path, csv_columns,
                 data_scaler, label_encoders, categorical_cols,
                 dirty_df, clean_df) = preprocess_data(dataset_name)
                oracle_split = None

        except Exception as e:
            print(f"\n[错误] 数据集 {dataset_name} 预处理失败: {e}")
            traceback.print_exc()
            continue

        # --- 遍历启用的版本 ---
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
                print(f"\n[错误] 版本 {version_key} 运行失败，已跳过:")
                print(f"  {e}")
                traceback.print_exc()
                continue

        # --- 跨版本对比 ---
        if len(all_reports) >= 2:
            try:
                cross_version_comparison(dataset_name, all_reports)
            except Exception as e:
                print(f"\n[错误] 跨版本对比失败: {e}")
                traceback.print_exc()
        elif len(all_reports) == 1:
            print("\n  只有 1 个版本完成，跳过跨版本对比")
        else:
            print("\n  没有版本完成，跳过跨版本对比")

    # --- 全局结束 ---
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"全部完成! 总耗时: {total_elapsed:.2f}s")
    print(f"结果目录: {os.path.join(_PROJECT_ROOT, 'results', 'demandclean')}")
    print(f"日志文件: {log_path}")
    print(f"{'='*70}")

    # 关闭 TeeLogger，恢复 stdout/stderr
    sys.stdout = tee_stdout.terminal
    sys.stderr = tee_stderr.terminal
    tee_stdout.close()
    tee_stderr.close()
    print(f"[日志] 完整日志已保存: {log_path}")


if __name__ == "__main__":
    main()
