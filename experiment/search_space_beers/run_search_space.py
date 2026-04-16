#!/usr/bin/env python3
"""
搜索空间可视化实验
==================

验证 DemandClean 清洗策略的搜索空间理论：
在 Authenticity-Diversity 空间中映射所有可能策略的性能分布。

实验方法:
1. 四个极端点: NoFix, FullFix, DeleteFix, RelaxFix
2. 随机采样: 遍历所有动作概率组合 (no_action, repair, delete, replace_nearby)
3. DQN 推理: 标注 DemandFix 在搜索空间中的位置
4. 热力图: 展示性能 (Accuracy) 在搜索空间中的分布

理论搜索空间模型 (月牙形):

    多样性 ↑
        │
     FullFix  ●─────────────────● DeleteFix (多样性下界)
        │╲   搜索空间      │
        │ ╲  (月牙形)     │
        │  ╲             ╱
        │   ● DemandFix ╱
        │    ╲         ╱
        │     ╲       ╱
    NoFix ●──┼──────●─────┼────→ 真实性
        │    RelaxFix

基于原始 search_space_experiment.py 重写，
核心改造：用发行版 demandclean/ API 替换旧的 TensorFlow DQN，消除 TF 依赖。
所有可视化函数完整保留，逻辑和风格不做修改，仅更新保存路径。
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

# alphashape / shapely (搜索空间边界)
try:
    import alphashape
    from shapely.geometry import Point
    HAS_ALPHASHAPE = True
except ImportError:
    HAS_ALPHASHAPE = False
    print("[WARN] alphashape/shapely 未安装，部分图表将跳过 alpha shape 裁剪")
    print("       pip install alphashape shapely")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')
import sys
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

# 实验参数
N_RANDOM_SAMPLES = 1000    # 随机采样数
N_DQN_RUNS = 1             # DQN 推理轮数
SEMANTIC_RATE = 0.15        # 语义错误注入率
SYNTACTIC_RATE = 0.20       # 句法错误注入率

RUN_EXPERIMENT = False      # True=运行实验, False=只读取已有结果重新画图
LOAD_MODEL = True           # True=加载已训练模型, False=重新训练DQN

# 极端策略名称集合（用于区分极端点 vs 随机采样）
EXTREME_NAMES = {'NoFix', 'FullFix', 'DeleteFix', 'RelaxFix'}


def _is_random(name):
    """判断是否为随机采样（非极端点、非DQN）"""
    return name not in EXTREME_NAMES and not name.startswith('DemandFix')

# 导入 DemandClean API
from demandclean.api.demand_clean import DemandClean

np.random.seed(2024)


# =============================================================================
# 1. 数据准备
# =============================================================================
def load_and_preprocess():
    """加载 Beer 数据"""
    clean_path = os.path.join(DATA_DIR, 'clean.csv')
    df = pd.read_csv(clean_path)
    df = df[df['abv'].notna() & df['ibu'].notna()].reset_index(drop=True)
    df['is_ipa'] = df['style'].apply(
        lambda s: 1 if pd.notna(s) and 'ipa' in str(s).lower() else 0)
    print(f"  数据加载: {len(df)} 行, IPA={df['is_ipa'].sum()}")
    return df


def inject_errors(X_clean, y, semantic_rate=0.15, syntactic_rate=0.20, seed=2024):
    """
    注入两种错误（仅 IBU 列）

    - 语义错误: 用同列已有值替换（偏好距离较近的值）
    - 句法错误: 添加高斯噪声
    """
    rng = np.random.RandomState(seed)
    X_dirty = X_clean.copy()
    n = len(X_clean)
    error_list = []  # [(idx, col, type, true_value, dirty_value), ...]
    selected_rows = set()

    all_ibu = X_clean[:, 1]

    # 语义错误
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

    # 句法错误
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

    print(f"  注入错误: 语义={sum(1 for e in error_list if e[2]=='semantic')}, "
          f"句法={sum(1 for e in error_list if e[2]=='syntactic')}")
    return X_dirty, error_list


# =============================================================================
# 2. 策略执行
# =============================================================================
def apply_strategy(X_dirty, X_clean, y, error_list, actions, knn_imputer):
    """
    执行清洗策略

    动作:
      0 = no_action   (保持原样)
      1 = repair_value (用真值修复)
      2 = delete       (删除整行)
      3 = replace_nearby (语义错误=真值, 句法错误=KNN估计)
    """
    X_result = X_dirty.copy()
    keep_mask = np.ones(len(X_dirty), dtype=bool)

    # KNN 预计算
    X_knn = knn_imputer.transform(X_dirty)

    for i, (idx, col, err_type, true_val, dirty_val) in enumerate(error_list):
        action = actions[i]
        if action == 0:  # no_action
            pass
        elif action == 1:  # repair_value (真值)
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

    # 填充剩余 NaN
    col_means = np.nanmean(X_result, axis=0) if len(X_result) > 0 else np.zeros(X_dirty.shape[1])
    for c in range(X_result.shape[1]):
        nan_mask = np.isnan(X_result[:, c])
        if nan_mask.any():
            X_result[nan_mask, c] = col_means[c]

    return X_result, y_result, keep_mask


# =============================================================================
# 3. 指标计算
# =============================================================================
def compute_veracity(X_result, X_clean, keep_mask, col=1, tolerance=0.01):
    """真实性 = 正确值 / 总值数"""
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
    """多样性 = 样本保留率 × 方差保留率"""
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
    """SVM 线性分类准确率"""
    if len(X_train) < 5 or len(np.unique(y_train)) < 2:
        return 0.0
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


# =============================================================================
# 4. 采样与实验
# =============================================================================
def run_single_strategy(X_dirty, X_clean, y, error_list, actions, X_test, y_test,
                        knn_imputer, name='random'):
    """运行单个策略并返回结果字典"""
    X_result, y_result, keep_mask = apply_strategy(
        X_dirty, X_clean, y, error_list, actions, knn_imputer)

    if len(X_result) < 5:
        return None

    veracity = compute_veracity(X_result, X_clean, keep_mask)
    diversity = compute_diversity(X_result, X_clean, keep_mask)
    accuracy = evaluate_performance(X_result, y_result, X_test, y_test)

    # 统计动作分布
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
    """均匀随机生成动作"""
    return np.random.randint(0, 4, size=len(error_list))


def generate_extreme_actions(error_list, strategy):
    """生成极端策略动作"""
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
        raise ValueError(f"未知策略: {strategy}")


def generate_uniform_bias_configs(step=0.01):
    """
    生成所有动作概率组合（和为1）

    格式: [p_noaction, p_repair, p_delete, p_nearby]
    约束: p_nearby (replace_nearby) ∈ [0.90, 1.00]
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

    print(f"  概率组合数: {len(configs)}")
    return configs


def generate_biased_random_actions(error_list, bias):
    """根据概率偏置生成动作"""
    n = len(error_list)
    bias = np.array(bias)
    bias = bias / bias.sum()  # 归一化
    return np.random.choice([0, 1, 2, 3], size=n, p=bias)


def sample_search_space(X_dirty, X_clean, y, error_list, X_test, y_test,
                        n_samples=1000, save_interval=200, run_id=0):
    """
    采样搜索空间

    1. 运行 4 个极端策略
    2. 遍历概率组合进行偏置随机采样
    """
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(X_dirty)

    results = []

    # 极端策略
    print("  运行极端策略...", flush=True)
    for strategy in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
        actions = generate_extreme_actions(error_list, strategy)
        r = run_single_strategy(X_dirty, X_clean, y, error_list, actions,
                                X_test, y_test, knn_imputer, name=strategy)
        if r:
            results.append(r)
            print(f"    {strategy}: V={r['veracity']:.4f}, D={r['diversity']:.4f}, "
                  f"A={r['accuracy']:.4f}")

    # 偏置采样
    print("  生成概率组合...", flush=True)
    configs = generate_uniform_bias_configs(step=0.01)
    n_per_config = max(1, n_samples // len(configs))
    total = len(configs) * n_per_config
    print(f"  采样: {len(configs)} 组合 × {n_per_config} 次 = {total} 策略", flush=True)

    count = 0
    for config in tqdm(configs, desc='  采样进度'):
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
    """保存中间结果到 CSV"""
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

    # 累积保存：保留旧数据，追加新 Random，极端点取最新
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        # 保留旧的 Random 记录
        old_random = old_df[~old_df['name'].isin(EXTREME_NAMES)]
        # 新的极端点 + 新的 Random
        new_extreme = new_df[new_df['name'].isin(EXTREME_NAMES)]
        new_random = new_df[~new_df['name'].isin(EXTREME_NAMES)]
        combined = pd.concat([new_extreme, old_random, new_random], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)


# =============================================================================
# 5. DQN 训练与推理（使用 DemandClean 发行版 API）
# =============================================================================
def run_dqn_training_and_inference(X_dirty_train, X_clean_train, y_train,
                                   X_dirty_test, X_clean_test, y_test,
                                   error_list_test, X_test_split, y_test_split,
                                   knn_imputer, n_runs=1, load_model=True):
    """
    使用 DemandClean 发行版 API 进行 DQN 训练与推理

    替换原始 TensorFlow 实现:
    - 旧: from demand_clean_dqn import DemandCleanDQNAgent, DemandCleanEnv
    - 新: from demandclean.api.demand_clean import DemandClean

    阶段 1: 训练或加载模型
    阶段 2: 多轮推理，每轮记录 (veracity, diversity, accuracy, action_counts)
    """
    model_path = os.path.join(MODEL_DIR, 'search_space_dqn.pt')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 构造 DataFrame 用于 DemandClean API
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

    # 阶段 1: 训练或加载
    if load_model and os.path.exists(model_path):
        print(f"  [DQN] 加载模型: {model_path}", flush=True)
        dc.load(model_path)
    else:
        print(f"  [DQN] 训练模型 (episodes=500)...", flush=True)
        dc.fit(X_dirty_train, y_train, X_clean=X_clean_train, y_clean=y_train)
        dc.save(model_path)
        print(f"  [DQN] 模型已保存: {model_path}", flush=True)

    # 阶段 2: 多轮推理
    dqn_results = []
    for run in range(n_runs):
        print(f"  [DQN] 推理轮次 {run+1}/{n_runs}...", flush=True)

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
# 6. 可视化函数（完整保留原始逻辑，仅更新保存路径）
# =============================================================================
def _compute_alpha_shape(x_arr, y_arr, alpha_val=2.0, max_points=5000):
    """计算 alpha shape 边界（对大数据集自动子采样以加速）"""
    if not HAS_ALPHASHAPE:
        return None
    try:
        points = np.column_stack([x_arr, y_arr])
        # 子采样（alphashape 不需要全部点来确定边界）
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        return alphashape.alphashape(points, alpha=alpha_val)
    except Exception as e:
        print(f"  [WARN] Alpha shape 失败: {e}")
        return None


MAX_SCATTER_POINTS = 30000  # 散点图最大绘制点数（44万 → 3万，视觉效果不变）


def _subsample_results(random_results, max_n=MAX_SCATTER_POINTS):
    """对随机策略结果子采样以加速可视化"""
    if len(random_results) <= max_n:
        return random_results
    idx = np.random.choice(len(random_results), max_n, replace=False)
    return [random_results[i] for i in idx]


def _mask_outside_shape(shape, XX, YY, AA):
    """将 alpha shape 外的网格点设为 NaN（使用 shapely.vectorized 加速）"""
    if shape is None:
        return AA
    try:
        from shapely import vectorized as sv
        mask = sv.contains(shape, XX, YY)
        AA = AA.copy()
        AA[~mask] = np.nan
        return AA
    except ImportError:
        # 回退到 prepared geometry
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
    """设置学术论文风格"""
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
    搜索空间散点图 (双子图)

    左图: 搜索空间分布 + 极端点 + DemandFix
    右图: 性能热力图散点
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # 子采样以加速绘制（44万 → 3万）
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    a_rand = [r['accuracy'] for r in plot_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左图: 搜索空间 ---
    ax1.scatter(v_rand, d_rand, c='lightgray', s=3, alpha=0.4, label='Random strategies')

    # DQN 结果
    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        ax1.scatter(v_dqn, d_dqn, c='orange', s=60, marker='*',
                    zorder=5, label='DemandFix (DQN)')

    # 极端点
    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}
    for name, color in extreme_colors.items():
        if name in extreme_map:
            r = extreme_map[name]
            ax1.scatter(r['veracity'], r['diversity'], c=color, s=150,
                        marker=extreme_markers[name], zorder=10, edgecolors='black',
                        linewidths=1.5, label=name)

    # 边界连线
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

    # --- 右图: 性能热力图 ---
    if len(a_rand) > 0:
        sc = ax2.scatter(v_rand, d_rand, c=a_rand, cmap='RdYlBu_r', s=5, alpha=0.6,
                         norm=PowerNorm(gamma=0.5, vmin=min(a_rand), vmax=max(a_rand)))
        plt.colorbar(sc, ax=ax2, label='Classification Accuracy')

    # 极端点（空心）
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_search_space_grid_heatmap(results, output_path, dqn_results=None, grid_size=80):
    """
    网格热力图

    使用 alphashape 裁剪月牙形区域外的网格
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    v = np.array([r['veracity'] for r in random_results])
    d = np.array([r['diversity'] for r in random_results])
    a = np.array([r['accuracy'] for r in random_results])

    if len(v) < 10:
        print(f"  [跳过] 数据不足: {len(v)}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 网格
    v_grid = np.linspace(v.min(), v.max(), grid_size)
    d_grid = np.linspace(d.min(), d.max(), grid_size)
    VV, DD = np.meshgrid(v_grid, d_grid)

    # 插值
    AA = griddata((v, d), a, (VV, DD), method='linear')

    # Alpha shape 裁剪
    shape = _compute_alpha_shape(v, d)
    AA = _mask_outside_shape(shape, VV, DD, AA)

    # 绘制
    mesh = ax.pcolormesh(VV, DD, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # 等高线
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(VV, DD, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # 极端点
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_combined_figure(results, dqn_results, output_path):
    """
    组合图 (三子图)

    图 1: 搜索空间边界
    图 2: 性能热力图
    图 3: DemandFix 分布
    """
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # 子采样以加速绘制
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    a_rand = [r['accuracy'] for r in plot_results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}

    # --- 图 1: 搜索空间边界 ---
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

    # --- 图 2: 性能热力图 ---
    if len(a_rand) > 0:
        # 百分位归一化（vectorized）
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

    # --- 图 3: DemandFix 分布 ---
    ax3.scatter(v_rand, d_rand, c='lightgray', s=2, alpha=0.2)

    if dqn_results:
        v_dqn = [r['veracity'] for r in dqn_results]
        d_dqn = [r['diversity'] for r in dqn_results]
        a_dqn = [r['accuracy'] for r in dqn_results]

        sc3 = ax3.scatter(v_dqn, d_dqn, c=a_dqn, cmap='Oranges', s=80,
                          edgecolors='black', linewidths=0.5, zorder=5)
        plt.colorbar(sc3, ax=ax3, label='DQN Accuracy')

        # 平均值
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_action_pair_heatmap(results, action_x, action_y, output_path,
                             dqn_results=None, grid_size=50):
    """
    动作组合热力图

    X/Y 轴为两种动作的比例，颜色为性能
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    action_names = {0: 'no_action', 1: 'repair_value', 2: 'delete', 3: 'replace_nearby'}
    action_short = {0: 'noact', 1: 'repair', 2: 'delete', 3: 'nearby'}

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    if not random_results:
        return

    # 计算比例
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

    # 网格插值
    x_grid = np.linspace(x_arr.min(), x_arr.max(), grid_size)
    y_grid = np.linspace(y_arr.min(), y_arr.max(), grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    AA = griddata((x_arr, y_arr), a_arr, (XX, YY), method='linear')

    # Alpha shape 裁剪
    shape = _compute_alpha_shape(x_arr, y_arr)
    AA = _mask_outside_shape(shape, XX, YY, AA)

    mesh = ax.pcolormesh(XX, YY, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # 等高线
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(XX, YY, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # 极端点
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_all_action_pair_heatmaps(results, output_dir, dqn_results=None):
    """生成所有动作两两组合热力图 (C(4,2)=6 个)"""
    os.makedirs(output_dir, exist_ok=True)
    action_short = {0: 'noact', 1: 'repair', 2: 'delete', 3: 'nearby'}

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for ax, ay in pairs:
        filename = f'{action_short[ax]}_vs_{action_short[ay]}.png'
        output_path = os.path.join(output_dir, filename)
        plot_action_pair_heatmap(results, ax, ay, output_path, dqn_results)


def plot_repair_vs_nonrepair_heatmap(results, output_path, dqn_results=None, grid_size=50):
    """
    Repair vs Non-repair 热力图

    X = repair_value 比例
    Y = (delete + replace_nearby) 比例
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

    # 网格插值
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    AA = griddata((x, y), a, (XX, YY), method='linear')

    # Alpha shape 裁剪
    shape = _compute_alpha_shape(x, y)
    AA = _mask_outside_shape(shape, XX, YY, AA)

    mesh = ax.pcolormesh(XX, YY, AA, cmap='RdYlBu_r', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Classification Accuracy')

    # 等高线
    valid = ~np.isnan(AA)
    if valid.any():
        levels = np.linspace(np.nanmin(AA), np.nanmax(AA), 8)
        ax.contour(XX, YY, AA, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # 极端点
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_cost_heatmap(results, output_path, dqn_results=None, grid_size=80):
    """
    成本热力图

    X = Authenticity, Y = Diversity, 颜色 = repair_value 次数（修复成本）
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

    # 网格插值
    v_grid = np.linspace(v.min(), v.max(), grid_size)
    d_grid = np.linspace(d.min(), d.max(), grid_size)
    VV, DD = np.meshgrid(v_grid, d_grid)
    CC = griddata((v, d), costs, (VV, DD), method='linear')

    # Alpha shape 裁剪
    shape = _compute_alpha_shape(v, d)
    CC = _mask_outside_shape(shape, VV, DD, CC)

    mesh = ax.pcolormesh(VV, DD, CC, cmap='YlOrRd', shading='auto',
                         norm=PowerNorm(gamma=0.5))
    plt.colorbar(mesh, ax=ax, label='Truth Value Cost (repair_value count)')

    # 等高线（幂函数间隔）
    valid_costs = CC[~np.isnan(CC)]
    if len(valid_costs) > 0:
        c_min, c_max = valid_costs.min(), valid_costs.max()
        if c_max > c_min:
            n_levels = 8
            levels = c_min + (c_max - c_min) * np.power(
                np.linspace(0, 1, n_levels), 2)
            ax.contour(VV, DD, CC, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # 极端点
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
    print(f"  [图] {os.path.basename(output_path)}")


def plot_cost_scatter(results, output_path, dqn_results=None):
    """
    成本散点图 (双子图)

    左图: 搜索空间分布
    右图: 成本散点着色
    """
    from matplotlib.colors import PowerNorm
    set_academic_style()

    random_results = [r for r in results if _is_random(r['name'])]
    extreme_map = {r['name']: r for r in results if r['name'] in EXTREME_NAMES}

    # 子采样以加速绘制
    plot_results = _subsample_results(random_results)
    v_rand = [r['veracity'] for r in plot_results]
    d_rand = [r['diversity'] for r in plot_results]
    costs = [r['action_counts'].get(1, 0) for r in plot_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    extreme_colors = {'NoFix': 'red', 'FullFix': 'blue', 'DeleteFix': 'green', 'RelaxFix': 'purple'}
    extreme_markers = {'NoFix': 's', 'FullFix': '^', 'DeleteFix': 'D', 'RelaxFix': 'v'}

    # --- 左图: 分布 ---
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

    # --- 右图: 成本热力图散点 ---
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
    print(f"  [图] {os.path.basename(output_path)}")


# =============================================================================
# 7. 主函数
# =============================================================================
def main():
    print("=" * 70)
    print("搜索空间可视化实验")
    print("=" * 70)
    print(f"  RUN_EXPERIMENT={RUN_EXPERIMENT}, LOAD_MODEL={LOAD_MODEL}")
    print(f"  N_RANDOM_SAMPLES={N_RANDOM_SAMPLES}, N_DQN_RUNS={N_DQN_RUNS}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # ------------------------------------------------------------------
    # [1] 加载数据
    # ------------------------------------------------------------------
    print("\n[1] 加载数据")
    clean_df = load_and_preprocess()
    X_clean = clean_df[['abv', 'ibu']].values.astype(float)
    y = clean_df['is_ipa'].values

    # 标准化
    scaler = StandardScaler()
    X_clean_scaled = scaler.fit_transform(X_clean)

    # 划分训练/测试
    train_idx, test_idx = train_test_split(
        np.arange(len(X_clean_scaled)), test_size=0.3,
        random_state=42, stratify=y)
    X_test, y_test = X_clean_scaled[test_idx], y[test_idx]

    # ------------------------------------------------------------------
    # [2] 注入测试错误
    # ------------------------------------------------------------------
    print("\n[2] 注入测试错误 (seed=2024)")
    X_dirty_test_scaled, error_list_test = inject_errors(
        X_clean_scaled, y, SEMANTIC_RATE, SYNTACTIC_RATE, seed=2024)

    # 保存脏数据 CSV
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
    print(f"  脏数据已保存: {dirty_csv_path}")

    results = None
    dqn_results = None
    extreme_points = None

    if RUN_EXPERIMENT:
        # ------------------------------------------------------------------
        # [3] 随机采样策略空间
        # ------------------------------------------------------------------
        print("\n[3] 随机采样策略空间")
        results = sample_search_space(
            X_dirty_test_scaled, X_clean_scaled, y, error_list_test,
            X_test, y_test, n_samples=N_RANDOM_SAMPLES)

        # ------------------------------------------------------------------
        # [4] DQN 训练与推理
        # ------------------------------------------------------------------
        print("\n[4] DQN 训练与推理")
        # 注入训练错误 (不同 seed)
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
        # [5] 保存结果
        # ------------------------------------------------------------------
        print("\n[5] 保存结果")

        # 保存采样结果 CSV
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

        # 累积保存
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
        print(f"  采样结果: {csv_path} ({len(combined_df)} 条)")

        # 从 combined_df 重建 results（用于后续画图）
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

        # 保存 DQN 结果
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
        print(f"  DQN 结果: {dqn_csv_path}")

        # 保存极端点
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
        print(f"  极端点: {ep_path}")

    else:
        # ------------------------------------------------------------------
        # [3] 读取已有结果
        # ------------------------------------------------------------------
        print("\n[3] 读取已有结果")

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
            print(f"  采样结果: {len(results)} 条")
        else:
            print(f"  [ERROR] 未找到 {csv_path}，请设置 RUN_EXPERIMENT=True 运行实验")
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
            print(f"  DQN 结果: {len(dqn_results)} 条")
        else:
            print(f"  [WARN] 未找到 {dqn_csv_path}，DQN 点将不显示")
            dqn_results = []

        ep_path = os.path.join(RESULT_DIR, 'extreme_points.json')
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                extreme_points = json.load(f)
            print(f"  极端点: {list(extreme_points.keys())}")

    # ------------------------------------------------------------------
    # [6] 生成可视化
    # ------------------------------------------------------------------
    print("\n[6] 生成可视化")

    import time as _time

    _t0 = _time.time()
    plot_search_space_scatter(
        results, dqn_results,
        os.path.join(RESULT_DIR, 'search_space_scatter.png'))
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_search_space_grid_heatmap(
        results,
        os.path.join(RESULT_DIR, 'search_space_heatmap.png'),
        dqn_results=dqn_results)
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_combined_figure(
        results, dqn_results,
        os.path.join(RESULT_DIR, 'search_space_combined.png'))
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_cost_heatmap(
        results,
        os.path.join(RESULT_DIR, 'search_space_cost_heatmap.png'),
        dqn_results=dqn_results)
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_cost_scatter(
        results,
        os.path.join(RESULT_DIR, 'search_space_cost_scatter.png'),
        dqn_results=dqn_results)
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    _t0 = _time.time()
    plot_repair_vs_nonrepair_heatmap(
        results,
        os.path.join(RESULT_DIR, 'repair_vs_nonrepair.png'),
        dqn_results=dqn_results)
    print(f"  耗时: {_time.time()-_t0:.1f}s")

    # 动作组合热力图（可选，默认关闭以节省时间）
    # plot_all_action_pair_heatmaps(
    #     results,
    #     os.path.join(RESULT_DIR, 'action_pairs'),
    #     dqn_results=dqn_results)

    # ------------------------------------------------------------------
    # [7] 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    # 极端策略
    print(f"\n{'策略':<15} {'真实性':>8} {'多样性':>8} {'准确率':>8}")
    print("-" * 45)
    for r in results:
        if r['name'] in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
            print(f"{r['name']:<15} {r['veracity']:>8.4f} {r['diversity']:>8.4f} "
                  f"{r['accuracy']:>8.4f}")

    # DemandFix DQN
    if dqn_results:
        mean_v = np.mean([r['veracity'] for r in dqn_results])
        mean_d = np.mean([r['diversity'] for r in dqn_results])
        mean_a = np.mean([r['accuracy'] for r in dqn_results])
        print(f"\nDemandFix (DQN) 平均: V={mean_v:.4f}, D={mean_d:.4f}, A={mean_a:.4f}")

    print(f"\n总耗时: {time.time() - total_start:.2f}s")
    return results, dqn_results


if __name__ == '__main__':
    results, dqn_results = main()
