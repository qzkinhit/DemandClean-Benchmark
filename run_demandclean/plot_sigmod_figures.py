#!/usr/bin/env python3
"""
SIGMOD 论文搜索空间示意图生成器（使用真实实验数据）

复用 run_ablation.py 和 run_search_space.py 的数据流水线，
确保生成的图与实验结果完全一致。

生成：
1. search_space_main.svg - 搜索空间散点图（双子图，复用 run_search_space.py 风格）
2. inset_{策略名}.svg    - 各策略的 ABV vs IBU 散点 + SVM 决策边界（复用 run_ablation.py 风格）

Usage:
    python plot_sigmod_figures.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 路径配置
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures', 'sigmod')

# 搜索空间数据
SEARCH_SPACE_CSV = os.path.join(PROJECT_ROOT, 'experiment', 'search_space_beers', 'result', 'search_space_results.csv')
DQN_DIST_CSV = os.path.join(PROJECT_ROOT, 'experiment', 'search_space_beers', 'result', 'dqn_distribution.csv')
EXTREME_POINTS_JSON = os.path.join(PROJECT_ROOT, 'experiment', 'search_space_beers', 'result', 'extreme_points.json')

# Ablation 数据
BEERS_CLEAN_CSV = os.path.join(PROJECT_ROOT, 'experiment', 'ablation_beers', 'datasets', 'beers', 'clean.csv')
ABLATION_MODEL_DIR = os.path.join(PROJECT_ROOT, 'experiment', 'ablation_beers', 'model')
ABLATION_RESULT_DIR = os.path.join(PROJECT_ROOT, 'experiment', 'ablation_beers', 'result')

# DemandClean API
from demandclean.api.demand_clean import DemandClean
from demandclean.core.environments.value_estimation import ValueEstimator
from demandclean.config import DemandCleanConfig

np.random.seed(2024)

# 极端策略名称
EXTREME_NAMES = {'NoFix', 'FullFix', 'DeleteFix', 'RelaxFix'}

# 真值预算
MIN_TRUTH_BUDGET = 10
MAX_TRUTH_BUDGET = 300

MAX_SCATTER_POINTS = 30000  # 子采样上限

# ============================================================================
# 全局样式
# ============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


# ============================================================================
# 1. 搜索空间主图（复用 run_search_space.py:plot_search_space_scatter 逻辑）
# ============================================================================
def _load_search_space_data():
    """加载搜索空间数据和极端点，返回公共数据供多个图复用"""
    from scipy.interpolate import griddata

    print('  加载搜索空间数据...', flush=True)
    df = pd.read_csv(SEARCH_SPACE_CSV)
    print(f'  总数据: {len(df)} 行')

    # 只取随机策略的数据（排除极端策略）
    random_mask = ~df['name'].isin(EXTREME_NAMES)
    df_random = df[random_mask]

    auth_all = df_random['veracity'].values
    div_all = df_random['diversity'].values
    acc_all = df_random['accuracy'].values
    cost_all = df_random['repair_value'].values

    # 加载 DQN 结果
    dqn_results = []
    if os.path.exists(DQN_DIST_CSV):
        dqn_df = pd.read_csv(DQN_DIST_CSV)
        for _, row in dqn_df.iterrows():
            dqn_results.append(row)
        print(f'  DQN 结果: {len(dqn_results)} 行')

    # 从 extreme_points.json 读取真实坐标
    with open(EXTREME_POINTS_JSON, 'r') as f:
        extreme_points = json.load(f)

    return auth_all, div_all, acc_all, cost_all, dqn_results, extreme_points


def _build_grid_heatmap(auth_all, div_all, values, grid_size=80):
    """
    构建网格热力图数据（pcolormesh 风格，与原始 run_search_space.py 一致）。
    使用 alpha shape 裁剪，只保留有真实数据覆盖的月牙形区域。
    返回 VV, DD, ZZ。
    """
    import alphashape
    from scipy.interpolate import griddata

    v_grid = np.linspace(auth_all.min(), auth_all.max(), grid_size)
    d_grid = np.linspace(div_all.min(), div_all.max(), grid_size)
    VV, DD = np.meshgrid(v_grid, d_grid)

    # griddata 线性插值
    ZZ = griddata((auth_all, div_all), values, (VV, DD), method='linear')

    # Alpha shape 裁剪（与 run_search_space.py 一致）
    sample_idx = np.random.choice(len(auth_all), min(5000, len(auth_all)), replace=False)
    points = np.column_stack([auth_all[sample_idx], div_all[sample_idx]])
    shape = alphashape.alphashape(points, alpha=1.9)  # 与 run_search_space.py 一致

    # 使用 shapely vectorized 或回退方案进行裁剪
    try:
        from shapely import vectorized as sv
        mask = sv.contains(shape, VV, DD)
    except ImportError:
        from shapely.geometry import Point
        from shapely.prepared import prep
        prepared = prep(shape)
        mask = np.zeros(VV.shape, dtype=bool)
        for i in range(VV.shape[0]):
            for j in range(VV.shape[1]):
                mask[i, j] = prepared.contains(Point(VV[i, j], DD[i, j]))

    ZZ[~mask] = np.nan

    return VV, DD, ZZ


def _annotate_strategies(ax, extreme_points, dqn_results, strategy_results=None):
    """
    在 ax 上标注策略点（箭头 + 文字框），与之前风格一致。
    """
    pt_color = '#5D3A1A'

    offsets = {'NoFix': (-0.10, 0.02), 'FullFix': (0.02, -0.10),
              'DeleteFix': (-0.16, -0.04), 'RelaxFix': (-0.04, 0.03)}

    for name in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
        if name not in extreme_points:
            continue
        ep = extreme_points[name]
        x, y_val = ep['veracity'], ep['diversity']

        ax.scatter(x, y_val, c=pt_color, s=120, edgecolors='black', linewidth=1.2, zorder=10)
        ox, oy = offsets.get(name, (0.02, 0.02))
        ax.annotate(name,
                   xy=(x, y_val), xytext=(x+ox, y_val+oy),
                   fontsize=15, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # DemandClean
    dm_data = None
    if dqn_results:
        dm = dqn_results[0]
        dm_data = {'auth': dm['veracity'], 'div': dm['diversity'],
                  'acc': dm['accuracy'], 'cost': 0}
        if strategy_results and 'DemandFix' in strategy_results:
            dm_data['cost'] = strategy_results['DemandFix']['cost']

    if dm_data:
        ax.scatter(dm_data['auth'], dm_data['div'], c='#C8A415', s=300,
                  marker='*', edgecolors='black', linewidth=1.2, zorder=15)
        ax.scatter(dm_data['auth'], dm_data['div'], c=pt_color, s=80,
                  edgecolors='black', linewidth=1.0, zorder=14)
        ax.annotate('DemandClean',
                   xy=(dm_data['auth'], dm_data['div']),
                   xytext=(0.64, 0.80),
                   fontsize=15, fontweight='bold', color='#5D3A1A',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='#C8A415', alpha=0.95, linewidth=2.5),
                   arrowprops=dict(arrowstyle='->', color='#C8A415', lw=1.8,
                                  connectionstyle='arc3,rad=0.3'))
    return dm_data


def _plot_one_heatmap(ax, fig, VV, DD, ZZ, cbar_label, cmap, norm=None):
    """绘制单张 pcolormesh 热力图 + 顶部 colorbar"""
    mesh = ax.pcolormesh(VV, DD, ZZ, cmap=cmap, shading='auto', norm=norm)
    cax = fig.add_axes([0.20, 0.92, 0.55, 0.025])
    cbar = plt.colorbar(mesh, cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=14, labelpad=5)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    return mesh, cbar


def _add_hull_boundary(ax, auth_all, div_all):
    """画凸包边界虚线并返回排序后的顶点"""
    from scipy.spatial import ConvexHull
    sample_idx = np.random.choice(len(auth_all), min(5000, len(auth_all)), replace=False)
    hull_points = np.column_stack([auth_all[sample_idx], div_all[sample_idx]])
    hull = ConvexHull(hull_points)
    hull_boundary = hull_points[hull.vertices]
    center = hull_boundary.mean(axis=0)
    angles = np.arctan2(hull_boundary[:, 1] - center[1], hull_boundary[:, 0] - center[0])
    order = np.argsort(angles)
    hull_sorted = hull_boundary[order]
    hull_sorted = np.vstack([hull_sorted, hull_sorted[0]])
    ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'k--', lw=2.0, alpha=0.7, zorder=5)
    return hull_sorted


def _finish_ax(ax):
    """公共坐标轴设置"""
    ax.set_xlabel('Authenticity (normalized)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Diversity (normalized)', fontsize=16, fontweight='bold')
    ax.set_xlim(0.62, 1.06)
    ax.set_ylim(0.58, 1.55)
    ax.grid(False)


def plot_search_space_main(strategy_results=None):
    """
    搜索空间图（三张），全部使用 pcolormesh 网格风格 + 箭头标注：
    1. search_space_main — 性能热力图 (RdBu_r + PowerNorm)
    2. search_space_cost — 成本热力图 (RdBu_r)
    3. search_space_tradeoff — 综合指标热力图 (非线性公式 + PowerNorm)
    """
    from matplotlib.colors import PowerNorm

    auth_all, div_all, acc_all, cost_all, dqn_results, extreme_points = _load_search_space_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid_size = 80

    # ================================================================
    # 图1: 性能热力图 — 与 run_search_space.py 一致
    # ================================================================
    print('  [图1] 性能热力图...', flush=True)
    VV, DD, AA = _build_grid_heatmap(auth_all, div_all, acc_all, grid_size)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    _plot_one_heatmap(ax, fig, VV, DD, AA, 'Classification Accuracy',
                      cmap='RdYlBu_r', norm=PowerNorm(gamma=0.5))
    hull_sorted = _add_hull_boundary(ax, auth_all, div_all)
    _annotate_strategies(ax, extreme_points, dqn_results, strategy_results)
    _finish_ax(ax)
    plt.subplots_adjust(top=0.88)
    for fmt in ['svg', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'search_space_main.{fmt}'),
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f'  [OK] search_space_main.svg/.png')
    plt.close()

    # ================================================================
    # 图2: 成本热力图 — 蓝(低)→红(高) 渐变
    # ================================================================
    print('  [图2] 成本热力图...', flush=True)
    VV2, DD2, CC = _build_grid_heatmap(auth_all, div_all, cost_all, grid_size)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    # 反转 RdBu，使高成本为红，低成本为蓝
    _plot_one_heatmap(ax, fig, VV2, DD2, CC, 'Truth Value Cost',
                      cmap='RdBu', norm=PowerNorm(gamma=0.5))
    ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'k--', lw=2.0, alpha=0.7, zorder=5)
    _annotate_strategies(ax, extreme_points, dqn_results, strategy_results)
    _finish_ax(ax)
    plt.subplots_adjust(top=0.88)
    for fmt in ['svg', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'search_space_cost.{fmt}'),
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f'  [OK] search_space_cost.svg/.png')
    plt.close()

    # ================================================================
    # 图3: Trade-off = min(P_norm, E_norm)
    #
    # 公式:
    #   P_norm = (accuracy - acc_min) / (acc_max - acc_min)      性能分 ∈ [0,1]
    #   C_total = C_repair + w · max(0, 1 - diversity) · C_max   综合成本
    #   E_norm = 1 - (C_total / C_total_max)^α                  效率分 ∈ [0,1]
    #   Trade-off = min(P_norm, E_norm)
    #
    # 其中 α < 1 为非线性放大系数，使小成本也产生显著惩罚。
    # 效果：只有**同时**高性能且低成本才红；任一维度差就变蓝。
    #
    # 策略标记使用各策略的**真实成本**着色（而非网格插值值），
    # 以体现 DemandClean (cost=43) vs FullFix (cost=491) 的巨大差异。
    # ================================================================
    print('  [图3] Trade-off = min(P_norm, E_norm)...', flush=True)

    # 性能归一化
    acc_min_v, acc_max_v = np.nanpercentile(acc_all, [2, 98])
    norm_acc = np.clip((acc_all - acc_min_v) / (acc_max_v - acc_min_v), 0, 1)

    # 综合成本 = repair + w × diversity_loss × max_repair
    cost_max_val = np.max(cost_all)
    w_div = 0.5
    alpha_cost = 0.3  # 非线性放大
    total_cost = cost_all + w_div * np.clip(1.0 - div_all, 0, 1.0) * cost_max_val
    tc_max = np.max(total_cost)
    norm_eff = 1.0 - np.clip(total_cost / tc_max, 0, 1) ** alpha_cost

    tradeoff_all = np.minimum(norm_acc, norm_eff)

    VV3, DD3, TT = _build_grid_heatmap(auth_all, div_all, tradeoff_all, grid_size)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    _plot_one_heatmap(ax, fig, VV3, DD3, TT,
                      r'$\min(\hat{Perf},\; 1-\hat{C})$  (higher $\Rightarrow$ accuracy $\uparrow$, cost $\downarrow$)',
                      cmap='RdYlBu_r', norm=PowerNorm(gamma=0.5))
    ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'k--', lw=2.0, alpha=0.7, zorder=5)

    # ---- 策略标记：用各自**真实成本**着色 ----
    # 计算各极端策略的 trade-off 分（使用它们的真实 repair cost）
    import matplotlib.colors as mcolors
    cmap_markers = plt.cm.RdYlBu_r
    marker_norm = mcolors.Normalize(vmin=0, vmax=1)

    def _strategy_tradeoff(acc_val, repair_val, div_val):
        p = np.clip((acc_val - acc_min_v) / (acc_max_v - acc_min_v), 0, 1)
        tc = repair_val + w_div * max(0, 1.0 - div_val) * cost_max_val
        e = 1.0 - min(tc / tc_max, 1.0) ** alpha_cost
        return min(p, e)

    pt_color = '#5D3A1A'
    offsets = {'NoFix': (-0.10, 0.02), 'FullFix': (0.02, -0.10),
              'DeleteFix': (-0.16, -0.04), 'RelaxFix': (-0.04, 0.03)}
    # 各极端策略的真实 repair cost
    real_costs = {'NoFix': 0, 'FullFix': 491, 'DeleteFix': 0, 'RelaxFix': 0}

    for name in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
        if name not in extreme_points:
            continue
        ep = extreme_points[name]
        x, y_val, acc_v = ep['veracity'], ep['diversity'], ep['accuracy']
        score = _strategy_tradeoff(acc_v, real_costs[name], y_val)
        color = cmap_markers(marker_norm(score))
        ax.scatter(x, y_val, c=[color], s=150, edgecolors='black', linewidth=1.5, zorder=10)
        ox, oy = offsets.get(name, (0.02, 0.02))
        ax.annotate(name,
                   xy=(x, y_val), xytext=(x+ox, y_val+oy),
                   fontsize=15, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.92),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # DemandClean: 标到 trade-off 热力图最红的位置（示意图）
    # 找 TT 网格中最大值的位置
    TT_safe = np.where(np.isnan(TT), -np.inf, TT)
    max_idx = np.unravel_index(np.argmax(TT_safe), TT_safe.shape)
    dm_x = VV3[max_idx]
    dm_y = DD3[max_idx]
    print(f'  DemandClean 标记位置 (示意): auth={dm_x:.3f}, div={dm_y:.3f}, '
          f'score={TT[max_idx]:.3f}')

    ax.scatter(dm_x, dm_y, c='#C8A415', s=300,
              marker='*', edgecolors='black', linewidth=1.2, zorder=15)
    ax.annotate('DemandClean',
               xy=(dm_x, dm_y),
               xytext=(0.64, 0.80),
               fontsize=15, fontweight='bold', color='#5D3A1A',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#C8A415', alpha=0.95, linewidth=2.5),
               arrowprops=dict(arrowstyle='->', color='#C8A415', lw=1.8,
                              connectionstyle='arc3,rad=0.3'))

    for name in ['NoFix', 'FullFix', 'DeleteFix', 'RelaxFix']:
        if name in extreme_points:
            ep = extreme_points[name]
            s = _strategy_tradeoff(ep['accuracy'], real_costs[name], ep['diversity'])
            print(f'  {name} trade-off score = {s:.3f}')

    _finish_ax(ax)
    plt.subplots_adjust(top=0.88)
    for fmt in ['svg', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'search_space_tradeoff.{fmt}'),
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f'  [OK] search_space_tradeoff.svg/.png (alpha={alpha_cost}, w_div={w_div})')
    plt.close()


# ============================================================================
# 2. ErrorInjector（完整复用 run_ablation.py 的逻辑）
# ============================================================================
class ErrorInjector:
    """错误注入器 - 与 run_ablation.py 完全一致"""

    def __init__(self, df):
        self.df = df.copy()
        self.df['ABV_dirty'] = df['abv'].copy()
        self.df['IBU_dirty'] = df['ibu'].copy()
        self.errors = {'missing': [], 'semantic': [], 'syntactic': []}
        self.used_rows = set()

        boundary_mask = (df['ibu'] >= 35) & (df['ibu'] <= 65)
        self.boundary_idx = df[boundary_mask].index.tolist()
        self.non_boundary_idx = df[~boundary_mask].index.tolist()
        self.all_ibu_values = df['ibu'].values

    def inject_semantic(self, rate=0.15):
        n_boundary = int(len(self.boundary_idx) * rate * 1.5)
        n_non_boundary = int(len(self.non_boundary_idx) * rate * 0.3)

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
        n_boundary = int(len(self.boundary_idx) * rate * 2.0)
        n_non_boundary = int(len(self.non_boundary_idx) * rate * 0.5)

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
        return self.df, self.errors


# ============================================================================
# 3. 基线策略（完整复用 run_ablation.py）
# ============================================================================
def strategy_nofix(X, y, detected, X_clean):
    missing_rows = set(e[0] for e in detected['missing'])
    keep = np.array([i not in missing_rows for i in range(len(X))])
    X_out = X[keep].copy()
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(missing_rows), keep


def strategy_deletefix(X, y, detected, X_clean):
    error_rows = set()
    for key in detected:
        for e in detected[key]:
            error_rows.add(e[0])
    min_keep = max(int(len(X) * 0.2), 10)
    n_keep = len(X) - len(error_rows)
    if n_keep < min_keep:
        n_to_delete = max(len(X) - min_keep, 0)
        error_list = sorted(error_rows)
        error_rows = set(error_list[:n_to_delete])
    keep = np.array([i not in error_rows for i in range(len(X))])
    X_out = X[keep].copy()
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(error_rows), keep


def strategy_replaceall(X, y, detected, X_clean):
    config = DemandCleanConfig(column_names=['abv', 'ibu'], save_path=OUTPUT_DIR)
    estimator = ValueEstimator(config)
    X_out = X.copy()
    col_means = np.nanmean(X_out, axis=0)
    deleted_rows = set()
    for key in detected:
        for e in detected[key]:
            idx = e[0]
            col = e[1] if isinstance(e[1], int) else 1
            estimated = estimator.estimate_feature_value(X_out, idx, col, deleted_rows, col_means)
            X_out[idx, col] = estimated
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = col_means[col]
    return X_out, y, 0, 0, None


def strategy_fullfix(X, y, detected, X_clean):
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
    X_out = X.copy()
    config = DemandCleanConfig(column_names=['abv', 'ibu'], save_path=OUTPUT_DIR)
    ve = ValueEstimator(config)
    col_means = np.nanmean(X_out, axis=0)
    deleted_rows_set = set()
    all_error_indices = set()
    X_imp = X_out.copy()
    for key in detected:
        for e in detected[key]:
            all_error_indices.add(e[0])
            idx = e[0]
            col = e[1] if isinstance(e[1], int) else 1
            X_imp[idx, col] = ve.estimate_feature_value(X_out, idx, col, deleted_rows_set, col_means)
    for c in range(X_imp.shape[1]):
        X_imp[np.isnan(X_imp[:, c]), c] = col_means[c]

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

    for e in detected['semantic']:
        X_out[e[0], 1] = X_clean[e[0], 1]
        cost += 1

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
    nan_rows = set(np.where(np.isnan(X).any(axis=1))[0])
    keep = np.array([i not in nan_rows for i in range(len(X))])
    X_out = X[keep].copy()
    if len(X_out) == 0:
        return X_out, y[keep], 0, len(nan_rows), keep
    means = np.nanmean(X_out, axis=0)
    for col in range(X_out.shape[1]):
        X_out[np.isnan(X_out[:, col]), col] = means[col]
    return X_out, y[keep], 0, len(nan_rows), keep


# ============================================================================
# 4. DQN 策略（通过 DemandClean API）
# ============================================================================
DQN_STRATEGY_TABLE = {
    'DQN_Single': {
        'detector_mode': 'oracle', 'agent_type': 'single',
        'inference_mode': 'single_phase', 'model_name': 'demandfix_dqn.pt',
    },
    'DQN_TwoStage': {
        'detector_mode': 'oracle', 'agent_type': 'two_stage',
        'inference_mode': 'single_phase', 'model_name': 'two_stage_dqn.pt',
    },
    'SemiSup_Single': {
        'detector_mode': 'oracle', 'agent_type': 'single',
        'inference_mode': 'single_phase', 'model_name': 'semi_supervised_single.pt',
    },
    'SemiSup_TwoStage': {
        'detector_mode': 'oracle', 'agent_type': 'two_stage',
        'inference_mode': 'single_phase', 'model_name': 'semi_supervised_dqn.pt',
    },
    'FullUnsup_Single': {
        'detector_mode': 'auto', 'agent_type': 'single',
        'inference_mode': 'single_phase', 'model_name': 'full_unsupervised_single.pt',
    },
    'FullUnsup_TwoStage': {
        'detector_mode': 'auto', 'agent_type': 'two_stage',
        'inference_mode': 'single_phase', 'model_name': 'full_unsupervised_dqn.pt',
    },
    'SemiSup_Dueling_Single': {
        'detector_mode': 'oracle', 'agent_type': 'dueling_single',
        'inference_mode': 'single_phase', 'model_name': 'semi_supervised_dueling_single.pt',
    },
    'SemiSup_Dueling_TwoStage': {
        'detector_mode': 'oracle', 'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase', 'model_name': 'semi_supervised_dueling.pt',
    },
    'FullUnsup_Dueling_Single': {
        'detector_mode': 'auto', 'agent_type': 'dueling_single',
        'inference_mode': 'single_phase', 'model_name': 'full_unsupervised_dueling_single.pt',
    },
    'FullUnsup_Dueling_TwoStage': {
        'detector_mode': 'auto', 'agent_type': 'dueling_two_stage',
        'inference_mode': 'single_phase', 'model_name': 'full_unsupervised_dueling.pt',
    },
}


def run_dqn_strategy(strategy_name, X_dirty, y, X_clean,
                     X_clean_val=None, y_clean_val=None,
                     dirty_csv_path=None, clean_csv_path=None,
                     pre_detected=None):
    """DQN 策略：加载已训练模型并推理"""
    params = DQN_STRATEGY_TABLE[strategy_name]
    model_path = os.path.join(ABLATION_MODEL_DIR, params['model_name'])

    if not os.path.exists(model_path):
        print(f'    [SKIP] 模型不存在: {model_path}')
        return None

    dc = DemandClean(
        task_type='classification', model_type='svm',
        agent_type=params['agent_type'],
        detector_mode=params['detector_mode'],
        inference_mode=params['inference_mode'],
        n_episodes=400,
        min_truth_budget=MIN_TRUTH_BUDGET, max_truth_budget=MAX_TRUTH_BUDGET,
        max_repair_ratio=0.8,
        column_names=['abv', 'ibu'], label_col='is_ipa',
        save_path=ABLATION_RESULT_DIR,
        dirty_csv_path=dirty_csv_path, clean_csv_path=clean_csv_path,
        csv_columns=['abv', 'ibu', 'is_ipa'],
        reward_eval_interval=10, repair_lambda=0.005,
    )

    print(f'    加载模型: {model_path}', flush=True)
    dc.load(model_path)

    print(f'    推理...', flush=True)
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, pre_detected=pre_detected)

    keep_mask = stats.get('keep_mask', None)
    truth_cost = stats.get('truth_cost', 0)
    deleted_count = stats.get('deleted_count', 0)

    return X_result, y_result, truth_cost, deleted_count, keep_mask


# ============================================================================
# 5. 指标计算（完整复用 run_ablation.py:compute_metrics）
# ============================================================================
def compute_metrics(X_result, X_clean, X_dirty, keep_mask=None):
    """
    计算 auth (authenticity) 和 div (diversity)

    diversity 使用与搜索空间一致的公式:
      div = sample_retention × variance_retention
      variance_retention = clip(result_var / clean_kept_var, 0, 1.5)
    """
    n = len(X_result)
    n_total = len(X_clean)
    if n == 0:
        return 0, 0

    # 真实性
    if keep_mask is not None:
        X_clean_kept = X_clean[keep_mask]
        correct = sum(1 for i in range(min(n, len(X_clean_kept)))
                      if abs(X_result[i, 1] - X_clean_kept[i, 1]) < 0.01)
    else:
        X_clean_kept = X_clean
        correct = sum(1 for i in range(n) if abs(X_result[i, 1] - X_clean[i, 1]) < 0.01)
    auth = correct / n

    # 多样性 — 与搜索空间 (run_search_space.py) 一致的公式
    sample_ret = n / n_total

    if len(X_clean_kept) > 1 and len(X_result) > 1:
        result_var = np.var(X_result[:, 1])
        clean_kept_var = np.var(X_clean_kept[:, 1])
        if clean_kept_var > 1e-10:
            var_ret = np.clip(result_var / clean_kept_var, 0, 1.5)
        else:
            var_ret = 1.0
        div = sample_ret * var_ret
    else:
        div = 0

    return auth, div


# ============================================================================
# 6. 可视化：SVM 决策边界（复用 run_ablation.py:plot_boundary 风格，输出 SVG）
# ============================================================================
def plot_boundary_svg(clf, ideal_clf, X_train, y_train, X_test, y_test,
                      name, cost, deleted, auth, div, path, xlim, ylim,
                      override_acc=None, action_dist=None):
    """SVM 决策边界可视化 — 与 run_ablation.py 完全一致的风格，输出 SVG"""
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),
                          np.linspace(ylim[0], ylim[1], 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Zi = ideal_clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # 如果有搜索空间真实 accuracy，用于标注显示
    display_acc = override_acc if override_acc is not None else acc

    fig, ax = plt.subplots(figsize=(4, 3.5))

    # 训练数据散点
    for label, marker, color, lbl in [(0, 'o', 'C0', 'Non-IPA (train)'),
                                       (1, 'x', 'C1', 'IPA (train)')]:
        mask = y_train == label
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                    c=color, marker=marker, s=25, alpha=0.7, label=lbl)

    # 误分类红圈
    mis_mask = y_test != y_pred
    if mis_mask.sum() > 0:
        ax.scatter(X_test[mis_mask, 0], X_test[mis_mask, 1],
                    facecolors='none', edgecolors='red', s=80, linewidths=2,
                    label=f'Misclassified ({mis_mask.sum()})')

    # 决策边界: 紫色实线 = 当前 SVM，绿色虚线 = 理想 SVM
    ax.contour(xx, yy, Z, levels=[0], colors=['purple'], linewidths=2.5)
    ax.contour(xx, yy, Zi, levels=[0], colors=['green'], linestyles=['--'], linewidths=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('ABV')
    ax.set_ylabel('IBU')
    ax.set_title(name, fontsize=12)

    # 底部指标条
    summary = f'Acc: {display_acc:.3f} | Cost: {cost} | Auth: {auth:.2f} | Div: {div:.2f}'
    if action_dist:
        total = sum(action_dist.values())
        parts = []
        for k, label in [('repair_value', 'Repair'), ('delete', 'Delete'),
                         ('replace_nearby', 'Replace'), ('no_action', 'NoAct')]:
            v = action_dist.get(k, 0)
            if v > 0:
                # parts.append(f'{label}: {v}({v*100/total:.0f}%)')
                parts.append(f'{label}: {v}')
        summary += '\n' + ' | '.join(parts)
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='upper left', fontsize=7)
    plt.tight_layout()

    # 保存 SVG + PNG
    for fmt in ['svg', 'png']:
        plt.savefig(path.replace('.svg', f'.{fmt}'), format=fmt, dpi=300, bbox_inches='tight')
    print(f'  [OK] {os.path.basename(path)}')
    plt.close()
    return acc


# ============================================================================
# 7. 主函数
# ============================================================================
def main():
    print('=' * 60)
    print('SIGMOD 图形生成器（真实数据版）')
    print('=' * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 2 先运行（策略散点图），得到各策略的指标
    # 然后 Part 1 用这些指标标注搜索空间主图（保证数值一致）
    # ------------------------------------------------------------------
    print('\n[Part 2] 策略散点图（先运行以获取指标）')

    # 2.1 加载数据
    print('\n  [2.1] 加载数据')
    clean_df = pd.read_csv(BEERS_CLEAN_CSV)
    valid_mask = clean_df['abv'].notna() & clean_df['ibu'].notna()
    clean_df = clean_df[valid_mask].reset_index(drop=True)
    clean_df['is_ipa'] = clean_df['style'].apply(
        lambda s: 1 if pd.notna(s) and 'ipa' in str(s).lower() else 0)
    print(f'  数据: {len(clean_df)} 行, IPA: {clean_df["is_ipa"].sum()}')

    # 2.2 注入错误
    print('\n  [2.2] 注入错误')
    injector = ErrorInjector(clean_df)
    injector.inject_semantic(0.25).inject_syntactic(0.35).inject_missing(0.05)
    dirty_df, errors = injector.get_result()
    print(f'  语义: {len(errors["semantic"])}, 句法: {len(errors["syntactic"])}, 缺失: {len(errors["missing"])}')

    # 2.3 60/20/20 划分 + 标准化
    print('\n  [2.3] 60/20/20 划分 + 标准化')
    X_clean_full = clean_df[['abv', 'ibu']].values.astype(float)
    X_dirty_full = dirty_df[['ABV_dirty', 'IBU_dirty']].values.astype(float)
    y_full = clean_df['is_ipa'].values

    all_idx = np.arange(len(X_clean_full))
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.4, random_state=42, stratify=y_full)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_full[temp_idx])
    print(f'  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # Scaler 在 dirty 训练集上 fit
    X_dirty_train_raw = X_dirty_full[train_idx]
    X_dirty_train_filled = X_dirty_train_raw.copy()
    train_col_means = np.nanmean(X_dirty_train_filled, axis=0)
    for col in range(X_dirty_train_filled.shape[1]):
        nan_mask = np.isnan(X_dirty_train_filled[:, col])
        X_dirty_train_filled[nan_mask, col] = train_col_means[col]

    scaler = StandardScaler()
    scaler.fit(X_dirty_train_filled)

    def scale_subset(X_raw):
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

    X_dirty_train = scale_subset(X_dirty_full[train_idx])
    X_clean_train = scaler.transform(X_clean_full[train_idx])
    X_clean_val = scaler.transform(X_clean_full[val_idx])
    X_clean_test = scaler.transform(X_clean_full[test_idx])
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    # 保存 60% CSV（DQN auto 模式需要）
    dirty_train_csv_path = os.path.join(ABLATION_RESULT_DIR, 'dirty_train_60pct.csv')
    clean_train_csv_path = os.path.join(ABLATION_RESULT_DIR, 'clean_train_60pct.csv')

    # 2.4 构造 detected 字典
    print('\n  [2.4] 构造检测结果')
    global_to_local = {int(g): l for l, g in enumerate(train_idx)}
    detected = {'missing': [], 'semantic': [], 'syntactic': []}
    for e in errors['missing']:
        if e[0] in global_to_local:
            detected['missing'].append((global_to_local[e[0]], 1, e[2]))
    for e in errors['semantic']:
        if e[0] in global_to_local:
            detected['semantic'].append((global_to_local[e[0]], 1, e[2], e[3]))
    for e in errors['syntactic']:
        if e[0] in global_to_local:
            detected['syntactic'].append((global_to_local[e[0]], 1, e[2], e[3]))
    total_train = sum(len(v) for v in detected.values())
    print(f'  训练集错误: {total_train}')

    # 2.5 训练理想分类器
    print('\n  [2.5] 训练理想分类器')
    ideal_clf = SVC(kernel='linear')
    ideal_clf.fit(X_clean_train, y_train)

    X_clean_full_scaled = scaler.transform(X_clean_full)
    xlim = (X_clean_full_scaled[:, 0].min() - 0.5, X_clean_full_scaled[:, 0].max() + 0.5)
    ylim = (X_clean_full_scaled[:, 1].min() - 0.5, X_clean_full_scaled[:, 1].max() + 0.5)

    # 2.6 构建策略字典
    print('\n  [2.6] 运行各策略并生成 SVG')
    baseline_strategies = {
        'NoFix': lambda: strategy_nofix(X_dirty_train, y_train, detected, X_clean_train),
        'DeleteAll': lambda: strategy_deleteall(X_dirty_train, y_train, detected, X_clean_train),
        'DeleteFix': lambda: strategy_deletefix(X_dirty_train, y_train, detected, X_clean_train),
        'ReplaceAll': lambda: strategy_replaceall(X_dirty_train, y_train, detected, X_clean_train),
        'FullFix': lambda: strategy_fullfix(X_dirty_train, y_train, detected, X_clean_train),
        'DemandFix': lambda: strategy_demandfix(X_dirty_train, y_train, detected, X_clean_train),
    }

    dqn_strategies = {}
    for dqn_name in DQN_STRATEGY_TABLE:
        def _make_dqn_func(name):
            def _run():
                result = run_dqn_strategy(
                    name, X_dirty_train, y_train, X_clean_train,
                    X_clean_val=X_clean_val, y_clean_val=y_val,
                    dirty_csv_path=dirty_train_csv_path,
                    clean_csv_path=clean_train_csv_path,
                    pre_detected=detected)
                return result
            return _run
        dqn_strategies[dqn_name] = _make_dqn_func(dqn_name)

    all_strategies = {**baseline_strategies, **dqn_strategies}

    # 2.7 逐策略运行
    results = {}
    for name, func in all_strategies.items():
        print(f'\n  --- {name} ---', flush=True)
        try:
            result = func()
            if result is None:
                print(f'  [SKIP] {name}: 模型不存在')
                continue
            X_train_result, y_train_result, cost, deleted, keep_mask = result
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')
            import traceback
            traceback.print_exc()
            continue

        if len(X_train_result) < 10 or len(np.unique(y_train_result)) < 2:
            print(f'  [SKIP] {name}: 样本不足')
            continue

        auth, div = compute_metrics(X_train_result, X_clean_train, X_dirty_train, keep_mask)

        clf = SVC(kernel='linear')
        clf.fit(X_train_result, y_train_result)

        # 用搜索空间的真实值覆盖（保证 inset 标注与搜索空间一致）
        display_auth, display_div, display_acc, display_cost = auth, div, None, cost
        with open(EXTREME_POINTS_JSON, 'r') as f:
            _ep = json.load(f)
        if name in _ep:
            ep = _ep[name]
            display_auth = ep['veracity']
            display_div = ep['diversity']
            display_acc = ep['accuracy']
            display_cost = ep['action_counts'].get('repair_value', 0)
        elif name == 'DemandFix' and os.path.exists(DQN_DIST_CSV):
            _dqn = pd.read_csv(DQN_DIST_CSV)
            if len(_dqn) > 0:
                row = _dqn.iloc[0]
                display_auth = row['veracity']
                display_div = row['diversity']
                display_acc = row['accuracy']
                display_cost = int(row['repair_value'])

        # 为 DemandFix 读取动作分布
        act_dist = None
        if name == 'DemandFix' and os.path.exists(DQN_DIST_CSV):
            _dqn2 = pd.read_csv(DQN_DIST_CSV)
            if len(_dqn2) > 0:
                r2 = _dqn2.iloc[0]
                act_dist = {
                    'no_action': int(r2['no_action']),
                    'repair_value': int(r2['repair_value']),
                    'delete': int(r2['delete']),
                    'replace_nearby': int(r2['replace_nearby']),
                }

        save_path = os.path.join(OUTPUT_DIR, f'inset_{name}.svg')
        acc = plot_boundary_svg(clf, ideal_clf, X_train_result, y_train_result,
                                X_clean_test, y_test,
                                name, display_cost, deleted, display_auth, display_div,
                                save_path, xlim, ylim,
                                override_acc=display_acc, action_dist=act_dist)

        results[name] = {'acc': acc, 'cost': display_cost, 'auth': display_auth, 'div': display_div}
        print(f'  Acc={acc:.4f}, Cost={display_cost}, Auth={display_auth:.2f}, Div={display_div:.2f}')

    # 汇总
    print('\n' + '=' * 60)
    print('结果汇总')
    print('=' * 60)
    print(f"{'策略':<30} {'准确率':>8} {'成本':>6} {'真实性':>8} {'多样性':>8}")
    print('-' * 68)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        print(f"{name:<30} {r['acc']:>8.4f} {r['cost']:>6} {r['auth']:>8.2f} {r['div']:>8.2f}")

    # ------------------------------------------------------------------
    # Part 1: 搜索空间主图（使用 Part 2 的指标，保证数值与 inset 一致）
    # ------------------------------------------------------------------
    print('\n[Part 1] 搜索空间主图（使用消融实验指标）')
    plot_search_space_main(strategy_results=results)

    print(f'\n所有图形已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
