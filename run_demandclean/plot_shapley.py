#!/usr/bin/env python3
"""
DemandClean Shapley 归因分析可视化

生成学术风格的归因分析图表：
1. 动作重要性热力图
2. 错误类型重要性热力图
3. v3 vs v6 动作对比柱状图
4. 归一化贡献堆叠图

Usage:
    python plot_shapley.py --versions v3,v6 --output ./figures/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

# 学术风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 颜色方案（学术友好）
COLORS = {
    'v3': '#2E86AB',  # 蓝色
    'v6': '#E94F37',  # 红色
    'actions': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261'],
    'errors': ['#E76F51', '#F4A261', '#E9C46A', '#2A9D8F'],
}

DATASETS = ['adult', 'beers', 'bike', 'breast_cancer', 'har', 'mercedes', 'nasa', 'smartfactory', 'soilmoisture']
VERSIONS = {'v3': 'v3_oracle_plain_single', 'v6': 'v6_auto_dueling_two'}
ACTIONS = ['no_action', 'repair_value', 'delete', 'replace_nearby']
ERROR_TYPES = ['missing', 'semantic', 'syntactic', 'label_noise']

ACTION_LABELS = {
    'no_action': 'No Action',
    'repair_value': 'Repair Value',
    'delete': 'Delete',
    'replace_nearby': 'Replace Nearby'
}
ERROR_LABELS = {
    'missing': 'Missing',
    'semantic': 'Semantic',
    'syntactic': 'Syntactic',
    'label_noise': 'Label Noise'
}
DATASET_LABELS = {
    'adult': 'Adult', 'beers': 'Beers', 'bike': 'Bike',
    'breast_cancer': 'Breast Cancer', 'har': 'HAR',
    'mercedes': 'Mercedes', 'nasa': 'NASA',
    'smartfactory': 'SmartFactory', 'soilmoisture': 'SoilMoisture'
}


def load_shapley_data(base_dir: str) -> dict:
    """加载所有 Shapley 数据"""
    data = {}
    for ver_short, ver_full in VERSIONS.items():
        data[ver_short] = {}
        for ds in DATASETS:
            path = os.path.join(base_dir, ds, ver_full, 'evaluation/shapley/shapley_results.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data[ver_short][ds] = json.load(f)
    return data


def normalize_values(values: np.ndarray) -> np.ndarray:
    """归一化到 [-1, 1] 范围，保持符号"""
    max_abs = np.max(np.abs(values))
    if max_abs > 0:
        return values / max_abs
    return values


def plot_action_heatmap(data: dict, output_dir: str):
    """图1: 动作重要性热力图（v3 和 v6 并排）"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for idx, (ver, ax) in enumerate(zip(['v3', 'v6'], axes)):
        # 构建矩阵
        matrix = np.zeros((len(DATASETS), len(ACTIONS)))
        for i, ds in enumerate(DATASETS):
            if ds in data[ver]:
                action_sh = data[ver][ds].get('action_shapley', {})
                for j, act in enumerate(ACTIONS):
                    matrix[i, j] = action_sh.get(act, 0)

        # 归一化（按行）
        for i in range(len(DATASETS)):
            row_max = np.max(np.abs(matrix[i, :]))
            if row_max > 0:
                matrix[i, :] = matrix[i, :] / row_max

        # 绘制热力图
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', norm=norm)

        # 标签
        ax.set_xticks(range(len(ACTIONS)))
        ax.set_xticklabels([ACTION_LABELS[a] for a in ACTIONS], rotation=45, ha='right')
        if idx == 0:
            ax.set_yticks(range(len(DATASETS)))
            ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS])
        ax.set_title(f'{ver.upper()}', fontweight='bold')

        # 添加数值标注
        for i in range(len(DATASETS)):
            for j in range(len(ACTIONS)):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=7, color=color)

    # 颜色条
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Normalized Shapley Value', fontsize=10)

    fig.suptitle('Action Importance (Shapley Values)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shapley_action_heatmap.pdf'))
    plt.savefig(os.path.join(output_dir, 'shapley_action_heatmap.png'))
    plt.close()
    print('  生成: shapley_action_heatmap.pdf/png')


def plot_error_type_heatmap(data: dict, output_dir: str):
    """图2: 错误类型重要性热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for idx, (ver, ax) in enumerate(zip(['v3', 'v6'], axes)):
        matrix = np.zeros((len(DATASETS), len(ERROR_TYPES)))
        for i, ds in enumerate(DATASETS):
            if ds in data[ver]:
                err_sh = data[ver][ds].get('error_type_shapley', {})
                for j, err in enumerate(ERROR_TYPES):
                    matrix[i, j] = err_sh.get(err, 0)

        # 归一化
        for i in range(len(DATASETS)):
            row_max = np.max(np.abs(matrix[i, :]))
            if row_max > 0:
                matrix[i, :] = matrix[i, :] / row_max

        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(matrix, cmap='PiYG', aspect='auto', norm=norm)

        ax.set_xticks(range(len(ERROR_TYPES)))
        ax.set_xticklabels([ERROR_LABELS[e] for e in ERROR_TYPES], rotation=45, ha='right')
        if idx == 0:
            ax.set_yticks(range(len(DATASETS)))
            ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS])
        ax.set_title(f'{ver.upper()}', fontweight='bold')

        for i in range(len(DATASETS)):
            for j in range(len(ERROR_TYPES)):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Normalized Shapley Value', fontsize=10)

    fig.suptitle('Error Type Importance (Shapley Values)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shapley_error_type_heatmap.pdf'))
    plt.savefig(os.path.join(output_dir, 'shapley_error_type_heatmap.png'))
    plt.close()
    print('  生成: shapley_error_type_heatmap.pdf/png')


def plot_action_comparison_bar(data: dict, output_dir: str):
    """图3: v3 vs v6 动作选择对比（分组柱状图）"""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(DATASETS))
    width = 0.35

    # 计算每个数据集的主导动作
    v3_dominant = []
    v6_dominant = []

    for ds in DATASETS:
        # v3
        if ds in data['v3']:
            action_sh = data['v3'][ds].get('action_shapley', {})
            if action_sh and any(action_sh.values()):
                best = max(action_sh, key=lambda k: abs(action_sh.get(k, 0)))
                v3_dominant.append(ACTIONS.index(best))
            else:
                v3_dominant.append(-1)
        else:
            v3_dominant.append(-1)

        # v6
        if ds in data['v6']:
            action_sh = data['v6'][ds].get('action_shapley', {})
            if action_sh and any(action_sh.values()):
                best = max(action_sh, key=lambda k: abs(action_sh.get(k, 0)))
                v6_dominant.append(ACTIONS.index(best))
            else:
                v6_dominant.append(-1)
        else:
            v6_dominant.append(-1)

    # 绘制
    colors_v3 = [COLORS['actions'][i] if i >= 0 else '#cccccc' for i in v3_dominant]
    colors_v6 = [COLORS['actions'][i] if i >= 0 else '#cccccc' for i in v6_dominant]

    bars1 = ax.bar(x - width/2, [1]*len(DATASETS), width, color=colors_v3,
                   edgecolor='black', linewidth=0.5, label='V3')
    bars2 = ax.bar(x + width/2, [1]*len(DATASETS), width, color=colors_v6,
                   edgecolor='black', linewidth=0.5, label='V6')

    # 添加动作名称标注
    for i, (v3_idx, v6_idx) in enumerate(zip(v3_dominant, v6_dominant)):
        if v3_idx >= 0:
            ax.text(i - width/2, 0.5, ACTION_LABELS[ACTIONS[v3_idx]].replace(' ', '\n'),
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        if v6_idx >= 0:
            ax.text(i + width/2, 0.5, ACTION_LABELS[ACTIONS[v6_idx]].replace(' ', '\n'),
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=45, ha='right')
    ax.set_ylabel('Dominant Action')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])

    # 图例
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=COLORS['actions'][i],
                                      edgecolor='black', label=ACTION_LABELS[ACTIONS[i]])
                       for i in range(len(ACTIONS))]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, framealpha=0.9)

    ax.set_title('Dominant Action Comparison: V3 vs V6', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shapley_action_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'shapley_action_comparison.png'))
    plt.close()
    print('  生成: shapley_action_comparison.pdf/png')


def plot_stacked_contribution(data: dict, output_dir: str):
    """图4: 动作贡献堆叠图（绝对值占比）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (ver, ax) in enumerate(zip(['v3', 'v6'], axes)):
        # 计算绝对值占比
        contributions = {act: [] for act in ACTIONS}

        for ds in DATASETS:
            if ds in data[ver]:
                action_sh = data[ver][ds].get('action_shapley', {})
                total = sum(abs(action_sh.get(a, 0)) for a in ACTIONS)
                for act in ACTIONS:
                    val = abs(action_sh.get(act, 0))
                    contributions[act].append(val / total * 100 if total > 0 else 0)
            else:
                for act in ACTIONS:
                    contributions[act].append(0)

        # 堆叠柱状图
        x = np.arange(len(DATASETS))
        bottom = np.zeros(len(DATASETS))

        for i, act in enumerate(ACTIONS):
            ax.bar(x, contributions[act], bottom=bottom, label=ACTION_LABELS[act],
                   color=COLORS['actions'][i], edgecolor='white', linewidth=0.5)
            bottom += contributions[act]

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=45, ha='right')
        ax.set_ylabel('Contribution (%)')
        ax.set_ylim(0, 100)
        ax.set_title(f'{ver.upper()}', fontweight='bold')

        if idx == 1:
            ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('Action Contribution Distribution', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shapley_action_stacked.pdf'))
    plt.savefig(os.path.join(output_dir, 'shapley_action_stacked.png'))
    plt.close()
    print('  生成: shapley_action_stacked.pdf/png')


def plot_radar_chart(data: dict, output_dir: str, dataset: str = 'beers'):
    """图5: 单数据集雷达图（动作 + 错误类型）"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(projection='polar'))

    categories = ACTIONS + ERROR_TYPES
    labels = [ACTION_LABELS.get(c, ERROR_LABELS.get(c, c)) for c in categories]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for idx, (ver, ax) in enumerate(zip(['v3', 'v6'], axes)):
        if dataset not in data[ver]:
            continue

        action_sh = data[ver][dataset].get('action_shapley', {})
        error_sh = data[ver][dataset].get('error_type_shapley', {})

        values = []
        for cat in categories:
            if cat in ACTIONS:
                values.append(action_sh.get(cat, 0))
            else:
                values.append(error_sh.get(cat, 0))

        # 归一化
        max_abs = max(abs(v) for v in values) if values else 1
        values = [v / max_abs for v in values]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[ver], label=ver.upper())
        ax.fill(angles, values, alpha=0.25, color=COLORS[ver])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'{ver.upper()} - {DATASET_LABELS[dataset]}', fontweight='bold', pad=20)

    fig.suptitle(f'Shapley Attribution Radar: {DATASET_LABELS[dataset]}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shapley_radar_{dataset}.pdf'))
    plt.savefig(os.path.join(output_dir, f'shapley_radar_{dataset}.png'))
    plt.close()
    print(f'  生成: shapley_radar_{dataset}.pdf/png')


def main():
    parser = argparse.ArgumentParser(description='DemandClean Shapley 归因分析可视化')
    parser.add_argument('--output', type=str, default='./figures/shapley/',
                        help='输出目录')
    parser.add_argument('--radar_dataset', type=str, default='beers',
                        help='雷达图使用的数据集')
    args = parser.parse_args()

    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'results', 'demandclean')

    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('DemandClean Shapley 归因分析可视化')
    print(f'  输出目录: {output_dir}')
    print()

    # 加载数据
    data = load_shapley_data(base_dir)

    # 生成图表
    print('生成图表...')
    plot_action_heatmap(data, output_dir)
    plot_error_type_heatmap(data, output_dir)
    plot_action_comparison_bar(data, output_dir)
    plot_stacked_contribution(data, output_dir)
    plot_radar_chart(data, output_dir, args.radar_dataset)

    print(f'\n完成！图表已保存到: {output_dir}')


if __name__ == '__main__':
    main()
