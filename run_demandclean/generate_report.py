#!/usr/bin/env python3
"""
DemandClean 实验结果汇总报告生成器

从 report.json 文件中提取实验结果，生成结构化 Markdown 文档。
支持两种调用方式:
  1. 独立调用:  python generate_report.py --datasets beers,adult --versions v3,v6
  2. run.sh 透传: python generate_report.py --dataset beers --versions v3,v6
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ============================================================================
# 版本短名到全名的映射
# ============================================================================
SHORT_NAME_MAP = {
    'v1': 'v1_oracle_dueling_single',
    'v2': 'v2_oracle_dueling_two',
    'v3': 'v3_oracle_plain_single',
    'v4': 'v4_oracle_plain_two',
    'v5': 'v5_auto_dueling_single',
    'v6': 'v6_auto_dueling_two',
    'v7': 'v7_auto_plain_single',
    'v8': 'v8_auto_plain_two',
}

ALL_DATASETS = [
    'beers', 'adult', 'bike', 'breast_cancer', 'har',
    'mercedes', 'nasa', 'smartfactory', 'soilmoisture',
]

# 默认启用的版本（与 run_demandclean_base.py --versions 默认值一致）
DEFAULT_VERSIONS = ['v5']

# ============================================================================
# 工具函数
# ============================================================================


def resolve_version(name: str) -> str:
    """将版本名（短名或全名）解析为全名"""
    name = name.strip()
    if name in SHORT_NAME_MAP:
        return SHORT_NAME_MAP[name]
    # 已经是全名
    if name in SHORT_NAME_MAP.values():
        return name
    return name


def version_short(full_name: str) -> str:
    """全名转短名，用于显示"""
    for short, full in SHORT_NAME_MAP.items():
        if full == full_name:
            return short
    return full_name


def fmt(val, digits=4) -> str:
    """安全格式化数字"""
    if val is None:
        return '—'
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int,)):
        return f'{val:,}'
    if isinstance(val, float):
        if abs(val) >= 100:
            return f'{val:.1f}'
        if abs(val) >= 10:
            return f'{val:.2f}'
        return f'{val:.{digits}f}'
    return str(val)


def safe_get(d: dict, *keys, default=None):
    """安全的多级字典取值"""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, None)
        if current is None:
            return default
    return current


# ============================================================================
# 数据加载
# ============================================================================


def find_project_root() -> str:
    """定位项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_reports(datasets: list, versions: list, results_dir: str) -> dict:
    """
    加载所有 (dataset, version) 组合的 report.json

    Returns:
        {(dataset, version_full): report_dict, ...}
    """
    reports = {}
    for ds in datasets:
        for ver in versions:
            ver_full = resolve_version(ver)
            report_path = os.path.join(
                results_dir, ds, ver_full, 'report', f'{ver_full}_report.json'
            )
            if not os.path.exists(report_path):
                print(f'  [跳过] {ds}/{ver_full}: 报告文件不存在')
                continue
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                reports[(ds, ver_full)] = report
            except Exception as e:
                print(f'  [警告] {ds}/{ver_full}: 读取报告失败 - {e}')
    return reports


# ============================================================================
# 各板块生成函数
# ============================================================================


def section_header(datasets, versions, reports) -> str:
    """报告头部"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ver_labels = ', '.join(versions)
    ds_labels = ', '.join(datasets)

    # 从报告中提取训练轮数
    episodes_set = set()
    for r in reports.values():
        ep = r.get('n_episodes')
        if ep:
            episodes_set.add(ep)
    ep_str = ', '.join(str(e) for e in sorted(episodes_set)) if episodes_set else '—'

    lines = [
        '# DemandClean 实验报告\n',
        f'> 生成时间: {now}  ',
        f'> 数据集: {ds_labels} | 版本: {ver_labels} | 训练轮数: {ep_str}\n',
    ]
    return '\n'.join(lines) + '\n'


def section_overview(datasets, reports) -> str:
    """一、数据集概览"""
    lines = [
        '## 一、数据集概览\n',
        '| 数据集 | 行数 | 列数 | 任务类型 | 错误数(GT) | 错误率 |',
        '|--------|------|------|---------|-----------|--------|',
    ]

    seen = set()
    for (ds, ver), r in reports.items():
        if ds in seen:
            continue
        seen.add(ds)
        shape = r.get('data_shape', [0, 0])
        rows, cols = shape[0], shape[1] if len(shape) > 1 else 0
        task = r.get('task_type', '—')
        gt_total = safe_get(r, 'detector_accuracy', 'ground_truth_total', default=0)
        total_cells = rows * cols
        err_rate = f'{gt_total / total_cells * 100:.1f}%' if total_cells > 0 else '—'
        lines.append(
            f'| {ds} | {rows:,} | {cols} | {task} | {gt_total:,} | {err_rate} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_runtime(reports) -> str:
    """二、运行时间"""
    lines = [
        '## 二、运行时间\n',
        '| 数据集 | 版本 | 训练轮数 | 耗时 |',
        '|--------|------|---------|------|',
    ]
    for (ds, ver), r in reports.items():
        ep = r.get('n_episodes', '—')
        elapsed = r.get('elapsed_time', 0)
        if elapsed >= 3600:
            t_str = f'{elapsed:.0f}s ({elapsed / 3600:.1f}h)'
        elif elapsed >= 60:
            t_str = f'{elapsed:.0f}s ({elapsed / 60:.1f}min)'
        else:
            t_str = f'{elapsed:.1f}s'
        lines.append(f'| {ds} | {version_short(ver)} | {ep} | {t_str} |')

    return '\n'.join(lines) + '\n\n'


def section_detector(reports) -> str:
    """三、检测器性能（仅 auto 版本）"""
    # 筛选 auto 版本
    auto_reports = {k: v for k, v in reports.items() if v.get('detector_mode') == 'auto'}
    if not auto_reports:
        return ''

    lines = [
        '## 三、检测器性能 (Auto 版本)\n',
        '| 数据集 | 版本 | Overall P/R/F1 | Missing F1 | Syntactic F1 | Semantic F1 |',
        '|--------|------|---------------|-----------|-------------|------------|',
    ]
    for (ds, ver), r in auto_reports.items():
        da = r.get('detector_accuracy', {})
        ov = da.get('overall', {})
        mi = da.get('missing', {})
        sy = da.get('syntactic', {})
        se = da.get('semantic', {})

        def prf(d):
            p, rc, f = d.get('precision', 0), d.get('recall', 0), d.get('f1', 0)
            return f'{p:.3f}/{rc:.3f}/{f:.3f}'

        lines.append(
            f'| {ds} | {version_short(ver)} | {prf(ov)} | {fmt(mi.get("f1", 0))} '
            f'| {fmt(sy.get("f1", 0))} | {fmt(se.get("f1", 0))} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_ml_performance(reports) -> str:
    """四、下游 ML 性能对比（核心表）"""
    lines = ['## 四、下游 ML 性能对比\n']

    # 按任务类型分组
    classification = [(k, r) for k, r in reports.items() if r.get('task_type') == 'classification']
    regression = [(k, r) for k, r in reports.items() if r.get('task_type') == 'regression']
    clustering = [(k, r) for k, r in reports.items() if r.get('task_type') == 'clustering']

    if classification:
        lines.extend([
            '### 分类任务 (Accuracy ↑)\n',
            '| 数据集 | 版本 | 模型 | NoFix | DemandClean | FullFix | DeleteAll |',
            '|--------|------|------|-------|-------------|---------|-----------|',
        ])
        for (ds, ver), r in classification:
            bl = r.get('baseline_results', {})
            model = r.get('model_type', 'rf')
            # 寻找准确率字段
            def get_acc(strategy):
                s = bl.get(strategy, {})
                for key in ['rf_accuracy', 'xgboost_accuracy', 'accuracy']:
                    if key in s:
                        return s[key]
                # 尝试 auth/div 格式（新版）
                for key in s:
                    if 'accuracy' in key.lower():
                        return s[key]
                return None

            nf = get_acc('NoFix')
            dc = get_acc('DemandClean')
            ff = get_acc('FullFix')
            da = get_acc('DeleteAll')
            lines.append(
                f'| {ds} | {version_short(ver)} | {model} | {fmt(nf)} | {fmt(dc)} '
                f'| {fmt(ff)} | {fmt(da)} |'
            )
        lines.append('')

    if regression:
        lines.extend([
            '### 回归任务 (R² ↑)\n',
            '| 数据集 | 版本 | 模型 | NoFix | DemandClean | FullFix |',
            '|--------|------|------|-------|-------------|---------|',
        ])
        for (ds, ver), r in regression:
            bl = r.get('baseline_results', {})
            model = r.get('model_type', 'rf')

            def get_r2(strategy):
                s = bl.get(strategy, {})
                for key in ['rf_r2', 'ridge_r2', 'r2']:
                    if key in s:
                        return s[key]
                return None

            nf = get_r2('NoFix')
            dc = get_r2('DemandClean')
            ff = get_r2('FullFix')
            lines.append(
                f'| {ds} | {version_short(ver)} | {model} | {fmt(nf)} | {fmt(dc)} | {fmt(ff)} |'
            )
        lines.append('')

    if clustering:
        lines.extend([
            '### 聚类任务 (Silhouette / ARI ↑)\n',
            '| 数据集 | 版本 | NoFix Sil | NoFix ARI | DC Sil | DC ARI | FullFix Sil | FullFix ARI |',
            '|--------|------|-----------|-----------|--------|--------|-------------|-------------|',
        ])
        for (ds, ver), r in clustering:
            bl = r.get('baseline_results', {})

            def get_cluster(strategy):
                s = bl.get(strategy, {})
                sil = s.get('silhouette', s.get('kmeans_silhouette', None))
                ari = s.get('ari', s.get('kmeans_ari', None))
                return sil, ari

            nf_s, nf_a = get_cluster('NoFix')
            dc_s, dc_a = get_cluster('DemandClean')
            ff_s, ff_a = get_cluster('FullFix')
            lines.append(
                f'| {ds} | {version_short(ver)} | {fmt(nf_s)} | {fmt(nf_a)} '
                f'| {fmt(dc_s)} | {fmt(dc_a)} | {fmt(ff_s)} | {fmt(ff_a)} |'
            )
        lines.append('')

    return '\n'.join(lines) + '\n'


def section_traditional(reports) -> str:
    """五、传统清洗指标"""
    lines = [
        '## 五、传统清洗指标\n',
        '| 数据集 | 版本 | Precision | Recall | F1 | EDR | Hybrid Dist |',
        '|--------|------|-----------|--------|-----|-----|-------------|',
    ]
    for (ds, ver), r in reports.items():
        gs = r.get('getscoreml_results', {})
        p = gs.get('precision', None)
        rc = gs.get('recall', None)
        f1 = gs.get('f1_score', None)
        edr = gs.get('edr', None)
        hd = gs.get('hybrid_distance', None)
        lines.append(
            f'| {ds} | {version_short(ver)} | {fmt(p)} | {fmt(rc)} '
            f'| {fmt(f1)} | {fmt(edr)} | {fmt(hd)} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_clean4ml(reports) -> str:
    """六、Clean4ML 多模型测评"""
    lines = ['## 六、Clean4ML 多模型测评\n']

    classification = [(k, r) for k, r in reports.items() if r.get('task_type') == 'classification']
    regression = [(k, r) for k, r in reports.items() if r.get('task_type') == 'regression']
    clustering = [(k, r) for k, r in reports.items() if r.get('task_type') == 'clustering']

    if classification:
        lines.extend([
            '### 分类 (Accuracy)\n',
            '| 数据集 | 版本 | RF | LR | SVM | KNN | DT | GB |',
            '|--------|------|----|-----|-----|-----|----|----|',
        ])
        for (ds, ver), r in classification:
            gs = r.get('getscoreml_results', {})
            rf = gs.get('ml_rf_accuracy', None)
            lr = gs.get('ml_lr_accuracy', None)
            svm = gs.get('ml_svm_accuracy', None)
            knn = gs.get('ml_knn_accuracy', None)
            dt = gs.get('ml_dt_accuracy', None)
            gb = gs.get('ml_gb_accuracy', None)
            lines.append(
                f'| {ds} | {version_short(ver)} | {fmt(rf)} | {fmt(lr)} '
                f'| {fmt(svm)} | {fmt(knn)} | {fmt(dt)} | {fmt(gb)} |'
            )
        lines.append('')

    if regression:
        lines.extend([
            '### 回归 (R²)\n',
            '| 数据集 | 版本 | RF | Ridge | Lasso | KNN | GB |',
            '|--------|------|-------|-------|-------|------|-----|',
        ])
        for (ds, ver), r in regression:
            gs = r.get('getscoreml_results', {})
            # 回归指标可能是 r2 或 mse, 优先取 r2
            rf = gs.get('ml_rf_r2', gs.get('ml_rf_accuracy', None))
            ridge = gs.get('ml_lr_r2', gs.get('ml_lr_accuracy', None))
            lasso = gs.get('ml_lasso_r2', None)
            knn = gs.get('ml_knn_r2', gs.get('ml_knn_accuracy', None))
            gb = gs.get('ml_gb_r2', gs.get('ml_gb_accuracy', None))
            lines.append(
                f'| {ds} | {version_short(ver)} | {fmt(rf)} | {fmt(ridge)} '
                f'| {fmt(lasso)} | {fmt(knn)} | {fmt(gb)} |'
            )
        lines.append('')

    if clustering:
        lines.extend([
            '### 聚类\n',
            '| 数据集 | 版本 | KMeans Sil | KMeans ARI | Aggl. ARI | Spectral ARI |',
            '|--------|------|-----------|-----------|-----------|-------------|',
        ])
        for (ds, ver), r in clustering:
            gs = r.get('getscoreml_results', {})
            ks = gs.get('ml_kmeans_silhouette', None)
            ka = gs.get('ml_kmeans_ari', None)
            aa = gs.get('ml_agglomerative_ari', None)
            sa = gs.get('ml_spectral_ari', None)
            lines.append(
                f'| {ds} | {version_short(ver)} | {fmt(ks)} | {fmt(ka)} '
                f'| {fmt(aa)} | {fmt(sa)} |'
            )
        lines.append('')

    return '\n'.join(lines) + '\n'


def section_snoopy(reports) -> str:
    """七、Snoopy 上界"""
    lines = [
        '## 七、Snoopy 上界\n',
        '| 数据集 | 版本 | UB_dirty | UB_cleaned | UB_clean | improvement |',
        '|--------|------|----------|------------|----------|-------------|',
    ]
    for (ds, ver), r in reports.items():
        gs = r.get('getscoreml_results', {})
        avail = gs.get('snoopy_snoopy_available', False)
        if not avail:
            lines.append(f'| {ds} | {version_short(ver)} | — | — | — | — |')
            continue
        dirty = gs.get('snoopy_upper_bound_dirty', None)
        cleaned = gs.get('snoopy_upper_bound_cleaned', None)
        clean = gs.get('snoopy_upper_bound_clean', None)
        imp = gs.get('snoopy_upper_bound_improvement', None)
        lines.append(
            f'| {ds} | {version_short(ver)} | {fmt(dirty)} | {fmt(cleaned)} '
            f'| {fmt(clean)} | {fmt(imp)} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_tolerance(reports) -> str:
    """八、模型容忍度"""
    lines = [
        '## 八、模型容忍度\n',
        '| 数据集 | 版本 | 模型 | 先验容忍度 | 后验容忍度 |',
        '|--------|------|------|----------|----------|',
    ]
    for (ds, ver), r in reports.items():
        gs = r.get('getscoreml_results', {})
        model = r.get('model_type', '—')
        prior = gs.get('tolerance_tolerance_prior', None)
        post = gs.get('tolerance_tolerance_post', None)
        lines.append(
            f'| {ds} | {version_short(ver)} | {model} | {fmt(prior)} | {fmt(post)} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_cost(reports) -> str:
    """九、Ground Truth 成本"""
    lines = [
        '## 九、Ground Truth 成本\n',
        '| 数据集 | 版本 | 真值使用 | RAHA成本 | 总成本 | 总cell数 | 成本率 |',
        '|--------|------|---------|---------|--------|---------|--------|',
    ]
    for (ds, ver), r in reports.items():
        # 优先从 ground_truth_cost_summary 取
        gcs = r.get('ground_truth_cost_summary', {})
        gs = r.get('getscoreml_results', {})

        gt_used = r.get('ground_truth_used', gs.get('ground_truth_cost', 0))
        raha_cost = gcs.get('raha_detection_cost', '—')
        total_cost = gcs.get('total_cost', gt_used)
        total_cells = gcs.get('total_data_cells', gs.get('total_cells', '—'))
        cost_ratio = gcs.get('cost_ratio', gs.get('cost_ratio', None))

        cr_str = f'{cost_ratio:.4f}' if isinstance(cost_ratio, (int, float)) else '—'
        lines.append(
            f'| {ds} | {version_short(ver)} | {fmt(gt_used)} | {fmt(raha_cost)} '
            f'| {fmt(total_cost)} | {fmt(total_cells)} | {cr_str} |'
        )

    return '\n'.join(lines) + '\n\n'


def section_shapley(reports) -> str:
    """十、Shapley 分析"""
    # 只生成有 shapley_results 的报告
    shapley_reports = {k: v for k, v in reports.items() if v.get('shapley_results')}
    if not shapley_reports:
        return ''

    lines = ['## 十、Shapley 分析\n']

    # 动作重要性
    lines.extend([
        '### 动作重要性\n',
        '| 数据集 | 版本 | 最重要 | repair_value | delete | replace_nearby | no_action |',
        '|--------|------|--------|-------------|--------|---------------|-----------|',
    ])
    for (ds, ver), r in shapley_reports.items():
        sh = r.get('shapley_results', {})
        action_sh = sh.get('action_shapley', {})
        if not action_sh:
            continue

        actions = {
            'repair_value': action_sh.get('repair_value', 0),
            'delete': action_sh.get('delete', 0),
            'replace_nearby': action_sh.get('replace_nearby', 0),
            'no_action': action_sh.get('no_action', 0),
        }
        best = max(actions, key=lambda k: abs(actions[k])) if any(actions.values()) else '—'
        lines.append(
            f'| {ds} | {version_short(ver)} | **{best}** '
            f'| {fmt(actions["repair_value"])} | {fmt(actions["delete"])} '
            f'| {fmt(actions["replace_nearby"])} | {fmt(actions["no_action"])} |'
        )
    lines.append('')

    # 错误类型重要性
    has_error_type = any(
        r.get('shapley_results', {}).get('error_type_shapley')
        for r in shapley_reports.values()
    )
    if has_error_type:
        lines.extend([
            '### 错误类型重要性\n',
            '| 数据集 | 版本 | 最重要 | syntactic | semantic | missing | label_noise |',
            '|--------|------|--------|----------|----------|---------|-------------|',
        ])
        for (ds, ver), r in shapley_reports.items():
            sh = r.get('shapley_results', {})
            err_sh = sh.get('error_type_shapley', {})
            if not err_sh:
                continue

            types = {
                'syntactic': err_sh.get('syntactic', 0),
                'semantic': err_sh.get('semantic', 0),
                'missing': err_sh.get('missing', 0),
                'label_noise': err_sh.get('label_noise', 0),
            }
            best = max(types, key=lambda k: abs(types[k])) if any(types.values()) else '—'
            lines.append(
                f'| {ds} | {version_short(ver)} | **{best}** '
                f'| {fmt(types["syntactic"])} | {fmt(types["semantic"])} '
                f'| {fmt(types["missing"])} | {fmt(types["label_noise"])} |'
            )
        lines.append('')

    return '\n'.join(lines) + '\n'


# ============================================================================
# 主入口
# ============================================================================


def generate_report(datasets: list, versions: list, output_path: str,
                    results_dir: str) -> str:
    """
    主函数: 加载报告 → 拼接各板块 → 写入 Markdown

    Returns: 生成的报告文件路径
    """
    print(f'加载报告: datasets={datasets}, versions={versions}')
    reports = load_reports(datasets, versions, results_dir)

    if not reports:
        print('[错误] 没有找到任何有效的报告文件')
        return ''

    print(f'  成功加载 {len(reports)} 个报告')

    # 拼接各板块
    md_parts = [
        section_header(datasets, versions, reports),
        section_overview(datasets, reports),
        section_runtime(reports),
        section_detector(reports),
        section_ml_performance(reports),
        section_traditional(reports),
        section_clean4ml(reports),
        section_snoopy(reports),
        section_tolerance(reports),
        section_cost(reports),
        section_shapley(reports),
    ]

    md = '\n'.join(part for part in md_parts if part)

    # 写入文件
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f'  报告已生成: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='DemandClean 实验结果汇总报告生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 独立使用
    python generate_report.py --datasets beers,adult --versions v3,v6
    python generate_report.py --datasets beers --versions v3,v6 --output my_report.md

    # run.sh 透传 (自动兼容)
    python generate_report.py --dataset beers --versions v3,v6
    python generate_report.py --all_datasets --versions v3,v6
        """,
    )

    # 独立模式参数
    parser.add_argument(
        '--datasets', type=str, default='',
        help='数据集列表，逗号分隔 (如 beers,adult)',
    )
    parser.add_argument(
        '--versions', type=str, default='',
        help='版本列表，逗号分隔 (如 v3,v6)',
    )
    parser.add_argument(
        '--output', type=str, default='',
        help='输出文件路径 (默认自动命名)',
    )

    # run.sh 透传兼容参数
    parser.add_argument('--dataset', type=str, default='beers', help=argparse.SUPPRESS)
    parser.add_argument('--all_datasets', action='store_true', help=argparse.SUPPRESS)

    # 忽略的透传参数（不影响报告生成）
    parser.add_argument('--n_episodes', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--verbose', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--missing_rate', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--semantic_rate', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--syntactic_rate', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--label_rate', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--resume', type=str, default='auto', help=argparse.SUPPRESS)
    parser.add_argument('--apply_raha_truth', type=str, default='true', help=argparse.SUPPRESS)
    parser.add_argument('--count_raha_cost', type=str, default='true', help=argparse.SUPPRESS)
    parser.add_argument('--visualize_only', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # --- 确定数据集列表 ---
    if args.datasets:
        # 独立模式: --datasets beers,adult
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    elif args.all_datasets:
        # 透传模式: --all_datasets
        datasets = ALL_DATASETS
    else:
        # 透传模式: --dataset beers
        datasets = [args.dataset]

    # --- 确定版本列表 ---
    if args.versions:
        versions = [v.strip() for v in args.versions.split(',') if v.strip()]
    else:
        versions = DEFAULT_VERSIONS

    # --- 确定输出路径 ---
    project_root = find_project_root()
    results_dir = os.path.join(project_root, 'results', 'demandclean')

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_root, output_path)
    else:
        # 自动命名: report_{timestamp}_{versions}_{datasets}.md
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ver_tag = '_'.join(versions)
        ds_tag = '_'.join(datasets) if len(datasets) <= 3 else f'{len(datasets)}ds'
        filename = f'report_{ts}_{ver_tag}_{ds_tag}.md'
        output_path = os.path.join(results_dir, filename)

    # --- 生成报告 ---
    result = generate_report(datasets, versions, output_path, results_dir)
    if not result:
        sys.exit(1)


if __name__ == '__main__':
    main()
