"""
集成测试
========

使用真实数据测试完整流程。
"""

import sys
import os
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# import pytest  # Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from demandclean import DemandClean


# 数据集路径
DATASETS_PATH = os.path.join(_PROJECT_ROOT, 'experiment', 'ablation_beers', 'datasets')


def load_beers_data():
    """加载 beers 数据集（分类任务）"""
    clean_path = os.path.join(DATASETS_PATH, 'beers/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'beers/dirty.csv')

    if not os.path.exists(clean_path):
        return None, None, None

    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)

    # 选择数值列
    feature_cols = ['abv', 'ibu']
    target_col = 'style'

    # 处理百分比字符串（如 '0.09%' -> 0.09）
    def convert_to_float(x):
        if isinstance(x, str):
            if x.endswith('%'):
                try:
                    return float(x[:-1])
                except ValueError:
                    return np.nan
            try:
                return float(x)
            except ValueError:
                return np.nan
        return x

    for col in feature_cols:
        clean_df[col] = clean_df[col].apply(convert_to_float)
        dirty_df[col] = dirty_df[col].apply(convert_to_float)

    # 提取特征和标签
    X_clean = clean_df[feature_cols].values.astype(np.float64)
    X_dirty = dirty_df[feature_cols].values.astype(np.float64)

    # 处理标签（转换为数值）
    le = LabelEncoder()
    y = le.fit_transform(clean_df[target_col].values)

    return X_clean, X_dirty, y


def load_synthetic_data():
    """生成合成数据用于测试"""
    np.random.seed(42)

    # 生成分类数据
    n_samples = 200
    n_features = 4

    # 生成两类数据
    X_clean = np.vstack([
        np.random.randn(n_samples // 2, n_features) + np.array([1, 1, 0, 0]),
        np.random.randn(n_samples // 2, n_features) + np.array([-1, -1, 0, 0])
    ])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # 注入错误
    X_dirty = X_clean.copy()
    # 缺失值
    for _ in range(10):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] = np.nan
    # 句法错误（添加噪声）
    for _ in range(20):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] += np.random.randn() * 3

    return X_clean, X_dirty, y


def test_integration_synthetic_single_stage():
    """合成数据单阶段测试"""
    print("\n" + "=" * 50)
    print("集成测试: 合成数据 - 单阶段")
    print("=" * 50)

    X_clean, X_dirty, y = load_synthetic_data()
    print(f"数据: {X_dirty.shape}")

    # 只用前100行快速测试
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )

    # 训练
    dc.fit(X_dirty, y, verbose=False)
    assert dc.is_fitted
    print("✓ 训练完成")

    # 单阶段清洗
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ 清洗完成: {len(X_result)} 行")
    print(f"  动作: {stats['action_counts']}")
    print(f"  真值成本: {stats['truth_cost']}")

    return True


def test_integration_synthetic_two_stage():
    """合成数据两阶段测试"""
    print("\n" + "=" * 50)
    print("集成测试: 合成数据 - 两阶段")
    print("=" * 50)

    X_clean, X_dirty, y = load_synthetic_data()

    # 只用前100行快速测试
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='two_stage',
        n_episodes=5
    )

    # 训练
    dc.fit(X_dirty, y, verbose=False)
    print("✓ 训练完成")

    # 第一阶段: 计划
    plan = dc.plan(X_dirty, y, verbose=False)
    print(f"✓ 计划生成: {len(plan)} 个需要真值")

    # 获取需要真值的位置
    positions = dc.get_plan_positions()

    # 从 X_clean 提取真值
    true_values = {}
    for idx, col in positions:
        true_values[(idx, col)] = X_clean[idx, col]

    # 第二阶段: 执行
    X_result, y_result, keep_mask = dc.execute(X_dirty, true_values, verbose=False)
    print(f"✓ 执行完成: {len(X_result)} 行")

    return True


def test_integration_beers_data():
    """真实数据 (beers) 测试"""
    print("\n" + "=" * 50)
    print("集成测试: Beers 数据")
    print("=" * 50)

    X_clean, X_dirty, y = load_beers_data()
    if X_clean is None:
        print("⚠ 无法加载 beers 数据，跳过")
        return True

    print(f"数据: clean={X_clean.shape}, dirty={X_dirty.shape}")

    # 只用前100行快速测试
    n = min(100, len(X_clean))
    X_clean, X_dirty, y = X_clean[:n], X_dirty[:n], y[:n]

    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=5
    )

    # 训练
    dc.fit(X_dirty, y, verbose=False)
    print("✓ 训练完成")

    # 检测错误
    detected = dc.detect_errors(X_dirty, X_clean, verbose=False)
    print(f"✓ 检测到: missing={len(detected['missing'])}, "
          f"semantic={len(detected['semantic'])}, syntactic={len(detected['syntactic'])}")

    # 清洗
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ 清洗完成: {len(X_result)} 行")
    print(f"  动作: {stats['action_counts']}")

    return True


def test_integration_regression():
    """回归任务测试"""
    print("\n" + "=" * 50)
    print("集成测试: 回归任务")
    print("=" * 50)

    np.random.seed(42)
    n_samples = 100
    n_features = 4

    # 生成回归数据
    X_clean = np.random.randn(n_samples, n_features)
    y = X_clean[:, 0] * 2 + X_clean[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1

    # 注入错误
    X_dirty = X_clean.copy()
    for _ in range(10):
        i, j = np.random.randint(0, n_samples), np.random.randint(0, n_features)
        X_dirty[i, j] = np.nan

    dc = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='single_stage',
        n_episodes=5
    )

    # 训练
    dc.fit(X_dirty, y, verbose=False)
    print("✓ 训练完成")

    # 清洗
    X_result, y_result, stats = dc.clean(X_dirty, y, X_clean, verbose=False)
    print(f"✓ 清洗完成: {len(X_result)} 行")
    print(f"  动作: {stats['action_counts']}")

    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean 集成测试")
    print("=" * 60)

    tests = [
        ("合成数据 - 单阶段", test_integration_synthetic_single_stage),
        ("合成数据 - 两阶段", test_integration_synthetic_two_stage),
        ("Beers 数据", test_integration_beers_data),
        ("回归任务", test_integration_regression),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"集成测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
