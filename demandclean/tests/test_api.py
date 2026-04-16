"""
DemandClean API 测试
====================
"""

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
import tempfile
import os
from demandclean import DemandClean
from demandclean.utils.model_io import ModelIO


def test_demandclean_init_classification():
    """测试分类任务初始化"""
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )

    assert dc.config.task_type.value == 'classification'
    assert dc.config.model_type.value == 'svm'
    assert dc.config.agent_type.value == 'single'
    assert not dc.is_fitted
    print("✓ 分类任务初始化测试通过")


def test_demandclean_init_regression():
    """测试回归任务初始化"""
    dc = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='two_stage',
        n_episodes=5
    )

    assert dc.config.task_type.value == 'regression'
    assert dc.config.model_type.value == 'ridge'
    assert dc.config.agent_type.value == 'two_stage'
    print("✓ 回归任务初始化测试通过")


def test_demandclean_fit():
    """测试训练功能"""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=3
    )

    dc.fit(X, y, verbose=False)
    assert dc.is_fitted
    assert dc.agent is not None
    print("✓ 训练功能测试通过")


def test_demandclean_fit_with_semantic_errors():
    """测试带语义错误训练"""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    semantic_errors = [(5, 0), (10, 1), (15, 2)]

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=3
    )

    dc.fit(X, y, semantic_errors=semantic_errors, verbose=False)
    assert dc.is_fitted
    print("✓ 带语义错误训练测试通过")


def test_demandclean_detect_errors():
    """测试错误检测"""
    np.random.seed(42)
    X_dirty = np.random.randn(50, 4)
    X_dirty[5, 0] = np.nan
    X_dirty[10, 1] = np.nan
    y = np.random.randint(0, 2, 50)

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X_dirty, y, verbose=False)

    detected = dc.detect_errors(X_dirty, verbose=False)
    assert 'missing' in detected
    assert 'semantic' in detected
    assert 'syntactic' in detected
    assert len(detected['missing']) >= 2  # 至少有我们注入的 2 个
    print(f"✓ 错误检测测试通过: {len(detected['missing'])} 个缺失值")


def test_demandclean_get_config():
    """测试获取配置"""
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=100,
        repair_lambda=0.05
    )

    config = dc.get_config()
    assert config.n_episodes == 100
    assert config.repair_lambda == 0.05
    print("✓ 获取配置测试通过")


def test_demandclean_get_training_history():
    """测试获取训练历史"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X, y, verbose=False)

    history = dc.get_training_history()
    assert 'episode' in history
    assert 'score' in history
    assert 'reward' in history
    assert len(history['episode']) == 3
    print("✓ 获取训练历史测试通过")


def test_demandclean_save_load():
    """测试模型保存和加载"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    # 训练
    dc = DemandClean(
        task_type='classification',
        model_type='svm',
        n_episodes=3
    )
    dc.fit(X, y, verbose=False)

    # 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pt')
        dc.save(model_path)
        assert ModelIO.agent_model_exists(model_path)

        # 加载
        dc2 = DemandClean(
            task_type='classification',
            model_type='svm'
        )
        dc2.load(model_path)
        assert dc2.is_fitted

    print("✓ 模型保存加载测试通过")


def test_demandclean_not_fitted_error():
    """测试未训练时调用错误"""
    dc = DemandClean()

    try:
        dc.clean(np.random.randn(10, 3), np.random.randint(0, 2, 10),
                 np.random.randn(10, 3), verbose=False)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "未训练" in str(e)
        print("✓ 未训练错误测试通过")


def test_demandclean_plan_not_fitted_error():
    """测试未训练时 plan 调用错误"""
    dc = DemandClean()

    try:
        dc.plan(np.random.randn(10, 3), np.random.randint(0, 2, 10), verbose=False)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "未训练" in str(e)
        print("✓ plan 未训练错误测试通过")


def test_demandclean_execute_no_plan_error():
    """测试未 plan 时 execute 错误"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)

    dc = DemandClean(n_episodes=3)
    dc.fit(X, y, verbose=False)

    try:
        dc.execute(X, {}, verbose=False)
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "plan" in str(e).lower()
        print("✓ execute 无 plan 错误测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("DemandClean API 测试")
    print("=" * 50 + "\n")

    test_demandclean_init_classification()
    test_demandclean_init_regression()
    test_demandclean_fit()
    test_demandclean_fit_with_semantic_errors()
    test_demandclean_detect_errors()
    test_demandclean_get_config()
    test_demandclean_get_training_history()
    test_demandclean_save_load()
    test_demandclean_not_fitted_error()
    test_demandclean_plan_not_fitted_error()
    test_demandclean_execute_no_plan_error()

    print("\n所有测试通过!")
