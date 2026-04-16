"""
检测器模块测试
==============
"""

import sys
import ossys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.detectors import ErrorInjector, RahaBasedDetector


def test_error_injector_init():
    """测试错误注入器初始化"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    assert injector.X_base.shape == (100, 5)
    assert injector.y_base.shape == (100,)
    print("✓ 错误注入器初始化测试通过")


def test_error_injector_missing():
    """测试缺失值注入"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(missing_rate=0.1)

    assert np.isnan(X_dirty).sum() > 0
    assert len(injected['missing']) > 0
    print(f"✓ 缺失值注入测试通过: {len(injected['missing'])} 个缺失值")


def test_error_injector_semantic():
    """测试语义错误注入"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(semantic_rate=0.1)

    assert len(injected['semantic']) > 0
    # 语义错误会替换原值
    diff_count = np.sum(X != X_dirty)
    assert diff_count >= len(injected['semantic'])
    print(f"✓ 语义错误注入测试通过: {len(injected['semantic'])} 个语义错误")


def test_error_injector_syntactic():
    """测试句法错误注入"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(syntactic_rate=0.15)

    assert len(injected['syntactic']) > 0
    print(f"✓ 句法错误注入测试通过: {len(injected['syntactic'])} 个句法错误")


def test_error_injector_combined():
    """测试组合错误注入"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    total_errors = len(injected['missing']) + len(injected['semantic']) + len(injected['syntactic'])
    assert total_errors > 0
    print(f"✓ 组合错误注入测试通过: 共 {total_errors} 个错误")


def test_error_injector_build_error_list():
    """测试错误列表构建"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    error_list = injector.build_error_list(injected)

    assert len(error_list) > 0
    assert all('idx' in e and 'col' in e and 'type' in e for e in error_list)
    print(f"✓ 错误列表构建测试通过: {len(error_list)} 个错误")


def test_raha_detector_init():
    """测试 RAHA 检测器初始化（RahaBasedDetector 是 AutoDetector 的别名）"""
    detector = RahaBasedDetector()

    assert detector.is_fitted == False
    assert isinstance(detector.col_stats, dict)
    print("✓ RAHA 检测器初始化测试通过")


def test_raha_detector_col_stats():
    """测试列统计量计算"""
    np.random.seed(42)
    X = np.random.randn(100, 3)

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)

    assert len(detector.col_stats) == 3
    assert 'mean' in detector.col_stats[0]
    assert 'std' in detector.col_stats[0]
    print("✓ 列统计量计算测试通过")


def test_raha_detector_detect_missing():
    """测试缺失值检测"""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # 注入一些 NaN
    X[5, 0] = np.nan
    X[10, 1] = np.nan
    X[20, 2] = np.nan

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)
    detected = detector.detect(X, verbose=False)

    assert len(detected['missing']) == 3
    print("✓ 缺失值检测测试通过")


def test_raha_detector_detect_with_semantic():
    """测试带语义错误位置的检测"""
    np.random.seed(42)
    X = np.random.randn(100, 3)

    detector = RahaBasedDetector()
    detector._compute_col_stats(X)

    semantic_positions = [(5, 0), (10, 1), (15, 2)]
    detected = detector.detect(X, semantic_positions=semantic_positions, verbose=False)

    assert len(detected['semantic']) == 3
    print("✓ 语义错误检测测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("检测器模块测试")
    print("=" * 50 + "\n")

    test_error_injector_init()
    test_error_injector_missing()
    test_error_injector_semantic()
    test_error_injector_syntactic()
    test_error_injector_combined()
    test_error_injector_build_error_list()
    test_raha_detector_init()
    test_raha_detector_col_stats()
    test_raha_detector_detect_missing()
    test_raha_detector_detect_with_semantic()

    print("\n所有测试通过!")
