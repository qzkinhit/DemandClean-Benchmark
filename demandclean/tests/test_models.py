"""
模型适配器测试
==============
"""

import sys
sys.path.insert(0, '/Users/qianzekai/PycharmProjects/TolerDM')

# import pytest  # Optional
import numpy as np
from demandclean.models import create_model_adapter
from demandclean.config import ModelType, TaskType


def test_create_svm_adapter():
    """测试创建 SVM 适配器"""
    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    assert adapter is not None
    print("✓ SVM 适配器创建测试通过")


def test_create_random_forest_adapter():
    """测试创建 RandomForest 适配器"""
    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    assert adapter is not None
    print("✓ RandomForest 适配器创建测试通过")


def test_create_linear_adapter():
    """测试创建 Linear 适配器"""
    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    assert adapter is not None
    print("✓ Linear 适配器创建测试通过")


def test_create_ridge_adapter():
    """测试创建 Ridge 适配器"""
    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    assert adapter is not None
    print("✓ Ridge 适配器创建测试通过")


def test_create_adapter_with_string():
    """测试使用字符串创建适配器"""
    adapter = create_model_adapter('svm', 'classification')
    assert adapter is not None
    print("✓ 字符串创建适配器测试通过")


def test_svm_fit_predict():
    """测试 SVM 训练和预测"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    assert all(p in [0, 1] for p in pred)
    print("✓ SVM 训练预测测试通过")


def test_svm_evaluate():
    """测试 SVM 评估"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    score = adapter.evaluate(X, y)
    assert 0 <= score <= 1  # 准确率在 0-1 之间
    print(f"✓ SVM 评估测试通过: 准确率={score:.4f}")


def test_svm_distance_to_boundary():
    """测试 SVM 边界距离"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ SVM 边界距离测试通过: {distance[:3]}")


def test_svm_feature_importance():
    """测试 SVM 特征重要性"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    importance = adapter.get_feature_importance()
    assert len(importance) == 5
    assert np.isclose(importance.sum(), 1.0)  # 归一化
    print(f"✓ SVM 特征重要性测试通过: {importance}")


def test_random_forest_fit_predict():
    """测试 RandomForest 训练和预测"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ RandomForest 训练预测测试通过")


def test_random_forest_distance_to_boundary():
    """测试 RandomForest 边界距离"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ RandomForest 边界距离测试通过: {distance[:3]}")


def test_linear_fit_predict():
    """测试 Linear 回归训练和预测"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ Linear 回归训练预测测试通过")


def test_linear_evaluate():
    """测试 Linear 回归评估"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.LINEAR, TaskType.REGRESSION)
    adapter.fit(X, y)

    score = adapter.evaluate(X, y)
    assert score <= 0  # 负 MSE
    print(f"✓ Linear 回归评估测试通过: 负MSE={score:.4f}")


def test_ridge_fit_predict():
    """测试 Ridge 回归训练和预测"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    adapter.fit(X, y)

    pred = adapter.predict(X)
    assert len(pred) == 100
    print("✓ Ridge 回归训练预测测试通过")


def test_ridge_distance_to_boundary():
    """测试 Ridge 回归影响度"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    adapter = create_model_adapter(ModelType.RIDGE, TaskType.REGRESSION)
    adapter.fit(X, y)

    distance = adapter.get_distance_to_boundary(X[:5])
    assert len(distance) == 5
    assert all(0 <= d <= 1 for d in distance)
    print(f"✓ Ridge 回归影响度测试通过: {distance[:3]}")


def test_model_clone():
    """测试模型克隆"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    adapter = create_model_adapter(ModelType.SVM, TaskType.CLASSIFICATION)
    adapter.fit(X, y)

    cloned = adapter.clone()
    assert not cloned._is_fitted  # 克隆应该是未训练的
    print("✓ 模型克隆测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("模型适配器测试")
    print("=" * 50 + "\n")

    test_create_svm_adapter()
    test_create_random_forest_adapter()
    test_create_linear_adapter()
    test_create_ridge_adapter()
    test_create_adapter_with_string()
    test_svm_fit_predict()
    test_svm_evaluate()
    test_svm_distance_to_boundary()
    test_svm_feature_importance()
    test_random_forest_fit_predict()
    test_random_forest_distance_to_boundary()
    test_linear_fit_predict()
    test_linear_evaluate()
    test_ridge_fit_predict()
    test_ridge_distance_to_boundary()
    test_model_clone()

    print("\n所有测试通过!")
