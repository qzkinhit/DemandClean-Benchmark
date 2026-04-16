"""
DemandClean 模块测试脚本
========================

测试模块是否能正确导入和基本功能是否正常。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_imports():
    """测试所有模块是否能正确导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # 1. 测试主包导入
    try:
        from demandclean import DemandClean, DemandCleanConfig
        print("✓ 主包导入成功: DemandClean, DemandCleanConfig")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 主包导入失败: {e}")
        tests_failed += 1

    # 2. 测试配置模块
    try:
        from demandclean.config import DemandCleanConfig, TaskType, ModelType, AgentType
        print("✓ 配置模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 配置模块导入失败: {e}")
        tests_failed += 1

    # 3. 测试模型适配器
    try:
        from demandclean.models import (
            ModelAdapter, create_model_adapter,
            SVMAdapter, RandomForestAdapter,
            LinearAdapter, RidgeAdapter
        )
        print("✓ 模型适配器模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 模型适配器模块导入失败: {e}")
        tests_failed += 1

    # 4. 测试状态提取器
    try:
        from demandclean.core.state import (
            StateExtractor,
            ClassificationStateExtractor,
            RegressionStateExtractor
        )
        print("✓ 状态提取器模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 状态提取器模块导入失败: {e}")
        tests_failed += 1

    # 5. 测试 DQN Agent
    try:
        from demandclean.core.agents import (
            BaseAgent,
            SingleStageDQNAgent,
            TwoStageDQNAgent
        )
        print("✓ DQN Agent 模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ DQN Agent 模块导入失败: {e}")
        tests_failed += 1

    # 6. 测试环境模块
    try:
        from demandclean.core.environments import (
            CleaningEnv,
            TwoPhaseCleaningEnv
        )
        print("✓ 环境模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 环境模块导入失败: {e}")
        tests_failed += 1

    # 7. 测试检测器模块
    try:
        from demandclean.detectors import (
            ErrorInjector,
            RahaBasedDetector
        )
        print("✓ 检测器模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 检测器模块导入失败: {e}")
        tests_failed += 1

    # 8. 测试训练模块
    try:
        from demandclean.training import Trainer
        print("✓ 训练模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 训练模块导入失败: {e}")
        tests_failed += 1

    # 9. 测试推理模块
    try:
        from demandclean.inference import (
            SinglePhaseInference,
            TwoPhaseInference
        )
        print("✓ 推理模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 推理模块导入失败: {e}")
        tests_failed += 1

    # 10. 测试工具模块
    try:
        from demandclean.utils import (
            DemandCleanLogger,
            ModelIO,
            Metrics
        )
        print("✓ 工具模块导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 工具模块导入失败: {e}")
        tests_failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {tests_passed} 通过, {tests_failed} 失败")
    print("=" * 60)

    return tests_failed == 0


def test_config_creation():
    """测试配置对象创建"""
    print("\n" + "=" * 60)
    print("测试配置对象创建")
    print("=" * 60)

    try:
        from demandclean.config import DemandCleanConfig, TaskType, ModelType, AgentType

        # 默认配置
        config = DemandCleanConfig()
        print(f"✓ 默认配置创建成功")
        print(f"  任务类型: {config.task_type.value}")
        print(f"  模型类型: {config.model_type.value}")
        print(f"  Agent类型: {config.agent_type.value}")

        # 自定义配置
        config = DemandCleanConfig(
            task_type=TaskType.REGRESSION,
            model_type=ModelType.RIDGE,
            agent_type=AgentType.SINGLE_STAGE,
            n_episodes=100,
            max_truth_budget=50
        )
        print(f"✓ 自定义配置创建成功")
        print(f"  任务类型: {config.task_type.value}")
        print(f"  模型类型: {config.model_type.value}")
        print(f"  Agent类型: {config.agent_type.value}")
        print(f"  训练轮数: {config.n_episodes}")
        print(f"  真值预算上限: {config.max_truth_budget}")

        return True
    except Exception as e:
        print(f"✗ 配置创建失败: {e}")
        return False


def test_model_adapter_creation():
    """测试模型适配器创建"""
    print("\n" + "=" * 60)
    print("测试模型适配器创建")
    print("=" * 60)

    try:
        from demandclean.models import create_model_adapter
        from demandclean.config import ModelType, TaskType

        # 分类模型
        for model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]:
            adapter = create_model_adapter(model_type, TaskType.CLASSIFICATION)
            print(f"✓ {model_type.value} 分类适配器创建成功")

        # 回归模型
        for model_type in [ModelType.LINEAR, ModelType.RIDGE]:
            adapter = create_model_adapter(model_type, TaskType.REGRESSION)
            print(f"✓ {model_type.value} 回归适配器创建成功")

        return True
    except Exception as e:
        print(f"✗ 模型适配器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demandclean_api():
    """测试 DemandClean API"""
    print("\n" + "=" * 60)
    print("测试 DemandClean API 创建")
    print("=" * 60)

    try:
        from demandclean import DemandClean

        # 分类任务
        dc = DemandClean(
            task_type='classification',
            model_type='svm',
            agent_type='two_stage',
            n_episodes=10
        )
        print(f"✓ DemandClean (分类, SVM, 两阶段) 创建成功")

        # 回归任务
        dc = DemandClean(
            task_type='regression',
            model_type='ridge',
            agent_type='single_stage',
            n_episodes=10,
            max_truth_budget=20
        )
        print(f"✓ DemandClean (回归, Ridge, 单阶段) 创建成功")

        return True
    except Exception as e:
        print(f"✗ DemandClean API 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean 模块测试")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config_creation()
    all_passed &= test_model_adapter_creation()
    all_passed &= test_demandclean_api()

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 60)
