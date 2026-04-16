"""
环境测试
========
"""

import sys
import ossys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.core.environments import CleaningEnv, TwoPhaseCleaningEnv
from demandclean.core.state import ClassificationStateExtractor
from demandclean.models import create_model_adapter
from demandclean.config import DemandCleanConfig, ModelType, TaskType
from demandclean.detectors import ErrorInjector


def create_test_env():
    """创建测试环境的辅助函数"""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    # 注入错误
    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(0.05, 0.1, 0.15)
    error_list = injector.build_error_list(injected)

    # 创建组件
    config = DemandCleanConfig(
        task_type=TaskType.CLASSIFICATION,
        model_type=ModelType.SVM
    )
    model_adapter = create_model_adapter(config.model_type, config.task_type)
    state_extractor = ClassificationStateExtractor(model_adapter, config)

    return X_dirty, y, error_list, model_adapter, state_extractor, config


def test_cleaning_env_init():
    """测试清洗环境初始化"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    assert env is not None
    print("✓ 清洗环境初始化测试通过")


def test_cleaning_env_reset():
    """测试清洗环境重置"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    assert state.shape == (8,)
    assert not np.isnan(state).any()
    print(f"✓ 清洗环境重置测试通过: state shape={state.shape}")


def test_cleaning_env_step():
    """测试清洗环境步进"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # 测试每种动作
    for action in [0, 1, 2, 3]:
        env.reset()
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (8,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        print(f"  动作 {action}: reward={reward:.4f}, done={done}")

    print("✓ 清洗环境步进测试通过")


def test_cleaning_env_episode():
    """测试清洗环境完整 episode"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state
        if done:
            break

    assert steps > 0
    print(f"✓ 清洗环境完整 episode 测试通过: {steps} 步, reward={total_reward:.4f}")


def test_cleaning_env_action_counts():
    """测试清洗环境动作统计"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    env.reset()

    # 执行一些动作
    for _ in range(10):
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    counts = env.get_action_counts()
    assert 'no_action' in counts
    assert 'repair_value' in counts
    assert 'delete' in counts
    assert 'replace_nearby' in counts
    print(f"✓ 清洗环境动作统计测试通过: {counts}")


def test_cleaning_env_get_cleaned_data():
    """测试获取清洗后数据"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    env.reset()

    # 执行一些动作
    for _ in range(10):
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    X_clean, y_clean, keep_mask = env.get_cleaned_data()
    assert len(X_clean) == len(y_clean)
    assert len(keep_mask) == len(y)
    print(f"✓ 获取清洗后数据测试通过: {len(X_clean)} 行")


def test_two_phase_env_init():
    """测试两阶段环境初始化"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    assert env is not None
    print("✓ 两阶段环境初始化测试通过")


def test_two_phase_env_plan():
    """测试两阶段环境计划生成"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # 执行到结束，生成计划
    while True:
        action = np.random.randint(0, 4)
        next_state, _, done, _ = env.step(action)
        if done:
            break

    plan = env.get_repair_plan()
    assert isinstance(plan, list)
    print(f"✓ 两阶段环境计划生成测试通过: {len(plan)} 个计划项")


def test_two_phase_env_execute():
    """测试两阶段环境执行计划"""
    X_dirty, y, error_list, model_adapter, state_extractor, config = create_test_env()

    env = TwoPhaseCleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)
    state = env.reset()

    # 执行到结束
    while True:
        action = np.random.randint(0, 4)
        _, _, done, _ = env.step(action)
        if done:
            break

    plan = env.get_repair_plan()
    positions = env.get_plan_positions()

    # 创建真值
    true_values = {}
    for idx, col in positions:
        true_values[(idx, col)] = np.random.randn()

    # 执行计划
    X_result, y_result = env.execute_repair_plan(X_dirty, true_values)
    assert X_result is not None
    print(f"✓ 两阶段环境执行计划测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("环境测试")
    print("=" * 50 + "\n")

    test_cleaning_env_init()
    test_cleaning_env_reset()
    test_cleaning_env_step()
    test_cleaning_env_episode()
    test_cleaning_env_action_counts()
    test_cleaning_env_get_cleaned_data()
    test_two_phase_env_init()
    test_two_phase_env_plan()
    test_two_phase_env_execute()

    print("\n所有测试通过!")
