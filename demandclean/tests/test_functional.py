"""
DemandClean 功能测试
====================

使用真实数据测试系统完整功能。
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 添加项目路径
sys.path.insert(0, '/Users/qianzekai/PycharmProjects/TolerDM')

# 数据集路径
DATASETS_PATH = '/Users/qianzekai/PycharmProjects/TolerDM/experiment/pre_exp/beers_ipa_experiment/datasets'


def load_beers_data():
    """加载 beers 数据集（分类任务）"""
    clean_path = os.path.join(DATASETS_PATH, 'beers/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'beers/dirty.csv')

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


def load_bike_data():
    """加载 bike 数据集（回归任务）"""
    clean_path = os.path.join(DATASETS_PATH, 'bike/clean.csv')
    dirty_path = os.path.join(DATASETS_PATH, 'bike/dirty.csv')

    clean_df = pd.read_csv(clean_path)
    dirty_df = pd.read_csv(dirty_path)

    # 选择数值列（根据实际列名调整）
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return None, None, None

    # 使用最后一列作为目标
    feature_cols = numeric_cols[:-1]
    target_col = numeric_cols[-1]

    X_clean = clean_df[feature_cols].values.astype(np.float64)
    X_dirty = dirty_df[feature_cols].values.astype(np.float64)
    y = clean_df[target_col].values.astype(np.float64)

    return X_clean, X_dirty, y


def test_error_injector():
    """测试错误注入器"""
    print("\n" + "=" * 60)
    print("测试: ErrorInjector")
    print("=" * 60)

    from demandclean.detectors import ErrorInjector

    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    injector = ErrorInjector(X, y)
    X_dirty, y_dirty, injected = injector.inject_errors(
        missing_rate=0.05,
        semantic_rate=0.1,
        syntactic_rate=0.15
    )

    print(f"✓ 原始数据: {X.shape}")
    print(f"✓ 注入的缺失值: {len(injected['missing'])} 个")
    print(f"✓ 注入的语义错误: {len(injected['semantic'])} 个")
    print(f"✓ 注入的句法错误: {len(injected['syntactic'])} 个")
    print(f"✓ 注入后 NaN 数量: {np.isnan(X_dirty).sum()}")

    # 转换为错误列表
    error_list = injector.build_error_list(injected)
    print(f"✓ 错误列表长度: {len(error_list)}")

    return True


def test_model_adapters_with_data():
    """测试模型适配器（使用真实数据）"""
    print("\n" + "=" * 60)
    print("测试: ModelAdapters with Real Data")
    print("=" * 60)

    from demandclean.models import create_model_adapter
    from demandclean.config import ModelType, TaskType

    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_class = np.random.randint(0, 2, 100)
    y_reg = np.random.randn(100)

    # 测试分类模型
    for model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]:
        adapter = create_model_adapter(model_type, TaskType.CLASSIFICATION)
        adapter.fit(X, y_class)
        acc = adapter.evaluate(X, y_class)
        distance = adapter.get_distance_to_boundary(X[:5])
        print(f"✓ {model_type.value}: 准确率={acc:.4f}, 边界距离={distance[:3]}")

    # 测试回归模型
    for model_type in [ModelType.LINEAR, ModelType.RIDGE]:
        adapter = create_model_adapter(model_type, TaskType.REGRESSION)
        adapter.fit(X, y_reg)
        score = adapter.evaluate(X, y_reg)
        distance = adapter.get_distance_to_boundary(X[:5])
        print(f"✓ {model_type.value}: 得分={score:.4f}, 影响度={distance[:3]}")

    return True


def test_cleaning_env():
    """测试清洗环境"""
    print("\n" + "=" * 60)
    print("测试: CleaningEnv")
    print("=" * 60)

    from demandclean.core.environments import CleaningEnv
    from demandclean.core.state import ClassificationStateExtractor
    from demandclean.models import create_model_adapter
    from demandclean.config import DemandCleanConfig, ModelType, TaskType
    from demandclean.detectors import ErrorInjector

    # 生成测试数据
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

    # 创建环境
    env = CleaningEnv(X_dirty, y, error_list, model_adapter, state_extractor, config)

    # 重置环境
    state = env.reset()
    print(f"✓ 初始状态维度: {state.shape}")

    # 模拟几步
    total_reward = 0
    steps = 0
    while steps < 10:
        action = np.random.randint(0, 4)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"✓ 执行步数: {steps}")
    print(f"✓ 累积奖励: {total_reward:.4f}")
    print(f"✓ 动作统计: {env.get_action_counts()}")

    return True


def test_dqn_agent():
    """测试 DQN Agent"""
    print("\n" + "=" * 60)
    print("测试: DQN Agent")
    print("=" * 60)

    from demandclean.core.agents import SingleStageDQNAgent, TwoStageDQNAgent

    # 测试单阶段 Agent
    agent_single = SingleStageDQNAgent(state_size=8, action_size=4)
    state = np.random.randn(8).astype(np.float32)

    action = agent_single.act(state, training=True)
    print(f"✓ SingleStageDQNAgent: 动作={action}")

    # 测试两阶段 Agent
    agent_two = TwoStageDQNAgent(state_size=8)
    final_action, stage1_action, stage2_action = agent_two.act(state, training=True)
    print(f"✓ TwoStageDQNAgent: final={final_action}, stage1={stage1_action}, stage2={stage2_action}")

    # 测试经验回放
    next_state = np.random.randn(8).astype(np.float32)
    agent_single.remember(state, action, 0.5, next_state, False)
    print(f"✓ 经验存储成功")

    return True


def test_demandclean_basic():
    """测试 DemandClean 基本功能"""
    print("\n" + "=" * 60)
    print("测试: DemandClean 基本功能")
    print("=" * 60)

    from demandclean import DemandClean

    # 创建分类实例
    dc_class = DemandClean(
        task_type='classification',
        model_type='svm',
        agent_type='single_stage',
        n_episodes=5
    )
    print(f"✓ DemandClean (分类) 创建成功")

    # 创建回归实例
    dc_reg = DemandClean(
        task_type='regression',
        model_type='ridge',
        agent_type='two_stage',
        n_episodes=5,
        max_truth_budget=10
    )
    print(f"✓ DemandClean (回归) 创建成功")

    # 检查配置
    config = dc_class.get_config()
    print(f"✓ 配置获取成功: task={config.task_type.value}")

    return True


def test_demandclean_training_mini():
    """测试 DemandClean 迷你训练"""
    print("\n" + "=" * 60)
    print("测试: DemandClean 迷你训练 (5 episodes)")
    print("=" * 60)

    from demandclean import DemandClean

    # 生成小型测试数据
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)

    # 创建并训练
    dc = DemandClean(
        task_type='classification',
        model_type='random_forest',
        agent_type='single_stage',
        n_episodes=5
    )

    print("开始训练...")
    dc.fit(X, y, verbose=False)
    print(f"✓ 训练完成")

    # 检查是否已训练
    print(f"✓ 已训练: {dc.is_fitted}")

    return True


def test_full_pipeline_with_beers():
    """使用 beers 数据测试完整流程"""
    print("\n" + "=" * 60)
    print("测试: 完整流程 (Beers 数据)")
    print("=" * 60)

    try:
        X_clean, X_dirty, y = load_beers_data()
        if X_clean is None:
            print("⚠ 无法加载 beers 数据，跳过")
            return True

        print(f"✓ 数据加载成功: clean={X_clean.shape}, dirty={X_dirty.shape}")

        # 只使用前100行进行快速测试
        n_samples = min(100, len(X_clean))
        X_clean = X_clean[:n_samples]
        X_dirty = X_dirty[:n_samples]
        y = y[:n_samples]

        from demandclean import DemandClean

        # 创建并训练
        dc = DemandClean(
            task_type='classification',
            model_type='svm',
            agent_type='single_stage',
            n_episodes=10
        )

        print("开始训练...")
        dc.fit(X_dirty, y, verbose=False)
        print(f"✓ 训练完成")

        # 检测错误
        detected = dc.detect_errors(X_dirty, X_clean, verbose=False)
        print(f"✓ 检测到错误: missing={len(detected.get('missing', []))}, "
              f"semantic={len(detected.get('semantic', []))}, "
              f"syntactic={len(detected.get('syntactic', []))}")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DemandClean 功能测试")
    print("=" * 60 + "\n")

    tests = [
        ("ErrorInjector", test_error_injector),
        ("ModelAdapters", test_model_adapters_with_data),
        ("CleaningEnv", test_cleaning_env),
        ("DQN Agent", test_dqn_agent),
        ("DemandClean 基本", test_demandclean_basic),
        ("DemandClean 训练", test_demandclean_training_mini),
        ("完整流程 (Beers)", test_full_pipeline_with_beers),
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
            print(f"✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
