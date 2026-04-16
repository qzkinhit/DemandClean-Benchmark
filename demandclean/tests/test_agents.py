"""
DQN Agent 测试
===============
"""

import sys
import ossys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import pytest  # Optional
import numpy as np
from demandclean.core.agents import SingleStageDQNAgent, TwoStageDQNAgent


def test_single_stage_agent_init():
    """测试单阶段 Agent 初始化"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)

    assert agent.state_size == 8
    assert agent.action_size == 4
    assert agent.epsilon == 1.0
    print("✓ 单阶段 Agent 初始化测试通过")


def test_single_stage_agent_act_training():
    """测试单阶段 Agent 训练时动作"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=True)
    assert action in [0, 1, 2, 3]
    print(f"✓ 单阶段 Agent 训练动作测试通过: 动作={action}")


def test_single_stage_agent_act_inference():
    """测试单阶段 Agent 推理时动作"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    agent.epsilon = 0  # 关闭探索
    state = np.random.randn(8).astype(np.float32)

    action = agent.act(state, training=False)
    assert action in [0, 1, 2, 3]
    print(f"✓ 单阶段 Agent 推理动作测试通过: 动作={action}")


def test_single_stage_agent_remember():
    """测试单阶段 Agent 经验回放存储"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, memory_size=100)
    state = np.random.randn(8).astype(np.float32)
    next_state = np.random.randn(8).astype(np.float32)

    agent.remember(state, 1, 0.5, next_state, False)
    assert len(agent.memory) == 1
    print("✓ 单阶段 Agent 经验存储测试通过")


def test_single_stage_agent_replay():
    """测试单阶段 Agent 经验回放"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, memory_size=100)

    # 添加足够的经验
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember(state, np.random.randint(4), np.random.randn(), next_state, False)

    # 执行回放
    agent.replay(batch_size=32)
    print("✓ 单阶段 Agent 经验回放测试通过")


def test_single_stage_agent_update_target():
    """测试单阶段 Agent 目标网络更新"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)
    agent.update_target_model()
    print("✓ 单阶段 Agent 目标网络更新测试通过")


def test_single_stage_agent_epsilon_decay():
    """测试单阶段 Agent 探索率衰减"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4, epsilon_decay=0.99)
    initial_eps = agent.epsilon

    # 添加经验并回放
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember(state, np.random.randint(4), np.random.randn(), next_state, False)

    agent.replay(batch_size=32)
    assert agent.epsilon < initial_eps
    print(f"✓ 单阶段 Agent 探索率衰减测试通过: {initial_eps:.3f} -> {agent.epsilon:.3f}")


def test_two_stage_agent_init():
    """测试两阶段 Agent 初始化"""
    agent = TwoStageDQNAgent(state_size=8)

    assert agent.state_size == 8
    assert agent.stage1_action_size == 3  # no_action, 处理, delete
    assert agent.stage2_action_size == 2  # repair_value, replace_nearby
    print("✓ 两阶段 Agent 初始化测试通过")


def test_two_stage_agent_act_training():
    """测试两阶段 Agent 训练时动作"""
    agent = TwoStageDQNAgent(state_size=8)
    state = np.random.randn(8).astype(np.float32)

    final_action, stage1_action, stage2_action = agent.act(state, training=True)
    assert final_action in [0, 1, 2, 3]
    assert stage1_action in [0, 1, 2]
    # stage2_action 可能是 None（如果 stage1 不是"处理"）
    print(f"✓ 两阶段 Agent 训练动作测试通过: final={final_action}, s1={stage1_action}, s2={stage2_action}")


def test_two_stage_agent_act_inference():
    """测试两阶段 Agent 推理时动作"""
    agent = TwoStageDQNAgent(state_size=8)
    agent.epsilon = 0
    state = np.random.randn(8).astype(np.float32)

    final_action, stage1_action, stage2_action = agent.act(state, training=False)
    assert final_action in [0, 1, 2, 3]
    print(f"✓ 两阶段 Agent 推理动作测试通过: final={final_action}")


def test_two_stage_agent_remember():
    """测试两阶段 Agent 经验回放存储"""
    agent = TwoStageDQNAgent(state_size=8, memory_size=100)
    state = np.random.randn(8).astype(np.float32)
    next_state = np.random.randn(8).astype(np.float32)

    agent.remember_stage1(state, 1, 0.5, next_state, False)
    agent.remember_stage2(state, 0, 0.3, next_state, False)

    assert len(agent.stage1_memory) == 1
    assert len(agent.stage2_memory) == 1
    print("✓ 两阶段 Agent 经验存储测试通过")


def test_two_stage_agent_replay():
    """测试两阶段 Agent 经验回放"""
    agent = TwoStageDQNAgent(state_size=8, memory_size=100)

    # 添加足够的经验
    for _ in range(64):
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        agent.remember_stage1(state, np.random.randint(3), np.random.randn(), next_state, False)
        agent.remember_stage2(state, np.random.randint(2), np.random.randn(), next_state, False)

    # 执行回放
    agent.replay(batch_size=32)
    print("✓ 两阶段 Agent 经验回放测试通过")


def test_agent_get_set_weights():
    """测试获取和设置权重"""
    agent = SingleStageDQNAgent(state_size=8, action_size=4)

    weights = agent.get_weights()
    assert weights is not None

    agent.set_weights(weights)
    print("✓ Agent 权重获取设置测试通过")


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("DQN Agent 测试")
    print("=" * 50 + "\n")

    test_single_stage_agent_init()
    test_single_stage_agent_act_training()
    test_single_stage_agent_act_inference()
    test_single_stage_agent_remember()
    test_single_stage_agent_replay()
    test_single_stage_agent_update_target()
    test_single_stage_agent_epsilon_decay()
    test_two_stage_agent_init()
    test_two_stage_agent_act_training()
    test_two_stage_agent_act_inference()
    test_two_stage_agent_remember()
    test_two_stage_agent_replay()
    test_agent_get_set_weights()

    print("\n所有测试通过!")
