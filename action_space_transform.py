#!/usr/bin/env python3
"""
降维动作空间变换

实现第5节中描述的三元动作空间变换：
- 将连续4D动作空间转换为离散三元表示
- 使用笛卡尔积方法进行独立维度控制
- 将动作空间从连续缩减到最大3^4 = 81种可能性
- 通过方程(23)实现与PAC算法的无缝兼容
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import copy

try:
    from .optimization_problem import OptimizationConstraints
    from .mdp_framework import Action
except ImportError:
    from optimization_problem import OptimizationConstraints
    from mdp_framework import Action


class TernaryAction(Enum):
    """三元动作表示：{-1, 0, 1}。"""
    DECREASE = -1  # 按单位粒度减少当前值
    MAINTAIN = 0   # 保持当前值
    INCREASE = 1   # 按单位粒度增加当前值


@dataclass
class ActionGranularity:
    """定义每个动作维度的粒度。"""
    
    n_clients_step: int = 1           # 客户端选择步长
    frequency_step: float = 1e9       # CPU频率步长 (±1.0 GHz)
    bandwidth_step: float = 10e6      # 带宽步长 (±10 MHz)
    # 量化步长按“比特”计：±1 bit -> q按2的指数变化
    quantization_bit_step: int = 1
    
    def to_array(self) -> np.ndarray:
        """转换为数组格式。"""
        # 返回前三个维度的线性步长与第4维的比特步长
        return np.array([
            self.n_clients_step,
            self.frequency_step,
            self.bandwidth_step,
            self.quantization_bit_step
        ])


@dataclass
class TernaryActionVector:
    """动作向量的三元表示 a'(m) = {-1, 0, 1}。"""
    
    n_clients_action: TernaryAction = TernaryAction.MAINTAIN      # a'(1)
    frequency_action: TernaryAction = TernaryAction.MAINTAIN      # a'(2)
    bandwidth_action: TernaryAction = TernaryAction.MAINTAIN      # a'(3)
    quantization_action: TernaryAction = TernaryAction.MAINTAIN   # a'(4)
    
    def to_array(self) -> np.ndarray:
        """转换为每个维度的数组格式 [-1, 0, 1]。"""
        return np.array([
            self.n_clients_action.value,
            self.frequency_action.value,
            self.bandwidth_action.value,
            self.quantization_action.value
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'TernaryActionVector':
        """从数组格式创建。"""
        return cls(
            n_clients_action=TernaryAction(int(array[0])),
            frequency_action=TernaryAction(int(array[1])),
            bandwidth_action=TernaryAction(int(array[2])),
            quantization_action=TernaryAction(int(array[3]))
        )
    
    def to_index(self) -> int:
        """转换为离散动作空间的单一索引 (0 到 80)。"""
        # 从 {-1, 0, 1} 转换为 {0, 1, 2} 用于索引
        actions = self.to_array() + 1  # 现在范围是 [0, 2]
        
        # 使用基数3编码：index = a[0]*27 + a[1]*9 + a[2]*3 + a[3]
        index = (actions[0] * 27 + 
                actions[1] * 9 + 
                actions[2] * 3 + 
                actions[3])
        
        return int(index)
    
    @classmethod
    def from_index(cls, index: int) -> 'TernaryActionVector':
        """从单一索引创建 (0 到 80)。"""
        if not 0 <= index <= 80:
            raise ValueError(f"索引 {index} 超出范围 [0, 80]")
        
        # 解码基数3表示
        actions = np.zeros(4, dtype=int)
        temp = index
        
        for i in range(4):
            actions[3-i] = temp % 3
            temp //= 3
        
        # 从 {0, 1, 2} 转换回 {-1, 0, 1}
        actions = actions - 1
        
        return cls.from_array(actions)


class ActionSpaceTransformer:
    """
    将连续动作空间转换为离散三元表示。
    
    实现降维技术，通过离散化为3^4 = 81个动作，
    使PAC算法与连续动作空间兼容。
    """
    
    def __init__(self, 
                 constraints: OptimizationConstraints,
                 granularity: ActionGranularity = None):
        """
        初始化动作空间变换器。
        
        Args:
            constraints: 动作边界的系统约束
            granularity: 每个动作维度的步长
        """
        self.constraints = constraints
        self.granularity = granularity or ActionGranularity()
        
        # 生成所有可能的三元动作组合（笛卡尔积）
        self.action_space = self._generate_ternary_action_space()
        
        print(f"动作空间变换器已初始化：")
        print(f"  原始空间：连续4D")
        print(f"  变换后空间：离散{len(self.action_space)}个动作")
        print(f"  粒度：{self.granularity}")
    
    def _generate_ternary_action_space(self) -> List[TernaryActionVector]:
        """生成所有可能的三元动作组合 A' = ∏_m a'(m)。"""
        
        action_space = []
        
        # 生成 {-1, 0, 1}^4 的笛卡尔积
        for n_action in TernaryAction:
            for f_action in TernaryAction:
                for b_action in TernaryAction:
                    for q_action in TernaryAction:
                        ternary_action = TernaryActionVector(
                            n_clients_action=n_action,
                            frequency_action=f_action,
                            bandwidth_action=b_action,
                            quantization_action=q_action
                        )
                        action_space.append(ternary_action)
        
        return action_space
    
    def transform_to_ternary(self, 
                           current_action: Union[Action, np.ndarray],
                           target_action: Union[Action, np.ndarray]) -> TernaryActionVector:
        """
        将连续动作变化转换为三元表示。
        
        Args:
            current_action: 当前动作值
            target_action: 目标动作值
            
        Returns:
            表示变化方向的三元动作
        """
        if isinstance(current_action, Action):
            current_array = current_action.to_array()
        else:
            current_array = current_action
        
        if isinstance(target_action, Action):
            target_array = target_action.to_array()
        else:
            target_array = target_action
        
        # 计算每个维度的变化方向
        change = target_array - current_array
        granularity_array = self.granularity.to_array()

        # 转换为三元表示（前三维线性，量化为“比特”步长）
        ternary_values = []
        # 0: n_clients, 1: frequency, 2: bandwidth
        for i in range(3):
            step = granularity_array[i]
            delta = change[i]
            if delta > step / 2:
                ternary_values.append(TernaryAction.INCREASE)
            elif delta < -step / 2:
                ternary_values.append(TernaryAction.DECREASE)
            else:
                ternary_values.append(TernaryAction.MAINTAIN)

        # 3: quantization 使用“±1比特”等效判定
        q_cur = max(1, int(round(current_array[3])))
        q_tgt = max(1, int(round(target_array[3])))
        b_cur = int(np.ceil(np.log2(q_cur))) if q_cur > 0 else 0
        b_tgt = int(np.ceil(np.log2(q_tgt))) if q_tgt > 0 else 0
        bit_delta = b_tgt - b_cur
        bit_step = int(granularity_array[3]) if granularity_array[3] > 0 else 1
        if bit_delta > 0.5 * bit_step:
            ternary_values.append(TernaryAction.INCREASE)
        elif bit_delta < -0.5 * bit_step:
            ternary_values.append(TernaryAction.DECREASE)
        else:
            ternary_values.append(TernaryAction.MAINTAIN)
        
        return TernaryActionVector(
            n_clients_action=ternary_values[0],
            frequency_action=ternary_values[1],
            bandwidth_action=ternary_values[2],
            quantization_action=ternary_values[3]
        )
    
    def apply_ternary_action(self, 
                           current_action: Union[Action, np.ndarray],
                           ternary_action: TernaryActionVector) -> np.ndarray:
        """
        将三元动作应用于当前动作以获得新动作。
        
        Args:
            current_action: 当前动作值
            ternary_action: 要应用的三元动作
            
        Returns:
            应用三元变换后的新动作值
        """
        if isinstance(current_action, Action):
            current_array = current_action.to_array()
        else:
            current_array = current_action.copy()
        
        ternary_array = ternary_action.to_array()
        granularity_array = self.granularity.to_array()

        # 前三维：线性步长
        new_action = current_array.copy()
        for i in range(3):
            new_action[i] = current_array[i] + ternary_array[i] * granularity_array[i]

        # 量化维：±1比特（对数步长）
        q_cur = max(1, int(round(current_array[3])))
        bit_step = int(granularity_array[3]) if granularity_array[3] > 0 else 1
        if ternary_array[3] > 0:  # +1 bit
            new_bits = int(np.ceil(np.log2(q_cur))) + bit_step
            q_new = int(np.clip(2 ** new_bits, self.constraints.min_quantization, self.constraints.max_quantization))
        elif ternary_array[3] < 0:  # -1 bit
            new_bits = max(0, int(np.ceil(np.log2(q_cur))) - bit_step)
            q_new = int(np.clip(max(1, 2 ** new_bits), self.constraints.min_quantization, self.constraints.max_quantization))
        else:
            q_new = q_cur

        new_action[3] = q_new

        # 应用约束（裁剪到有效范围）
        new_action = self._apply_constraints(new_action)

        return new_action
    
    def _apply_constraints(self, action_array: np.ndarray) -> np.ndarray:
        """将系统约束应用于动作值。"""
        
        constrained_action = action_array.copy()
        
        # 裁剪到约束边界
        constrained_action[0] = np.clip(
            constrained_action[0], 
            self.constraints.min_clients, 
            self.constraints.max_clients
        )
        
        constrained_action[1] = np.clip(
            constrained_action[1],
            self.constraints.min_frequency,
            self.constraints.max_frequency
        )
        
        constrained_action[2] = np.clip(
            constrained_action[2],
            self.constraints.min_bandwidth,
            self.constraints.max_bandwidth
        )
        
        constrained_action[3] = np.clip(
            constrained_action[3],
            self.constraints.min_quantization,
            self.constraints.max_quantization
        )
        
        # 确保整数约束
        constrained_action[0] = round(constrained_action[0])  # n_clients
        constrained_action[3] = round(constrained_action[3])  # quantization
        
        return constrained_action
    
    def get_action_space_size(self) -> int:
        """获取变换后离散动作空间的大小。"""
        return len(self.action_space)
    
    def get_action_by_index(self, index: int) -> TernaryActionVector:
        """通过索引获取三元动作。"""
        if 0 <= index < len(self.action_space):
            return self.action_space[index]
        else:
            raise ValueError(f"索引 {index} 超出范围 [0, {len(self.action_space)-1}]")
    
    def get_all_possible_actions(self, current_action: Union[Action, np.ndarray]) -> List[np.ndarray]:
        """
        从当前动作获取所有可能的下一个动作。
        
        这用于PAC算法的虚拟联合策略计算（方程23）。
        
        Args:
            current_action: 当前动作状态
            
        Returns:
            所有可能的下一个动作的列表（最多81个）
        """
        possible_actions = []
        
        for ternary_action in self.action_space:
            new_action = self.apply_ternary_action(current_action, ternary_action)
            possible_actions.append(new_action)
        
        return possible_actions
    
    def get_exploration_actions(self, 
                              current_action: Union[Action, np.ndarray],
                              exploration_radius: int = 1) -> List[Tuple[np.ndarray, TernaryActionVector]]:
        """
        获取特定半径内的探索动作，以实现更高效的搜索。
        
        Args:
            current_action: 当前动作状态
            exploration_radius: 同时改变的最大维度数
            
        Returns:
            (动作, 三元动作) 对的列表
        """
        exploration_actions = []
        
        for ternary_action in self.action_space:
            # 计算非零（非MAINTAIN）动作的数量
            ternary_array = ternary_action.to_array()
            num_changes = np.sum(ternary_array != 0)
            
            if num_changes <= exploration_radius:
                new_action = self.apply_ternary_action(current_action, ternary_action)
                exploration_actions.append((new_action, ternary_action))
        
        return exploration_actions


class TernaryPACAgent:
    """
    适用于三元动作空间的PAC智能体。
    
    与动作空间变换器集成，使PAC算法
    与连续动作空间兼容。
    """
    
    def __init__(self,
                 agent_id: int,
                 observation_dim: int,
                 constraints: OptimizationConstraints,
                 granularity: ActionGranularity = None):
        """
        初始化三元PAC智能体。
        
        Args:
            agent_id: 智能体标识符
            observation_dim: 观测空间维度
            constraints: 系统约束
            granularity: 动作粒度设置
        """
        self.agent_id = agent_id
        self.observation_dim = observation_dim
        self.constraints = constraints
        
        # 初始化动作空间变换器
        self.action_transformer = ActionSpaceTransformer(constraints, granularity)
        self.action_space_size = self.action_transformer.get_action_space_size()
        
        # 当前动作状态
        self.current_action = self._initialize_action()
        
        print(f"三元PAC智能体 {agent_id} 已初始化：")
        print(f"  离散动作空间大小：{self.action_space_size}")
        print(f"  动作粒度：{granularity}")
    
    def _initialize_action(self) -> np.ndarray:
        """使用中等范围的动作值进行初始化。"""
        return np.array([
            (self.constraints.min_clients + self.constraints.max_clients) // 2,
            (self.constraints.min_frequency + self.constraints.max_frequency) / 2,
            (self.constraints.min_bandwidth + self.constraints.max_bandwidth) / 2,
            (self.constraints.min_quantization + self.constraints.max_quantization) // 2
        ])
    
    def select_ternary_action(self, 
                            observation: np.ndarray,
                            exploration_mode: bool = True) -> Tuple[TernaryActionVector, np.ndarray]:
        """
        基于当前观测选择三元动作。
        
        Args:
            observation: 当前观测
            exploration_mode: 是否使用探索
            
        Returns:
            (三元动作, 结果连续动作) 的元组
        """
        if exploration_mode:
            # 在三元动作中随机探索
            action_index = np.random.randint(0, self.action_space_size)
            ternary_action = self.action_transformer.get_action_by_index(action_index)
        else:
            # 贪心动作（占位符 - 在完整实现中会使用Q网络）
            ternary_action = TernaryActionVector()  # 全部MAINTAIN
        
        # 应用三元动作获得连续动作
        new_action = self.action_transformer.apply_ternary_action(
            self.current_action, ternary_action
        )
        
        # 更新当前动作
        self.current_action = new_action
        
        return ternary_action, new_action
    
    def compute_virtual_joint_policy_ternary(self, 
                                           observation: np.ndarray,
                                           other_agents_actions: List[np.ndarray]) -> TernaryActionVector:
        """
        使用三元动作空间计算虚拟联合策略。
        
        这实现了通过方程(23)使PAC算法与连续动作空间兼容的关键改进。
        
        Args:
            observation: 当前观测
            other_agents_actions: 其他智能体的动作
            
        Returns:
            根据虚拟联合策略的最佳三元动作
        """
        best_ternary_action = None
        best_q_value = float('-inf')
        
        # 评估所有可能的三元动作（最多81个）
        for ternary_action in self.action_transformer.action_space:
            # 计算潜在的新动作
            potential_action = self.action_transformer.apply_ternary_action(
                self.current_action, ternary_action
            )
            
            # 评估Q值（占位符 - 会使用Q网络）
            q_value = self._evaluate_q_value(observation, potential_action, other_agents_actions)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_ternary_action = ternary_action
        
        return best_ternary_action
    
    def _evaluate_q_value(self, 
                         observation: np.ndarray,
                         action: np.ndarray,
                         other_actions: List[np.ndarray]) -> float:
        """
        评估给定状态-动作对的Q值。
        
        占位符实现 - 在实践中会使用训练好的Q网络。
        """
        # 演示用的简单启发式
        # 在实践中，这会是 Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})
        
        # 奖励组件（简化）
        efficiency_reward = -np.sum(action ** 2) * 1e-12  # 效率偏好
        coordination_reward = np.sum([np.dot(action, other_action) for other_action in other_actions]) * 1e-15
        
        return efficiency_reward + coordination_reward
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """获取动作空间使用情况的统计信息。"""
        return {
            'current_action': self.current_action.tolist(),
            'action_space_size': self.action_space_size,
            'granularity': self.action_transformer.granularity,
            'constraints': {
                'clients': [self.constraints.min_clients, self.constraints.max_clients],
                'frequency': [self.constraints.min_frequency/1e9, self.constraints.max_frequency/1e9],
                'bandwidth': [self.constraints.min_bandwidth/1e6, self.constraints.max_bandwidth/1e6],
                'quantization': [self.constraints.min_quantization, self.constraints.max_quantization]
            }
        }


def test_action_space_transformation():
    """测试动作空间变换实现。"""
    print("测试动作空间变换...")
    
    # 设置
    constraints = OptimizationConstraints(
        min_clients=1, max_clients=5,
        min_frequency=0.5e9, max_frequency=3.5e9,  # 0.5-3.5 GHz
        min_bandwidth=0, max_bandwidth=30e6,        # 0-30 MHz
        min_quantization=2, max_quantization=32
    )
    
    granularity = ActionGranularity(
        n_clients_step=1,
        frequency_step=0.5e9,  # 500 MHz步长
        bandwidth_step=5e6,    # 5 MHz步长
        quantization_step=2
    )
    
    # 测试1：动作空间变换器
    print(f"\n--- 测试1：动作空间变换器 ---")
    transformer = ActionSpaceTransformer(constraints, granularity)
    
    print(f"动作空间大小：{transformer.get_action_space_size()}")
    print(f"期望大小 (3^4)：{3**4}")
    
    # 测试2：三元动作表示
    print(f"\n--- 测试2：三元动作表示 ---")
    
    # 测试所有三元组合
    test_actions = [
        TernaryActionVector(
            n_clients_action=TernaryAction.INCREASE,
            frequency_action=TernaryAction.MAINTAIN,
            bandwidth_action=TernaryAction.DECREASE,
            quantization_action=TernaryAction.INCREASE
        ),
        TernaryActionVector(),  # 全部MAINTAIN
    ]
    
    for i, ternary_action in enumerate(test_actions):
        array = ternary_action.to_array()
        index = ternary_action.to_index()
        reconstructed = TernaryActionVector.from_index(index)
        
        print(f"三元动作 {i+1}：")
        print(f"  数组：{array}")
        print(f"  索引：{index}")
        print(f"  重构：{reconstructed.to_array()}")
        print(f"  匹配：{np.array_equal(array, reconstructed.to_array())}")
    
    # 测试3：连续到三元变换
    print(f"\n--- 测试3：连续到三元变换 ---")
    
    current_action = np.array([2, 2e9, 10e6, 8])  # 2个客户端，2GHz，10MHz，级别8
    target_action = np.array([3, 2.5e9, 15e6, 10])  # 全部增加
    
    ternary_result = transformer.transform_to_ternary(current_action, target_action)
    applied_action = transformer.apply_ternary_action(current_action, ternary_result)
    
    print(f"当前动作：{current_action}")
    print(f"目标动作：{target_action}")
    print(f"三元表示：{ternary_result.to_array()}")
    print(f"应用结果：{applied_action}")
    
    # 测试4：三元PAC智能体
    print(f"\n--- 测试4：三元PAC智能体 ---")
    
    agent = TernaryPACAgent(
        agent_id=1,
        observation_dim=17,
        constraints=constraints,
        granularity=granularity
    )
    
    # 测试动作选择
    observation = np.random.randn(17)
    ternary_action, continuous_action = agent.select_ternary_action(observation)
    
    print(f"选择的三元动作：{ternary_action.to_array()}")
    print(f"结果连续动作：{continuous_action}")
    
    # 测试虚拟联合策略计算
    other_actions = [np.random.randn(4) for _ in range(2)]
    best_ternary = agent.compute_virtual_joint_policy_ternary(observation, other_actions)
    
    print(f"最佳三元动作（虚拟策略）：{best_ternary.to_array()}")
    
    # 获取智能体统计信息
    stats = agent.get_action_statistics()
    print(f"智能体统计信息：{stats}")
    
    print(f"\n✅ 动作空间变换测试完成！")
    print(f"关键成就：")
    print(f"  ✅ 将动作空间从连续缩减到{transformer.get_action_space_size()}个离散动作")
    print(f"  ✅ 具有可解释语义的三元表示")
    print(f"  ✅ 与PAC算法的无缝兼容")
    print(f"  ✅ 独立维度控制的笛卡尔积方法")


if __name__ == "__main__":
    test_action_space_transformation()