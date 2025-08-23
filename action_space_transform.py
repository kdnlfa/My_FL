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
    