#!/usr/bin/env python3
"""
多服务联邦学习的马尔可夫决策过程框架

实现了方程(15)-(18)中描述的MDP公式:
- 观察空间: o_{r,t} = {Z_{r,t}(ω), Θ_{r,t}, B_t}
- 动作空间: a_{r,t} = {n_{r,t}, f_{r,t}, B_{r,t}, q_{r,t}}
- 奖励函数: rwd_{r,t} 包含对抗因子 Φ_{r,t}
- 多智能体环境的状态转移动态
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("gym导入失败")

import random
from collections import deque
import copy

try:
    from .optimization_problem import OptimizationConstraints
    from .communication import SystemMetrics, ClientMetrics
    from .quantization import QuantizationModule
except ImportError:
    # 对于直接执行，使用绝对导入
    from optimization_problem import OptimizationConstraints
    from communication import SystemMetrics, ClientMetrics
    from quantization import QuantizationModule


@dataclass
class FLTrainingState:
    """联邦学习训练状态 Z_{r,t}(ω)."""
    
    round_t: int = 0                    # 训练轮次 t
    loss: float = 2.0                   # 模型损失 L_r(ω_{r,t})
    accuracy: float = 0.1               # 模型准确率 Γ_{r,t}
    quantization_level: int = 8         # 量化等级 q_{r,t}
    
    def to_array(self) -> np.ndarray:
        """转换为NumPy数组供强化学习代理使用."""
        return np.array([self.round_t, self.loss, self.accuracy, self.quantization_level])


@dataclass 
class SystemState:
    """系统状态 Θ_{r,t}."""
    
    total_delay: float = 0.0            # T_{r,t}^total
    total_energy: float = 0.0           # E_{r,t}^total  
    communication_volume: int = 0       # vol_{r,t}
    
    def to_array(self) -> np.ndarray:
        """转换为NumPy数组供强化学习代理使用."""
        return np.array([self.total_delay, self.total_energy, self.communication_volume])


@dataclass
class Observation:
    """服务提供商r在轮次t的完整观察 o_{r,t}."""
    
    service_id: int
    fl_state: FLTrainingState = field(default_factory=FLTrainingState)
    system_state: SystemState = field(default_factory=SystemState)
    bandwidth_allocations: Dict[int, float] = field(default_factory=dict)  # 所有服务的带宽分配 B_t
    
    def to_array(self) -> np.ndarray:
        """转换为扁平化NumPy数组，实现公式(15): o_{r,t} = {Z_{r,t}(ω), Θ_{r,t}, B_t}"""
        # Z_{r,t}(ω) = {t, L_r(ω_{r,t}), Γ_{r,t}, q_{r,t}} - FL训练状态(4个元素)
        z_rt = self.fl_state.to_array()
        
        # Θ_{r,t} = {T_{r,t}^total, E_{r,t}^total, vol_{r,t}} - 系统状态(3个元素)
        theta_rt = self.system_state.to_array()
        
        # B_t = {B_{r,t}}_{r∈R} - 所有服务的带宽分配
        # 按服务ID顺序排列带宽分配
        max_services = len(self.bandwidth_allocations) if self.bandwidth_allocations else 3
        bandwidth_array = np.zeros(max_services)
        for service_id, bandwidth in self.bandwidth_allocations.items():
            if service_id < max_services:
                bandwidth_array[service_id] = bandwidth
        
        # 归一化处理
        z_rt_normalized = np.array([
            z_rt[0] / 100.0,    # 轮次归一化
            z_rt[1],            # 损失值
            z_rt[2],            # 准确率
            z_rt[3] / 32.0      # 量化级别归一化
        ])
        
        theta_rt_normalized = np.array([
            theta_rt[0] / 100.0,    # 延迟归一化
            theta_rt[1] / 1000.0,   # 能耗归一化  
            theta_rt[2] / 100000.0  # 通信量归一化
        ])
        
        # 带宽按系统约束最大值归一化（与动作空间一致）
        max_bw = 1.0
        try:
            # 在环境中可访问到约束的最大带宽；若不可用，退化到论文30MHz
            max_bw = float(30e6)
        except Exception:
            max_bw = float(30e6)
        bandwidth_normalized = bandwidth_array / max_bw
        
        # 连接所有部分: Z_{r,t} + Θ_{r,t} + B_t
        obs_array = np.concatenate([z_rt_normalized, theta_rt_normalized, bandwidth_normalized])
        return obs_array
    
    @property
    def observation_size(self) -> int:
        """观察向量的大小."""
        max_services = len(self.bandwidth_allocations) if self.bandwidth_allocations else 3
        return 4 + 3 + max_services  # Z_{r,t} + Θ_{r,t} + B_t


@dataclass
class ClientConfig:
    """客户端配置，用于计算能量和延迟"""
    mu_i: float = 1e-28       # 有效电容常数（公式8）
    c_ir: float = 1000        # 每样本CPU周期数（公式8,9）
    dataset_size: int = 1000  # 本地数据集大小 |D_{i,r}|
    channel_gain: float = 1e-3 # 信道增益 g_{i,t}（公式10）


@dataclass
class Action:
    """
    动作类，对应公式(16): a_{r,t} = {n_{r,t}, f_{r,t}, B_{r,t}, q_{r,t}}
    
    实现论文中的三元动作空间变换：
    - 将连续动作空间离散化为三元表示 {-1, 0, 1}
    - 0表示保持当前值，-1和1分别对应按单位粒度增减当前值
    - 动作空间A' = ∏_m a'_(m)，最多3^4=81种可能性
    """
    n_clients: int          # n_{r,t} - 客户端选择数量
    cpu_frequency: float    # f_{r,t} - CPU频率
    bandwidth: float        # B_{r,t} - 带宽分配
    quantization_level: int # q_{r,t} - 量化级别
    
    def to_array(self) -> np.ndarray:
        """转换为NumPy数组."""
        return np.array([self.n_clients, self.cpu_frequency, self.bandwidth, self.quantization_level])
    
    def to_dict(self) -> dict:
        """转换为字典."""
        return {
            'n_clients': self.n_clients,
            'cpu_frequency': self.cpu_frequency,
            'bandwidth': self.bandwidth,
            'quantization_level': self.quantization_level
        }
    
    
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Action':
        """从动作数组创建Action对象（使用物理量级的退化边界）。"""
        return cls(
            n_clients=int(np.clip(array[0], 1, 100)),
            cpu_frequency=float(np.clip(array[1], 1e8, 3e9)),
            bandwidth=float(np.clip(array[2], 1e5, 1e8)),
            quantization_level=int(np.clip(array[3], 1, 32))
        )

    @classmethod
    def from_array_with_constraints(cls, array: np.ndarray, constraints: OptimizationConstraints) -> 'Action':
        """结合系统约束从数组创建Action对象（统一到物理单位边界）。"""
        return cls(
            n_clients=int(np.clip(array[0], constraints.min_clients, constraints.max_clients)),
            cpu_frequency=float(np.clip(array[1], constraints.min_frequency, constraints.max_frequency)),
            bandwidth=float(np.clip(array[2], constraints.min_bandwidth, constraints.max_bandwidth)),
            quantization_level=int(np.clip(array[3], constraints.min_quantization, constraints.max_quantization))
        )



class AdversarialFactor:
    """
    实现方程(18)中的对抗因子 Φ_{r,t}(q).
    
    模拟服务提供商之间对通信资源的竞争.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        初始化对抗因子.
        
        参数:
            epsilon: 小常数，防止除零
        """
        self.epsilon = epsilon
    
    def calculate(self, 
                 service_id: int,
                 actions: Dict[int, Action],
                 communication_volumes: Dict[int, int]) -> float:
        """
        计算服务r的对抗因子.
        
        实现: Φ_{r,t}(q) = (n_{r,t} * q_{r,t}) / (ε * vol_{r,t} + Σ_{j∈R/{r}} n_{j,t} * q_{j,t})
        
        参数:
            service_id: 服务提供商ID r
            actions: 所有服务提供商的动作
            communication_volumes: 所有服务的通信量
            
        返回:
            对抗因子值
        """
        if service_id not in actions:
            return 0.0
        
        current_action = actions[service_id]
        current_volume = communication_volumes.get(service_id, 1)
        
        # 分子: n_{r,t} * q_{r,t}
        numerator = current_action.n_clients * current_action.quantization_level
        
        # 分母: ε * vol_{r,t} + Σ_{j∈R/{r}} n_{j,t} * q_{j,t}
        other_services_sum = 0
        for other_service_id, other_action in actions.items():
            if other_service_id != service_id:
                other_services_sum += other_action.n_clients * other_action.quantization_level
        
        denominator = self.epsilon * current_volume + other_services_sum
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class RewardFunction:
    """
    实现方程(17)中的奖励函数，按照论文参数表设置权重.
    
    rwd_{r,t} = σ_1 * Γ_{r,t} + σ_2 * Φ_{r,t}(q) - σ_3 * E_{r,t}^total - σ_4 * T_{r,t}^total
    
    论文权重因子：
    - σ_1: [100, 100, 100] for [r1, r2, r3]
    - σ_2: [4.8, 31.25, 12.5] for [r1, r2, r3]
    - σ_3, σ_4: [0.8, 25, 16.6] for [r1, r2, r3]
    """
    
    def __init__(self, 
                 sigma_1: float = 100.0,   # 准确率权重（论文默认值）
                 sigma_2: float = 4.8,     # 对抗因子权重（论文r1默认值）
                 sigma_3: float = 0.8,     # 能量惩罚权重（论文r1默认值）
                 sigma_4: float = 0.8,     # 延迟惩罚权重（论文r1默认值）
                 constraints: Optional[OptimizationConstraints] = None):
        """
        初始化奖励函数，使用论文参数表中的权重.
        
        参数:
            sigma_1: 准确率项权重（论文：100）
            sigma_2: 对抗因子权重（论文：r1=4.8, r2=31.25, r3=12.5）
            sigma_3: 能量惩罚权重（论文：r1=0.8, r2=25, r3=16.6）
            sigma_4: 延迟惩罚权重（论文：r1=0.8, r2=25, r3=16.6）
        """
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3
        self.sigma_4 = sigma_4
        # 奖励中的能耗/时延采用“基于约束的归一化”，避免数量级失衡
        self.constraints = constraints
        
        self.adversarial_factor = AdversarialFactor()
        
        print(f"🎯 奖励函数初始化 - σ₁={sigma_1}, σ₂={sigma_2}, σ₃={sigma_3}, σ₄={sigma_4} | 归一化: {'constraints' if constraints else 'none'}")
    
    @classmethod
    def create_for_service(cls, service_id: int, constraints: Optional[OptimizationConstraints] = None) -> 'RewardFunction':
        """
        为特定服务创建奖励函数，使用论文中对应的权重因子.
        
        参数:
            service_id: 服务提供商ID (1, 2, 3)
            
        返回:
            配置好的奖励函数实例
        """
        # 论文参数表中的权重因子
        sigma_1_values = [100.0, 100.0, 100.0]       # 所有服务相同
        sigma_2_values = [4.8, 31.25, 12.5]          # 不同服务不同值
        sigma_3_values = [0.8, 25.0, 16.6]           # 不同服务不同值  
        sigma_4_values = [0.8, 25.0, 16.6]           # 与sigma_3相同
        
        # 服务ID从1开始，数组索引从0开始
        idx = service_id - 1
        if idx < 0 or idx >= 3:
            print(f"⚠️  警告：服务ID {service_id} 超出范围，使用默认权重")
            idx = 0
        
        return cls(
            sigma_1=sigma_1_values[idx],
            sigma_2=sigma_2_values[idx],
            sigma_3=sigma_3_values[idx],
            sigma_4=sigma_4_values[idx],
            constraints=constraints
        )
    
    def calculate(self,
                 service_id: int,
                 observation: Observation,
                 action: Action,
                 all_actions: Dict[int, Action],
                 communication_volumes: Dict[int, int]) -> float:
        """
        计算服务提供商r在当前轮次的奖励，实现公式(17).
        
        参数:
            service_id: 服务提供商ID
            observation: 当前观察
            action: 采取的动作
            all_actions: 所有服务提供商的动作
            communication_volumes: 所有服务的通信量
            
        返回:
            奖励值
        """
        # 从观察中提取组件
        accuracy = observation.fl_state.accuracy  # Γ_{r,t}
        total_energy = observation.system_state.total_energy  # E_{r,t}^total
        total_delay = observation.system_state.total_delay    # T_{r,t}^total
        
        # 计算对抗因子 Φ_{r,t}(q) - 公式(18)
        adversarial_value = self.adversarial_factor.calculate(
            service_id, all_actions, communication_volumes
        )
        
        # 将能量与时延按系统约束进行归一化，避免数量级爆炸
        if self.constraints is not None:
            max_e = max(1e-12, float(self.constraints.max_energy))
            max_t = max(1e-12, float(self.constraints.max_delay))
            normalized_energy = total_energy / max_e
            normalized_delay = total_delay / max_t
        else:
            # 回退：不做单位放大，直接使用物理量（J与s），建议传入constraints
            normalized_energy = total_energy
            normalized_delay = total_delay

        # 稳健性：裁剪到[0, 2]范围内，避免异常值主导训练；适度放宽上界以保留波动
        normalized_energy = float(np.clip(normalized_energy, 0.0, 2.0))
        normalized_delay = float(np.clip(normalized_delay, 0.0, 2.0))
        
        # 计算奖励 - 公式(17): rwd_{r,t} = σ₁Γ_{r,t} + σ₂Φ_{r,t}(q) - σ₃E_{r,t}^total - σ₄T_{r,t}^total
        reward = (self.sigma_1 * accuracy +
                 self.sigma_2 * adversarial_value -
                 self.sigma_3 * normalized_energy -
                 self.sigma_4 * normalized_delay)
        
        return reward


class MultiServiceFLEnvironment:
    """
    多服务联邦学习环境.
    
    实现多服务提供商联邦学习的MDP框架.
    """
    
    def __init__(self,
                 service_ids: List[int],
                 constraints: OptimizationConstraints,
                 max_rounds: int = 100,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        初始化多服务FL环境.
        
        参数:
            service_ids: 服务提供商ID列表
            constraints: 系统约束
            max_rounds: 最大训练轮数
            reward_weights: 自定义奖励函数权重
        """
        # super().__init__()  # 移除gym.Env继承
        
        self.service_ids = service_ids
        self.num_services = len(service_ids)
        self.constraints = constraints
        self.max_rounds = max_rounds
        
        # 初始化组件
        self.quantization_module = QuantizationModule()
        self.system_metrics = SystemMetrics()
        
        # 为每个服务初始化奖励函数（按照论文参数表）
        self.reward_functions = {}
        for service_id in service_ids:
            # 使用论文中特定服务的权重因子，并注入约束用于归一化
            self.reward_functions[service_id] = RewardFunction.create_for_service(service_id, constraints=self.constraints)
            
            # 如果提供了自定义权重，则覆盖
            if reward_weights:
                custom_weights = reward_weights.get(str(service_id), reward_weights)
                self.reward_functions[service_id] = RewardFunction(
                    sigma_1=custom_weights.get('sigma_1', 100.0),
                    sigma_2=custom_weights.get('sigma_2', 4.8),
                    sigma_3=custom_weights.get('sigma_3', 0.8),
                    sigma_4=custom_weights.get('sigma_4', 0.8),
                    constraints=self.constraints
                )
        
        # 定义观察和动作空间
        self._setup_spaces()
        
        # 初始化客户端配置
        self._setup_clients()
        
        # 环境状态
        self.current_round = 0
        self.observations = {}
        self.actions_history = deque(maxlen=10)
        self.rewards_history = deque(maxlen=100)
        
        # 初始化观察
        self.reset()
    
    def _setup_clients(self):
        """设置客户端配置和服务映射"""
        # 客户端配置（基于论文的典型值）
        self.client_configs = {}
        for i in range(1, 6):  # 5个客户端
            self.client_configs[i] = ClientConfig(
                mu_i=1e-28 + i * 1e-29,  # 略有差异的电容常数
                c_ir=1000 + i * 100,     # 不同的CPU周期需求
                dataset_size=800 + i * 200,  # 不同的数据集大小
                channel_gain=1e-3 * (0.8 + i * 0.1)  # 不同的信道条件
            )
        
        # 服务-客户端映射（每个服务关联不同客户端）
        self.service_client_mapping = {
            1: [1, 2, 5],      # 服务1关联客户端1,2,5
            2: [2, 3, 4],      # 服务2关联客户端2,3,4
            3: [1, 3, 4, 5]    # 服务3关联客户端1,3,4,5
        }
    
    def _setup_spaces(self):
        """设置观察和动作空间，严格按照论文公式(15)和(16)."""
        # 动作空间(物理单位，来源于 OptimizationConstraints)
        self.action_space = spaces.Box(
            low=np.array([
                self.constraints.min_clients,
                self.constraints.min_frequency,
                self.constraints.min_bandwidth,
                self.constraints.min_quantization
            ], dtype=np.float32),
            high=np.array([
                self.constraints.max_clients,
                self.constraints.max_frequency,
                self.constraints.max_bandwidth,
                self.constraints.max_quantization
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # 观察空间(公式15): o_{r,t} = {Z_{r,t}(ω), Θ_{r,t}, B_t}
        obs_dim = 4 + 3 + self.num_services
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
    
    def select_clients(self, service_id: int, action: Action) -> List[int]:
        """根据action.n_clients动态选择客户端，实现聚合权重 κ_{i,r}"""
        available_clients = self.service_client_mapping[service_id]
        return random.sample(available_clients, min(action.n_clients, len(available_clients)))
    
    def calculate_computation_energy(self, client_id: int, service_id: int, action: Action) -> float:
        """实现公式(8): E_{i,r,t}^{cmp} = μ_i * c_{i,r} * |D_{i,r}| * f_{i,r,t}^2"""
        client_config = self.client_configs[client_id]
        mu_i = client_config.mu_i
        c_ir = client_config.c_ir  
        dataset_size = client_config.dataset_size
        frequency = action.cpu_frequency
        return mu_i * c_ir * dataset_size * (frequency ** 2)
    
    def calculate_computation_delay(self, client_id: int, service_id: int, action: Action) -> float:
        """实现公式(9): T_{i,r,t}^{cmp} = c_{i,r} * |D_{i,r}| / f_{i,r,t}"""
        client_config = self.client_configs[client_id]
        c_ir = client_config.c_ir
        dataset_size = client_config.dataset_size
        frequency = action.cpu_frequency
        return c_ir * dataset_size / frequency
    
    def calculate_adversarial_factor(self, service_id: int, action: Action, all_actions: Dict[int, Action]) -> float:
        """实现公式(18)的对抗因子Φ_{r,t}(q)"""
        epsilon = 0.01  # 常数ε
        
        # 当前服务的 n_{r,t} * q_{r,t}
        current_nq = action.n_clients * action.quantization_level
        
        # 当前服务的通信量 vol_{r,t}
        # 简化计算，实际应该从通信模块获取
        model_size = 50000 + service_id * 10000
        bits_per_param = max(1, np.ceil(np.log2(action.quantization_level)) + 1)
        current_volume = model_size * bits_per_param + 32
        
        # 其他服务的 Σ_{j∈R/{r}} n_{j,t} * q_{j,t}
        other_services_nq = 0
        for other_service_id, other_action in all_actions.items():
            if other_service_id != service_id:
                other_services_nq += other_action.n_clients * other_action.quantization_level
        
        # Φ_{r,t}(q) = (n_{r,t} * q_{r,t}) / (ε * vol_{r,t} + Σ_{j∈R/{r}} n_{j,t} * q_{j,t})
        denominator = epsilon * current_volume + other_services_nq
        adversarial_factor = current_nq / max(denominator, 1e-6)  # 避免除零
        
        return adversarial_factor
    
    def check_constraints(self, service_id: int, action: Action, current_obs: Observation) -> bool:
        """检查所有约束条件C1-C5"""
        # C1: 能耗和延迟约束（使用系统约束）
        if (current_obs.system_state.total_energy > self.constraints.max_energy or
            current_obs.system_state.total_delay > self.constraints.max_delay):
            return False
        
        # C2: 客户端选择范围 n_{r,t} ∈ [min, max]
        if not (self.constraints.min_clients <= action.n_clients <= self.constraints.max_clients):
            return False
        
        # C3: CPU频率范围 f^min ≤ f_{r,t} ≤ f^max
        if not (self.constraints.min_frequency <= action.cpu_frequency <= self.constraints.max_frequency):
            return False
        
        # C4: 带宽约束 B^min ≤ Σ_r B_{r,t} ≤ B^max（此处仅检查单个动作在边界内）
        # 这里暂时跳过全局带宽检查，因为需要所有服务的信息
        
        # C5: 量化级别范围 q_{r,t}
        if not (self.constraints.min_quantization <= action.quantization_level <= self.constraints.max_quantization):
            return False
        
        return True
    
    def reset(self) -> Dict[int, np.ndarray]:
        """
        重置环境到初始状态.
        
        返回:
            所有服务的初始观察
        """
        self.current_round = 0
        self.observations = {}
        self.actions_history.clear()
        self.rewards_history.clear()
        
        # 为每个服务初始化观察
        for service_id in self.service_ids:
            # 使用默认值初始化
            fl_state = FLTrainingState(
                round_t=0,
                loss=2.0,  # 初始高损失
                accuracy=0.1,  # 初始低准确率
                quantization_level=8
            )
            
            system_state = SystemState(
                total_delay=0.0,
                total_energy=0.0,
                communication_volume=0
            )
            
            # 平等的初始带宽分配
            initial_bandwidth = self.constraints.max_bandwidth / self.num_services
            bandwidth_allocations = {sid: initial_bandwidth for sid in self.service_ids}
            
            observation = Observation(
                service_id=service_id,
                fl_state=fl_state,
                system_state=system_state,
                bandwidth_allocations=bandwidth_allocations
            )
            
            self.observations[service_id] = observation
        
        # 返回观察数组
        return {service_id: obs.to_array() for service_id, obs in self.observations.items()}
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], 
                                                          Dict[int, float], 
                                                          Dict[int, bool], 
                                                          Dict[int, Dict]]:
        """
        执行环境中的一个步骤.
        
        参数:
            actions: 每个服务提供商的动作
            
        返回:
            元组(observations, rewards, dones, infos)
        """
        # 将NumPy数组转换为Action对象，严格按照公式(16)
        action_objects = {}
        for service_id, action_array in actions.items():
            # 离散化整数动作
            action_array_discrete = action_array.copy()
            action_array_discrete[0] = round(action_array_discrete[0])  # n_clients
            action_array_discrete[3] = round(action_array_discrete[3])  # quantization_level
            action_objects[service_id] = Action.from_array_with_constraints(action_array_discrete, self.constraints)
        
        # 将动作存储在历史记录中
        self.actions_history.append(action_objects.copy())
        
        # 模拟联邦学习轮次
        new_observations, communication_volumes = self._simulate_fl_round(action_objects)
        
        # 计算奖励
        rewards = {}
        for service_id in self.service_ids:
            reward = self.reward_functions[service_id].calculate(
                service_id=service_id,
                observation=new_observations[service_id],
                action=action_objects[service_id],
                all_actions=action_objects,
                communication_volumes=communication_volumes
            )
            rewards[service_id] = reward
        
        # 存储奖励
        self.rewards_history.append(rewards.copy())
        
        # 更新当前观察
        self.observations = new_observations
        
        # 检查回合是否结束
        self.current_round += 1
        dones = {service_id: self.current_round >= self.max_rounds for service_id in self.service_ids}
        
        # 附加信息
        infos = {}
        for service_id in self.service_ids:
            infos[service_id] = {
                'round': self.current_round,
                'communication_volume': communication_volumes.get(service_id, 0),
                'constraints_satisfied': self._check_constraints(service_id, action_objects[service_id])
            }
        
        # 将观察转换为数组
        observation_arrays = {service_id: obs.to_array() for service_id, obs in new_observations.items()}
        
        return observation_arrays, rewards, dones, infos
    
    def _simulate_fl_round(self, actions: Dict[int, Action]) -> Tuple[Dict[int, Observation], Dict[int, int]]:
        """
        参数:
            actions: 所有服务提供商采取的动作
            
        返回:
            元组(new_observations, communication_volumes)
        """
        new_observations = {}
        communication_volumes = {}
        
        for service_id in self.service_ids:
            action = actions[service_id]
            current_obs = self.observations[service_id]
            
            # 模拟模型训练进展
            # 更多客户端和更高的量化 -> 更好的准确率，更高的成本
            accuracy_improvement = (
                0.01 * np.log(max(1, action.n_clients)) +
                0.005 * np.log(max(1, action.quantization_level)) +
                0.002 * np.log(max(1e-6, action.cpu_frequency / 1e9)) +      # 频率影响（GHz对数）
                0.002 * np.log(1.0 + action.bandwidth / 1e6) +               # 带宽影响（MHz对数）
                np.random.normal(0, 0.02)  # 略增噪声以体现探索波动
            )
            
            new_accuracy = min(1.0, current_obs.fl_state.accuracy + accuracy_improvement)
            new_loss = max(0.1, current_obs.fl_state.loss * (1 - accuracy_improvement))
            
            # 使用量化模块计算通信量，实现公式(7)
            model_size = 50000 + service_id * 10000  # 不同模型大小
            # 通信位宽上限限制为8bit，避免过大位宽导致能耗/时延失真
            bits_per_param = int(min(8, max(1, np.ceil(np.log2(action.quantization_level)) + 1)))
            comm_volume = model_size * bits_per_param + 32
            communication_volumes[service_id] = comm_volume
            
            # 动态选择客户端，实现聚合权重 κ_{i,r}
            selected_clients = self.select_clients(service_id, action)
            
            # 初始化累积能量和延迟
            total_comp_energy = 0.0
            total_comm_energy = 0.0
            client_delays = []
            
            # 对每个选中的客户端计算精确的能量和延迟
            for client_id in selected_clients:
                # 计算计算能量（公式8）
                comp_energy = self.calculate_computation_energy(client_id, service_id, action)
                
                # 简化通信能量计算（由于没有传输功率参数）
                client_config = self.client_configs[client_id]
                
                # 传输速率简化计算
                bandwidth = action.bandwidth
                channel_gain = client_config.channel_gain
                default_power = 0.1  # 默认传输功率
                noise_power = 1e-9

                # 避免除零错误：当带宽为0或极小时，使用最小带宽值
                min_safe_bandwidth = 1e3  # 1kHz 最小安全带宽
                safe_bandwidth = max(bandwidth, min_safe_bandwidth)

                snr = (channel_gain * default_power) / (safe_bandwidth * noise_power)
                transmission_rate = safe_bandwidth * np.log2(1 + snr) if snr > 0 else 1e3
                
                # 通信延迟（公式11）
                comm_delay = comm_volume / transmission_rate
                
                # 通信能量（简化）
                comm_energy = comm_delay * default_power
                
                total_comp_energy += comp_energy
                total_comm_energy += comm_energy
                
                # 计算延迟（公式9）
                comp_delay = self.calculate_computation_delay(client_id, service_id, action)
                
                client_delays.append(comp_delay + comm_delay)
            
            # 总能量（公式13a）：E_{r,t}^{total} = Σ(E_{i,r,t}^{com} + E_{i,r,t}^{cmp})
            total_energy = total_comp_energy + total_comm_energy
            
            # 总延迟（公式13b）：T_{r,t}^{total} = max(T_{i,r,t}^{cmp} + T_{i,r,t}^{com})
            total_delay = max(client_delays) if client_delays else 0.0
            
            # 创建新的观察
            new_fl_state = FLTrainingState(
                round_t=self.current_round,
                loss=new_loss,
                accuracy=new_accuracy,
                quantization_level=action.quantization_level
            )
            
            new_system_state = SystemState(
                total_delay=total_delay,
                total_energy=total_energy,
                communication_volume=comm_volume
            )
            
            # 更新带宽分配(公开信息)
            bandwidth_allocations = {sid: actions[sid].bandwidth for sid in self.service_ids}
            
            new_observation = Observation(
                service_id=service_id,
                fl_state=new_fl_state,
                system_state=new_system_state,
                bandwidth_allocations=bandwidth_allocations
            )
            
            new_observations[service_id] = new_observation
        
        return new_observations, communication_volumes
    
    def _check_constraints(self, service_id: int, action: Action) -> bool:
        """检查动作是否满足约束."""
        obs = self.observations[service_id]
        
        # 能量约束
        if obs.system_state.total_energy > self.constraints.max_energy:
            return False
        
        # 延迟约束
        if obs.system_state.total_delay > self.constraints.max_delay:
            return False
        
        # 动作边界
        if not (self.constraints.min_clients <= action.n_clients <= self.constraints.max_clients):
            return False
        
        if not (self.constraints.min_frequency <= action.cpu_frequency <= self.constraints.max_frequency):
            return False
        
        if not (self.constraints.min_bandwidth <= action.bandwidth <= self.constraints.max_bandwidth):
            return False
        
        if not (self.constraints.min_quantization <= action.quantization_level <= self.constraints.max_quantization):
            return False
        
        return True
    
    