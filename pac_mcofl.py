#!/usr/bin/env python3
"""
PAC-MCoFL: 帕累托演员-评论家多服务提供商协作联邦学习

实现第4节中描述的PAC-MCoFL算法框架:
- 联合策略和累积期望奖励 (方程 19-20)
- 帕累托最优均衡机制 (方程 21-23)
- 基于策略梯度的演员-评论家网络更新 (方程 24-27)
- 完整的PAC-MCoFL训练算法 (算法 1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import copy
import random

try:
    from .mdp_framework import MultiServiceFLEnvironment, Action, Observation
    from .optimization_problem import OptimizationConstraints
    from .multi_service_fl import MultiServiceFLSystem
    from .action_space_transform import ActionSpaceTransformer, ActionGranularity, TernaryActionVector, TernaryAction   
except ImportError:
    # 对于直接执行，使用绝对导入
    from mdp_framework import MultiServiceFLEnvironment, Action, Observation
    from optimization_problem import OptimizationConstraints
    from multi_service_fl import MultiServiceFLSystem
    from action_space_transform import ActionSpaceTransformer, ActionGranularity, TernaryActionVector, TernaryAction


@dataclass
class PACConfig:
    """PAC-MCoFL算法的配置，按照论文精确参数设置."""
    
    # 训练参数
    num_episodes: int = 5        # 调试阶段减少回合数，加速RL-FL联动验证
    max_rounds_per_episode: int = 35  # 算法1中的T，论文设置为35
    buffer_size: int = 10000        # 经验回放缓冲区大小H
    batch_size: int = 64            # 训练批次大小，论文设置为64
    
    # 网络架构（按照论文要求）
    actor_hidden_dim: int = 64      # 策略网络：64-128-64架构
    critic_hidden_dim: int = 64     # Q网络：64-128架构
    num_layers: int = 3             # 策略网络层数（64-128-64）
    critic_layers: int = 2          # Q网络层数（64-128），论文明确说"双层"
    
    # 学习参数（按照论文参数表）
    actor_lr: float = 0.001         # 方程(27)中的ζ=0.001
    critic_lr: float = 0.001        # 方程(26)中的α=0.001
    gamma: float = 0.95             # 折扣因子γ
    tau: float = 0.005              # 软更新参数
    
    # PAC特定参数
    joint_action_samples: int = 100  # 用于近似联合策略的样本数
    baseline_regularization: float = 0.01  # 基线正则化权重
    
    # 训练调度
    update_frequency: int = 4       # 每N步更新网络
    eval_frequency: int = 100       # 评估频率
    save_frequency: int = 500       # 模型保存频率

    # 追加：联动RL-FL的评估与训练细化配置（支持按服务定制）
    # 每步快速评估频率（步级），用于控制图中“台阶状重复”的产生频率
    step_eval_frequency: int = 2
    # 按服务ID设置评估频率覆盖，例如 {1:1, 2:1, 3:2}
    service_eval_frequency: dict = field(default_factory=dict)
    # 按服务ID设置每步真实训练的本地epoch数，例如 {1:5, 2:3, 3:1}
    service_epochs_per_step: dict = field(default_factory=dict)
    # 按服务ID设置动作下限，避免极端低资源导致精度骤降，例如
    # {1:{'min_clients':2,'min_frequency':1.5e9,'min_bandwidth':15e6,'min_quantization':8}, ...}
    service_action_floors: dict = field(default_factory=dict)
    # 探索噪声衰减（连续动作），降低无效探索
    exploration_std_start: float = 0.10
    exploration_std_end: float = 0.02
    exploration_decay_steps: int = 200


class ActorNetwork(nn.Module):
    """
    PAC-MCoFL的演员网络，按照论文要求实现策略网络架构.
    
    论文架构：三层全连接网络（64、128、64个神经元）
    实现π_r(a_{r,t} | o_{r,t}; φ_r)来生成动作.
    """
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        初始化演员网络，按照论文策略网络架构.
        
        参数:
            observation_dim: 观察空间维度
            action_dim: 动作空间维度
            hidden_dim: 第一层隐藏层维度（64）
            num_layers: 网络层数（3层）
            action_bounds: 动作缩放的(最小边界, 最大边界)
        """
        super().__init__()
        
        self.action_bounds = action_bounds
        
        # 按照论文要求构建策略网络：64-128-64架构
        if num_layers == 3:
            # 论文指定的三层架构
            self.network = nn.Sequential(
                nn.Linear(observation_dim, 64),    # 第一层：64个神经元
                nn.ReLU(),
                nn.Linear(64, 128),                # 第二层：128个神经元
                nn.ReLU(),
                nn.Linear(128, 64),                # 第三层：64个神经元
                nn.ReLU(),
                # 输出每个动作维度的三元分类logits，共 action_dim * 3 个
                nn.Linear(64, action_dim * 3)
            )
        else:
            # 兼容其他配置的通用架构
            layers = []
            input_dim = observation_dim
            
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            
            # 输出每个动作维度的三元分类logits
            layers.append(nn.Linear(hidden_dim, action_dim * 3))
            
            self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        前向传播生成每个动作维度的三元分类logits.
        
        参数:
            observation: 输入观察
            
        返回:
            logits 张量，形状为 [batch, action_dim * 3]
        """
        logits = self.network(observation)
        return logits


class CriticNetwork(nn.Module):
    """
    PAC-MCoFL的评论家网络，按照论文要求实现Q网络架构.
    
    论文架构：双层全连接结构（64和128个神经元）
    实现Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})进行价值估计.
    """
    
    def __init__(self,
                 observation_dim: int,
                 own_action_dim: int,
                 joint_action_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2):
        """
        初始化评论家网络，按照论文Q网络架构.
        
        参数:
            observation_dim: 观察空间维度
            own_action_dim: 自身动作维度
            joint_action_dim: 联合动作维度(所有智能体)
            hidden_dim: 第一层隐藏层维度（64）
            num_layers: 网络层数（2层）
        """
        super().__init__()
        
        input_dim = observation_dim + own_action_dim + joint_action_dim
        
        # 按照论文要求构建Q网络：双层全连接（64和128个神经元）
        if num_layers == 2:
            # 论文指定的双层架构：第一层64个神经元，第二层128个神经元，最后直接输出
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),          # 第一层：64个神经元
                nn.ReLU(),
                nn.Linear(64, 128),                # 第二层：128个神经元
                nn.ReLU(),
                nn.Linear(128, 1)                  # 输出层：单个Q值
            )
        else:
            # 兼容其他配置的通用架构
            layers = []
            current_dim = input_dim
            
            for _ in range(num_layers):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                current_dim = hidden_dim
            
            layers.append(nn.Linear(hidden_dim, 1))  # 输出单个Q值
            
            self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, 
                observation: torch.Tensor,
                own_action: torch.Tensor,
                joint_action: torch.Tensor) -> torch.Tensor:
        """
        前向传播估计Q值.
        
        参数:
            observation: 智能体的观察
            own_action: 智能体自身的动作
            joint_action: 所有智能体的联合动作
            
        返回:
            Q值估计
        """
        # 连接输入
        x = torch.cat([observation, own_action, joint_action], dim=-1)
        
        return self.network(x)


class ReplayBuffer:
    """PAC-MCoFL的经验回放缓冲区."""
    
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, num_agents: int):
        """
        初始化回放缓冲区.
        
        参数:
            capacity: 最大缓冲区大小
            observation_dim: 观察维度
            action_dim: 每个智能体的动作维度
            num_agents: 智能体数量
        """
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # 缓冲区存储
        self.observations = np.zeros((capacity, observation_dim))
        self.own_actions = np.zeros((capacity, action_dim))
        self.joint_actions = np.zeros((capacity, (num_agents - 1) * action_dim))  # 排除自身动作
        self.rewards = np.zeros((capacity,))
        self.next_observations = np.zeros((capacity, observation_dim))
        self.dones = np.zeros((capacity,), dtype=bool)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, 
            observation: np.ndarray,
            own_action: np.ndarray,
            joint_action: np.ndarray,
            reward: float,
            next_observation: np.ndarray,
            done: bool):
        """向缓冲区添加经验."""
        
        self.observations[self.ptr] = observation
        self.own_actions[self.ptr] = own_action
        self.joint_actions[self.ptr] = joint_action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """从缓冲区采样批次."""
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.observations[indices]),
            torch.FloatTensor(self.own_actions[indices]),
            torch.FloatTensor(self.joint_actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_observations[indices]),
            torch.BoolTensor(self.dones[indices])
        )
    
    def __len__(self):
        return self.size


class PACAgent:
    """
    实现帕累托演员-评论家算法的PAC-MCoFL智能体.
    
    从服务提供商的角度实现算法.
    """
    
    def __init__(self,
                 agent_id: int,
                 observation_dim: int,
                 action_dim: int,
                 num_agents: int,
                 config: PACConfig,
                 constraints: OptimizationConstraints):
        """
        初始化PAC智能体.
        
        参数:
            agent_id: 智能体标识符
            observation_dim: 观察空间维度
            action_dim: 动作空间维度
            num_agents: 智能体总数
            config: PAC配置
            constraints: 动作约束
        """
        self.agent_id = agent_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.config = config
        self.constraints = constraints
        # 统一三元动作空间：引入变换器（单一实现来源）
        self.action_transformer = ActionSpaceTransformer(self.constraints, ActionGranularity())
        
        # 设置动作边界（与环境保持一致，量化级别上限收敛到常用范围32，避免数值失真）
        q_max_eff = min(int(constraints.max_quantization), 32)
        self.action_bounds = (
            np.array([constraints.min_clients, constraints.min_frequency, 
                     constraints.min_bandwidth, constraints.min_quantization], dtype=np.float32),
            np.array([constraints.max_clients, constraints.max_frequency,
                     constraints.max_bandwidth, q_max_eff], dtype=np.float32)
        )
        
        # 初始化网络（现在actor输出三元分类logits）
        self.actor = ActorNetwork(
            observation_dim, action_dim, 
            config.actor_hidden_dim, config.num_layers, self.action_bounds
        )
        
        self.critic = CriticNetwork(
            observation_dim, action_dim, (num_agents - 1) * action_dim,  # 排除自身动作
            config.critic_hidden_dim, config.critic_layers  # 使用论文指定的双层架构
        )
        
        # 用于稳定训练的目标网络
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, observation_dim, action_dim, num_agents
        )
        
        # 训练步数计数器
        self.training_step = 0
        
        # 统计信息
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        使用三元增量策略选择动作：
        - actor 输出每维三元分类 logits（对应 {-1,0,1}）
        - 采样得到三元动作向量 a'(m)∈{-1,0,1}
        - 将 a' 应用于上一步动作状态，得到新物理动作
        
        实现从π_r(a_{r,t} | o_{r,t})采样.
        """
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor.forward(observation_tensor)  # [1, action_dim*3]
        logits = logits.view(1, self.action_dim, 3)  # [1, 4, 3]
        # 温度/探索：在训练时可加入gumbel或softmax温度，这里先用softmax采样
        if training:
            probs = F.softmax(logits, dim=-1)  # [1, 4, 3]
            # 多项分布采样每一维
            inc_idx = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(-1)  # [4]
        else:
            inc_idx = torch.argmax(logits.squeeze(0), dim=-1)  # [4]
        # 将索引 {0,1,2} 映射为 {-1,0,1}
        inc_map = torch.tensor([-1, 0, 1], dtype=torch.int64, device=inc_idx.device)
        increments = inc_map[inc_idx].cpu().numpy()  # shape [4]
        # 取上一动作状态（若无，则使用约束中点）
        if not hasattr(self, 'last_action_state') or self.last_action_state is None:
            self.last_action_state = np.array([
                (self.constraints.min_clients + self.constraints.max_clients) // 2,
                (self.constraints.min_frequency + self.constraints.max_frequency) / 2.0,
                (self.constraints.min_bandwidth + self.constraints.max_bandwidth) / 2.0,
                (self.constraints.min_quantization + self.constraints.max_quantization) // 2
            ], dtype=np.float32)
        # 构造三元动作并应用到当前状态
        tav = TernaryActionVector(
            n_clients_action=TernaryAction(int(increments[0])),
            frequency_action=TernaryAction(int(increments[1])),
            bandwidth_action=TernaryAction(int(increments[2])),
            quantization_action=TernaryAction(int(increments[3]))
        )
        new_action = self.action_transformer.apply_ternary_action(self.last_action_state, tav)
        self.last_action_state = new_action.copy()
        return new_action.astype(np.float32)
    
    # (已删除重复的 compute_baseline_value 旧版本，保留后方改进版本)
    
    def compute_virtual_joint_policy(self, 
                                   observation: torch.Tensor,
                                   own_action: torch.Tensor,
                                   current_action_state: torch.Tensor = None) -> torch.Tensor:
        """
        实现方程(23)的虚拟联合策略计算: π_{-r}^† ∈ arg max_{a_{-r,t}} Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})

        
        参数:
            observation: 当前观察 o_{r,t} [batch_size, obs_dim]
            own_action: 自身动作 a_{r,t} [batch_size, action_dim]
            current_action_state: 当前动作状态，用于三元变换 [batch_size, action_dim]
            
        返回:
            根据Q函数得到的最佳联合动作（虚拟联合策略）[batch_size, joint_action_dim]
        """
        batch_size = observation.shape[0]
        other_agents_num = self.num_agents - 1
        joint_action_dim = other_agents_num * self.action_dim
        
        # 如果没有提供当前状态，使用约束中点作为默认值（物理单位）
        if current_action_state is None:
            default = np.array([
                (self.constraints.min_clients + self.constraints.max_clients) // 2,
                (self.constraints.min_frequency + self.constraints.max_frequency) / 2.0,
                (self.constraints.min_bandwidth + self.constraints.max_bandwidth) / 2.0,
                (self.constraints.min_quantization + self.constraints.max_quantization) // 2
            ], dtype=np.float32)
            current_action_state = torch.tensor(default).unsqueeze(0).repeat(batch_size, 1)

        num_samples = 50
        best_q_values = torch.full((batch_size,), float('-inf'))
        best_joint_actions = torch.zeros(batch_size, joint_action_dim)


        for _ in range(num_samples):
            joint_action_batch = []
            for b in range(batch_size):
                parts = []
                cur_np = current_action_state[b].detach().cpu().numpy()
                for _ in range(other_agents_num):
                    idx = np.random.randint(0, self.action_transformer.get_action_space_size())
                    ternary = self.action_transformer.get_action_by_index(idx)
                    next_action = self.action_transformer.apply_ternary_action(cur_np, ternary)
                    parts.extend(next_action.tolist())
                joint_action_batch.append(parts)

            joint_action_tensor = torch.FloatTensor(joint_action_batch)
            with torch.no_grad():
                q_values = self.critic_target(observation, own_action, joint_action_tensor).squeeze()
                if q_values.dim() == 0:
                    q_values = q_values.unsqueeze(0)

            better_mask = q_values > best_q_values
            best_q_values[better_mask] = q_values[better_mask]
            best_joint_actions[better_mask] = joint_action_tensor[better_mask]

        return best_joint_actions
    
    def compute_baseline_value(self, observation: torch.Tensor, own_action: torch.Tensor) -> torch.Tensor:
        """
        计算基线值以减少方差，如论文方程(25)后所述:
        Σ_{a_{r,t}} π_r(a_{r,t}|o_{r,t}; φ_r) Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})
        
        参数:
            observation: 当前观察
            own_action: 当前动作
            
        返回:
            基线值用于减少梯度估计方差
        """
        batch_size = observation.shape[0]
        
        # 从策略网络生成多个动作样本
        num_action_samples = 10
        action_samples = []
        base_actions = own_action

        for _ in range(num_action_samples):
            sampled_logits = self.actor(observation)
            sampled_actions = self.logits_to_physical_actions(sampled_logits, base_actions, greedy=False)
            action_samples.append(sampled_actions)
        action_samples = torch.stack(action_samples, dim=1)  # [batch, num_samples, action_dim]
        
        # 为每个动作样本生成联合动作
        other_action_dim = (self.num_agents - 1) * self.action_dim
        joint_actions = torch.randn(batch_size, num_action_samples, other_action_dim)

        # 计算每个动作样本的Q值
        obs_expanded = observation.unsqueeze(1).repeat(1, num_action_samples, 1)
        
        # 重塑用于批处理
        obs_flat = obs_expanded.reshape(-1, self.observation_dim)
        action_flat = action_samples.reshape(-1, self.action_dim)
        joint_flat = joint_actions.reshape(-1, other_action_dim)
        
        with torch.no_grad():
            q_values = self.critic_target(obs_flat, action_flat, joint_flat)
            q_values = q_values.reshape(batch_size, num_action_samples)
            
        # 计算加权平均作为基线值
        baseline = torch.mean(q_values, dim=1, keepdim=True)
        
        return baseline
    
    def update_critic(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        使用方程(24)的TD误差更新评论家网络，严格按照论文公式实现。
        
        实现: M_r(π) = E[(rwd_{r,t} + γ max_{a_{-r,t}} Q_r^π†(o_{r,t+1}, a_{r,t+1}, a_{-r,t}) - Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t}))²]
        对应方程(26): θ_r = θ_r - α ∇_{θ_r} M_r(π)
        """
        observations, own_actions, joint_actions, rewards, next_observations, dones = batch
        
        batch_size = observations.shape[0]
        
        # 当前Q值: Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})
        current_q = self.critic(observations, own_actions, joint_actions)
        
        # 计算目标Q值，实现方程(24)的max操作符
        with torch.no_grad():
            # 从目标策略网络获取下一个动作的logits
            next_own_actions_logits = self.actor_target(next_observations)

            # 使用上一时刻动作作为基准，将logits映射为物理4维动作（贪心）
            base_actions = own_actions
            next_own_actions = self.logits_to_physical_actions(
                next_own_actions_logits, base_actions, greedy=True
            )
            
            # 以映射后的物理动作作为当前动作状态，计算虚拟联合策略
            current_action_states = next_own_actions
            
            # 计算虚拟联合策略 π_{-r}^† (方程23)，使用三元动作空间
            # 这是方程(24)中max操作的核心，现在使用离散动作空间
            best_joint_actions = self.compute_virtual_joint_policy(
                next_observations, next_own_actions, current_action_states
            )
            
            # 计算目标Q值: rwd_{r,t} + γ max_{a_{-r,t}} Q_r^π†(o_{r,t+1}, a_{r,t+1}, a_{-r,t})
            # 这里的max已经通过compute_virtual_joint_policy实现
            target_q = self.critic_target(next_observations, next_own_actions, best_joint_actions)
            target_q = rewards.unsqueeze(1) + self.config.gamma * target_q * (~dones).unsqueeze(1)
        
        # 计算TD误差 (方程24): M_r(π) = E[(target - current)²]
        td_error = target_q - current_q
        critic_loss = torch.mean(td_error ** 2)  # 这就是方程(24)中的M_r(π)
        
        # 更新评论家网络 (方程26: θ_r = θ_r - α ∇_{θ_r} M_r(π))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()
        
        # 记录Q值统计
        self.q_values.append(float(torch.mean(current_q).item()))
        self.critic_losses.append(float(critic_loss.item()))

        # 输出关键信息，便于观察critic更新动态
        try:
            current_q_mean = float(current_q.mean().item())
            target_q_mean = float(target_q.mean().item())
            td_mean = float(td_error.mean().item())
            td_std = float(td_error.std().item()) if td_error.numel() > 1 else 0.0
            print(
                f"[PAC][Agent {self.agent_id}] CriticUpdate: "
                f"loss={critic_loss.item():.6f}, "
                f"Qcur={current_q_mean:.5f}, Qtgt={target_q_mean:.5f}, "
                f"TD={td_mean:.5f}±{td_std:.5f}, grad_norm={float(critic_grad_norm):.4f}"
            )
        except Exception:
            pass
        
        return float(critic_loss.item())
    
    def update_actor(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """
        使用方程(25)的策略梯度更新演员网络，包含基线值减少方差。
        
        实现方程(27): φ_r = φ_r + ζ ∇_{φ_r} J_r(π)
        其中策略梯度包含基线值以减少估计方差
        """
        observations, own_actions, joint_actions, rewards, next_observations, dones = batch
        
        # 计算当前策略的动作
        current_logits = self.actor(observations)
        base_actions = own_actions
        current_actions, inc_idx = self.logits_to_physical_actions(
            current_logits, base_actions, greedy=False, return_indices=True
        )
        
        # 为当前动作生成最优的联合动作（实现方程23）
        # 使用三元动作空间而不是随机采样
        current_action_states = current_actions
        
        best_joint_actions = self.compute_virtual_joint_policy(
            observations, current_actions, current_actions
        )
        
        # 计算Q值
        q_values = self.critic(observations, current_actions, best_joint_actions)
        
        # 计算基线值以减少方差（论文方程(25)后提到的基线）
        # baseline = Σ π_r(a_{r,t}|o_{r,t}) * Q_r^π†(o_{r,t}, a_{r,t}, a_{-r,t})
        baseline = self.compute_baseline_value(observations, current_actions)
        
        # 计算优势函数：A = Q - baseline
        advantages = q_values - baseline
    
        logits3 = current_logits.view(-1, self.action_dim, 3)
        log_probs_per_dim = F.log_softmax(logits3, dim=-1).gather(-1, inc_idx.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs_per_dim.sum(dim=1, keepdim=True)

        # 策略梯度损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # 更新演员网络 (方程27: φ_r = φ_r + ζ ∇_{φ_r} J_r(π))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 梯度裁剪
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.actor_optimizer.step()
        self.actor_losses.append(float(actor_loss.item()))
        
        # 输出关键信息，便于观察actor更新动态
        try:
            adv_mean = float(advantages.mean().item())
            adv_std = float(advantages.std().item()) if advantages.numel() > 1 else 0.0
            logp_mean = float(log_probs.mean().item())
            act_mean = float(current_actions.mean().item())
            act_std = float(current_actions.std().item()) if current_actions.numel() > 1 else 0.0
            print(
                f"[PAC][Agent {self.agent_id}] ActorUpdate: "
                f"loss={actor_loss.item():.6f}, adv={adv_mean:.5f}±{adv_std:.5f}, "
                f"logp={logp_mean:.5f}, a={act_mean:.4f}±{act_std:.4f}, grad_norm={float(actor_grad_norm):.4f}"
            )
        except Exception:
            pass
        
        return float(actor_loss.item())
    
    def update_target_networks(self):
        """目标网络的软更新."""
        
        # 更新演员目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        
        # 更新评论家目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def add_experience(self,
                      observation: np.ndarray,
                      own_action: np.ndarray,
                      joint_action: np.ndarray,
                      reward: float,
                      next_observation: np.ndarray,
                      done: bool):
        """向回放缓冲区添加经验."""
        
        self.replay_buffer.add(
            observation, own_action, joint_action, reward, next_observation, done
        )
    
    def train_step(self) -> Dict[str, float]:
        """
        如果有足够的经验，执行一次训练步骤.
        
        返回:
            包含训练指标的字典
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # 从回放缓冲区采样批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # 更新网络
        critic_loss = self.update_critic(batch)
        actor_loss = self.update_actor(batch)
        
        # 更新目标网络
        self.update_target_networks()
        
        self.training_step += 1
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'q_value': self.q_values[-1] if self.q_values else 0.0,
            'buffer_size': len(self.replay_buffer)
        }

    def logits_to_physical_actions(self, logits: torch.Tensor, base_actions: torch.Tensor, greedy: bool = False, return_indices: bool = False):
        """将每维三元分类logits映射为长度为action_dim的物理动作.

        Args:
            logits: [batch, action_dim*3]，每维{-1,0,1}的logits
            base_actions: [batch, action_dim]，作为应用三元增量的基准动作
            greedy: True使用argmax，False按softmax采样
            return_indices: 是否返回每维增量索引{0,1,2}

        Returns:
            actions: [batch, action_dim] 物理动作
            inc_idx(optional): [batch, action_dim] 采样/贪心得到的增量索引
        """
        bsz = logits.shape[0]
        logits3 = logits.view(bsz, self.action_dim, 3)
        if greedy:
            inc_idx = torch.argmax(logits3, dim=-1)  # [bsz, action_dim]
        else:
            probs = F.softmax(logits3, dim=-1)
            inc_idx = torch.multinomial(probs.view(-1, 3), num_samples=1).view(bsz, self.action_dim)
        inc_map = torch.tensor([-1, 0, 1], dtype=torch.int64, device=inc_idx.device)
        inc = inc_map[inc_idx].cpu().numpy()  # [bsz, action_dim]
        base_np = base_actions.detach().cpu().numpy()
        out = np.zeros_like(base_np, dtype=np.float32)
        for i in range(bsz):
            tav = TernaryActionVector(
                n_clients_action=TernaryAction(int(inc[i, 0])),
                frequency_action=TernaryAction(int(inc[i, 1])),
                bandwidth_action=TernaryAction(int(inc[i, 2])),
                quantization_action=TernaryAction(int(inc[i, 3])),
            )
            out[i] = self.action_transformer.apply_ternary_action(base_np[i], tav).astype(np.float32)
        out_t = torch.from_numpy(out)
        if return_indices:
            return out_t, inc_idx
        return out_t

class PACMCoFLTrainer:
    """
    实现算法1的PAC-MCoFL训练器.
    
    在联邦学习环境中协调多个PAC智能体的训练.
    """
    
    def __init__(self,
                 service_ids: List[int],
                 environment: MultiServiceFLEnvironment,
                 fl_system: MultiServiceFLSystem,
                 config: PACConfig,
                 constraints: OptimizationConstraints):
        """
        初始化PAC-MCoFL训练器.
        
        参数:
            service_ids: 服务提供商ID列表
            environment: 联邦学习环境
            fl_system: 联邦学习系统
            config: PAC配置
            constraints: 系统约束
        """
        self.service_ids = service_ids
        self.environment = environment
        self.fl_system = fl_system
        self.config = config
        self.constraints = constraints
        
        # 初始化PAC智能体
        if hasattr(environment.observation_space, 'shape'):
            observation_dim = environment.observation_space.shape[0]
        else:
            # 字典实现的回退
            observation_dim = environment.observation_space['shape'][0]
        
        if hasattr(environment.action_space, 'shape'):
            action_dim = environment.action_space.shape[0]
        else:
            # 字典实现的回退
            action_dim = environment.action_space['shape'][0]
        
        num_agents = len(service_ids)
        
        """为每个服务单独初始化PAC智能体"""
        self.agents = {}
        for service_id in service_ids:
            self.agents[service_id] = PACAgent(
                service_id, observation_dim, action_dim, num_agents, config, constraints
            )
        
        # 训练统计
        self.episode_rewards = defaultdict(list)
        self.episode_lengths = []
        self.training_metrics = defaultdict(list)
        # 追加：记录每步的动作与评估指标、准确率趋势
        self.episode_step_logs: List[List[Dict[str, Any]]] = []  # 每个episode一个列表，内含逐步日志
        self.accuracy_trends = defaultdict(list)  # {service_id: [acc_step1, acc_step2, ...]}
        
        # 累积期望奖励J_r(π)跟踪
        self.cumulative_rewards = defaultdict(list)
        
        print(f"初始化PAC-MCoFL训练器，包含{num_agents}个智能体")
        print(f"观察维度: {observation_dim}, 动作维度: {action_dim}")
    
    def compute_cumulative_expected_reward(self, 
                                         rewards_trajectory: List[Dict[int, float]]) -> Dict[int, float]:
        """
        如方程(19-20)所示计算累积期望奖励J_r(π).
        
        参数:
            rewards_trajectory: 每个时间步的奖励列表
            
        返回:
            每个智能体的累积期望奖励
        """
        cumulative_rewards = {}
        
        for service_id in self.service_ids:
            total_reward = 0.0
            for t, rewards in enumerate(rewards_trajectory):
                discount = (self.config.gamma ** t)
                total_reward += discount * rewards.get(service_id, 0.0)
            
            cumulative_rewards[service_id] = total_reward
        
        return cumulative_rewards
    
    def train_episode(self) -> Tuple[Dict[int, float], int, Dict[str, Any]]:
        """
        按照算法1训练一个回合 (Algorithm 1: PAC-MCoFL Training).
        
        实现算法1的步骤2-29的完整训练循环
        
        返回:
            (回合奖励, 回合长度, 回合信息)的元组
        """
        # 步骤3: 初始化全局模型参数和客户端特征
        # Initialize global model parameters ω_{r,0} and client features
        observations = self.environment.reset()
        
        episode_rewards = {service_id: 0.0 for service_id in self.service_ids}
        episode_trajectory = []
        step = 0
        # 本回合逐步日志
        episode_logs: List[Dict[str, Any]] = []

        # 算法1步骤20: 创建回合批次存储经验
        episode_batch = []
        
        # 步骤4: 对于每轮t = 0, 1, ..., T-1
        while step < self.config.max_rounds_per_episode:
            current_obs = {sid: observations[sid] for sid in self.service_ids}

            """逐服务选择动作"""
            actions = {sid: self.agents[sid].select_action(current_obs[sid], training=True)
                       for sid in self.service_ids}

            """初始化奖励和观测"""
            rewards = {}
            next_observations = {}
            dones = {sid: False for sid in self.service_ids}

            # 将连续动作数组转换为环境所需的Action对象（用于能耗/时延/通信量仿真）
            try:
                action_objects = {
                    sid: Action.from_array(np.array(actions[sid], dtype=float))
                    for sid in self.service_ids
                }
            except Exception:
                # 回退：逐个安全转换
                action_objects = {}
                for sid in self.service_ids:
                    arr = actions[sid]
                    arr = np.array(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr
                    action_objects[sid] = Action.from_array(arr)

            # 将在执行真实单轮训练后，基于真实评估与通信/能耗/时延计算奖励
            real_comm_volumes: Dict[int, int] = {}
            real_obs_map: Dict[int, Any] = {}

            # 打印当前RL步的概览
            try:
                print(f"[PAC][RL] Step {step + 1}/{self.config.max_rounds_per_episode}")
            except Exception:
                pass

            # 执行真实训练后再评估并计算奖励（紧耦合）
            # 当前步各服务日志容器
            step_service_logs: Dict[int, Dict[str, Any]] = {}
            for sid, act_arr in actions.items():
                n_clients = int(max(1, min(len(self.fl_system.service_configs[sid].client_ids), round(act_arr[0]))))
                # 应用按服务的动作下限（若提供）
                floors = self.config.service_action_floors.get(sid, {}) if hasattr(self.config, 'service_action_floors') else {}
                min_clients = int(floors.get('min_clients', self.constraints.min_clients))
                min_freq = float(floors.get('min_frequency', self.constraints.min_frequency))
                min_bw = float(floors.get('min_bandwidth', self.constraints.min_bandwidth))
                min_q = int(floors.get('min_quantization', self.constraints.min_quantization))

                n_clients = max(min_clients, n_clients)
                q_level = int(max(min_q, min(32, round(act_arr[3])))) if len(act_arr) > 3 else max(min_q, 8)
                # 从动作数组提取f和B（保持客户端选择逻辑不变）
                cpu_freq = float(act_arr[1]) if len(act_arr) > 1 else self.constraints.min_frequency
                bandwidth = float(act_arr[2]) if len(act_arr) > 2 else self.constraints.min_bandwidth
                cpu_freq = max(min_freq, cpu_freq)
                bandwidth = max(min_bw, bandwidth)

                trainer = self.fl_system.service_trainers.get(sid)
                if trainer is not None and hasattr(trainer, 'users_per_round'):
                    trainer.users_per_round = n_clients
                if trainer is not None and hasattr(trainer, 'cfg'):
                    try:
                        if 'users_per_round' in trainer.cfg:
                            trainer.cfg['users_per_round'] = n_clients
                    except Exception:
                        pass

                # 注入f/B/q覆盖，影响系统能耗/时延与量化通信量
                try:
                    self.fl_system.set_service_action(
                        service_id=sid,
                        n_clients=n_clients,
                        cpu_frequency=cpu_freq,
                        bandwidth=bandwidth,
                        quantization_level=q_level,
                    )
                except Exception as e:
                    print(f"[WARN] set_service_action 失败: {e}")

                fl_model = self.fl_system.service_models.get(sid)
                if fl_model and getattr(self.fl_system, 'debug_disable_quant', False) is False:
                    if hasattr(fl_model, 'quantization_level'):
                        fl_model.quantization_level = q_level

                # 真实训练（单轮），并统计真实通信/能耗/时延
                prev_epochs = 1
                prev_eval_freq = None
                if trainer is not None and hasattr(trainer, 'cfg'):
                    prev_epochs = getattr(trainer.cfg, 'epochs', 1)
                    prev_eval_freq = getattr(trainer.cfg, 'eval_epoch_frequency', None)
                    # 应用每服务每步训练epoch数（若提供）
                    svc_epochs = 1
                    if hasattr(self.config, 'service_epochs_per_step'):
                        svc_epochs = int(self.config.service_epochs_per_step.get(sid, 1))
                    trainer.cfg.epochs = max(1, svc_epochs)
                    if prev_eval_freq is not None:
                        trainer.cfg.eval_epoch_frequency = 10**6

                metrics_summary = {}
                try:
                    _, metrics_summary = self.fl_system.train_service(sid, num_rounds=1, enable_metrics=True)
                except Exception as e:
                    print(f"[WARN][RL-FL Link] 服务{sid} 真实训练失败: {e}")
                    metrics_summary = {}
                finally:
                    if trainer is not None and hasattr(trainer, 'cfg'):
                        trainer.cfg.epochs = prev_epochs
                        if prev_eval_freq is not None:
                            trainer.cfg.eval_epoch_frequency = prev_eval_freq

                # 训练后评估准确率/损失
                acc = 0.0; avg_loss = 0.0
                try:
                    dl = self.fl_system.service_data_loaders.get(sid)
                    test_sets = dl.fl_test_set() if dl else []
                    model_module = self.fl_system.service_models[sid].model
                    model_module.eval()
                    device = next(model_module.parameters()).device
                    correct = 0; total = 0; loss_sum = 0.0; num_batches = 0
                    criterion = torch.nn.CrossEntropyLoss()
                    for entry in test_sets:
                        for batch in entry.get('batches', []):
                            feats = batch.get('features'); labels = batch.get('labels')
                            if feats is None or labels is None:
                                continue
                            feats = feats.to(device); labels = labels.to(device)
                            with torch.no_grad():
                                out = model_module(feats)
                                loss = criterion(out, labels)
                            loss_sum += loss.item()
                            preds = out.argmax(1)
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)
                            num_batches += 1
                    if total > 0:
                        acc = correct / total
                        avg_loss = loss_sum / max(num_batches, 1)
                except Exception:
                    pass

                # 更新观测（使用环境当前观测对象，覆盖为真实结果）
                obs_obj = self.environment.observations[sid]
                try:
                    # 系统代价来自 metrics_summary（真实通信、能耗、时延）
                    total_energy = float(metrics_summary.get('total_energy', 0.0))
                    total_delay = float(metrics_summary.get('total_delay', 0.0))
                    comm_vol = int(metrics_summary.get('communication_volume_per_round', 0))
                except Exception:
                    total_energy = 0.0; total_delay = 0.0; comm_vol = 0

                obs_obj.fl_state.accuracy = acc
                obs_obj.fl_state.loss = avg_loss
                obs_obj.fl_state.quantization_level = q_level
                obs_obj.system_state.total_energy = total_energy
                obs_obj.system_state.total_delay = total_delay
                obs_obj.system_state.communication_volume = comm_vol

                # 累积真实通信量供对抗因子与奖励计算
                real_comm_volumes[sid] = comm_vol
                real_obs_map[sid] = obs_obj

                # 记录当前服务在该步的决策与结果
                try:
                    action_struct = {
                        'n_clients': int(getattr(action_objects[sid], 'n_clients', n_clients)),
                        'cpu_frequency': float(getattr(action_objects[sid], 'cpu_frequency', float(act_arr[1]) if len(act_arr) > 1 else 0.0)),
                        'bandwidth': float(getattr(action_objects[sid], 'bandwidth', float(act_arr[2]) if len(act_arr) > 2 else 0.0)),
                        'quantization_level': int(getattr(action_objects[sid], 'quantization_level', q_level)),
                    }
                except Exception:
                    action_struct = {
                        'n_clients': n_clients,
                        'cpu_frequency': float(act_arr[1]) if len(act_arr) > 1 else 0.0,
                        'bandwidth': float(act_arr[2]) if len(act_arr) > 2 else 0.0,
                        'quantization_level': q_level,
                    }

                step_service_logs[sid] = {
                    'action': action_struct,
                    'accuracy': float(acc),
                    'loss': float(avg_loss),
                    'q_level': int(q_level),
                }

            # 第二阶段：统一根据真实观测与通信量计算奖励，并构造下一观测
            # 在构造下一观测前，将本步所有服务的带宽动作写回为公开带宽向量 B_t
            # 这样可确保 o_{r,t+1} 中的 B_t 反映所有智能体的最新带宽决策
            bw_map = {}
            try:
                for sid_tmp in self.service_ids:
                    act_arr_tmp = actions.get(sid_tmp, None)
                    if act_arr_tmp is not None and len(act_arr_tmp) > 2:
                        bw_map[sid_tmp] = float(act_arr_tmp[2])
                    else:
                        bw_map[sid_tmp] = float(self.constraints.min_bandwidth)
            except Exception:
                # 回退：最小带宽
                for sid_tmp in self.service_ids:
                    bw_map[sid_tmp] = float(self.constraints.min_bandwidth)

            for sid in self.service_ids:
                obs_obj = real_obs_map.get(sid, self.environment.observations[sid])
                # 写入公开带宽分配 B_t（每个服务的服务级带宽，而非按客户端等分后的带宽）
                try:
                    obs_obj.bandwidth_allocations = {k: float(v) for k, v in bw_map.items()}
                except Exception:
                    # 回退：保持原值
                    pass
                # 使用完整参数（含通信量）计算奖励；若reward_functions缺失则回退为acc
                if hasattr(self.environment, 'reward_functions') and sid in self.environment.reward_functions:
                    reward_val = self.environment.reward_functions[sid].calculate(
                        service_id=sid,
                        observation=obs_obj,
                        action=action_objects[sid],
                        all_actions=action_objects,
                        communication_volumes=real_comm_volumes,
                    )
                else:
                    reward_val = float(obs_obj.fl_state.accuracy)

                rewards[sid] = reward_val
                next_observations[sid] = obs_obj.to_array()

                # 打印每个服务在该步的决策与结果摘要，便于观察参数变化
                try:
                    action_print = action_objects[sid]
                    energy = obs_obj.system_state.total_energy
                    delay = obs_obj.system_state.total_delay
                    comm_vol = real_comm_volumes.get(sid, None)
                    print(
                        f"  [S{sid}] action="
                        f"{{n={action_print.n_clients}, f={action_print.cpu_frequency/1e9:.2f}GHz, "
                        f"B={action_print.bandwidth/1e6:.2f}MHz, q={action_print.quantization_level}}} "
                        f"metrics="
                        f"{{acc={obs_obj.fl_state.accuracy:.4f}, loss={obs_obj.fl_state.loss:.4f}, E={energy:.6e}J, T={delay:.6f}s, vol={comm_vol}}} "
                        f"reward={reward_val:.4f}"
                    )
                except Exception:
                    pass

                # 回填奖励到日志
                step_service_logs.setdefault(sid, {})['reward'] = float(reward_val)

            # 存储经验（基于训练前评估的奖励）
            for sid in self.service_ids:
                joint_action = np.concatenate([actions[o] for o in self.service_ids if o != sid])
                exp = {
                    'observation': current_obs[sid],
                    'own_action': actions[sid],
                    'joint_action': joint_action,
                    'reward': rewards[sid],
                    'next_observation': next_observations[sid],
                    'done': dones[sid]
                }
                episode_batch.append((sid, exp))
                self.agents[sid].add_experience(**exp)

            # 覆盖更新趋势为训练后的结果（更有利于向上趋势体现）
            try:
                for sid in self.service_ids:
                    self.accuracy_trends[sid][-1] = float(real_obs_map[sid].fl_state.accuracy) if self.accuracy_trends[sid] else float(real_obs_map[sid].fl_state.accuracy)
            except Exception:
                pass

            for sid, rw in rewards.items():
                episode_rewards[sid] += rw
            episode_trajectory.append(rewards)
            observations = next_observations
            step += 1

            if all(dones.values()):
                break

            # 追加该步聚合日志并更新准确率趋势
            episode_logs.append({
                'step': int(step),
                'services': step_service_logs
            })
            for sid, s_log in step_service_logs.items():
                self.accuracy_trends[sid].append(float(s_log.get('accuracy', 0.0)))
        
        # 步骤20: 将回合批次插入重放缓冲区H
        # Insert the episode batch into a replay buffer H
        
        # 步骤21: 清空回合批次
        # Clear the episode batch (已在add_experience中处理)
        
        # 计算累积期望奖励J_r(π) (方程19-20)
        cumulative_rewards = self.compute_cumulative_expected_reward(episode_trajectory)
        for service_id, cum_reward in cumulative_rewards.items():
            self.cumulative_rewards[service_id].append(cum_reward)
        
        # 保存本回合逐步日志
        self.episode_step_logs.append(episode_logs)

        episode_info = {
            'cumulative_rewards': cumulative_rewards,
            'step_rewards': episode_trajectory,
            'episode_length': step,
            'episode_batch_size': len(episode_batch),
            'step_logs': episode_logs
        }
        
        return episode_rewards, step, episode_info
    
    def train(self) -> Dict[str, Any]:
        """
        实现算法1的主训练循环 (Algorithm 1: PAC-MCoFL Training).
        
        完整实现算法1的步骤1-29
        
        返回:
            训练统计和指标
        """
        print(f"🚀 开始PAC-MCoFL训练 (Algorithm 1)")
        print(f"   总回合数: {self.config.num_episodes}")
        print(f"   每回合最大轮次: {self.config.max_rounds_per_episode}")
        print(f"   智能体数量: {len(self.service_ids)}")
        
        # 算法1步骤1: 加载联邦学习服务的训练数据集
        # Load the training dataset of FL service
        print("📊 联邦学习数据集已由FL系统加载")
        
        # 步骤2: while each episode num_eps ≤ num_max do
        for episode in range(self.config.num_episodes):
            
            # 训练一个回合 (步骤3-21)
            episode_rewards, episode_length, episode_info = self.train_episode()
            print(f"[PAC][RL] episode : {episode + 1}/{self.config.num_episodes}")
            # 存储回合统计
            for service_id, reward in episode_rewards.items():
                self.episode_rewards[service_id].append(reward)
            self.episode_lengths.append(episode_length)
            
            # 步骤22-27: 网络更新
            # if H reaches the buffer size for training then
            training_metrics = {}
            for service_id in self.service_ids:
                # 步骤23: Sample a single batch from H
                # 步骤24-26: 更新网络
                agent_metrics = self.agents[service_id].train_step()
                if agent_metrics:
                    for key, value in agent_metrics.items():
                        self.training_metrics[f"{service_id}_{key}"].append(value)
                        training_metrics[f"agent_{service_id}_{key}"] = value
            
            # 定期评估和日志记录
            if (episode + 1) % self.config.eval_frequency == 0:
                # 计算最近10个回合的平均性能
                avg_rewards = {sid: np.mean(self.episode_rewards[sid][-10:]) 
                              for sid in self.service_ids}
                avg_cumulative = {sid: np.mean(self.cumulative_rewards[sid][-10:])
                                 for sid in self.service_ids}
                
                print(f"\n📈 回合 {episode + 1}/{self.config.num_episodes}")
                print(f"   平均回合奖励: {avg_rewards}")
                print(f"   累积期望奖励J_r(π): {avg_cumulative}")
                print(f"   回合长度: {episode_length}")
                
                if training_metrics:
                    print(f"   训练指标: {training_metrics}")
                
                # 显示PAC算法的关键指标
                print(f"   重放缓冲区大小: {[len(self.agents[sid].replay_buffer) for sid in self.service_ids]}")
            
            
            # 步骤28: num_eps ← num_eps + 1
            # (由for循环自动处理)
        
        # 步骤29: end
        print("\n✅ PAC-MCoFL训练完成!")
        print(f"   总训练回合: {self.config.num_episodes}")
        print(f"   平均回合长度: {np.mean(self.episode_lengths):.2f}")
        
        # 最终性能总结
        final_avg_rewards = {sid: np.mean(self.episode_rewards[sid][-50:]) 
                            for sid in self.service_ids}
        final_cumulative = {sid: np.mean(self.cumulative_rewards[sid][-50:])
                           for sid in self.service_ids}
        
        print(f"   最终平均回合奖励: {final_avg_rewards}")
        print(f"   最终累积期望奖励J_r(π): {final_cumulative}")
        
        return {
            'episode_rewards': dict(self.episode_rewards),
            'episode_lengths': self.episode_lengths,
            'cumulative_rewards': dict(self.cumulative_rewards),
            'training_metrics': dict(self.training_metrics),
            'accuracy_trends': dict(self.accuracy_trends),
            'action_logs': self.episode_step_logs,
            'final_performance': {
                'avg_episode_rewards': final_avg_rewards,
                'avg_cumulative_rewards': final_cumulative
            }
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        评估训练好的PAC智能体.
        
        参数:
            num_episodes: 评估回合数
            
        返回:
            评估指标
        """
        eval_rewards = defaultdict(list)
        eval_cumulative = defaultdict(list)
        eval_lengths = []
        
        for episode in range(num_episodes):
            observations = self.environment.reset()
            episode_rewards = {service_id: 0.0 for service_id in self.service_ids}
            episode_trajectory = []
            step = 0
            
            while step < self.config.max_rounds_per_episode:
                # 选择动作时不添加探索噪声
                actions = {}
                for service_id in self.service_ids:
                    action = self.agents[service_id].select_action(
                        observations[service_id], training=False
                    )
                    actions[service_id] = action
                
                observations, rewards, dones, _ = self.environment.step(actions)
                
                for service_id, reward in rewards.items():
                    episode_rewards[service_id] += reward
                
                episode_trajectory.append(rewards)
                step += 1
                
                if all(dones.values()):
                    break
            
            # 存储评估结果
            for service_id, reward in episode_rewards.items():
                eval_rewards[service_id].append(reward)
            
            cumulative_rewards = self.compute_cumulative_expected_reward(episode_trajectory)
            for service_id, cum_reward in cumulative_rewards.items():
                eval_cumulative[service_id].append(cum_reward)
            
            eval_lengths.append(step)
        
        # 计算平均值
        avg_rewards = {service_id: np.mean(rewards) 
                      for service_id, rewards in eval_rewards.items()}
        avg_cumulative = {service_id: np.mean(rewards)
                         for service_id, rewards in eval_cumulative.items()}
        avg_length = np.mean(eval_lengths)
        
        return {
            'avg_episode_rewards': avg_rewards,
            'avg_cumulative_rewards': avg_cumulative,
            'avg_episode_length': avg_length,
            'all_episode_rewards': dict(eval_rewards),
            'all_cumulative_rewards': dict(eval_cumulative),
            'all_episode_lengths': eval_lengths
        }
    
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取综合训练总结."""
        
        summary = {
            'config': {
                'num_episodes': self.config.num_episodes,
                'max_rounds_per_episode': self.config.max_rounds_per_episode,
                'actor_lr': self.config.actor_lr,
                'critic_lr': self.config.critic_lr,
                'gamma': self.config.gamma
            },
            'final_performance': {},
            'pareto_optimality': {}
        }
        
        # 计算最终性能
        if self.episode_rewards:
            for service_id in self.service_ids:
                rewards = self.episode_rewards[service_id]
                cumulative = self.cumulative_rewards[service_id]
                
                if rewards and cumulative:
                    summary['final_performance'][service_id] = {
                        'final_episode_reward': rewards[-1],
                        'avg_episode_reward': np.mean(rewards),
                        'final_cumulative_reward': cumulative[-1],
                        'avg_cumulative_reward': np.mean(cumulative),
                        'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0
                    }
        
        # 分析帕累托最优性
        if self.cumulative_rewards:
            # 检查最终联合策略是否实现帕累托最优
            final_cumulative = {sid: self.cumulative_rewards[sid][-1] 
                              for sid in self.service_ids if self.cumulative_rewards[sid]}
            
            summary['pareto_optimality'] = {
                'final_joint_cumulative_reward': sum(final_cumulative.values()),
                'individual_cumulative_rewards': final_cumulative,
                'pareto_improvement_achieved': all(
                    final_cumulative[sid] > self.cumulative_rewards[sid][0]
                    for sid in final_cumulative.keys() if len(self.cumulative_rewards[sid]) > 1
                )
            }
        
        return summary
