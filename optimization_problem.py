#!/usr/bin/env python3
"""
多服务提供商优化问题模块

实现了方程(14)中描述的优化问题：
- 具有客户端选择、量化和资源分配的多目标优化
- 处理能量、延迟、带宽和计算资源的约束条件
- 联邦学习性能的成本函数建模
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
import warnings


@dataclass
class OptimizationConstraints:
    """优化约束条件的容器(C1-C5)。"""
    
    # C1: 能量和延迟约束
    max_energy: float = 1.0  # E_r^max
    max_delay: float = 10.0  # T_r^max
    
    # C2: 客户端选择约束
    min_clients: int = 1
    max_clients: int = 5
    
    # C3: CPU频率约束  
    # 论文表：0.5–3.5 GHz（服务级平均频率）
    min_frequency: float = 0.5e9
    max_frequency: float = 3.5e9
    
    # C4: 带宽约束
    # 论文表：0–30 MHz（服务级带宽）
    min_bandwidth: float = 0.0
    max_bandwidth: float = 30e6
    
    # C5: 量化级别约束
    min_quantization: int = 1
    # 论文表：1–2^32（比特位宽上限转化为级数上限）
    max_quantization: int = 2**32


@dataclass
class DecisionVariables:
    """方程(14)中决策变量的容器。"""
    
    n_r_t: int          # 选定的客户端数量
    f_r_t: float        # CPU频率分配
    B_r_t: float        # 带宽分配
    q_r_t: int          # 量化级别
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便于操作。"""
        return {
            'n_r_t': self.n_r_t,
            'f_r_t': self.f_r_t,
            'B_r_t': self.B_r_t,
            'q_r_t': self.q_r_t
        }
    
    @classmethod
    def from_array(cls, x: np.ndarray) -> 'DecisionVariables':
        """从优化数组[n, f, B, q]创建。"""
        return cls(
            n_r_t=int(x[0]),
            f_r_t=float(x[1]),
            B_r_t=float(x[2]),
            q_r_t=int(x[3])
        )
    
    def to_array(self) -> np.ndarray:
        """转换为优化数组。"""
        return np.array([self.n_r_t, self.f_r_t, self.B_r_t, self.q_r_t])


class MultiObjectiveCostFunction:
    """
    方程(14)的多目标成本函数Υ。
    
    平衡训练精度、通信量、能量消耗和延迟。
    """
    
    def __init__(self, 
                 weights: Dict[str, float] = None,
                 discount_factor: float = 0.95):
        """
        初始化成本函数。
        
        参数:
            weights: 不同目标的权重
            discount_factor: 时间折扣因子γ
        """
        self.weights = weights or {
            'loss': 1.0,           # 模型损失权重
            'volume': 0.1,         # 通信量权重  
            'energy': 0.5,         # 能量消耗权重
            'delay': 0.3           # 延迟权重
        }
        self.gamma = discount_factor
        
    def __call__(self, 
                 loss: float,
                 communication_volume: int,
                 total_energy: float,
                 total_delay: float,
                 round_t: int = 0) -> float:
        """
        计算多目标成本。
        
        实现: γ^t * Υ(L_r(ω_{r,t}), vol_{r,t}, E_{r,t}^total, T_{r,t}^total)
        
        参数:
            loss: 模型损失L_r(ω_{r,t})
            communication_volume: 通信量vol_{r,t}
            total_energy: 总能耗E_{r,t}^total
            total_delay: 总延迟T_{r,t}^total
            round_t: 当前轮次(用于折扣因子)
            
        返回:
            多目标成本值
        """
        # 将目标归一化到类似的尺度
        normalized_loss = loss
        normalized_volume = communication_volume / 1e6  # 缩放到MB
        normalized_energy = total_energy * 1e6  # 缩放到μJ
        normalized_delay = total_delay * 1e3   # 缩放到ms
        
        # 多目标成本(越低越好)
        cost = (self.weights['loss'] * normalized_loss +
                self.weights['volume'] * normalized_volume +
                self.weights['energy'] * normalized_energy +
                self.weights['delay'] * normalized_delay)
        
        # 应用时间折扣
        discounted_cost = (self.gamma ** round_t) * cost
        
        return discounted_cost
    
    def evaluate_trajectory(self, 
                           trajectory: List[Dict[str, float]], 
                           total_rounds: int) -> float:
        """
        评估整个训练轨迹的成本。
        
        参数:
            trajectory: 每轮次的指标列表
            total_rounds: 总轮次数T
            
        返回:
            总折扣成本
        """
        total_cost = 0.0
        
        for t, metrics in enumerate(trajectory):
            round_cost = self(
                loss=metrics['loss'],
                communication_volume=metrics['communication_volume'],
                total_energy=metrics['total_energy'],
                total_delay=metrics['total_delay'],
                round_t=t
            )
            total_cost += round_cost
        
        return total_cost


class ServiceProviderOptimizer:
    """
    单一服务提供商优化引擎。
    
    解决一个服务提供商的约束优化问题。
    """
    
    def __init__(self, 
                 service_id: int,
                 constraints: OptimizationConstraints,
                 cost_function: MultiObjectiveCostFunction):
        """
        初始化服务提供商的优化器。
        
        参数:
            service_id: 服务提供商标识符
            constraints: 优化约束条件
            cost_function: 多目标成本函数
        """
        self.service_id = service_id
        self.constraints = constraints
        self.cost_function = cost_function
        
        # 当前状态
        self.current_variables: Optional[DecisionVariables] = None
        self.optimization_history: List[Dict[str, Any]] = []
    
    def evaluate_system_metrics(self, 
                               variables: DecisionVariables,
                               system_state: Dict[str, Any]) -> Dict[str, float]:
        """
        评估给定决策变量的系统指标。
        
        参数:
            variables: 要评估的决策变量
            system_state: 当前系统状态(模型权重、客户端数据等)
            
        返回:
            包含计算指标的字典
        """
        # 这将与通信和量化模块集成
        # 现在，我们提供一个简化的模拟
        
        # 提取参数
        n_clients = variables.n_r_t
        frequency = variables.f_r_t
        bandwidth = variables.B_r_t
        quantization = variables.q_r_t
        
        # 模拟模型损失(随着更多客户端和更高的量化而减少)
        base_loss = system_state.get('base_loss', 2.0)
        loss_reduction = 0.1 * np.log(n_clients) + 0.05 * np.log(quantization)
        model_loss = max(0.1, base_loss - loss_reduction)
        
        # 通信量(来自量化模块)
        model_size = system_state.get('model_size', 100000)
        bits_per_param = max(1, np.ceil(np.log2(quantization)) + 1)
        communication_volume = model_size * bits_per_param + 32
        
        # 能量消耗(来自通信模块)
        dataset_size = system_state.get('avg_dataset_size', 1000)
        mu_i = system_state.get('mu_i', 1e-28)
        c_ir = system_state.get('c_ir', 1000)
        
        # 每个客户端的计算能耗
        comp_energy_per_client = mu_i * c_ir * dataset_size * (frequency ** 2)
        total_comp_energy = n_clients * comp_energy_per_client
        
        # 通信能耗(简化)
        transmission_power = 0.1  # 瓦特
        channel_gain = system_state.get('channel_gain', 1e-3)
        noise_power = 1e-9
        
        # 香农容量
        snr = (channel_gain * transmission_power) / (bandwidth * noise_power)
        transmission_rate = bandwidth * np.log2(1 + snr)
        comm_delay = communication_volume / transmission_rate if transmission_rate > 0 else float('inf')
        comm_energy = comm_delay * transmission_power
        
        total_energy = total_comp_energy + (n_clients * comm_energy)
        
        # 总延迟(客户端间的最大值)
        comp_delay = (c_ir * dataset_size) / frequency
        total_delay = comp_delay + comm_delay
        
        return {
            'loss': model_loss,
            'communication_volume': communication_volume,
            'total_energy': total_energy,
            'total_delay': total_delay,
            'n_clients': n_clients,
            'frequency': frequency,
            'bandwidth': bandwidth,
            'quantization': quantization
        }
    
    def check_constraints(self, variables: DecisionVariables, metrics: Dict[str, float]) -> bool:
        """
        检查决策变量是否满足所有约束。
        
        参数:
            variables: 要检查的决策变量
            metrics: 计算的系统指标
            
        返回:
            如果满足所有约束则返回True
        """
        # C1: 能量和延迟约束
        if metrics['total_energy'] > self.constraints.max_energy:
            return False
        if metrics['total_delay'] > self.constraints.max_delay:
            return False
        
        # C2: 客户端选择约束
        if not (self.constraints.min_clients <= variables.n_r_t <= self.constraints.max_clients):
            return False
        
        # C3: CPU频率约束
        if not (self.constraints.min_frequency <= variables.f_r_t <= self.constraints.max_frequency):
            return False
        
        # C4: 带宽约束  
        if not (self.constraints.min_bandwidth <= variables.B_r_t <= self.constraints.max_bandwidth):
            return False
        
        # C5: 量化约束
        if not (self.constraints.min_quantization <= variables.q_r_t <= self.constraints.max_quantization):
            return False
        
        return True
    
    def objective_function(self, x: np.ndarray, system_state: Dict[str, Any], round_t: int = 0) -> float:
        """
        优化的目标函数。
        
        参数:
            x: 决策变量数组[n, f, B, q]
            system_state: 当前系统状态
            round_t: 当前轮次
            
        返回:
            成本值(待最小化)
        """
        try:
            variables = DecisionVariables.from_array(x)
            metrics = self.evaluate_system_metrics(variables, system_state)
            
            # 检查约束
            if not self.check_constraints(variables, metrics):
                return 1e6  # 违反约束的大惩罚
            
            # 计算成本
            cost = self.cost_function(
                loss=metrics['loss'],
                communication_volume=metrics['communication_volume'],
                total_energy=metrics['total_energy'],
                total_delay=metrics['total_delay'],
                round_t=round_t
            )
            
            return cost
            
        except Exception as e:
            warnings.warn(f"目标函数中的错误: {e}")
            return 1e6
    
    def optimize_single_round(self, 
                            system_state: Dict[str, Any],
                            round_t: int = 0,
                            method: str = 'differential_evolution') -> Tuple[DecisionVariables, Dict[str, float]]:
        """
        优化单轮的决策变量。
        
        参数:
            system_state: 当前系统状态
            round_t: 当前轮次号
            method: 优化方法('differential_evolution', 'scipy', 'grid_search')
            
        返回:
            (最优变量, 指标)的元组
        """
        if method == 'differential_evolution':
            return self._optimize_differential_evolution(system_state, round_t)
        elif method == 'scipy':
            return self._optimize_scipy(system_state, round_t)
        elif method == 'grid_search':
            return self._optimize_grid_search(system_state, round_t)
        else:
            raise ValueError(f"未知的优化方法: {method}")
    
    def _optimize_differential_evolution(self, 
                                       system_state: Dict[str, Any], 
                                       round_t: int) -> Tuple[DecisionVariables, Dict[str, float]]:
        """使用差分进化优化(适用于混合整数问题)。"""
        
        # 定义边界: [n_min, n_max], [f_min, f_max], [B_min, B_max], [q_min, q_max]
        bounds = [
            (self.constraints.min_clients, self.constraints.max_clients),
            (self.constraints.min_frequency, self.constraints.max_frequency),
            (self.constraints.min_bandwidth, self.constraints.max_bandwidth),
            (self.constraints.min_quantization, self.constraints.max_quantization)
        ]
        
        # n和q的整数约束
        integrality = [True, False, False, True]
        
        if not SCIPY_AVAILABLE:
            warnings.warn("scipy不可用，使用随机优化")
            return self._random_optimization(system_state, round_t)
            
        result = differential_evolution(
            func=lambda x: self.objective_function(x, system_state, round_t),
            bounds=bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            integrality=integrality
        )
        
        optimal_variables = DecisionVariables.from_array(result.x)
        metrics = self.evaluate_system_metrics(optimal_variables, system_state)
        
        # 存储优化历史
        self.optimization_history.append({
            'round': round_t,
            'variables': optimal_variables.to_dict(),
            'metrics': metrics,
            'cost': result.fun,
            'success': result.success
        })
        
        return optimal_variables, metrics
    
    def _optimize_scipy(self, 
                       system_state: Dict[str, Any], 
                       round_t: int) -> Tuple[DecisionVariables, Dict[str, float]]:
        """使用scipy最小化优化(处理连续变量表现良好)。"""
        
        # 初始猜测
        x0 = np.array([
            (self.constraints.min_clients + self.constraints.max_clients) // 2,
            (self.constraints.min_frequency + self.constraints.max_frequency) / 2,
            (self.constraints.min_bandwidth + self.constraints.max_bandwidth) / 2,
            (self.constraints.min_quantization + self.constraints.max_quantization) // 2
        ])
        
        # 边界
        bounds = [
            (self.constraints.min_clients, self.constraints.max_clients),
            (self.constraints.min_frequency, self.constraints.max_frequency),
            (self.constraints.min_bandwidth, self.constraints.max_bandwidth),
            (self.constraints.min_quantization, self.constraints.max_quantization)
        ]
        
        result = minimize(
            fun=lambda x: self.objective_function(x, system_state, round_t),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # 四舍五入整数变量
        x_rounded = result.x.copy()
        x_rounded[0] = round(x_rounded[0])  # n_r_t
        x_rounded[3] = round(x_rounded[3])  # q_r_t
        
        optimal_variables = DecisionVariables.from_array(x_rounded)
        metrics = self.evaluate_system_metrics(optimal_variables, system_state)
        
        return optimal_variables, metrics
    
    def _optimize_grid_search(self, 
                            system_state: Dict[str, Any], 
                            round_t: int) -> Tuple[DecisionVariables, Dict[str, float]]:
        """使用网格搜索优化(穷举但保证找到全局最优)。"""
        
        # 定义搜索网格
        n_grid = list(range(self.constraints.min_clients, 
                           min(self.constraints.max_clients + 1, 21)))  # 限制为20以提高效率
        f_grid = np.linspace(self.constraints.min_frequency, 
                           self.constraints.max_frequency, 10)
        B_grid = np.linspace(self.constraints.min_bandwidth, 
                           self.constraints.max_bandwidth, 10)
        q_grid = [2, 4, 8, 16, 32]  # 常见量化级别
        
        best_cost = float('inf')
        best_variables = None
        best_metrics = None
        
        total_combinations = len(n_grid) * len(f_grid) * len(B_grid) * len(q_grid)
        print(f"网格搜索: 评估{total_combinations}个组合...")
        
        for n in n_grid:
            for f in f_grid:
                for B in B_grid:
                    for q in q_grid:
                        x = np.array([n, f, B, q])
                        cost = self.objective_function(x, system_state, round_t)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_variables = DecisionVariables.from_array(x)
                            best_metrics = self.evaluate_system_metrics(best_variables, system_state)
        
        return best_variables, best_metrics


class MultiServiceOptimizer:
    """
    多服务提供商优化协调器。
    
    处理服务间的协调和资源分配。
    """
    
    def __init__(self, 
                 service_ids: List[int],
                 global_constraints: OptimizationConstraints,
                 cost_functions: Dict[int, MultiObjectiveCostFunction]):
        """
        初始化多服务优化器。
        
        参数:
            service_ids: 服务提供商ID列表
            global_constraints: 全局系统约束
            cost_functions: 每个服务的成本函数
        """
        self.service_ids = service_ids
        self.global_constraints = global_constraints
        
        # 创建单个优化器
        self.optimizers = {}
        for service_id in service_ids:
            self.optimizers[service_id] = ServiceProviderOptimizer(
                service_id=service_id,
                constraints=global_constraints,
                cost_function=cost_functions[service_id]
            )
    
    def optimize_all_services(self, 
                            system_states: Dict[int, Dict[str, Any]],
                            round_t: int = 0,
                            method: str = 'sequential') -> Dict[int, Tuple[DecisionVariables, Dict[str, float]]]:
        """
        联合优化所有服务。
        
        参数:
            system_states: 每个服务的系统状态
            round_t: 当前轮次
            method: 协调方法('sequential', 'parallel', 'iterative')
            
        返回:
            将service_id映射到(variables, metrics)的字典
        """
        if method == 'sequential':
            return self._optimize_sequential(system_states, round_t)
        elif method == 'parallel':
            return self._optimize_parallel(system_states, round_t)
        elif method == 'iterative':
            return self._optimize_iterative(system_states, round_t)
        else:
            raise ValueError(f"未知的协调方法: {method}")
    
    def _optimize_sequential(self, 
                           system_states: Dict[int, Dict[str, Any]], 
                           round_t: int) -> Dict[int, Tuple[DecisionVariables, Dict[str, float]]]:
        """顺序优化(服务一个接一个地优化)。"""
        
        results = {}
        remaining_bandwidth = self.global_constraints.max_bandwidth
        
        for service_id in self.service_ids:
            # 基于先前分配更新带宽约束
            service_constraints = OptimizationConstraints(
                max_energy=self.global_constraints.max_energy,
                max_delay=self.global_constraints.max_delay,
                min_clients=self.global_constraints.min_clients,
                max_clients=self.global_constraints.max_clients,
                min_frequency=self.global_constraints.min_frequency,
                max_frequency=self.global_constraints.max_frequency,
                min_bandwidth=self.global_constraints.min_bandwidth,
                max_bandwidth=min(remaining_bandwidth, self.global_constraints.max_bandwidth),
                min_quantization=self.global_constraints.min_quantization,
                max_quantization=self.global_constraints.max_quantization
            )
            
            self.optimizers[service_id].constraints = service_constraints
            
            variables, metrics = self.optimizers[service_id].optimize_single_round(
                system_states[service_id], round_t
            )
            
            results[service_id] = (variables, metrics)
            remaining_bandwidth -= variables.B_r_t
        
        return results
    
    def _optimize_parallel(self, 
                         system_states: Dict[int, Dict[str, Any]], 
                         round_t: int) -> Dict[int, Tuple[DecisionVariables, Dict[str, float]]]:
        """并行优化(服务独立优化)。"""
        
        results = {}
        for service_id in self.service_ids:
            variables, metrics = self.optimizers[service_id].optimize_single_round(
                system_states[service_id], round_t
            )
            results[service_id] = (variables, metrics)
        
        # 检查全局带宽约束
        total_bandwidth = sum(vars.B_r_t for vars, _ in results.values())
        if total_bandwidth > self.global_constraints.max_bandwidth:
            # 按比例缩小带宽分配
            scale_factor = self.global_constraints.max_bandwidth / total_bandwidth
            for service_id, (variables, metrics) in results.items():
                variables.B_r_t *= scale_factor
                # 使用缩放后的带宽重新计算指标
                metrics = self.optimizers[service_id].evaluate_system_metrics(
                    variables, system_states[service_id]
                )
                results[service_id] = (variables, metrics)
        
        return results
    
    def _optimize_iterative(self, 
                          system_states: Dict[int, Dict[str, Any]], 
                          round_t: int,
                          max_iterations: int = 5) -> Dict[int, Tuple[DecisionVariables, Dict[str, float]]]:
        """迭代优化(服务迭代优化直到收敛)。"""
        
        results = {}
        
        # 使用并行优化初始化
        results = self._optimize_parallel(system_states, round_t)
        
        for iteration in range(max_iterations):
            prev_results = results.copy()
            
            # 考虑其他服务的决策更新每个服务
            for service_id in self.service_ids:
                # 计算剩余带宽
                other_bandwidth = sum(
                    vars.B_r_t for sid, (vars, _) in results.items() 
                    if sid != service_id
                )
                available_bandwidth = self.global_constraints.max_bandwidth - other_bandwidth
                
                # 更新约束
                service_constraints = OptimizationConstraints(
                    max_energy=self.global_constraints.max_energy,
                    max_delay=self.global_constraints.max_delay,
                    min_clients=self.global_constraints.min_clients,
                    max_clients=self.global_constraints.max_clients,
                    min_frequency=self.global_constraints.min_frequency,
                    max_frequency=self.global_constraints.max_frequency,
                    min_bandwidth=self.global_constraints.min_bandwidth,
                    max_bandwidth=min(available_bandwidth, self.global_constraints.max_bandwidth),
                    min_quantization=self.global_constraints.min_quantization,
                    max_quantization=self.global_constraints.max_quantization
                )
                
                self.optimizers[service_id].constraints = service_constraints
                
                variables, metrics = self.optimizers[service_id].optimize_single_round(
                    system_states[service_id], round_t
                )
                results[service_id] = (variables, metrics)
            
            # 检查收敛性
            converged = True
            for service_id in self.service_ids:
                prev_vars = prev_results[service_id][0]
                curr_vars = results[service_id][0]
                
                # 检查变量是否显著变化
                if (abs(prev_vars.B_r_t - curr_vars.B_r_t) > 1e3 or
                    abs(prev_vars.f_r_t - curr_vars.f_r_t) > 1e6 or
                    prev_vars.n_r_t != curr_vars.n_r_t or
                    prev_vars.q_r_t != curr_vars.q_r_t):
                    converged = False
                    break
            
            if converged:
                print(f"迭代优化在{iteration + 1}次迭代后收敛")
                break
        
        return results


def test_optimization_problem():
    """优化问题模块的测试函数。"""
    print("测试多服务提供商优化...")
    
    # 创建约束条件
    constraints = OptimizationConstraints(
        max_energy=0.1,      # 100 mJ
        max_delay=5.0,       # 5秒
        max_clients=10,
        max_bandwidth=1e7    # 10 MHz
    )
    
    # 创建成本函数
    cost_functions = {
        1: MultiObjectiveCostFunction(weights={'loss': 2.0, 'volume': 0.1, 'energy': 0.5, 'delay': 0.3}),
        2: MultiObjectiveCostFunction(weights={'loss': 1.5, 'volume': 0.2, 'energy': 0.3, 'delay': 0.5})
    }
    
    # 创建系统状态
    system_states = {
        1: {
            'base_loss': 2.5,
            'model_size': 50000,
            'avg_dataset_size': 1000,
            'mu_i': 1e-28,
            'c_ir': 1000,
            'channel_gain': 1e-3
        },
        2: {
            'base_loss': 1.8,
            'model_size': 30000,
            'avg_dataset_size': 800,
            'mu_i': 1.2e-28,
            'c_ir': 1200,
            'channel_gain': 1.2e-3
        }
    }
    
    # 测试单一服务优化
    print("\n测试单一服务优化...")
    optimizer1 = ServiceProviderOptimizer(1, constraints, cost_functions[1])
    variables1, metrics1 = optimizer1.optimize_single_round(system_states[1], method='differential_evolution')
    
    print(f"服务1最优变量: {variables1.to_dict()}")
    print(f"服务1指标: {metrics1}")
    
    # 测试多服务优化
    print("\n测试多服务优化...")
    multi_optimizer = MultiServiceOptimizer([1, 2], constraints, cost_functions)
    results = multi_optimizer.optimize_all_services(system_states, method='iterative')
    
    for service_id, (variables, metrics) in results.items():
        print(f"\n服务 {service_id}:")
        print(f"  变量: {variables.to_dict()}")
        print(f"  损失: {metrics['loss']:.4f}")
        print(f"  能耗: {metrics['total_energy']:.6f} J")
        print(f"  延迟: {metrics['total_delay']:.6f} s")
        print(f"  通信量: {metrics['communication_volume']} 比特")
    
    print("\n✅ 优化问题测试完成!")


if __name__ == "__main__":
    test_optimization_problem()