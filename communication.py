#!/usr/bin/env python3
"""
联邦学习通信模型

实现论文中方程(8)-(13)描述的通信延迟和能耗计算：
- 计算能耗和延迟
- 使用FDMA的通信能耗和延迟
- 总系统能耗和延迟聚合
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import math


class CommunicationModel:
    """
    建模联邦学习中的通信延迟和能耗。
    
    实现论文中的方程(8)-(13)：
    - 计算能耗/延迟：E^cmp, T^cmp
    - 通信能耗/延迟：E^com, T^com  
    - 总系统指标聚合
    """
    
    def __init__(self, noise_power_density: float = 1e-9):
        """
        初始化通信模型。
        
        参数:
            noise_power_density: 单边白噪声功率谱密度 N_0
        """
        self.N_0 = noise_power_density
    
    def calculate_computation_energy(self, 
                                   mu_i: float, 
                                   c_ir: float, 
                                   dataset_size: int, 
                                   frequency: float) -> float:
        """
        计算客户端i的计算能耗。
        
        实现方程(8)：E^cmp_{i,r,t} = μ_i * c_{i,r} * |D_{i,r}| * f_{i,r,t}^2
        
        参数:
            mu_i: 由芯片架构决定的有效电容常数
            c_ir: 客户端i服务r每个样本所需的CPU周期数
            dataset_size: 本地数据集大小 |D_{i,r}|
            frequency: CPU周期频率 f_{i,r,t}
            
        返回:
            计算能耗（焦耳）
        """
        energy = mu_i * c_ir * dataset_size * (frequency ** 2)
        return energy
    
    def calculate_computation_delay(self, 
                                  c_ir: float, 
                                  dataset_size: int, 
                                  frequency: float) -> float:
        """
        计算客户端i的计算延迟。
        
        实现方程(9)：T^cmp_{i,r,t} = c_{i,r} * |D_{i,r}| / f_{i,r,t}
        
        参数:
            c_ir: 每个样本所需的CPU周期数
            dataset_size: 本地数据集大小
            frequency: CPU周期频率
            
        返回:
            计算延迟（秒）
        """
        delay = (c_ir * dataset_size) / frequency
        return delay
    
    def calculate_transmission_rate(self, 
                                  bandwidth: float, 
                                  channel_gain: float, 
                                  transmit_power: float) -> float:
        """
        使用香农公式计算传输速率。
        
        实现方程(10)：v_{i,r,t} = B_{i,r,t} * log_2(1 + g_{i,t}*p_{i,t}/(B_{i,r,t}*N_0))
        
        参数:
            bandwidth: 分配的带宽 B_{i,r,t}
            channel_gain: 信道增益 g_{i,t}
            transmit_power: 传输功率 p_{i,t}
            
        返回:
            传输速率（比特每秒）
        """
        snr = (channel_gain * transmit_power) / (bandwidth * self.N_0)
        rate = bandwidth * math.log2(1 + snr)
        return rate
    
    def calculate_communication_delay(self, 
                                    communication_volume: int, 
                                    transmission_rate: float) -> float:
        """
        计算通信延迟。
        
        实现方程(11)：T^com_{i,r,t} = vol_{i,r,t} / v_{i,r,t}
        
        参数:
            communication_volume: 要传输的数据量（比特）
            transmission_rate: 传输速率（比特每秒）
            
        返回:
            通信延迟（秒）
        """
        if transmission_rate <= 0:
            return float('inf')
        
        delay = communication_volume / transmission_rate
        return delay
    
    def calculate_communication_energy(self, 
                                     communication_delay: float, 
                                     transmit_power: float) -> float:
        """
        计算通信能耗。
        
        实现方程(12)：E^com_{i,r,t} = T^com_{i,r,t} * p_{i,t}
        
        参数:
            communication_delay: 通信延迟 T^com_{i,r,t}
            transmit_power: 传输功率 p_{i,t}
            
        返回:
            通信能耗（焦耳）
        """
        energy = communication_delay * transmit_power
        return energy


class ClientMetrics:
    """客户端特定指标的容器。"""
    
    def __init__(self, client_id: int, service_id: int):
        self.client_id = client_id
        self.service_id = service_id
        self.computation_energy = 0.0
        self.computation_delay = 0.0
        self.communication_energy = 0.0
        self.communication_delay = 0.0
        self.total_energy = 0.0
        self.total_delay = 0.0
    
    def update_computation_metrics(self, energy: float, delay: float):
        """更新计算指标。"""
        self.computation_energy = energy
        self.computation_delay = delay
        self._update_totals()
    
    def update_communication_metrics(self, energy: float, delay: float):
        """更新通信指标。""" 
        self.communication_energy = energy
        self.communication_delay = delay
        self._update_totals()
    
    def _update_totals(self):
        """更新总指标。"""
        self.total_energy = self.computation_energy + self.communication_energy
        self.total_delay = self.computation_delay + self.communication_delay


class SystemMetrics:
    """
    聚合联邦学习的系统级指标。
    
    实现方程(13a)和(13b)的总能耗和延迟。
    """
    
    def __init__(self):
        self.client_metrics: Dict[int, ClientMetrics] = {}
        self.comm_model = CommunicationModel()
    
    def add_client_metrics(self, client_metrics: ClientMetrics):
        """添加客户端指标。"""
        self.client_metrics[client_metrics.client_id] = client_metrics
    
    def calculate_client_metrics(self,
                                client_id: int,
                                service_id: int,
                                # 计算参数
                                mu_i: float,
                                c_ir: float, 
                                dataset_size: int,
                                cpu_frequency: float,
                                # 通信参数
                                communication_volume: int,
                                bandwidth: float,
                                channel_gain: float,
                                transmit_power: float) -> ClientMetrics:
        """
        计算单个客户端的综合指标。
        
        参数:
            client_id: 客户端标识符
            service_id: 服务标识符
            mu_i: 有效电容常数
            c_ir: 每个样本的CPU周期数
            dataset_size: 本地数据集大小
            cpu_frequency: CPU频率
            communication_volume: 要传输的数据量
            bandwidth: 分配的带宽
            channel_gain: 信道增益
            transmit_power: 传输功率
            
        返回:
            包含所有计算指标的ClientMetrics对象
        """
        metrics = ClientMetrics(client_id, service_id)
        
        # 计算计算指标
        comp_energy = self.comm_model.calculate_computation_energy(
            mu_i, c_ir, dataset_size, cpu_frequency
        )
        comp_delay = self.comm_model.calculate_computation_delay(
            c_ir, dataset_size, cpu_frequency
        )
        metrics.update_computation_metrics(comp_energy, comp_delay)
        
        # 计算通信指标
        transmission_rate = self.comm_model.calculate_transmission_rate(
            bandwidth, channel_gain, transmit_power
        )
        comm_delay = self.comm_model.calculate_communication_delay(
            communication_volume, transmission_rate
        )
        comm_energy = self.comm_model.calculate_communication_energy(
            comm_delay, transmit_power
        )
        metrics.update_communication_metrics(comm_energy, comm_delay)
        
        return metrics
    
    def calculate_service_total_energy(self, service_id: int) -> float:
        """
        计算服务的总能耗。
        
        实现方程(13a)：E^total_{r,t} = Σ_{i∈N}(E^com_{i,r,t} + E^cmp_{i,r,t})
        
        参数:
            service_id: 服务标识符
            
        返回:
            服务的总能耗
        """
        total_energy = 0.0
        for metrics in self.client_metrics.values():
            if metrics.service_id == service_id:
                total_energy += metrics.total_energy
        
        return total_energy
    
    def calculate_service_total_delay(self, service_id: int) -> float:
        """
        计算服务的总延迟。
        
        实现方程(13b)：T^total_{r,t} = max_{i∈N}(T^cmp_{i,r,t} + T^com_{i,r,t})
        
        参数:
            service_id: 服务标识符
            
        返回:
            服务的总延迟（所有客户端中的最大值）
        """
        max_delay = 0.0
        for metrics in self.client_metrics.values():
            if metrics.service_id == service_id:
                max_delay = max(max_delay, metrics.total_delay)
        
        return max_delay
    
    def get_service_metrics_summary(self, service_id: int) -> Dict[str, Any]:
        """
        获取服务的综合指标摘要。
        
        返回:
            包含所有相关指标的字典
        """
        service_clients = [m for m in self.client_metrics.values() 
                          if m.service_id == service_id]
        
        if not service_clients:
            return {}
        
        total_energy = self.calculate_service_total_energy(service_id)
        total_delay = self.calculate_service_total_delay(service_id)
        
        # 计算平均指标
        avg_comp_energy = np.mean([m.computation_energy for m in service_clients])
        avg_comp_delay = np.mean([m.computation_delay for m in service_clients])
        avg_comm_energy = np.mean([m.communication_energy for m in service_clients])
        avg_comm_delay = np.mean([m.communication_delay for m in service_clients])
        
        return {
            'service_id': service_id,
            'num_clients': len(service_clients),
            'total_energy': total_energy,
            'total_delay': total_delay,
            'avg_computation_energy': avg_comp_energy,
            'avg_computation_delay': avg_comp_delay,
            'avg_communication_energy': avg_comm_energy,
            'avg_communication_delay': avg_comm_delay,
            'client_metrics': {m.client_id: {
                'computation_energy': m.computation_energy,
                'computation_delay': m.computation_delay,
                'communication_energy': m.communication_energy,
                'communication_delay': m.communication_delay,
                'total_energy': m.total_energy,
                'total_delay': m.total_delay
            } for m in service_clients}
        }


def test_communication_model():
    """通信模型的测试函数。"""
    print("测试通信模型...")
    
    # 初始化系统指标
    system = SystemMetrics()
    
    # 测试参数
    clients = [
        {'id': 1, 'service': 1, 'mu': 1e-28, 'c': 1000, 'dataset_size': 1000, 
         'freq': 1e9, 'bandwidth': 1e6, 'gain': 1e-3, 'power': 0.1},
        {'id': 2, 'service': 1, 'mu': 1.2e-28, 'c': 1200, 'dataset_size': 1200, 
         'freq': 1.2e9, 'bandwidth': 1.2e6, 'gain': 1.2e-3, 'power': 0.12},
    ]
    
    communication_volume = 50000  # 比特
    
    # 为每个客户端计算指标
    for client in clients:
        metrics = system.calculate_client_metrics(
            client['id'], client['service'],
            client['mu'], client['c'], client['dataset_size'], client['freq'],
            communication_volume, client['bandwidth'], client['gain'], client['power']
        )
        system.add_client_metrics(metrics)
        
        print(f"客户端 {client['id']}:")
        print(f"  计算：能耗={metrics.computation_energy:.6f}焦耳，延迟={metrics.computation_delay:.6f}秒")
        print(f"  通信：能耗={metrics.communication_energy:.6f}焦耳，延迟={metrics.communication_delay:.6f}秒")
        print(f"  总计：能耗={metrics.total_energy:.6f}焦耳，延迟={metrics.total_delay:.6f}秒")
    
    # 计算服务级指标
    service_summary = system.get_service_metrics_summary(1)
    print(f"\n服务1摘要:")
    print(f"  总能耗: {service_summary['total_energy']:.6f}焦耳")
    print(f"  总延迟: {service_summary['total_delay']:.6f}秒")
    print(f"  客户端数量: {service_summary['num_clients']}")


if __name__ == "__main__":
    test_communication_model()
