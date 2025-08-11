#!/usr/bin/env python3
"""
联邦学习量化模块

实现论文中方程(5)-(7)描述的量化过程：
- 模型参数的q级量化
- 通信量计算
- 支持不同的量化策略
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
import math


class QuantizationModule:
    """
    实现联邦学习中的模型参数量化。
    
    基于论文中的方程(5)-(7)：
    - Ψ_q(ω_d) = ||ω||_p · sgn(ω_d) · Ξ_q(ω_d, q)
    - 通信量计算
    """
    
    def __init__(self, p_norm: int = 2):
        """
        初始化量化模块。
        
        参数:
            p_norm: 用于标准化的p范数 (默认: 2为L2范数)
        """
        self.p_norm = p_norm
    
    def quantize_parameters(self, parameters: torch.Tensor, q: int) -> Tuple[torch.Tensor, float]:
        """
        使用q级量化对模型参数进行量化。
        
        参数:
            parameters: 要量化的模型参数张量
            q: 量化级别
            
        返回:
            (量化后参数, 范数值) 的元组
        """
        # 如果需要，展平参数
        original_shape = parameters.shape
        params_flat = parameters.flatten()
        
        # 计算参数向量的p范数
        if self.p_norm == float('inf'):
            norm_value = torch.max(torch.abs(params_flat)).item()
        else:
            norm_value = torch.norm(params_flat, p=self.p_norm).item()
        
        if norm_value == 0:
            return torch.zeros_like(parameters), 0.0
        
        # 初始化量化参数
        quantized_flat = torch.zeros_like(params_flat)
        
        # 对每个元素应用量化
        for i, param_d in enumerate(params_flat):
            quantized_flat[i] = self._quantize_element(param_d.item(), norm_value, q)
        
        # 重新调整为原始形状
        quantized_params = quantized_flat.reshape(original_shape)
        
        return quantized_params, norm_value
    
    def _quantize_element(self, omega_d: float, norm_value: float, q: int) -> float:
        """
        量化单个参数元素。
        
        实现方程(5): Ψ_q(ω_d) = ||ω||_p · sgn(ω_d) · Ξ_q(ω_d, q)
        """
        if norm_value == 0:
            return 0.0
            
        # 计算标准化绝对值
        e = abs(omega_d) / norm_value
        
        # 找到u使得 u/q <= e <= (u+1)/q
        u = min(int(e * q), q - 1)
        
        # 计算 P(e, q) = eq - u
        P_eq = e * q - u
        
        # 生成随机变量 ε ~ Uniform(0, 1)
        epsilon = np.random.uniform(0, 1)
        
        # 根据方程(6)应用随机量化
        if epsilon <= 1 - P_eq:
            xi_q = u / q
        else:
            xi_q = (u + 1) / q
        
        # 返回带符号的量化值
        return norm_value * np.sign(omega_d) * xi_q
    
    def calculate_communication_volume(self, param_size: int, q: int) -> int:
        """
        计算量化参数的通信量。
        
        实现方程(7): vol_{r,t} = |ω|(⌈log_2(q)⌉ + 1) + 32
        
        参数:
            param_size: 参数数量 |ω|
            q: 量化级别
            
        返回:
            通信量(比特)
        """
        # 每个量化元素使用 ⌈log_2(q)⌉ + 1 比特 (包括符号位)
        bits_per_element = math.ceil(math.log2(q)) + 1
        
        # 总量: 参数比特 + 32比特用于梯度范数
        volume = param_size * bits_per_element + 32
        
        return volume
    
    def get_compression_ratio(self, param_size: int, q: int) -> float:
        """
        计算相对于32位浮点表示的压缩比。
        
        参数:
            param_size: 参数数量
            q: 量化级别
            
        返回:
            压缩比 (原始大小 / 压缩大小)
        """
        original_volume = param_size * 32  # 每个浮点数32比特
        compressed_volume = self.calculate_communication_volume(param_size, q)
        
        return original_volume / compressed_volume


class AdaptiveQuantization:
    """
    自适应量化策略，根据训练进度调整量化级别。
    """
    
    def __init__(self, initial_q: int = 4, max_q: int = 32, min_q: int = 2):
        """
        初始化自适应量化。
        
        参数:
            initial_q: 初始量化级别
            max_q: 最大量化级别
            min_q: 最小量化级别
        """
        self.current_q = initial_q
        self.max_q = max_q
        self.min_q = min_q
        self.quantizer = QuantizationModule()
    
    def update_quantization_level(self, accuracy_improvement: float, threshold: float = 0.01):
        """
        根据精度改善情况更新量化级别。
        
        参数:
            accuracy_improvement: 模型精度改善程度
            threshold: 调整量化级别的阈值
        """
        if accuracy_improvement > threshold:
            # 进展良好，可以承受更多压缩
            self.current_q = max(self.min_q, self.current_q - 1)
        else:
            # 进展不佳，需要减少压缩
            self.current_q = min(self.max_q, self.current_q + 1)
    
    def quantize(self, parameters: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        使用当前量化级别对参数进行量化。
        
        返回:
            (量化后参数, 范数值, 量化级别) 的元组
        """
        quantized_params, norm_value = self.quantizer.quantize_parameters(
            parameters, self.current_q
        )
        return quantized_params, norm_value, self.current_q


def test_quantization():
    """量化模块的测试函数。"""
    print("测试量化模块...")
    
    # 创建测试参数
    params = torch.randn(100, 50)  # 5000个参数
    q = 8  # 8级量化
    
    quantizer = QuantizationModule()
    
    # 测试量化
    quantized_params, norm_value = quantizer.quantize_parameters(params, q)
    
    # 计算通信量
    param_size = params.numel()
    volume = quantizer.calculate_communication_volume(param_size, q)
    compression_ratio = quantizer.get_compression_ratio(param_size, q)
    
    print(f"原始参数形状: {params.shape}")
    print(f"量化级别: {q}")
    print(f"范数值: {norm_value:.4f}")
    print(f"通信量: {volume} 比特")
    print(f"压缩比: {compression_ratio:.2f}倍")
    
    # 测试自适应量化
    adaptive_q = AdaptiveQuantization(initial_q=4)
    quantized_adaptive, norm_adaptive, q_level = adaptive_q.quantize(params)
    print(f"自适应量化级别: {q_level}")


if __name__ == "__main__":
    test_quantization()