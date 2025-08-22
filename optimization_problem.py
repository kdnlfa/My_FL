from dataclasses import dataclass

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