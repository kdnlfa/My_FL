#!/usr/bin/env python3
"""
论文实验复现模块

基于论文的实验设计
复现包含5个客户端、3个服务提供商的多服务联邦学习实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
import copy

# 导入现有模块
from multi_service_fl import MultiServiceFLSystem, ServiceProviderConfig, ClientResourceConfig, QuantizedFLModel
from optimization_problem import OptimizationConstraints
from mdp_framework import MultiServiceFLEnvironment, Action, Observation
from pac_mcofl import PACMCoFLTrainer, PACConfig
from quantization import QuantizationModule
from communication import SystemMetrics


@dataclass
class PaperExperimentConfig:
    """论文实验配置，严格按照论文参数表设置"""
    
    # 基本设置（论文Table 1）
    N: int = 5  # 客户端数量
    R: int = 3  # 任务数量（服务提供商数量）
    rho: float = 1.0  # 非独立同分布数据程度（IID）
    tau: int = 3  # 联邦学习本地更新步数
    # T: int = 35  # 联邦学习全局训练轮次
    T: int = 5  # 联邦学习全局训练轮次

    
    # 通信参数
    g_i_t_range: Tuple[float, float] = (-73.0, -63.0)  # 信道增益 [dB]
    N_0_range: Tuple[float, float] = (-174.0, -124.0)  # 噪声功率谱密度 [dBm/Hz]
    p_i_t_range: Tuple[float, float] = (10.0, 33.0)  # 客户端发射功率 [dBm]
    
    # 计算参数
    mu_i: float = 1e-27  # 有效切换电容常数
    c_i_1_range: Tuple[float, float] = (6.07e5, 7.41e5)  # 每样本CPU周期消耗量(r1)
    c_i_2_range: Tuple[float, float] = (6.07e5, 7.41e5)  # 每样本CPU周期消耗量(r2)
    c_i_3_range: Tuple[float, float] = (1.10e8, 1.34e8)  # 每样本CPU周期消耗量(r3)
    
    # 权重因子（论文方程17的σ参数）
    sigma_1_values: List[float] = field(default_factory=lambda: [100.0, 100.0, 100.0])  # r1,r2,r3
    sigma_2_values: List[float] = field(default_factory=lambda: [4.8, 31.25, 12.5])
    sigma_3_values: List[float] = field(default_factory=lambda: [0.8, 25.0, 16.6])
    sigma_4_values: List[float] = field(default_factory=lambda: [0.8, 25.0, 16.6])
    
    # PAC算法参数
    zeta: float = 0.001  # 行动者网络学习率
    alpha: float = 0.001  # 评论家网络学习率
    
    # 抖动因子
    sigma_q: float = 0.25  # 量化抖动因子
    sigma_f: float = 0.5  # CPU频率抖动因子
    
    # 系统约束
    f_min: float = 0.5e9  # 最小CPU频率 [Hz]
    f_max: float = 3.5e9  # 最大CPU频率 [Hz]
    B_min: float = 0.0  # 最小带宽 [Hz]
    B_max: float = 30e6  # 最大带宽 [Hz]
    
    # 数据集配置
    batch_size: int = 64
    learning_rate: float = 0.001
    
    def get_service_sigma_values(self, service_id: int) -> Dict[str, float]:
        """获取特定服务的权重因子"""
        idx = service_id - 1  # 服务ID从1开始，数组索引从0开始
        return {
            'sigma_1': self.sigma_1_values[idx],
            'sigma_2': self.sigma_2_values[idx], 
            'sigma_3': self.sigma_3_values[idx],
            'sigma_4': self.sigma_4_values[idx]
        }


class TaskModels:
    """论文中三个任务的模型架构定义"""
    
    @staticmethod
    def create_task_r1_model(num_classes: int = 10) -> nn.Module:
        """
        任务r1的模型：四层卷积神经网络(CNN)用于CIFAR-10
        卷积层：48、96、192、256个滤波器，3×3核尺寸
        全连接：512、64、10个神经元层
        """
        class TaskR1CNN(nn.Module):
            def __init__(self, num_classes=10):
                super(TaskR1CNN, self).__init__()
                
                # 四层卷积网络
                self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
                self.conv4 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
                
                # 池化层
                self.pool = nn.MaxPool2d(2, 2)
                
                # 全连接网络：512、64、10个神经元
                self.fc1 = nn.Linear(256 * 2 * 2, 512)  # CIFAR-10是32×32，经过4次池化后是2×2
                self.fc2 = nn.Linear(512, 64)
                self.fc3 = nn.Linear(64, num_classes)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # 卷积层
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = self.pool(self.relu(self.conv4(x)))
                
                # 展平
                x = x.view(-1, 256 * 2 * 2)
                
                # 全连接层
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                
                return x
        
        return TaskR1CNN(num_classes)
    
    @staticmethod
    def create_task_r2_model(num_classes: int = 10) -> nn.Module:
        """
        任务r2的模型：双层CNN架构用于FashionMNIST
        卷积层：32和64个5×5滤波器
        全连接：64和10个神经元
        """
        class TaskR2CNN(nn.Module):
            def __init__(self, num_classes=10):
                super(TaskR2CNN, self).__init__()
                
                # 双层卷积网络
                self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # FashionMNIST是灰度图
                self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
                
                # 池化层
                self.pool = nn.MaxPool2d(2, 2)
                
                # 全连接网络：64和10个神经元
                self.fc1 = nn.Linear(64 * 7 * 7, 64)  # FashionMNIST是28×28，经过2次池化后是7×7
                self.fc2 = nn.Linear(64, num_classes)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # 卷积层
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                
                # 展平
                x = x.view(-1, 64 * 7 * 7)
                
                # 全连接层
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        
        return TaskR2CNN(num_classes)
    
    @staticmethod 
    def create_task_r3_model(num_classes: int = 10) -> nn.Module:
        """
        任务r3的模型：双层全连接网络用于MNIST
        全连接：128和10个神经元
        """
        class TaskR3MLP(nn.Module):
            def __init__(self, num_classes=10):
                super(TaskR3MLP, self).__init__()
                
                # 双层全连接网络：128和10个神经元
                self.fc1 = nn.Linear(28 * 28, 128)  # MNIST是28×28灰度图
                self.fc2 = nn.Linear(128, num_classes)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # 展平输入
                x = x.view(-1, 28 * 28)
                
                # 全连接层
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        
        return TaskR3MLP(num_classes)


class DatasetLoader:
    """论文实验数据集加载器，支持IID分布"""
    
    def __init__(self, data_dir: str = "../data"):
        """
        初始化数据集加载器
        
        参数:
            data_dir: 数据集存储目录（使用上级目录的data文件夹）
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            # 如果上级目录不存在，使用当前目录
            self.data_dir = Path("./data")
            self.data_dir.mkdir(exist_ok=True)
        
        # 数据预处理配置
        self.transform_configs = {
            'cifar10': {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            },
            'fashionmnist': {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            },
            'mnist': {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            }
        }
    
    def load_datasets(self) -> Dict[str, Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]]:
        """
        加载三个数据集：CIFAR-10、FashionMNIST、MNIST
        
        返回:
            包含训练和测试数据集的字典
        """
        datasets = {}
        
        # CIFAR-10 (任务r1)
        cifar10_train = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=False,
            transform=self.transform_configs['cifar10']['train']
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=False,
            transform=self.transform_configs['cifar10']['test']
        )
        datasets['cifar10'] = (cifar10_train, cifar10_test)
        
        # FashionMNIST (任务r2)
        fashionmnist_train = torchvision.datasets.FashionMNIST(
            root=str(self.data_dir), train=True, download=False,
            transform=self.transform_configs['fashionmnist']['train']
        )
        fashionmnist_test = torchvision.datasets.FashionMNIST(
            root=str(self.data_dir), train=False, download=False,
            transform=self.transform_configs['fashionmnist']['test']
        )
        datasets['fashionmnist'] = (fashionmnist_train, fashionmnist_test)
        
        # MNIST (任务r3)
        mnist_train = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=True, download=False,
            transform=self.transform_configs['mnist']['train']
        )
        mnist_test = torchvision.datasets.MNIST(
            root=str(self.data_dir), train=False, download=False,
            transform=self.transform_configs['mnist']['test']
        )
        datasets['mnist'] = (mnist_train, mnist_test)
        
        return datasets
    

    def create_iid_client_splits(self, 
                                datasets: Dict[str, Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]],
                                num_clients: int = 5,
                                rho: float = 1.0) -> Dict[int, Dict[str, torch.utils.data.Dataset]]:
        """
        创建服务感知的IID数据分割：
        CIFAR-10 -> 客户端1,2
        FashionMNIST -> 客户端3,4
        MNIST -> 客户端5
        避免原实现把每个数据集切成5份只用其中部分，导致数据浪费。
        """
        mapping = {
            'cifar10': [1, 2],
            'fashionmnist': [3, 4],
            'mnist': [5]
        }
        client_datasets = {i: {} for i in range(1, num_clients + 1)}
        
        for dataset_name, (train_ds, test_ds) in datasets.items():
            target_clients = mapping.get(dataset_name, [])
            if not target_clients:
                continue
            train_size = len(train_ds)
            per_client = train_size // len(target_clients)
            torch.manual_seed(42)  # 可重现
            indices = torch.randperm(train_size).tolist()
            
            for idx, client_id in enumerate(target_clients):
                start = idx * per_client
                end = start + per_client if idx < len(target_clients) - 1 else train_size
                sub_idx = indices[start:end]
                client_datasets[client_id][dataset_name] = torch.utils.data.Subset(train_ds, sub_idx)
        
        print(f"✅ 服务感知IID数据分割完成 (ρ={rho})")
        for cid, ds_dict in client_datasets.items():
            if not ds_dict:
                continue
            total = sum(len(ds) for ds in ds_dict.values())
            print(f"  客户端 {cid}: {total} 样本 -> {[f'{k}:{len(v)}' for k,v in ds_dict.items()]}")
        return client_datasets


class PaperExperimentRunner:
    """论文实验执行器，集成所有组件进行完整实验"""
    
    def __init__(self, config: PaperExperimentConfig, output_dir: str = "./experimental_data"):
        """
        初始化实验执行器
        
        参数:
            config: 论文实验配置
            output_dir: 实验结果输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.dataset_loader = DatasetLoader()
        self.datasets = None
        self.client_datasets = None
        self.fl_system = None
        self.pac_trainer = None
        self.environment = None
        
        # 实验结果存储
        self.experiment_results = {}
        
        print(f"📊 论文实验执行器初始化完成")
        print(f"   输出目录: {self.output_dir}")
        print(f"   客户端数量: {self.config.N}")
        print(f"   服务提供商数量: {self.config.R}")
        print(f"   训练轮次: {self.config.T}")
    
    def setup_datasets(self):
        """设置和分割数据集"""
        print("\n🔄 设置数据集...")
        
        # 加载原始数据集
        self.datasets = self.dataset_loader.load_datasets()
        
        # 创建IID客户端分割
        self.client_datasets = self.dataset_loader.create_iid_client_splits(
            self.datasets, 
            num_clients=self.config.N,
            rho=self.config.rho
        )
    
    def setup_models_and_services(self):
        """设置模型和服务提供商配置"""
        print("\n🔄 设置模型和服务...")
        
        # 按照论文设计创建服务提供商配置
        service_configs = []
        
        # 服务1：CIFAR-10任务，客户端1-2
        service_configs.append(ServiceProviderConfig(
            service_id=1,
            name="CIFAR-10分类服务",
            client_ids=[1, 2],
            model_architecture={
                "type": "task_r1_cnn",
                "description": "四层CNN：48,96,192,256滤波器+512,64,10全连接",
                "dataset": "cifar10"
            },
            quantization_level=8,
            local_epochs=self.config.tau,
            learning_rate=self.config.learning_rate
        ))
        
        # 服务2：FashionMNIST任务，客户端3-4
        service_configs.append(ServiceProviderConfig(
            service_id=2,
            name="FashionMNIST分类服务", 
            client_ids=[3, 4],
            model_architecture={
                "type": "task_r2_cnn",
                "description": "双层CNN：32,64滤波器+64,10全连接",
                "dataset": "fashionmnist"
            },
            quantization_level=8,
            local_epochs=self.config.tau,
            learning_rate=self.config.learning_rate
        ))
        
        # 服务3：MNIST任务，客户端5
        service_configs.append(ServiceProviderConfig(
            service_id=3,
            name="MNIST分类服务",
            client_ids=[5],
            model_architecture={
                "type": "task_r3_mlp", 
                "description": "双层MLP：128,10神经元",
                "dataset": "mnist"
            },
            quantization_level=8,
            local_epochs=self.config.tau,
            learning_rate=self.config.learning_rate
        ))
        
        # 创建客户端资源配置（按照论文参数表）
        client_configs = {}
        for client_id in range(1, self.config.N + 1):
            # 根据客户端所属服务确定计算复杂度
            if client_id in [1, 2]:  # CIFAR-10客户端
                c_ir_range = self.config.c_i_1_range
            elif client_id in [3, 4]:  # FashionMNIST客户端  
                c_ir_range = self.config.c_i_2_range
            else:  # MNIST客户端
                c_ir_range = self.config.c_i_3_range
            
            # 随机采样参数（在论文给定范围内）
            np.random.seed(42 + client_id)  # 确保可重现性
            
            client_configs[client_id] = ClientResourceConfig(
                client_id=client_id,
                mu_i=self.config.mu_i,
                c_ir=np.random.uniform(*c_ir_range),
                max_frequency=self.config.f_max,
                max_power=10**(np.random.uniform(*self.config.p_i_t_range) / 10),  # dBm转瓦特
                channel_gain=10**(np.random.uniform(*self.config.g_i_t_range) / 10),  # dB转线性
                dataset_size=1000  # 假设每个客户端有1000个样本
            )
        
        # 创建多服务FL系统
        self.fl_system = MultiServiceFLSystem(
            service_configs,
            client_configs,
            debug_disable_quant=False,   # 暂时关闭量化影响，确保先把基础训练跑通
            debug_verbose=False        # 打印更多调试信息
        )
        
        print(f"✅ 创建了 {len(service_configs)} 个服务提供商")
        for config in service_configs:
            print(f"   服务{config.service_id}: {config.name}, 客户端{config.client_ids}")
    
    def setup_models_for_services(self):
        """为每个服务设置具体的模型和数据"""
        print("\n🔄 为服务设置模型...")
        
        # 服务1：CIFAR-10
        model_r1 = TaskModels.create_task_r1_model(num_classes=10)
        service1_train_datasets = {
            client_id: self.client_datasets[client_id]['cifar10'] 
            for client_id in [1, 2]
        }
        self.fl_system.setup_service(
            service_id=1,
            model=model_r1,
            train_datasets=service1_train_datasets
        )
        print(f"✅ 服务1设置完成：CIFAR-10任务，四层CNN模型")
        
        # 服务2：FashionMNIST
        model_r2 = TaskModels.create_task_r2_model(num_classes=10)
        service2_train_datasets = {
            client_id: self.client_datasets[client_id]['fashionmnist']
            for client_id in [3, 4]
        }
        self.fl_system.setup_service(
            service_id=2,
            model=model_r2,
            train_datasets=service2_train_datasets
        )
        print(f"✅ 服务2设置完成：FashionMNIST任务，双层CNN模型")
        
        # 服务3：MNIST
        model_r3 = TaskModels.create_task_r3_model(num_classes=10)
        service3_train_datasets = {
            client_id: self.client_datasets[client_id]['mnist']
            for client_id in [5]
        }
        self.fl_system.setup_service(
            service_id=3,
            model=model_r3,
            train_datasets=service3_train_datasets
        )
        print(f"✅ 服务3设置完成：MNIST任务，双层MLP模型")
    
    def setup_pac_environment(self):
        """设置PAC-MCoFL环境和训练器"""
        print("\n🔄 设置PAC-MCoFL环境...")
        
        # 创建约束条件（基于论文参数）
        constraints = OptimizationConstraints(
            max_energy=0.1,  # 100 mJ
            max_delay=5.0,   # 5秒
            max_clients=self.config.N,
            min_frequency=self.config.f_min,
            max_frequency=self.config.f_max,
            min_bandwidth=self.config.B_min,
            max_bandwidth=self.config.B_max
        )
        
        # 创建环境
        service_ids = [1, 2, 3]
        
        # 为每个服务创建权重配置（基于论文σ参数）
        per_service_reward_weights = {}
        for service_id in service_ids:
            sigma_values = self.config.get_service_sigma_values(service_id)
            # 环境支持传入按服务ID（字符串）索引的权重表
            per_service_reward_weights[str(service_id)] = sigma_values
        
        self.environment = MultiServiceFLEnvironment(
            service_ids=service_ids,
            constraints=constraints,
            max_rounds=self.config.T,
            reward_weights=per_service_reward_weights  # 为每个服务分别设置奖励权重
        )

        
        
        # 创建PAC配置（基于论文PAC算法参数）
        pac_config = PACConfig(
            num_episodes=2,  # 调试联动：先跑少量episode验证
            max_rounds_per_episode=15,  # 减少每episode轮数以加速
            buffer_size=2000,
            batch_size=8,
            actor_hidden_dim=64,    # 论文：策略网络64-128-64
            critic_hidden_dim=64,   # 论文：Q网络64-128
            num_layers=3,
            actor_lr=self.config.zeta,   # 论文：ζ=0.001
            critic_lr=self.config.alpha, # 论文：α=0.001
            gamma=0.95,
            joint_action_samples=100,
            update_frequency=4,
            # 三份训练方案（按服务）
            # 方案A（服务1，CIFAR-10）: 准确率优先 + 稳定性，提升训练量与资源下限，评估每步
            step_eval_frequency=1,
            service_eval_frequency={1: 1, 2: 1, 3: 2},
            service_epochs_per_step={1: 5, 2: 3, 3: 1},
            service_action_floors={
                1: { 'min_clients': 2, 'min_frequency': 1.5e9, 'min_bandwidth': 15e6, 'min_quantization': 8 },
                2: { 'min_clients': 2, 'min_frequency': 1.2e9, 'min_bandwidth': 10e6, 'min_quantization': 6 },
                3: { 'min_clients': 1, 'min_frequency': 1.0e9, 'min_bandwidth': 5e6,  'min_quantization': 4 }
            }
        )

        # pac_config = PACConfig(
        #     num_episodes=5,  # 调试联动：先跑少量episode验证
        #     max_rounds_per_episode=10,  # 减少每episode轮数以加速
        #     buffer_size=10000,
        #     batch_size=64,
        #     actor_hidden_dim=64,    # 论文：策略网络64-128-64
        #     critic_hidden_dim=64,   # 论文：Q网络64-128
        #     num_layers=3,
        #     actor_lr=self.config.zeta,   # 论文：ζ=0.001
        #     critic_lr=self.config.alpha, # 论文：α=0.001
        #     gamma=0.95,
        #     joint_action_samples=100,
        #     update_frequency=4,
        #     # 三份训练方案（按服务）
        #     # 方案A（服务1，CIFAR-10）: 准确率优先 + 稳定性，提升训练量与资源下限，评估每步
        #     step_eval_frequency=1,
        #     service_eval_frequency={1: 1, 2: 1, 3: 2},
        #     service_epochs_per_step={1: 5, 2: 3, 3: 1},
        #     service_action_floors={
        #         1: { 'min_clients': 2, 'min_frequency': 1.5e9, 'min_bandwidth': 15e6, 'min_quantization': 8 },
        #         2: { 'min_clients': 2, 'min_frequency': 1.2e9, 'min_bandwidth': 10e6, 'min_quantization': 6 },
        #         3: { 'min_clients': 1, 'min_frequency': 1.0e9, 'min_bandwidth': 5e6,  'min_quantization': 4 }
        #     }
        # )

        
        # 创建PAC训练器
        self.pac_trainer = PACMCoFLTrainer(
            service_ids=service_ids,
            environment=self.environment,
            fl_system=self.fl_system,
            config=pac_config,
            constraints=constraints
        )
        
        print(f"✅ PAC-MCoFL环境设置完成")
        print(f"   Q网络架构: {pac_config.critic_hidden_dim}-128节点（双层）")
        print(f"   策略网络架构: 64-128-64节点（三层）") 
        print(f"   学习率: 演员={pac_config.actor_lr}, 评论家={pac_config.critic_lr}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行完整的论文实验"""
        print("\n🚀 开始论文实验复现...")
        
        start_time = time.time()
        
        try:
            # 1. 设置数据集
            self.setup_datasets()
            
            # 2. 设置模型和服务
            self.setup_models_and_services()
            
            # 3. 为服务设置具体模型
            self.setup_models_for_services()
            
            # 4. 设置PAC环境
            self.setup_pac_environment()
            
            # 5. 先运行一次真实的联邦学习基线训练(预热)，确保模型权重发生有效更新
            # baseline_plan = {1: 5, 2: 3, 3: 1}
            # print(f"\n🔄 先进行基线联邦训练(预热): {baseline_plan} 轮，禁用量化评估干扰...")
            # saved_epochs = {}
            # for sid, trainer in self.fl_system.service_trainers.items():
            #     if hasattr(trainer, 'cfg'):
            #         saved_epochs[sid] = getattr(trainer.cfg, 'epochs', 1)
            #         trainer.cfg.epochs = max(3, saved_epochs[sid])
            # baseline_fl_training = {}
            # for sid, rounds in baseline_plan.items():
            #     try:
            #         self.fl_system.train_service(sid, num_rounds=rounds, enable_metrics=False)
            #         baseline_fl_training[sid] = (self.fl_system.service_models[sid], {'rounds': rounds})
            #     except Exception as e:
            #         print(f"[WARN] 基线预热失败 - 服务{sid}: {e}")
            # for sid, trainer in self.fl_system.service_trainers.items():
            #     try:
            #         if hasattr(trainer, 'cfg') and sid in saved_epochs:
            #             trainer.cfg.epochs = saved_epochs[sid]
            #     except Exception:
            #         pass

            baseline_fl_training = {}

            # 6. 运行PAC-MCoFL训练
            print(f"\n🔄 开始PAC-MCoFL训练...")
            training_results = self.pac_trainer.train()
            # 训练完成后立即持久化一次原始训练结果快照，便于排查奖励直线问题
            try:
                snapshot_path = self.output_dir / f"training_results_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(snapshot_path, 'w', encoding='utf-8') as sf:
                    import json as _json
                    def _to_serializable(o):
                        import numpy as _np
                        if isinstance(o, _np.ndarray):
                            return o.tolist()
                        if isinstance(o, (float, int, str, bool)) or o is None:
                            return o
                        if isinstance(o, dict):
                            return {k: _to_serializable(v) for k, v in o.items()}
                        if isinstance(o, list):
                            return [_to_serializable(x) for x in o]
                        return str(o)
                    _json.dump(_to_serializable(training_results), sf, ensure_ascii=False, indent=2)
                print(f"💾  已保存训练快照: {snapshot_path}")
            except Exception as e:
                print(f"[WARN] 保存训练快照失败: {e}")
            
            # 7. 评估训练结果（RL层面）
            print(f"\n🔄 评估训练结果...")
            evaluation_results = self.pac_trainer.evaluate(num_episodes=10)
            
            # 8. 获取训练总结
            training_summary = self.pac_trainer.get_training_summary()
            
            # 9. 评估联邦学习模型性能（真实模型）
            print(f"\n🔄 评估联邦学习模型性能...")
            model_performance = self.evaluate_model_performance()
            
            # 10. 编译实验结果
            end_time = time.time()
            
            self.experiment_results = {
                'config': {
                    'N': self.config.N,
                    'R': self.config.R,
                    'T': self.config.T,
                    'tau': self.config.tau,
                    'rho': self.config.rho,
                    'learning_rates': {
                        'fl': self.config.learning_rate,
                        'actor': self.config.zeta,
                        'critic': self.config.alpha
                    },
                    'model_architectures': {
                        'r1': "四层CNN(48,96,192,256)+FC(512,64,10)",
                        'r2': "双层CNN(32,64)+FC(64,10)", 
                        'r3': "双层MLP(128,10)"
                    }
                },
                'baseline_fl_training': {
                    sid: {
                        'metrics': metrics,
                        'has_model': sid in self.fl_system.service_models
                    } for sid, (model, metrics) in baseline_fl_training.items()
                },
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'model_performance': model_performance,  # 模型性能结果
                'training_summary': training_summary,
                'experiment_duration': end_time - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 9. 保存实验结果
            self.save_results()
            
            print(f"\n✅ 论文实验复现完成！")
            print(f"   总用时: {end_time - start_time:.2f} 秒")
            print(f"   结果保存在: {self.output_dir}")
            
            return self.experiment_results
            
        except Exception as e:
            print(f"\n❌ 实验执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    
    def evaluate_model_performance(self) -> Dict[str, Dict[str, float]]:
        """
        评估三个模型在各自数据集上的性能
        返回每个服务的准确率、损失等指标
        """
        results = {}
        
        print("📊 开始评估训练后的模型性能...")
        print("-" * 50)
        
        for service_id in range(1, 4):  # 三个服务
            service_name = {1: "CIFAR-10", 2: "FashionMNIST", 3: "MNIST"}[service_id]
            dataset_name = {1: "cifar10", 2: "fashionmnist", 3: "mnist"}[service_id]
            
            print(f"\n🔍 评估服务 {service_id} ({service_name}):")
            
            try:
                # ✅ 使用GPU设备保持与训练一致
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"   🔧 评估设备: {device}")
                
                # ✅ 使用训练后的模型而不是随机初始化的模型
                if hasattr(self.fl_system, 'service_models') and service_id in self.fl_system.service_models:
                    # 获取训练后的量化模型
                    fl_model = self.fl_system.service_models[service_id]
                    model = fl_model.model  # 提取PyTorch模型
                    
                    # 检查模型当前设备并确保在正确设备上
                    current_device = next(model.parameters()).device
                    if current_device != device:
                        model = model.to(device)
                        print(f"   🔄 模型从 {current_device} 移动到 {device}")
                    else:
                        print(f"   ✅ 使用训练后的模型 (设备: {device})")
                else:
                    # 如果没有训练后的模型，使用随机初始化模型作为后备
                    print(f"   ⚠️  警告：未找到训练后的模型，使用随机初始化模型")
                    if service_id == 1:
                        model = TaskModels.create_task_r1_model()
                    elif service_id == 2:
                        model = TaskModels.create_task_r2_model()
                    else:
                        model = TaskModels.create_task_r3_model()
                    model = model.to(device)
                
                # 获取测试数据
                test_dataset = self.datasets[dataset_name][1]  # [1]是测试集
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=64, shuffle=False
                )
                
                # 评估模型
                model.eval()
                correct = 0
                total = 0
                total_loss = 0.0
                criterion = nn.CrossEntropyLoss()
                
                with torch.no_grad():
                    batch_count = 0
                    for data, target in test_loader:
                        batch_count += 1
                        
                        # 确保数据也在正确的设备上
                        data = data.to(device)
                        target = target.to(device)
                        
                        # 对于MNIST，需要展平输入
                        if dataset_name == "mnist":
                            data = data.view(data.size(0), -1)
                        
                        output = model(data)
                        loss = criterion(output, target)
                        total_loss += loss.item()
                        
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                        
                        # 限制评估样本数量以节省时间
                        # if batch_count >= 20:  # 只评估前20个batch
                        #     break
                
                accuracy = 100. * correct / total if total > 0 else 0.0
                avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
                
                results[service_id] = {
                    'dataset': service_name,
                    'accuracy': accuracy,
                    'loss': avg_loss,
                    'correct': correct,
                    'total': total,
                    'batches_evaluated': batch_count
                }
                
                print(f"   ✅ 准确率: {accuracy:.2f}% ({correct}/{total})")
                print(f"   ✅ 平均损失: {avg_loss:.4f}")
                print(f"   ✅ 评估批次: {batch_count}")
                
            except Exception as e:
                print(f"   ❌ 评估失败: {str(e)}")
                results[service_id] = {
                    'dataset': service_name,
                    'accuracy': 0.0,
                    'loss': float('inf'),
                    'error': str(e)
                }
        
        return results

    def save_results(self):
        """保存实验结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存完整结果
        results_file = self.output_dir / f"paper_experiment_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 处理numpy类型序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            def serialize_dict(d):
                if isinstance(d, dict):
                    return {k: serialize_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [serialize_dict(item) for item in d]
                else:
                    return convert_numpy(d)
            
            serializable_results = serialize_dict(self.experiment_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 实验结果已保存: {results_file}")

        # 附加：保存强化学习训练过程中的动作日志与准确率趋势
        try:
            if 'training_results' in self.experiment_results:
                tr = self.experiment_results['training_results']
                # 保存动作日志
                if 'action_logs' in tr:
                    rl_logs_file = self.output_dir / f"rl_action_logs_{timestamp}.json"
                    with open(rl_logs_file, 'w', encoding='utf-8') as rf:
                        json.dump(tr['action_logs'], rf, indent=2, ensure_ascii=False)
                    print(f"💾 强化学习动作日志已保存: {rl_logs_file}")
        except Exception as e:
            print(f"[WARN] 保存RL过程日志失败: {e}")

        # 追加：绘制并保存三个服务的准确率趋势图（来自RL快速评估结果）
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt

            tr = self.experiment_results.get('training_results', {})
            acc_trends = tr.get('accuracy_trends', {})
            if acc_trends:
                fig, ax = plt.subplots(figsize=(8, 4))
                for sid_str, series in acc_trends.items():
                    try:
                        sid = int(sid_str)
                    except Exception:
                        sid = sid_str
                    ax.plot(range(1, len(series) + 1), series, label=f'Service {sid}')
                ax.set_xlabel('RL steps')
                ax.set_ylabel('Accuracy (quick eval)')
                ax.set_title('Accuracy Trend')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig_path = self.output_dir / f"accuracy_trend_rl_{timestamp}.png"
                fig.savefig(fig_path)
                plt.close(fig)
                print(f"🖼️  准确率趋势图已保存: {fig_path}")

            # 新增：绘制奖励值随训练过程的变化（每回合奖励）
            episode_rewards = tr.get('episode_rewards', {})
            if episode_rewards:
                fig_r, ax_r = plt.subplots(figsize=(8, 4))
                for sid_str, rewards in episode_rewards.items():
                    try:
                        sid = int(sid_str)
                    except Exception:
                        sid = sid_str
                    x = list(range(1, len(rewards) + 1))
                    ax_r.plot(x, rewards, marker='o', label=f'Service {sid}')
                ax_r.set_xlabel('Episode')
                ax_r.set_ylabel('Reward')
                ax_r.set_title('Episode Reward Trend')
                ax_r.legend()
                ax_r.grid(True, alpha=0.3)
                fig_r.tight_layout()
                reward_fig_path = self.output_dir / f"rl_reward_trends_{timestamp}.png"
                fig_r.savefig(reward_fig_path)
                plt.close(fig_r)
                print(f"🖼️  奖励趋势图已保存: {reward_fig_path}")

            # 新增：绘制按步奖励趋势（将每个episode的逐步奖励拼接）
            step_logs = tr.get('action_logs', [])
            if step_logs:
                # step_logs: List[episode] -> episode: List[{step, services:{sid:{reward,...}}}]
                per_service_step_rewards = {}
                for ep in step_logs:
                    for entry in ep:
                        if not isinstance(entry, dict) or 'services' not in entry:
                            continue
                        for sid, sdict in entry['services'].items():
                            try:
                                sid_int = int(sid)
                            except Exception:
                                sid_int = sid
                            per_service_step_rewards.setdefault(sid_int, []).append(float(sdict.get('reward', 0.0)))
                if per_service_step_rewards:
                    fig_sr, ax_sr = plt.subplots(figsize=(9, 4))
                    for sid, series in per_service_step_rewards.items():
                        ax_sr.plot(range(1, len(series) + 1), series, label=f'Service {sid}', alpha=0.9)
                    ax_sr.set_xlabel('RL steps (concatenated across episodes)')
                    ax_sr.set_ylabel('Reward (per step)')
                    ax_sr.set_title('Per-step Reward Trend')
                    ax_sr.legend()
                    ax_sr.grid(True, alpha=0.3)
                    fig_sr.tight_layout()
                    step_reward_fig_path = self.output_dir / f"rl_step_reward_trends_{timestamp}.png"
                    fig_sr.savefig(step_reward_fig_path)
                    plt.close(fig_sr)
                    print(f"🖼️  按步奖励趋势图已保存: {step_reward_fig_path}")

            # 可选：绘制累积期望奖励的变化
            cumulative_rewards = tr.get('cumulative_rewards', {})
            if cumulative_rewards:
                fig_c, ax_c = plt.subplots(figsize=(8, 4))
                for sid_str, cum_rewards in cumulative_rewards.items():
                    try:
                        sid = int(sid_str)
                    except Exception:
                        sid = sid_str
                    x = list(range(1, len(cum_rewards) + 1))
                    ax_c.plot(x, cum_rewards, marker='s', label=f'Service {sid}')
                ax_c.set_xlabel('Episode')
                ax_c.set_ylabel('Cumulative Reward J_r(π)')
                ax_c.set_title('Cumulative Expected Reward Trend')
                ax_c.legend()
                ax_c.grid(True, alpha=0.3)
                fig_c.tight_layout()
                cum_fig_path = self.output_dir / f"rl_cumulative_reward_trends_{timestamp}.png"
                fig_c.savefig(cum_fig_path)
                plt.close(fig_c)
                print(f"🖼️  累积期望奖励趋势图已保存: {cum_fig_path}")
        except Exception as e:
            print(f"[WARN] 绘制准确率趋势图失败: {e}")
    
    def print_experiment_summary(self):
        """打印实验结果摘要"""
        if not self.experiment_results:
            print("❌ 没有可用的实验结果")
            return
        
        print(f"\n" + "="*60)
        print(f"📊 论文实验复现结果摘要")
        print(f"="*60)
        
        # 配置信息
        config = self.experiment_results.get('config', {})
        print(f"\n🔧 实验配置:")
        print(f"   客户端数量: {config.get('N', 'N/A')}")
        print(f"   服务提供商数量: {config.get('R', 'N/A')}")
        print(f"   全局训练轮次: {config.get('T', 'N/A')}")
        print(f"   本地更新步数: {config.get('tau', 'N/A')}")
        print(f"   数据分布: IID (ρ={config.get('rho', 'N/A')})")
        
        # 模型架构
        models = config.get('model_architectures', {})
        print(f"\n🧠 模型架构:")
        for task, arch in models.items():
            print(f"   任务{task}: {arch}")
        
        # 模型性能结果（新增）
        model_performance = self.experiment_results.get('model_performance', {})
        if model_performance:
            print(f"\n🎯 模型性能结果:")
            total_accuracy = 0
            valid_services = 0
            
            for service_id, result in model_performance.items():
                if 'error' not in result:
                    print(f"   服务{service_id} ({result['dataset']}):")
                    print(f"     准确率: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
                    print(f"     损失: {result['loss']:.4f}")
                    print(f"     评估批次: {result['batches_evaluated']}")
                    total_accuracy += result['accuracy']
                    valid_services += 1
                else:
                    print(f"   服务{service_id} ({result['dataset']}): 评估失败")
            
            if valid_services > 0:
                avg_accuracy = total_accuracy / valid_services
                print(f"   📊 平均准确率: {avg_accuracy:.2f}%")
        
        # PAC算法性能结果
        eval_results = self.experiment_results.get('evaluation_results', {})
        if eval_results:
            print(f"\n📈 PAC算法性能:")
            
            avg_cumulative = eval_results.get('avg_cumulative_rewards', {})
            if avg_cumulative:
                print(f"   累积奖励J_r(π) (方程19-20):")
                for service_id, reward in avg_cumulative.items():
                    print(f"     服务{service_id}: {reward:.4f}")
            
            avg_episode = eval_results.get('avg_episode_rewards', {})
            if avg_episode:
                print(f"   平均回合奖励:")
                for service_id, reward in avg_episode.items():
                    print(f"     服务{service_id}: {reward:.4f}")
        
        # 训练信息
        duration = self.experiment_results.get('experiment_duration', 0)
        print(f"\n⏱️  训练用时: {duration:.2f} 秒")
        
        print(f"="*60)


def main():
    """主函数：运行论文实验复现"""
    
    # 创建实验配置
    config = PaperExperimentConfig()
    
    # 创建实验执行器
    runner = PaperExperimentRunner(config)
    
    # 运行实验
    results = runner.run_experiment()
    
    # 打印结果摘要
    runner.print_experiment_summary()
    
    return results


if __name__ == "__main__":
    results = main()
