#!/usr/bin/env python3
"""
将 FLSim 库与论文的系统模型相集成，以实现：
- 拥有共享网络资源的多个服务提供商
- 量化感知联邦学习
- 通信感知的客户端选择与资源分配
- 能量与延迟优化
"""


try:
    from flsim.trainers.sync_trainer import SyncTrainer
    from flsim.servers.sync_servers import SyncServer, SyncServerConfig
    from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig
    from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
    from flsim.active_user_selectors.simple_user_selector import UniformlyRandomActiveUserSelector, UniformlyRandomActiveUserSelectorConfig
    from flsim.data.data_provider import IFLDataProvider, IFLUserData, FLDataProviderFromList
    from flsim.data.data_sharder import SequentialSharder
    from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
    from flsim.interfaces.data_loader import IFLDataLoader
    from flsim.interfaces.model import IFLModel
    from flsim.utils.config_utils import fl_config_from_json
    from hydra.utils import instantiate
    FLSIM_AVAILABLE = True
    print("FLSim 模块导入成功")
except ImportError as e:
    print(f"FLSim 模块导入失败: {e}")
    FLSIM_AVAILABLE = False

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy
import json
from dataclasses import dataclass
from pathlib import Path

from quantization import QuantizationModule, AdaptiveQuantization
from communication import SystemMetrics, ClientMetrics

@dataclass
class ServiceProviderConfig:
    """服务提供商的配置"""
    service_id: int
    name: str
    client_ids: List[int]
    model_architecture: Dict[str, Any]
    quantization_level: int = 8
    local_epochs: int = 1
    learning_rate: float = 0.01
    users_per_round: int = 10 # 服务每轮选几个客户端

# 类中的参数最后都会影响到优化问题
@dataclass 
class ClientResourceConfig:
    """客户端资源配置"""
    client_id: int
    mu_i: float  # 有效电容常数
    c_ir: float  # 每个样本所需的CPU周期数
    max_frequency: float  # 最大CPU频率
    max_power: float  # 最大传输功率
    channel_gain: float  # 信道增益
    dataset_size: int  # 本地数据集大小


class MultiServiceFLDataLoader(IFLDataLoader):
    """
    多服务联邦学习数据加载器。
    扩展FLSim的数据加载器以支持多服务。
    """
    
    def __init__(self, 
                 datasets: Dict[int, torch.utils.data.Dataset],
                 service_assignments: Dict[int, int],  # client_id -> service_id
                 batch_size: int = 32,
                 drop_last: bool = False,
                 eval_datasets: Optional[Dict[int, torch.utils.data.Dataset]] = None,
                 test_datasets: Optional[Dict[int, torch.utils.data.Dataset]] = None,
                 train_ratio: float = 0.8,
                 eval_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        初始化多服务数据加载器。
        
        参数:
            datasets: 将client_id映射到数据集的字典
            service_assignments: 从client_id到service_id的映射
            batch_size: 训练批次大小
            drop_last: 是否丢弃最后一个不完整的批次
            eval_datasets: 可选的评估数据集
            test_datasets: 可选的测试数据集
            train_ratio: 训练集比例 (默认0.8)
            eval_ratio: 评估集比例 (默认0.1)
            test_ratio: 测试集比例 (默认0.1)
            random_seed: 随机种子，确保可重现性
        """
        self.datasets = datasets
        self.service_assignments = service_assignments
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 数据分割比例
        assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, "数据分割比例之和必须等于1"
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio   

        # 分割后的数据集
        self.train_datasets: Dict[int, torch.utils.data.Dataset] = {}
        self.eval_datasets: Dict[int, torch.utils.data.Dataset] = {}
        self.test_datasets: Dict[int, torch.utils.data.Dataset] = {}     
        
        # 如果提供了独立的评估和测试数据集，直接使用
        if eval_datasets is not None:
            self.eval_datasets = eval_datasets
        if test_datasets is not None:
            self.test_datasets = test_datasets

        # 进行数据分割
        self._split_all_datasets()  

        # 按服务分组客户端
        '''
        结果可能是
        self.service_clients = {
            1: [1, 2, 3],  # 服务1对应客户端1,2,3
            2: [4, 5, 6],  # 服务2对应客户端4,5,6
        }
        '''
        self.service_clients: Dict[int, List[int]] = {}
        for client_id, service_id in service_assignments.items():
            if service_id not in self.service_clients:
                self.service_clients[service_id] = []
            self.service_clients[service_id].append(client_id)
    
    def fl_train_set(self, **kwargs) -> List[Dict[str, Any]]:
        """返回所有客户端的训练数据。"""
        return self._create_data_batches(self.train_datasets, shuffle=True)
    
    def fl_eval_set(self, **kwargs):
        """
        返回评估数据。
        
        评估数据用于：
        1. 训练过程中监控模型性能
        2. 超参数调优
        3. 早停策略
        4. 检测过拟合
        
        特点：
        - 不参与训练，只用于评估
        - 不打乱数据顺序（确保评估结果可重现）
        - 不丢弃最后一批数据（确保所有数据都被评估）
        """
        return self._create_data_batches(self.eval_datasets, shuffle=False)
    
    def fl_test_set(self, **kwargs) -> List[Dict[str, Any]]:
        """
        返回测试数据。
        
        测试数据用于：
        1. 最终模型性能评估
        2. 与其他方法比较
        3. 发布模型前的最终验证
        
        特点：
        - 只在训练完全结束后使用
        - 不打乱数据顺序
        - 不丢弃任何数据
        - 代表模型在真实世界的表现
        """
        return self._create_data_batches(self.test_datasets, shuffle=False)

    def _split_dataset(self, dataset: torch.utils.data.Dataset, client_id: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        将单个客户端的数据集分割为训练/评估/测试三部分。
        
        参数:
            dataset: 要分割的数据集
            client_id: 客户端ID，用于生成确定性的随机种子
            
        返回:
            (train_dataset, eval_dataset, test_dataset)
        """        
        dataset_size = len(dataset)
        # 计算各部分大小
        train_size = int(self.train_ratio * dataset_size)
        eval_size = int(self.eval_ratio * dataset_size)
        test_size = dataset_size - train_size - eval_size  # 剩余的都给测试集

        # 使用确定性随机种子，确保每次分割结果一致
        generator = torch.Generator().manual_seed(self.random_seed + client_id)
        
        return torch.utils.data.random_split(
            dataset, 
            [train_size, eval_size, test_size],
            generator=generator
        )

    def _split_all_datasets(self):
        """为所有客户端分割数据集。"""
        for client_id, dataset in self.datasets.items():
            # 如果没有提供独立的评估/测试数据集，则进行分割
            if client_id not in self.eval_datasets or client_id not in self.test_datasets:
                train_ds, eval_ds, test_ds = self._split_dataset(dataset, client_id)
                # 只保存训练数据集（原始数据集现在只用于分割）
                self.train_datasets[client_id] = train_ds
                
                # 如果没有独立提供评估数据集，使用分割的结果
                if client_id not in self.eval_datasets:
                    self.eval_datasets[client_id] = eval_ds
                    
                # 如果没有独立提供测试数据集，使用分割的结果
                if client_id not in self.test_datasets:
                    self.test_datasets[client_id] = test_ds
            else:
                # 如果提供了独立的评估和测试数据集，训练集就是原始数据集
                self.train_datasets[client_id] = dataset

    def _create_data_batches(self, datasets: Dict[int, torch.utils.data.Dataset], shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        创建数据批次的通用方法。
        
        参数:
            datasets: 客户端数据集字典
            shuffle: 是否打乱数据
            
        返回:
            格式化的数据批次列表
        """
        data = []
        for client_id, dataset in datasets.items():
            # 为此客户端创建数据加载器
            client_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=self.drop_last if shuffle else False  # 评估/测试时不丢弃数据
            )
            # 将PyTorch的标准数据格式转换为FLSim框架要求的字典格式
            '''
                batch = (
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # features: [batch_size, feature_dim]
                    torch.tensor([0, 1])                      # labels: [batch_size]
                )
                转换后
                batch_dict = {
                    'features': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    'labels': torch.tensor([0, 1])
                }
            '''
            batches = []
            for batch in client_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    features, labels = batch
                    batches.append({
                        'features': features,
                        'labels': labels
                    })
            
            data.append({
                'client_id': client_id,
                'service_id': self.service_assignments[client_id],
                'batches': batches
            })
        
        return data

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集分割信息，用于调试和验证。
        
        返回:
            包含各客户端数据集大小信息的字典
        """
        info = {
            'total_clients': len(self.datasets),
            'data_split_ratios': {
                'train': self.train_ratio,
                'eval': self.eval_ratio, 
                'test': self.test_ratio
            },
            'clients_info': {}
        }
        for client_id in self.datasets.keys():
            original_size = len(self.datasets[client_id])
            train_size = len(self.train_datasets[client_id]) if client_id in self.train_datasets else 0
            eval_size = len(self.eval_datasets[client_id]) if client_id in self.eval_datasets else 0
            test_size = len(self.test_datasets[client_id]) if client_id in self.test_datasets else 0
            
            info['clients_info'][client_id] = {
                'original_size': original_size,
                'train_size': train_size,
                'eval_size': eval_size,
                'test_size': test_size,
                'service_id': self.service_assignments[client_id]
            }
    
        return info

class QuantizedFLModel(IFLModel):
    """
    FL Model wrapper that incorporates quantization.
    Extends FLSim's FLModel with quantization capabilities.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: Optional[str] = None,
                 quantization_level: int = 8):
        """
        Initialize quantized FL model.
        
        Args:
            model: PyTorch model
            device: Device to run on
            quantization_level: Quantization level for model parameters
        """
        self.model = model
        self.device = device
        self.quantization_level = quantization_level
        self.quantizer = QuantizationModule()
        
        # Store original parameters for comparison
        self.original_params = None
        self.quantized_params = None
        self.communication_volume = 0
    
    def fl_forward(self, batch):
        """Forward pass with quantized model."""
        features = batch['features']
        labels = batch['labels']
        
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
        output = self.model(features)
        loss = torch.nn.functional.cross_entropy(output, labels.long())
        
        # Return FLSim-compatible batch metrics
        if FLSIM_AVAILABLE:
            try:
                from flsim.utils.simple_batch_metrics import FLBatchMetrics
                return FLBatchMetrics(
                    loss=loss,
                    num_examples=labels.size(0),
                    predictions=output.detach().cpu(),
                    targets=labels.detach().cpu(),
                    model_inputs=features
                )
            except ImportError:
                pass
        
        # Fallback: create a simple object with required attributes
        class SimpleBatchMetrics:
            def __init__(self, loss, num_examples, predictions, targets):
                self.loss = loss
                self.num_examples = num_examples
                self.predictions = predictions
                self.targets = targets
        
        return SimpleBatchMetrics(
            loss=loss,
            num_examples=labels.size(0),
            predictions=output.detach().cpu(),
            targets=labels.detach().cpu()
        )
    
    def fl_create_training_batch(self, **kwargs):
        """Create training batch."""
        return kwargs
    
    def fl_get_module(self) -> nn.Module:
        """Get the underlying PyTorch module."""
        return self.model
    
    def fl_cuda(self):
        """Move model to CUDA."""
        if self.device:
            self.model = self.model.to(self.device)
    
    def get_eval_metrics(self, batch):
        """Get evaluation metrics."""
        with torch.no_grad():
            return self.fl_forward(batch)
    
    def get_num_examples(self, batch) -> int:
        """Get number of examples in batch."""
        return batch['labels'].size(0)
    
    def quantize_parameters(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Quantize model parameters and return communication volume.
        
        Returns:
            Tuple of (quantized_state_dict, communication_volume)
        """
        state_dict = self.model.state_dict()
        quantized_state_dict = {}
        total_volume = 0
        
        for name, param in state_dict.items():
            quantized_param, norm_value = self.quantizer.quantize_parameters(
                param, self.quantization_level
            )
            quantized_state_dict[name] = quantized_param
            
            # Calculate communication volume for this parameter
            param_volume = self.quantizer.calculate_communication_volume(
                param.numel(), self.quantization_level
            )
            total_volume += param_volume
        
        self.communication_volume = total_volume
        self.quantized_params = quantized_state_dict
        
        return quantized_state_dict, total_volume


class MultiServiceFLSystem:
    """
    Main system class for multi-service provider federated learning.
    
    Integrates quantization, communication modeling, and FLSim training.
    """
    
    def __init__(self, 
                 service_configs: List[ServiceProviderConfig],
                 client_configs: Dict[int, ClientResourceConfig],
                 base_config_path: Optional[str] = None):
        """
        Initialize multi-service FL system.
        
        Args:
            service_configs: List of service provider configurations
            client_configs: Dictionary of client resource configurations
            base_config_path: Path to base FLSim configuration
        """
        self.service_configs = {config.service_id: config for config in service_configs}
        self.client_configs = client_configs
        
        # Initialize subsystems
        self.system_metrics = SystemMetrics()
        self.service_models: Dict[int, QuantizedFLModel] = {}
        self.service_trainers: Dict[int, SyncTrainer] = {}
        self.service_data_providers: Dict[int, IFLDataProvider] = {}
        self.service_data_loaders: Dict[int, MultiServiceFLDataLoader] = {}
        
        # Load base configuration
        self.base_config = self._load_base_config(base_config_path)
        
        print(f"Initialized Multi-Service FL System with {len(service_configs)} services")
    
    def _load_base_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load base FLSim configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration based on FLSim examples
        return {
            "trainer": {
                "_base_": "base_sync_trainer",
                "epochs": 1,
                "server": {
                    "_base_": "base_sync_server",
                    "server_optimizer": {
                        "_base_": "base_fed_avg_with_lr",
                        "lr": 1.0,
                        "momentum": 0.9
                    },
                    "active_user_selector": {
                        "_base_": "base_uniformly_random_active_user_selector"
                    }
                },
                "client": {
                    "epochs": 1,
                    "optimizer": {
                        "_base_": "base_optimizer_sgd",
                        "lr": 0.01,
                        "momentum": 0
                    }
                },
                "users_per_round": 5,
                "eval_epoch_frequency": 1,
                "do_eval": True,
                "report_train_metrics_after_aggregation": True
            }
        }

    def setup_service(self, 
                 service_id: int,
                 model: nn.Module,
                 train_datasets: Dict[int, torch.utils.data.Dataset],
                 eval_datasets: Optional[Dict[int, torch.utils.data.Dataset]] = None,
                 test_datasets: Optional[Dict[int, torch.utils.data.Dataset]] = None):

        """
        设置具体服务及其模型和数据。
        
        参数:
            service_id: 服务标识符
            model: 该服务的PyTorch模型
            train_datasets: 每个客户端的训练数据集
            eval_datasets: 可选的评估数据集
            test_datasets: 可选的测试数据集
        """
        if service_id not in self.service_configs:
            raise ValueError(f"Service {service_id} not found in configurations")

        config = self.service_configs[service_id]
        
        # 创建量化FL模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fl_model = QuantizedFLModel(
            model=model,
            device=device,
            quantization_level=config.quantization_level
        )

        if torch.cuda.is_available():
            fl_model.fl_cuda()

        self.service_models[service_id] = fl_model

        # 创建服务分配映射
        service_assignments = {client_id: service_id for client_id in config.client_ids}

        # 过滤该服务的客户端数据集
        service_datasets = {cid: train_datasets[cid] for cid in config.client_ids 
                        if cid in train_datasets}


        # ✅ 使用 MultiServiceFLDataLoader 创建数据加载器
        data_loader = MultiServiceFLDataLoader(
            datasets=service_datasets,
            service_assignments=service_assignments,
            batch_size=32,
            eval_datasets=eval_datasets,
            test_datasets=test_datasets,
            train_ratio=0.8,
            eval_ratio=0.1,
            test_ratio=0.1
        )


        # 获取训练数据
        train_data = data_loader.fl_train_set()


        # 尝试使用FLSim
        if FLSIM_AVAILABLE:
            try:
                # 将 MultiServiceFLDataLoader 转换为 FLSim 兼容的数据提供器
                data_provider = self._create_flsim_data_provider_from_loader(data_loader, config)
                self.service_data_providers[service_id] = data_provider

                # 创建trainer配置
                trainer_config = self._create_trainer_config(config)
                
                # 使用 Hydra instantiate 创建trainer，传入模型
                trainer = instantiate(
                    trainer_config,
                    model=fl_model,
                    cuda_enabled=torch.cuda.is_available()
                )
                
                print(f"使用FLSim训练器设置服务 {service_id}")
                
            except Exception as e:
                print(f"FLSim trainer创建失败: {e}")
                
        
        self.service_trainers[service_id] = trainer
        self.service_data_loaders[service_id] = data_loader  # 保存数据加载器引用
        print(f"Set up service {service_id} with {len(service_datasets)} clients")

        # 打印数据分割信息
        info = data_loader.get_dataset_info()
        print("数据分割信息:")
        for client_id, client_info in info['clients_info'].items():
            print(f"  客户端{client_id}: 训练{client_info['train_size']}, "
                f"评估{client_info['eval_size']}, 测试{client_info['test_size']}")
    


    def _create_flsim_data_provider_from_loader(self, 
                                            data_loader: MultiServiceFLDataLoader,
                                            config: ServiceProviderConfig) -> IFLDataProvider:
        """从 MultiServiceFLDataLoader 创建 FLSim 兼容的数据提供器。"""
        
        # 获取训练数据
        train_data = data_loader.fl_train_set()
        eval_data = data_loader.fl_eval_set()
        test_data = data_loader.fl_test_set()
        
        # 创建 FLSim 兼容的数据提供器
        class MultiServiceDataProvider(IFLDataProvider):
            def __init__(self, train_data, eval_data, test_data):
                self.train_data = train_data
                self.eval_data = eval_data  
                self.test_data = test_data
                
            def train_users(self):
                return [d['client_id'] for d in self.train_data]
                
            def eval_users(self):
                return [d['client_id'] for d in self.eval_data]
                
            def test_users(self):
                return [d['client_id'] for d in self.test_data]
                
            # 实现其他必需的方法...
            # todo
            
        return MultiServiceDataProvider(train_data, eval_data, test_data)

    def _create_trainer_config(self, service_config: ServiceProviderConfig):
        """Create trainer configuration for a service."""
        # 基于FLSim的配置格式
        trainer_config = {
            "_target_": "flsim.trainers.sync_trainer.SyncTrainer",
            "epochs": 3,  # 总的FL轮数
            "server": {
                "_target_": "flsim.servers.sync_servers.SyncServerConfig",
                "server_optimizer": {
                    "_target_": "flsim.optimizers.server_optimizers.FedAvgWithLROptimizerConfig",
                    "lr": 1.0,
                    "momentum": 0.9
                },
                "active_user_selector": {
                    "_target_": "flsim.active_user_selectors.simple_user_selector.UniformlyRandomActiveUserSelectorConfig"
                }
            },
            "client": {
                "_target_": "flsim.clients.base_client.ClientConfig",
                "epochs": service_config.local_epochs,
                "optimizer": {
                    "_target_": "flsim.optimizers.local_optimizers.LocalOptimizerSGDConfig",
                    "lr": service_config.learning_rate,
                    "momentum": 0.0
                }
            },
            "users_per_round": min(service_config.users_per_round, len(service_config.client_ids)),
            "eval_epoch_frequency": 1,
            "do_eval": False,  # 暂时禁用评估以避免数据接口问题
            "report_train_metrics_after_aggregation": True
        }
        
        return trainer_config
    
    def train_service(self, 
                     service_id: int, 
                     num_rounds: int = 10,
                     enable_metrics: bool = True) -> Tuple[QuantizedFLModel, Dict[str, Any]]:
        """
        Train a specific service using federated learning.
        
        Args:
            service_id: Service to train
            num_rounds: Number of training rounds
            enable_metrics: Whether to calculate communication metrics
            
        Returns:
            Tuple of (trained_model, metrics_summary)
        """
        if service_id not in self.service_trainers:
            raise ValueError(f"Service {service_id} not set up")
        
        trainer = self.service_trainers[service_id]
        data_provider = self.service_data_providers[service_id]
        model = self.service_models[service_id]
        
        try:
            from flsim.interfaces.metrics_reporter import Channel
            from flsim.utils.example_utils import MetricsReporter
            metrics_reporter = MetricsReporter([Channel.STDOUT])
        except ImportError:
            metrics_reporter = None
        
        print(f"\nStarting federated training for service {service_id}...")

        # 使用FLSim trainer的完整接口
        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metrics_reporter=metrics_reporter,
            num_total_users=data_provider.num_train_users() if data_provider else 1,
            distributed_world_size=1
        )
        
        # Calculate communication and energy metrics if requested
        metrics_summary = {}
        if enable_metrics:
            metrics_summary = self._calculate_service_metrics(service_id, num_rounds)
        
        print(f"Completed training for service {service_id}")
        
        return final_model, metrics_summary
    
    def _calculate_service_metrics(self, service_id: int, num_rounds: int) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a service."""
        config = self.service_configs[service_id]
        model = self.service_models[service_id]
        
        # Calculate quantization metrics
        _, comm_volume = model.quantize_parameters()
        
        # Calculate per-client metrics
        service_metrics = []
        for client_id in config.client_ids:
            if client_id in self.client_configs:
                client_config = self.client_configs[client_id]
                
                # Calculate client metrics
                client_metrics = self.system_metrics.calculate_client_metrics(
                    client_id=client_id,
                    service_id=service_id,
                    mu_i=client_config.mu_i,
                    c_ir=client_config.c_ir,
                    dataset_size=client_config.dataset_size,
                    cpu_frequency=client_config.max_frequency,
                    communication_volume=comm_volume,
                    bandwidth=1e6,  # 1 MHz (can be made configurable)
                    channel_gain=client_config.channel_gain,
                    transmit_power=client_config.max_power
                )
                
                self.system_metrics.add_client_metrics(client_metrics)
                service_metrics.append(client_metrics)
        
        # Get service-level summary
        summary = self.system_metrics.get_service_metrics_summary(service_id)
        
        # Add quantization information
        total_params = sum(p.numel() for p in model.model.parameters())
        compression_ratio = self.quantizer.get_compression_ratio(
            total_params, config.quantization_level
        ) if hasattr(self, 'quantizer') else 0
        
        summary.update({
            'quantization_level': config.quantization_level,
            'communication_volume_per_round': comm_volume,
            'total_parameters': total_params,
            'compression_ratio': compression_ratio,
            'total_communication_volume': comm_volume * num_rounds
        })
        
        return summary
    
    def train_all_services(self, 
                          num_rounds: int = 10,
                          parallel: bool = False) -> Dict[int, Tuple[QuantizedFLModel, Dict[str, Any]]]:
        """
        Train all services.
        
        Args:
            num_rounds: Number of training rounds
            parallel: Whether to train services in parallel (not implemented)
            
        Returns:
            Dictionary mapping service_id to (model, metrics)
        """
        results = {}
        
        for service_id in self.service_configs.keys():
            if service_id in self.service_trainers:
                print(f"\n{'='*50}")
                print(f"Training Service {service_id}")
                print(f"{'='*50}")
                
                model, metrics = self.train_service(service_id, num_rounds)
                results[service_id] = (model, metrics)
        
        return results
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        summary = {
            'num_services': len(self.service_configs),
            'num_clients': len(self.client_configs),
            'service_summaries': {}
        }
        
        for service_id in self.service_configs.keys():
            service_summary = self.system_metrics.get_service_metrics_summary(service_id)
            summary['service_summaries'][service_id] = service_summary
        
        return summary


def create_sample_system() -> MultiServiceFLSystem:
    """Create a sample multi-service FL system for testing."""
    
    # Define service configurations
    service_configs = [
        ServiceProviderConfig(
            service_id=1,
            name="Image Classification Service", 
            client_ids=[1, 2, 3],
            model_architecture={"type": "cnn", "num_classes": 10},
            quantization_level=8,
            users_per_round=2
        ),
        ServiceProviderConfig(
            service_id=2,
            name="Text Classification Service",
            client_ids=[4, 5, 6],
            model_architecture={"type": "mlp", "num_classes": 5},
            quantization_level=4,
            users_per_round=2
        )
    ]
    
    # Define client resource configurations
    client_configs = {
        1: ClientResourceConfig(1, 1e-28, 1000, 1e9, 0.1, 1e-3, 1000),
        2: ClientResourceConfig(2, 1.2e-28, 1200, 1.2e9, 0.12, 1.2e-3, 1200),
        3: ClientResourceConfig(3, 0.8e-28, 800, 0.8e9, 0.08, 0.8e-3, 800),
        4: ClientResourceConfig(4, 1.1e-28, 1100, 1.1e9, 0.11, 1.1e-3, 1100),
        5: ClientResourceConfig(5, 0.9e-28, 900, 0.9e9, 0.09, 0.9e-3, 900),
        6: ClientResourceConfig(6, 1.3e-28, 1300, 1.3e9, 0.13, 1.3e-3, 1300),
    }
    
    return MultiServiceFLSystem(service_configs, client_configs)


if __name__ == "__main__":
    print("Multi-Service Federated Learning System")
    print("Creating sample system...")
    
    system = create_sample_system()
    print("Sample system created successfully!")
    
    # Print system configuration
    summary = system.get_system_summary()
    print(f"System Summary: {json.dumps(summary, indent=2)}")