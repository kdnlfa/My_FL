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
import math
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
        self.datasets = datasets
        self.service_assignments = service_assignments
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_seed = random_seed

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
    融合量化功能的联邦学习模型包装器。
    在FLSim的FLModel基础上扩展量化能力。
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: Optional[str] = None,
                 quantization_level: int = 8):
        """
        初始化量化联邦学习模型。
        
        参数:
            model: PyTorch模型
            device: 运行设备
            quantization_level: 模型参数的量化级别
        """
        self.model = model
        self.device = device
        self.quantization_level = quantization_level
        self.quantizer = QuantizationModule()
        
        # 存储原始参数用于比较
        self.original_params = None
        self.quantized_params = None
        self.communication_volume = 0
    
    def fl_forward(self, batch):
        """使用量化模型进行前向传播。"""
        features = batch['features']
        labels = batch['labels']
        
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
        output = self.model(features)
        loss = torch.nn.functional.cross_entropy(output, labels.long())
        
        # 返回FLSim兼容的批次指标
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
        
        return SimpleBatchMetrics(
            loss=loss,
            num_examples=labels.size(0),
            predictions=output.detach().cpu(),
            targets=labels.detach().cpu()
        )
    
    def fl_create_training_batch(self, **kwargs):
        """创建训练批次。"""
        return kwargs
    
    def fl_get_module(self) -> nn.Module:
        """获取底层的PyTorch模块。"""
        return self.model
    
    def fl_cuda(self):
        """将模型移动到CUDA设备。"""
        if self.device:
            self.model = self.model.to(self.device)
    
    def get_eval_metrics(self, batch):
        """获取评估指标。"""
        with torch.no_grad():
            return self.fl_forward(batch)
    
    def get_num_examples(self, batch) -> int:
        """获取批次中的样本数量。"""
        return batch['labels'].size(0)
    
    def quantize_parameters(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        量化模型参数并返回通信量。
        
        返回:
            元组 (量化后的状态字典, 通信量)
        """
        state_dict = self.model.state_dict()
        quantized_state_dict = {}
        total_volume = 0
        
        # 量化参数，降低通信开销
        for name, param in state_dict.items():
            quantized_param, norm_value = self.quantizer.quantize_parameters(
                param, self.quantization_level
            )
            quantized_state_dict[name] = quantized_param
            
            # 计算此参数的通信量
            param_volume = self.quantizer.calculate_communication_volume(
                param.numel(), self.quantization_level
            )
            total_volume += param_volume
        
        # 量化后的参数和通信量
        self.communication_volume = total_volume
        self.quantized_params = quantized_state_dict
        
        return quantized_state_dict, total_volume


class MultiServiceFLSystem:
    """
    多服务提供商联邦学习的主系统类。
    集成了量化、通信建模和 FLSim 训练。
    """
    
    def __init__(self,
                 service_configs: List[ServiceProviderConfig],
                 client_configs: Dict[int, ClientResourceConfig],
                 base_config_path: Optional[str] = None,
                 debug_disable_quant: bool = False,
                 debug_verbose: bool = False):
        """初始化多服务联邦学习系统。"""
        # 保存配置
        self.service_configs: Dict[int, ServiceProviderConfig] = {cfg.service_id: cfg for cfg in service_configs}
        self.client_configs: Dict[int, ClientResourceConfig] = client_configs

        # 子系统容器
        self.system_metrics = SystemMetrics()
        self.service_models: Dict[int, QuantizedFLModel] = {}
        self.service_trainers: Dict[int, SyncTrainer] = {}
        self.service_data_providers: Dict[int, IFLDataProvider] = {}
        self.service_data_loaders: Dict[int, MultiServiceFLDataLoader] = {}

        # 每轮由RL覆盖注入的服务动作参数（不改变客户端选择逻辑）
        # 结构: { service_id: { 'n_clients': int, 'cpu_frequency': float, 'bandwidth': float, 'quantization_level': int } }
        self.service_action_overrides: Dict[int, Dict[str, Any]] = {}

        # 基础配置与调试标志
        self.base_config = self._load_base_config(base_config_path)
        self.debug_disable_quant = debug_disable_quant
        self.debug_verbose = debug_verbose

        print(f"初始化多服务联邦学习系统，包含 {len(service_configs)} 个服务 (debug_disable_quant={debug_disable_quant}, debug_verbose={debug_verbose})")

    def _ensure_scalar_quant_channel(self, trainer, n_bits: int) -> None:
        """确保训练器与服务器使用标量量化通道，并设置位宽，用于上传前量化。"""
        try:
            from flsim.channels.scalar_quantization_channel import ScalarQuantizationChannel
        except Exception:
            return

        # 将 qLevel(级数) 映射为位数：bits = ceil(log2(q)), 并裁剪到 [1, 8]
        n_bits = int(max(1, min(8, n_bits)))

        # 若已是标量量化通道，则更新位宽及相关量化器
        if hasattr(trainer, 'channel') and isinstance(trainer.channel, ScalarQuantizationChannel):
            ch = trainer.channel
            # 更新 cfg 与内部量化边界/观察器
            try:
                ch.cfg.n_bits = n_bits
            except Exception:
                pass
            ch.quant_min = -(2 ** (n_bits - 1))
            ch.quant_max = (2 ** (n_bits - 1)) - 1
            ch.observer, ch.quantizer = ch.get_observers_and_quantizers()
            # 确保服务器也使用同一通道实例
            if hasattr(trainer, 'server') and hasattr(trainer.server, '_channel'):
                trainer.server._channel = ch
            return

        # 若当前为其它通道，则创建新的量化通道并替换
        try:
            new_channel = ScalarQuantizationChannel(n_bits=n_bits)
            # 训练器与服务器同时替换为同一实例，保证一致
            if hasattr(trainer, 'channel'):
                trainer.channel = new_channel
            if hasattr(trainer, 'server') and hasattr(trainer.server, '_channel'):
                trainer.server._channel = new_channel
        except Exception:
            # 忽略失败，保持原通道
            pass

    def set_service_action(self,
                           service_id: int,
                           n_clients: Optional[int] = None,
                           cpu_frequency: Optional[float] = None,
                           bandwidth: Optional[float] = None,
                           quantization_level: Optional[int] = None) -> None:
        """由RL在每步调用，注入本轮服务的动作参数以影响能耗/时延/通信量计算。
        注意：不改变FLSim内部客户端选择，仅用于系统指标与通信建模。
        """
        ov = self.service_action_overrides.get(service_id, {})
        if n_clients is not None:
            ov['n_clients'] = int(max(1, n_clients))
        if cpu_frequency is not None:
            ov['cpu_frequency'] = float(max(1e3, cpu_frequency))
        if bandwidth is not None:
            ov['bandwidth'] = float(max(0.0, bandwidth))
        if quantization_level is not None:
            ov['quantization_level'] = int(max(1, quantization_level))
        self.service_action_overrides[service_id] = ov
        
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base) if base else {}
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _get_service_overrides(self, service_id: int) -> Dict[str, Any]:
        services = self.base_config.get("services", {}) or {}
        return services.get(str(service_id), {}) or {}

    def _load_base_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载基础 FLSim 配置，仅从JSON文件读取。"""
        candidates: List[Path] = []
        if config_path:
            candidates.append(Path(config_path))
        else:
            # 优先使用与当前文件同级的 configs
            try:
                here = Path(__file__).resolve().parent
                candidates.append(here / "configs" / "flsim_base_config.json")
            except Exception:
                pass
            # 工作目录下的 My_FL/configs
            candidates.append(Path.cwd() / "My_FL" / "configs" / "flsim_base_config.json")
            # 工作目录下的 configs
            candidates.append(Path.cwd() / "configs" / "flsim_base_config.json")
        
        config_file = None
        for p in candidates:
            if p and p.exists():
                config_file = p
                break
        if config_file is None:
            tried = "\n".join(str(p) for p in candidates)
            raise FileNotFoundError(
                "配置文件不存在于以下任一路径:\n" + tried
            )
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 成功加载配置文件: {config_file}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise Exception(f"配置文件读取失败: {e}")

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
            raise ValueError(f"服务 {service_id} 未在配置中找到")

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

        # ✅ 使用 JSON 中的 DataLoader 配置（可被 services[service_id].data_loader_overrides 覆盖）
        dl_base = self.base_config.get("data_loader", {}) or {}
        dl_over = self._get_service_overrides(service_id).get("data_loader_overrides", {}) or {}
        dl_cfg = self._deep_merge(dl_base, dl_over)
        # ✅ 使用 MultiServiceFLDataLoader 创建数据加载器
        data_loader = MultiServiceFLDataLoader(
            datasets=service_datasets,
            service_assignments=service_assignments,
            batch_size=dl_cfg.get("batch_size", 32),
            drop_last=dl_cfg.get("drop_last", False),
            eval_datasets=eval_datasets,
            test_datasets=test_datasets,
            train_ratio=dl_cfg.get("train_ratio", 0.8),
            eval_ratio=dl_cfg.get("eval_ratio", 0.1),
            test_ratio=dl_cfg.get("test_ratio", 0.1)
        )


        # 获取训练数据
        train_data = data_loader.fl_train_set()

        # 尝试使用FLSim
        if not FLSIM_AVAILABLE:
            raise RuntimeError("FLSim 不可用，无法创建训练器。请检查安装。")

        # 将 MultiServiceFLDataLoader 转换为 FLSim 兼容的数据提供器（严格失败）
        data_provider = self._create_flsim_data_provider_from_loader(data_loader, config)
        self.service_data_providers[service_id] = data_provider

        # ✅ 训练器配置完全由 JSON（含 per-service 覆盖）决定
        trainer_config = self._create_trainer_config(config)
        try:
            # 将可能存在的嵌套 trainer_config 扁平化为 Hydra 期望的顶层字段
            def _flatten_trainer_cfg(cfg: dict) -> dict:
                if not isinstance(cfg, dict):
                    return cfg
                if "trainer_config" in cfg and isinstance(cfg["trainer_config"], dict):
                    inner = copy.deepcopy(cfg["trainer_config"])
                    # 使用外层的目标作为最终目标（SyncTrainer）
                    inner["_target_"] = cfg.get("_target_", inner.get("_target_"))
                    return inner
                return cfg

            hydra_payload = _flatten_trainer_cfg(trainer_config)

            # 基本校验：必须包含 _target_（SyncTrainer）
            if not isinstance(hydra_payload, dict) or "_target_" not in hydra_payload:
                raise ValueError("trainer 配置缺少 _target_")

            trainer = instantiate(
                hydra_payload,
                model=fl_model,
                cuda_enabled=torch.cuda.is_available()
            )
        except Exception as e:
            try:
                cfg_str = json.dumps(hydra_payload if 'hydra_payload' in locals() else trainer_config, ensure_ascii=False, indent=2)
            except Exception:
                cfg_str = str(hydra_payload if 'hydra_payload' in locals() else trainer_config)
            raise RuntimeError(f"实例化 FLSim Trainer 失败: {e}\n配置=\n{cfg_str}") from e

        if trainer is None:
            raise RuntimeError("FLSim Trainer 实例为 None，请检查配置与依赖。")

        print(f"✅ 使用FLSim训练器设置服务 {service_id}")

        self.service_trainers[service_id] = trainer
        self.service_data_loaders[service_id] = data_loader  # 保存数据加载器引用
        print(f"Set up service {service_id} with {len(service_datasets)} clients")
    


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
                # 保留原始原子数据（含 client_id、service_id、batches）
                self._raw_train_data = train_data
                self._raw_eval_data = eval_data
                self._raw_test_data = test_data

                # 预构建 IFLUserData 对象列表
                self._train_users = [self._create_fl_user_data(d["batches"], split="train") for d in train_data]
                self._eval_users = [self._create_fl_user_data(d["batches"], split="eval") for d in eval_data]
                self._test_users = [self._create_fl_user_data(d["batches"], split="eval") for d in test_data]  # 测试同样走 eval_data()

                # 可选：保留用户ID（client_id），供需要时使用
                self._train_user_ids = [d["client_id"] for d in train_data]
                self._eval_user_ids = [d["client_id"] for d in eval_data]
                self._test_user_ids = [d["client_id"] for d in test_data]
                
            def train_users(self):
                return self._train_users

            def eval_users(self):
                return self._eval_users

            def test_users(self):
                return self._test_users

            def train_user_ids(self):
                # 训练阶段若需要“索引”，返回 [0..n-1]
                return list(range(len(self._train_users)))

            def num_train_users(self):
                return len(self._train_users)

            def num_eval_users(self):
                return len(self._eval_users)

            def num_test_users(self):
                return len(self._test_users)


            def get_train_user(self, user_id: int):
                if 0 <= user_id < len(self._train_users):
                    return self._train_users[user_id]
                return self._create_empty_user_data()

            def get_eval_user(self, user_id: int):
                print(f"🔍 获取评估用户数据: user_id={user_id}")
                if 0 <= user_id < len(self._eval_users):
                    return self._eval_users[user_id]
                print(f"❌ 评估用户 {user_id} 不存在，有效范围: 0-{len(self._eval_users)-1}")
                return self._create_empty_user_data()

            def get_test_user(self, user_id: int):
                print(f"🔍 获取测试用户数据: user_id={user_id}")
                if 0 <= user_id < len(self._test_users):
                    return self._test_users[user_id]
                print(f"❌ 测试用户 {user_id} 不存在，有效范围: 0-{len(self._test_users)-1}")
                return self._create_empty_user_data()

            def _create_empty_user_data(self):
                empty_batch = {
                    "features": torch.zeros(1, 10),
                    "labels": torch.zeros(1, dtype=torch.long),
                }
                # 空用户走 eval split，避免参与训练
                return self._create_fl_user_data([empty_batch], split="eval")

            def _create_fl_user_data(self, batches, split: str = "train"):
                """将批次数据转换为FLSim用户数据格式; split in {'train','eval'}"""
                if not batches:
                    print("⚠️  警告: 批次数据为空，将注入虚拟批次")
                    batches = [{"features": torch.zeros(1, 10), "labels": torch.zeros(1, dtype=torch.long)}]

                from flsim.data.data_provider import IFLUserData

                class SimpleUserData(IFLUserData):
                    def __init__(self, batches, split):
                        self._split = split
                        if split == "train":
                            self._train_data = batches
                            self._eval_data = []
                        else:
                            self._train_data = []
                            self._eval_data = batches

                    def __iter__(self):
                        # 默认返回训练数据的迭代
                        for b in self._train_data:
                            yield b

                    def __len__(self):
                        return len(self._train_data) if self._split == "train" else len(self._eval_data)

                    def train_data(self):
                        for b in self._train_data:
                            yield b

                    def eval_data(self):
                        for b in self._eval_data:
                            yield b

                    def num_train_batches(self) -> int:
                        return len(self._train_data)

                    def num_eval_batches(self) -> int:
                        return len(self._eval_data)

                    def num_train_examples(self) -> int:
                        total = 0
                        for batch in self._train_data:
                            total += len(batch["labels"]) if isinstance(batch, dict) and "labels" in batch else len(batch)
                        return total

                    def num_eval_examples(self) -> int:
                        total = 0
                        for batch in self._eval_data:
                            total += len(batch["labels"]) if isinstance(batch, dict) and "labels" in batch else len(batch)
                        return total

                return SimpleUserData(batches, split) 
            
        return MultiServiceDataProvider(train_data, eval_data, test_data)
        

    def _create_trainer_config(self, service_config: ServiceProviderConfig):
        """从 JSON 读取 Trainer 配置，并应用 per-service 覆盖；按论文参数设置训练配置。"""
        base_trainer = copy.deepcopy(self.base_config.get("trainer", {}) or {})
        service_overrides = self._get_service_overrides(service_config.service_id).get("trainer_overrides", {}) or {}

        # 深度合并：全局 trainer + 当前服务覆盖
        trainer_config = self._deep_merge(base_trainer, service_overrides)

        # 按论文要求设置训练参数
        if isinstance(trainer_config, dict):
            inner = trainer_config.get("trainer_config")
            if isinstance(inner, dict):
                # 设置轮数为35（论文T=35）
                inner["epochs"] = 35
                
                # 设置每轮用户数为该服务的客户端总数
                if inner.get("users_per_round") is None:
                    inner["users_per_round"] = len(service_config.client_ids)
                
                # 设置本地训练轮数为3（论文τ=3）
                if inner.get("client", {}).get("epochs") is None:
                    if "client" not in inner:
                        inner["client"] = {}
                    inner["client"]["epochs"] = 3
                
                # 设置学习率为0.001（论文η_k=0.001）
                if inner.get("client", {}).get("optimizer", {}).get("lr") is None:
                    if "client" not in inner:
                        inner["client"] = {}
                    if "optimizer" not in inner["client"]:
                        inner["client"]["optimizer"] = {}
                    inner["client"]["optimizer"]["lr"] = 0.001
                    
            else:
                # 兼容旧格式（顶层直接设置参数）
                if trainer_config.get("users_per_round") is None:
                    trainer_config["users_per_round"] = len(service_config.client_ids)
                trainer_config["epochs"] = 35

        return trainer_config
    
    def train_service(self, 
                     service_id: int, 
                     num_rounds: int = 10,
                     enable_metrics: bool = True) -> Tuple[QuantizedFLModel, Dict[str, Any]]:
        """
        使用联邦学习训练指定的服务。
        
        参数:
            service_id: 要训练的服务ID
            num_rounds: 训练轮数
            enable_metrics: 是否计算通信指标
            
        返回:
            (训练后的模型, 指标摘要) 的元组
        """
        if service_id not in self.service_trainers:
            raise ValueError(f"服务 {service_id} 尚未设置")
        
        trainer = self.service_trainers[service_id]
        base_data_provider = self.service_data_providers[service_id]
        model = self.service_models[service_id]

        # 读取当前轮RL动作覆盖（若有）
        action_ov = self.service_action_overrides.get(service_id, {})

        # 1) 上传前量化：不再进行“入训量化”权重注入；改为在客户端本地训练完成、上传前通过通道进行量化。
        #    这里仅同步 qLevel 到模型记录与通道位宽设置。
        try:
            if 'quantization_level' in action_ov:
                model.quantization_level = int(action_ov['quantization_level'])
        except Exception:
            pass

        # 将 qLevel(级数) 转为通道位宽，并保证位宽在 [1, 8]
        try:
            q_level = int(action_ov.get('quantization_level', getattr(model, 'quantization_level', 8)))
            # bits ≈ ceil(log2(q_level))，避免 0
            n_bits = int(max(1, math.ceil(math.log2(max(2, q_level)))))
            self._ensure_scalar_quant_channel(trainer, n_bits)
            if self.debug_verbose:
                print(f"[DEBUG][S{service_id}] 配置上传前量化位宽: q_level={q_level} -> n_bits={n_bits}")
        except Exception as e:
            if self.debug_verbose:
                print(f"[DEBUG][S{service_id}] 配置上传前量化失败: {e}")

        # 2) n 的裁剪方案：基于RL覆盖的 n 限制本轮实际参与训练的用户数
        #    通过包装一个裁剪后的数据提供器实现，对FLSim透明。
        try:
            base_num_users = base_data_provider.num_train_users()
        except Exception:
            base_num_users = -1
        n_override = int(action_ov.get('n_clients', base_num_users)) if base_num_users > 0 else base_num_users
        if base_num_users > 0:
            n_effective = max(1, min(n_override, base_num_users))
        else:
            n_effective = base_num_users  # 维持不可用状态

        data_provider = base_data_provider
        if base_num_users > 0 and n_effective != base_num_users:
            from flsim.data.data_provider import IFLDataProvider as _IFLDP

            class _TrimmedDataProvider(_IFLDP):
                def __init__(self, base_dp, take_n: int):
                    self._base = base_dp
                    self._n = int(take_n)

                # 训练相关裁剪
                def train_users(self):
                    users = self._base.train_users()
                    try:
                        return users[: self._n]
                    except Exception:
                        return [self._base.get_train_user(i) for i in range(self._n)]

                def train_user_ids(self):
                    return list(range(self._n))

                def num_train_users(self):
                    return self._n

                def get_train_user(self, user_id: int):
                    if 0 <= user_id < self._n:
                        return self._base.get_train_user(user_id)
                    # 超界返回一个空用户，避免训练访问越界
                    return self._create_empty_user_data()

                # 评估/测试透传
                def eval_users(self):
                    return self._base.eval_users()

                def test_users(self):
                    return self._base.test_users()

                def num_eval_users(self):
                    return self._base.num_eval_users()

                def num_test_users(self):
                    return self._base.num_test_users()

                def get_eval_user(self, user_id: int):
                    return self._base.get_eval_user(user_id)

                def get_test_user(self, user_id: int):
                    return self._base.get_test_user(user_id)

                # 与我们内部构造的数据提供器对齐的空用户
                def _create_empty_user_data(self):
                    try:
                        from flsim.data.data_provider import IFLUserData as _IFLUser
                    except Exception:
                        _IFLUser = object  # 兜底

                    class _EmptyUser(_IFLUser):
                        def __iter__(self):
                            return iter([])

                        def __len__(self):
                            return 0

                        def train_data(self):
                            return iter([])

                        def eval_data(self):
                            return iter([])

                        def num_train_batches(self) -> int:
                            return 0

                        def num_eval_batches(self) -> int:
                            return 0

                        def num_train_examples(self) -> int:
                            return 0

                        def num_eval_examples(self) -> int:
                            return 0

                    return _EmptyUser()

            data_provider = _TrimmedDataProvider(base_data_provider, n_effective)
            print(f"[DEBUG][S{service_id}] 本轮训练将裁剪参与用户数: {base_num_users} -> {n_effective}")
        
        try:
            from flsim.interfaces.metrics_reporter import Channel
            from flsim.utils.example_utils import MetricsReporter
            metrics_reporter = MetricsReporter([Channel.STDOUT])
        except ImportError:
            metrics_reporter = None
        
        print(f"\n开始为服务 {service_id} 进行联邦学习训练...")

        # ---------------- 诊断阶段: 训练前统计 ----------------
        try:
            num_users = data_provider.num_train_users()
        except Exception:
            num_users = -1
        print(f"[DEBUG][S{service_id}] 训练用户数: {num_users}")
        # 统计前5个用户的批次数与样本数
        try:
            for u in range(min(5, num_users)):
                user = data_provider.get_train_user(u)
                if hasattr(user, 'num_train_batches'):
                    nb = user.num_train_batches()
                else:
                    nb = sum(1 for _ in user)
                if hasattr(user, 'num_train_examples'):
                    ne = user.num_train_examples()
                else:
                    ne = 0
                print(f"  用户{u}: batches={nb}, examples={ne}")
        except Exception as e:
            print(f"[DEBUG] 训练用户统计失败: {e}")

    # 取首个 batch 做前向，记录初始损失/精度（在入训量化之后）
        def _get_first_batch():
            try:
                for u in range(num_users):
                    user = data_provider.get_train_user(u)
                    if hasattr(user, 'num_train_batches') and user.num_train_batches() == 0:
                        continue
                    for batch in user.train_data():
                        feats = batch.get('features') if isinstance(batch, dict) else None
                        labels = batch.get('labels') if isinstance(batch, dict) else None
                        if feats is not None and labels is not None and len(labels) > 0:
                            return feats, labels
            except Exception:
                return None, None
            return None, None

        device = next(model.model.parameters()).device
        first_x, first_y = _get_first_batch()
        criterion = torch.nn.CrossEntropyLoss()
        if first_x is not None:
            first_x = first_x.to(device)
            first_y = first_y.to(device)
            model.model.eval()
            with torch.no_grad():
                try:
                    logits = model.model(first_x)
                    pre_loss = criterion(logits, first_y).item()
                    pre_acc = (logits.argmax(1) == first_y).float().mean().item()
                    print(f"[DEBUG][S{service_id}] 训练前单批: loss={pre_loss:.4f}, acc={pre_acc:.4f}")
                except Exception as e:
                    print(f"[DEBUG] 预训练批次前向失败: {e}")
        else:
            print(f"[DEBUG][S{service_id}] 未找到有效训练批次用于预检")

        # 记录首层参数用于差异比较
        try:
            first_param_before = next(model.model.parameters()).detach().cpu().clone()
        except Exception:
            first_param_before = None

        # 可选: 暂时禁用量化（调试）
        if self.debug_disable_quant:
            if hasattr(model, 'quantization_level'):
                print(f"[DEBUG][S{service_id}] 调试模式: 禁用量化过程 (quantization_level={model.quantization_level})")

        # ---------------- 调用FLSim训练 ----------------
        final_model, eval_score = trainer.train(
            data_provider=data_provider,
            metrics_reporter=metrics_reporter,
            num_total_users=data_provider.num_train_users() if data_provider else 1,
            distributed_world_size=1
        )

        # ---------------- 训练后诊断 ----------------
        try:
            first_param_after = next(model.model.parameters()).detach().cpu()
            if first_param_before is not None:
                delta = (first_param_after - first_param_before).pow(2).sum().sqrt().item()
                print(f"[DEBUG][S{service_id}] 首层参数L2变化: {delta:.6f}")
                if delta < 1e-6:
                    print(f"[WARN][S{service_id}] 权重几乎未变化，可能未发生有效训练！")
        except Exception as e:
            print(f"[DEBUG] 训练后参数比较失败: {e}")

        # 再次前向同一批次
        if first_x is not None:
            model.model.eval()
            with torch.no_grad():
                try:
                    logits2 = model.model(first_x)
                    post_loss = criterion(logits2, first_y).item()
                    post_acc = (logits2.argmax(1) == first_y).float().mean().item()
                    print(f"[DEBUG][S{service_id}] 训练后单批: loss={post_loss:.4f}, acc={post_acc:.4f}")
                except Exception as e:
                    print(f"[DEBUG] 训练后前向失败: {e}")

        # 快速测试集评估（限制前20批）
        try:
            dl = self.service_data_loaders.get(service_id)
            test_set_batches = dl.fl_test_set() if dl else []
            test_examples = 0
            test_correct = 0
            taken_batches = 0
            model.model.eval()
            with torch.no_grad():
                for entry in test_set_batches:
                    for batch in entry.get('batches', [])[:5]:  # 每个客户端最多取5批
                        feats = batch.get('features'); labels = batch.get('labels')
                        if feats is None or labels is None:
                            continue
                        feats = feats.to(device); labels = labels.to(device)
                        try:
                            out = model.model(feats)
                        except Exception as e:
                            print(f"[DEBUG] 测试批前向失败: {e}")
                            continue
                        preds = out.argmax(1)
                        test_correct += (preds == labels).sum().item()
                        test_examples += labels.size(0)
                        taken_batches += 1
                        if taken_batches >= 20:
                            break
                    if taken_batches >= 20:
                        break
            if test_examples > 0:
                print(f"[DEBUG][S{service_id}] 快速测试准确率: {test_correct/test_examples:.4f} ({test_correct}/{test_examples}) 在 {taken_batches} 批次")
            else:
                print(f"[DEBUG][S{service_id}] 无测试样本用于快速评估")
        except Exception as e:
            print(f"[DEBUG] 快速测试评估失败: {e}")
        
        # 计算通信和能量消耗
        metrics_summary = {}
        if enable_metrics:
            metrics_summary = self._calculate_service_metrics(service_id, num_rounds)

        print(f"完成服务 {service_id} 的训练")

        return final_model, metrics_summary
    
    def _calculate_service_metrics(self, service_id: int, num_rounds: int) -> Dict[str, Any]:
        """计算服务的综合指标。"""
        config = self.service_configs[service_id]
        model = self.service_models[service_id]

        # 读取当前轮的动作覆盖参数
        action_ov = self.service_action_overrides.get(service_id, {})

        # 若RL覆盖了量化级别，则应用到模型再计算通信量
        if 'quantization_level' in action_ov and not self.debug_disable_quant:
            try:
                model.quantization_level = int(action_ov['quantization_level'])
            except Exception:
                pass

        # 计算量化指标（通信量随q变化）
        _, comm_volume = model.quantize_parameters()

        # 计算每个客户端的指标
        service_metrics = []
        # 覆盖的客户端数用于带宽在客户端间的等分（不改变选择集合）
        n_clients_override = int(action_ov.get('n_clients', len(config.client_ids)))
        # 服务带宽按选用客户端数等分给单个客户端（FDMA简化）
        service_bandwidth = float(action_ov.get('bandwidth', 1e6))  # 默认1MHz
        per_client_bandwidth = max(service_bandwidth / max(n_clients_override, 1), 1e3)  # 至少1kHz避免除零
        for client_id in config.client_ids:
            if client_id in self.client_configs:
                client_config = self.client_configs[client_id]
                
                # 计算客户端指标
                client_metrics = self.system_metrics.calculate_client_metrics(
                    client_id=client_id,
                    service_id=service_id,
                    mu_i=client_config.mu_i,
                    c_ir=client_config.c_ir,
                    dataset_size=client_config.dataset_size,
                    # 使用RL覆盖的CPU频率，否则退回到客户端上限
                    cpu_frequency=float(action_ov.get('cpu_frequency', client_config.max_frequency)),
                    communication_volume=comm_volume,
                    # 使用RL覆盖的带宽（按客户端平均）
                    bandwidth=per_client_bandwidth,
                    channel_gain=client_config.channel_gain,
                    transmit_power=client_config.max_power
                )
                
                self.system_metrics.add_client_metrics(client_metrics)
                service_metrics.append(client_metrics)
        
        # 得到服务等级的总结
        summary = self.system_metrics.get_service_metrics_summary(service_id)
        
        # 添加量化信息
        total_params = sum(p.numel() for p in model.model.parameters())
        q_for_ratio = getattr(model, 'quantization_level', config.quantization_level)
        compression_ratio = (
            model.quantizer.get_compression_ratio(total_params, q_for_ratio)
            if hasattr(model, "quantizer") and model.quantizer is not None
            else 0
        )
        
        summary.update({
            # 报告模型当前量化级别（可能被RL覆盖）
            'quantization_level': getattr(model, 'quantization_level', config.quantization_level),
            'communication_volume_per_round': comm_volume,
            'total_parameters': total_params,
            'compression_ratio': compression_ratio,
            'total_communication_volume': comm_volume * num_rounds
        })

        # 将快速测试结果并入指标（若已计算）
        if 'test_correct' in locals():
            summary.update({
                'quick_test_accuracy': test_correct / test_examples if 'test_examples' in locals() and test_examples > 0 else None,
                'quick_test_examples': test_examples if 'test_examples' in locals() else None
            })
        
        return summary
    
    def train_all_services(self, 
                          num_rounds: int = 10,
                          parallel: bool = False) -> Dict[int, Tuple[QuantizedFLModel, Dict[str, Any]]]:
        """
        训练所有服务。
        
        参数:
            num_rounds: 训练轮数
            parallel: 是否并行训练服务（未实现）
            
        返回:
            将service_id映射到(模型, 指标)的字典
        """
        results = {}
        
        for service_id in self.service_configs.keys():
            if service_id in self.service_trainers:
                print(f"\n{'='*50}")
                print(f"训练服务 {service_id}")
                print(f"{'='*50}")
                
                model, metrics = self.train_service(service_id, num_rounds)
                results[service_id] = (model, metrics)
        
        return results
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取综合系统摘要。"""
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
    """创建用于测试的示例多服务联邦学习系统。"""
    
    # 定义服务配置
    service_configs = [
        ServiceProviderConfig(
            service_id=1,
            name="图像分类服务", 
            client_ids=[1, 2, 3],
            model_architecture={"type": "cnn", "num_classes": 10},
            quantization_level=8,
            users_per_round=2
        ),
        ServiceProviderConfig(
            service_id=2,
            name="文本分类服务",
            client_ids=[4, 5, 6],
            model_architecture={"type": "mlp", "num_classes": 5},
            quantization_level=4,
            users_per_round=2
        )
    ]
    
    # 定义客户端资源配置
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
    print("多服务联邦学习系统")
    print("创建示例系统...")
    
    system = create_sample_system()
    print("示例系统创建成功！")
    
    # 打印系统配置
    summary = system.get_system_summary()
    print(f"系统摘要: {json.dumps(summary, indent=2, ensure_ascii=False)}")