#!/usr/bin/env python3
"""
Multi-Service Provider Federated Learning Framework
Complete implementation of research paper sections 1-3

This package implements the comprehensive system model including:

Part I: System Model (Equations 1-13)
- Multi-service provider federated learning
- q-level model quantization strategies  
- Communication delay and energy consumption modeling
- FLSim library integration

Part II: Problem Modeling (Equation 14)
- Multi-objective optimization with constraints C1-C5
- Resource allocation (client selection, frequency, bandwidth, quantization)
- Service provider coordination strategies

Part III: MDP Framework (Equations 15-18)
- Observation space: FL state, system metrics, bandwidth allocations
- Action space: Decision variables with mixed integer/continuous
- Reward function with adversarial factor modeling
- Non-cooperative game theory and Nash equilibrium
- Multi-agent reinforcement learning (Independent DQN)
"""

__version__ = "2.0.0"
__author__ = "FL Research Team"

# Core system model (Part I)
from .quantization import QuantizationModule
from .communication import CommunicationModel
from .multi_service_fl import MultiServiceFLSystem

# Problem modeling (Part II) 
from .optimization_problem import (
    OptimizationConstraints, DecisionVariables, 
    MultiObjectiveCostFunction, ServiceProviderOptimizer, MultiServiceOptimizer
)

# MDP framework (Part III)
from .mdp_framework import (
    MultiServiceFLEnvironment, Action, Observation, 
    FLTrainingState, SystemState, RewardFunction, AdversarialFactor
)

# Game theory and MARL
from .game_theory import NonCooperativeGame, BestResponseCalculator
from .marl_interface import MARLTrainer, MARLConfig

# PAC-MCoFL (Part IV)
try:
    from .pac_mcofl import PACMCoFLTrainer, PACConfig, PACAgent
    PAC_AVAILABLE = True
except ImportError:
    PAC_AVAILABLE = False

__all__ = [
    # Core system model
    "QuantizationModule",
    "CommunicationModel", 
    "MultiServiceFLSystem",
    
    # Optimization framework
    "OptimizationConstraints",
    "DecisionVariables", 
    "MultiObjectiveCostFunction",
    "ServiceProviderOptimizer",
    "MultiServiceOptimizer",
    
    # MDP framework
    "MultiServiceFLEnvironment",
    "Action",
    "Observation", 
    "FLTrainingState",
    "SystemState",
    "RewardFunction",
    "AdversarialFactor",
    
    # Game theory & MARL
    "NonCooperativeGame",
    "BestResponseCalculator", 
    "MARLTrainer",
    "MARLConfig"
]

# Conditionally add PAC-MCoFL exports
if PAC_AVAILABLE:
    __all__.extend([
        "PACMCoFLTrainer",
        "PACConfig", 
        "PACAgent"
    ])