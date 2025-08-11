# Multi-Service Provider Federated Learning Framework

Complete implementation of the research paper's system model with advanced optimization and game theory components.

## 📋 Overview

This framework implements the complete multi-service provider federated learning system described in the research paper, including:

### **Part I: System Model** ✅ COMPLETED
- **Multi-Service Provider Architecture**: Support for multiple service providers with shared network resources
- **Quantization Module**: q-level parameter quantization with communication volume optimization (Equations 5-7)
- **Communication Model**: Energy consumption and delay calculation based on FDMA (Equations 8-13)
- **FLSim Integration**: Leverages Facebook's FLSim library for federated learning orchestration

### **Part II: Problem Modeling** ✅ COMPLETED
- **Multi-Objective Optimization**: Implements Equation (14) with discount factor and constraints (C1-C5)
- **Resource Allocation**: Client selection, CPU frequency, bandwidth, quantization level optimization
- **Constraint Handling**: Energy budget, delay limits, bandwidth allocation, computation capacity
- **Service Provider Coordination**: Sequential, parallel, and iterative optimization strategies

### **Part III: Markov Decision Process** ✅ COMPLETED
- **Observation Space**: FL training state Z_{r,t}(ω), system state Θ_{r,t}, bandwidth allocations B_t (Equation 15)
- **Action Space**: Decision variables {n_{r,t}, f_{r,t}, B_{r,t}, q_{r,t}} (Equation 16)
- **Reward Function**: Multi-component reward with adversarial factor Φ_{r,t}(q) (Equations 17-18)
- **Game Theory Integration**: Non-cooperative strategies and Nash equilibrium convergence
- **MARL Framework**: Independent DQN, experience replay, multi-agent coordination

### **Part IV: PAC-MCoFL Algorithm** ✅ COMPLETED
- **Joint Policy**: Cumulative expected rewards J_r(π) computation (Equations 19-20)
- **Pareto Optimal Equilibrium**: Virtual joint policy π_{-r}^† optimization (Equations 21-23)
- **Actor-Critic Networks**: Policy gradient and TD error minimization (Equations 24-27)
- **Complete Algorithm 1**: PAC-MCoFL training flow with federated learning integration
- **Pareto vs Nash**: Collaborative optimization superior to non-cooperative approaches

### **Part V: Action Space Transformation** ✅ COMPLETED
- **Dimensional Reduction**: Continuous 4D → Discrete 3^4 = 81 actions
- **Ternary Representation**: a'(m) = {-1, 0, 1} for interpretable semantics
- **Cartesian Product**: Independent control of each action dimension
- **PAC Compatibility**: Enables tractable virtual joint policy computation (Equation 23)
- **Algorithm Stability**: Significantly reduced Q-function space traversal

### **Part VI: Experimental Design** ✅ COMPLETED
- **Multi-Service Setup**: 5 clients coordinated by 3 service providers
- **Dataset Integration**: CIFAR-10, FashionMNIST, MNIST with task-specific models
- **Parameter Compliance**: Complete implementation of paper's parameter table
- **Performance Evaluation**: Equation (17) reward system with system-wide metrics
- **IID Data Distribution**: Configurable non-IID degree ρ = 1 (100% label coverage)

## 🏗️ Enhanced Architecture

### Core Components

1. **`quantization.py`**: q-level quantization (Equations 5-7)
2. **`communication.py`**: Energy/delay modeling (Equations 8-13)
3. **`multi_service_fl.py`**: FLSim integration (Equations 1-4)
4. **`optimization_problem.py`**: Multi-objective optimization (Equation 14)
5. **`mdp_framework.py`**: MDP formulation (Equations 15-18)
6. **`game_theory.py`**: Non-cooperative game theory
7. **`marl_interface.py`**: Multi-agent reinforcement learning
8. **`pac_mcofl.py`**: PAC-MCoFL algorithm (Equations 19-27, Algorithm 1)
9. **`action_space_transform.py`**: Ternary action space transformation (Section 5)
10. **`experimental_setup.py`**: Complete experimental design framework (Section 6)
11. **`comprehensive_example.py`**: Complete validation demonstration
12. **`pac_example.py`**: PAC-MCoFL specific demonstrations
13. **`comprehensive_experimental_demo.py`**: Action space + experimental demo

## 📊 Mathematical Model Implementation

### Federated Learning Process (Equations 1-4) ✅
- **Local Loss Function**: `L_{i,r}(ω) = (1/|D_{i,r}|) Σ_{ξ∈D_{i,r}} l(ω;ξ)`
- **Global Optimization**: `ω* = argmin_ω Σ_{i=1}^N κ_{i,r} L_{i,r}(ω)`
- **Local Updates**: `ω_{i,r,t}^(k) = ω_{i,r,t}^(k-1) - η_k ∇L_{i,r}(ω_{i,r,t}^(k-1))`
- **Global Aggregation**: `ω_{r,t+1} = Σ_{i=1}^N κ_{i,r} Ψ(ω_{i,r,t}^(τ))`

### Quantization Process (Equations 5-7) ✅
- **q-level Quantization**: `Ψ_q(ω_d) = ||ω||_p · sgn(ω_d) · Ξ_q(ω_d, q)`
- **Stochastic Mapping**: `Ξ_q` with probability-based discretization
- **Communication Volume**: `vol_{r,t} = |ω|(⌈log_2(q)⌉ + 1) + 32`

### Communication & Energy Model (Equations 8-13) ✅
- **Computation Energy**: `E^cmp_{i,r,t} = μ_i · c_{i,r} · |D_{i,r}| · f_{i,r,t}^2`
- **Computation Delay**: `T^cmp_{i,r,t} = c_{i,r} · |D_{i,r}| / f_{i,r,t}`
- **Transmission Rate**: `v_{i,r,t} = B_{i,r,t} log_2(1 + g_{i,t}p_{i,t}/(B_{i,r,t}N_0))`
- **Communication Delay**: `T^com_{i,r,t} = vol_{i,r,t} / v_{i,r,t}`
- **Communication Energy**: `E^com_{i,r,t} = T^com_{i,r,t} · p_{i,t}`
- **Total System Metrics**: Energy sum, maximum delay bottleneck

### Multi-Objective Optimization (Equation 14) ✅
- **Objective Function**: `min Σ_{t=0}^{T-1} γ^t Υ(L_r(ω_{r,t}), vol_{r,t}, E_{r,t}^total, T_{r,t}^total)`
- **Constraints**: C1 (energy/delay), C2 (clients), C3 (frequency), C4 (bandwidth), C5 (quantization)
- **Mixed-Integer Optimization**: Discrete (n, q) and continuous (f, B) variables
- **Service Coordination**: Independent, sequential, and iterative strategies

### MDP Framework (Equations 15-18) ✅
- **Observation Space**: `o_{r,t} = {Z_{r,t}(ω), Θ_{r,t}, B_t}`
- **Action Space**: `a_{r,t} = {n_{r,t}, f_{r,t}, B_{r,t}, q_{r,t}}`
- **Reward Function**: `rwd_{r,t} = σ_1 Γ_{r,t} + σ_2 Φ_{r,t}(q) - σ_3 E_{r,t}^total - σ_4 T_{r,t}^total`
- **Adversarial Factor**: `Φ_{r,t}(q) = (n_{r,t}q_{r,t})/(ε·vol_{r,t} + Σ_{j≠r} n_{j,t}q_{j,t})`

### PAC-MCoFL Algorithm (Equations 19-27, Algorithm 1) ✅
- **Joint Policy**: `π = (π_r, π_{-r})` with cumulative expected rewards `J_r(π)`
- **Cumulative Rewards**: `J_r(π) = E[Σ_{t=0}^{T-1} γ^t E_π[rwd_{r,t} | o_{r,0}, π]]`
- **Pareto Optimality**: `π_{-r}^† ∈ arg max J_r(π_r, π_{-r})` and `π_r ∈ arg max J_r(π_r, π_{-r}^†)`
- **Virtual Joint Policy**: `π_{-r}^† ∈ arg max_{a_{-r,t}} Q_r^{π†}(o_{r,t}, a_{r,t}, a_{-r,t})`
- **TD Error**: `M_r(π) = E[(rwd_{r,t} + γ max_{a_{-r,t}} Q_r^{π†}(...) - Q_r^{π†}(...))²]`
- **Policy Gradient**: `∇_{φ_r} J_r(π)` with baseline subtraction for variance reduction
- **Network Updates**: Actor `φ_r = φ_r + ζ ∇_{φ_r} J_r(π)`, Critic `θ_r = θ_r - α ∇_{θ_r} M_r(π)`

### Action Space Transformation (Section 5) ✅
- **Ternary Action Space**: `a'(m) = {-1, 0, 1}` for each dimension m ∈ {1,2,3,4}
- **Cartesian Product**: `A' = ∏_m a'(m)` with maximum 3^4 = 81 combinations
- **Dimension Control**: Independent granularity for clients, frequency, bandwidth, quantization
- **PAC Integration**: Enables tractable computation of virtual joint policy π_{-r}^†
- **Semantic Interpretation**: -1 (decrease), 0 (maintain), 1 (increase) current values

### Experimental Design (Section 6) ✅
- **Multi-Service Architecture**: 5 clients (N=5) coordinated by 3 service providers (R=3)
- **Task-Specific Models**: CNN-4layer (CIFAR-10), CNN-2layer (FashionMNIST), MLP-2layer (MNIST)
- **Parameter Table Compliance**: All 20+ parameters matching paper specifications
- **Communication Parameters**: Channel gain [-73,-63]dB, noise [-174,-124]dBm/Hz, power [10,33]dBm
- **Reward Function Weights**: Service-specific σ₁,σ₂,σ₃,σ₄ values for equation (17)
- **Network Architecture**: Q-network (64,128), Policy network (64,128,64) neurons

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision flsim hydra-core omegaconf numpy matplotlib scipy

# Navigate to project directory
cd My_FL
```

### Comprehensive Demonstration

```bash
# Run complete validation of all components
python comprehensive_example.py

# Run PAC-MCoFL specific demonstrations  
python pac_example.py

# Run experimental design demonstrations (Sections 5-6)
python comprehensive_experimental_demo.py
```

This demonstrates:
1. **Optimization Problem**: Multi-service resource allocation
2. **MDP Framework**: State transitions and reward calculations
3. **Game Theory**: Nash equilibrium convergence
4. **MARL Training**: Multi-agent learning with Independent DQN
5. **PAC-MCoFL**: Pareto Actor-Critic with joint policy optimization
6. **Action Space Transform**: Ternary representation with 81 discrete actions
7. **Experimental Design**: 5 clients, 3 services, complete parameter setup

### Individual Component Testing

```bash
# Test specific components
python optimization_problem.py  # Equation (14) validation
python mdp_framework.py        # Equations (15-18) validation
python game_theory.py          # Non-cooperative game
python marl_interface.py       # MARL training
python pac_mcofl.py            # PAC-MCoFL algorithm (Equations 19-27)
python action_space_transform.py  # Ternary action space (Section 5)
python experimental_setup.py      # Experimental design (Section 6)
```

### Custom Configuration

```python
from optimization_problem import OptimizationConstraints, MultiServiceOptimizer
from mdp_framework import MultiServiceFLEnvironment
from marl_interface import MARLTrainer, MARLConfig
from pac_mcofl import PACMCoFLTrainer, PACConfig
from action_space_transform import ActionSpaceTransformer, ActionGranularity
from experimental_setup import ExperimentalEnvironment, ExperimentalParameters

# Define constraints
constraints = OptimizationConstraints(
    max_energy=0.1,      # Energy budget
    max_delay=5.0,       # Delay limit  
    max_clients=10,      # Client limit
    max_bandwidth=1e7    # Bandwidth budget
)

# Create environment
environment = MultiServiceFLEnvironment([1, 2, 3], constraints)

# Configure MARL
config = MARLConfig(
    algorithm="independent_dqn",
    training_episodes=100,
    learning_rate=0.001
)

# Train agents with MARL
trainer = MARLTrainer(environment, config, constraints)
results = trainer.train()

# Or train with PAC-MCoFL (Pareto optimal)
pac_config = PACConfig(
    num_episodes=100,
    actor_lr=0.0003,
    critic_lr=0.001
)
pac_trainer = PACMCoFLTrainer(service_ids, environment, fl_system, pac_config, constraints)
pac_results = pac_trainer.train()
```

## 📈 Advanced Features

### Optimization Module
- ✅ Multi-objective cost function with configurable weights
- ✅ Mixed-integer optimization with multiple algorithms
- ✅ Constraint satisfaction checking
- ✅ Service coordination strategies
- ✅ Performance metric evaluation

### MDP Framework
- ✅ Gym-compatible environment interface
- ✅ Multi-agent observation/action spaces
- ✅ Adversarial factor computation
- ✅ State transition dynamics
- ✅ Reward function with multiple components

### Game Theory
- ✅ Strategy conjecture learning
- ✅ Best response calculation
- ✅ Nash equilibrium detection
- ✅ Convergence analysis
- ✅ Non-cooperative decision making

### MARL Interface
- ✅ Independent DQN implementation
- ✅ Experience replay buffer
- ✅ Epsilon-greedy exploration
- ✅ Target network updates
- ✅ Multi-agent coordination

### PAC-MCoFL Algorithm
- ✅ Pareto Actor-Critic networks (φ_r, θ_r parameters)
- ✅ Joint policy optimization π = (π_r, π_{-r})
- ✅ Virtual joint policy π_{-r}^† computation
- ✅ Cumulative expected rewards J_r(π) calculation
- ✅ Pareto optimal equilibrium achievement
- ✅ Superior performance vs Nash equilibrium

## 📁 Project Structure

```
My_FL/
├── __init__.py                    # Package initialization
├── quantization.py                # Quantization module (Eq. 5-7)
├── communication.py               # Communication model (Eq. 8-13)
├── multi_service_fl.py            # FLSim integration (Eq. 1-4)
├── optimization_problem.py        # Multi-objective optimization (Eq. 14)
├── mdp_framework.py               # MDP formulation (Eq. 15-18)
├── game_theory.py                 # Non-cooperative game theory
├── marl_interface.py              # Multi-agent RL interface
├── pac_mcofl.py                   # PAC-MCoFL algorithm (Eq. 19-27, Alg. 1)
├── comprehensive_example.py       # Complete demonstration
├── pac_example.py                 # PAC-MCoFL specific examples
├── example.py                     # Basic validation examples
├── config.json                    # System configuration
└── README.md                      # This file
```

## 🧪 Validation Results

The comprehensive example validates:

1. **Optimization Performance**: 
   - Resource allocation efficiency
   - Constraint satisfaction
   - Multi-service coordination

2. **MDP Accuracy**:
   - State transition correctness
   - Reward function computation
   - Action space coverage

3. **Game Theory**:
   - Nash equilibrium convergence
   - Strategy evolution dynamics
   - Payoff optimization

4. **MARL Training**:
   - Learning curve convergence
   - Multi-agent coordination
   - Performance improvement

5. **PAC-MCoFL Algorithm**:
   - Pareto optimal equilibrium achievement
   - Superior performance vs Nash equilibrium
   - Joint policy optimization effectiveness
   - Algorithm 1 implementation correctness

## 🔍 Key Equations Implemented

| Equation | Description | Implementation |
|----------|-------------|----------------|
| (1) | Local Loss Function | `QuantizedFLModel.fl_forward()` |
| (2) | Global Optimization | FLSim aggregation with weights |
| (3) | Local SGD Updates | FLSim client training |
| (4) | Global Aggregation | FLSim server aggregation |
| (5) | q-level Quantization | `QuantizationModule.quantize_parameters()` |
| (6) | Stochastic Mapping | `QuantizationModule._quantize_element()` |
| (7) | Communication Volume | `QuantizationModule.calculate_communication_volume()` |
| (8) | Computation Energy | `CommunicationModel.calculate_computation_energy()` |
| (9) | Computation Delay | `CommunicationModel.calculate_computation_delay()` |
| (10) | Transmission Rate | `CommunicationModel.calculate_transmission_rate()` |
| (11) | Communication Delay | `CommunicationModel.calculate_communication_delay()` |
| (12) | Communication Energy | `CommunicationModel.calculate_communication_energy()` |
| (13a) | Total Energy | `SystemMetrics.calculate_service_total_energy()` |
| (13b) | Total Delay | `SystemMetrics.calculate_service_total_delay()` |
| (14) | Multi-Objective Optimization | `MultiObjectiveCostFunction.__call__()` |
| (15) | Observation Space | `Observation.to_array()` |
| (16) | Action Space | `Action.to_array()` |
| (17) | Reward Function | `RewardFunction.calculate()` |
| (18) | Adversarial Factor | `AdversarialFactor.calculate()` |
| (19) | Cumulative Expected Reward | `PACMCoFLTrainer.compute_cumulative_expected_reward()` |
| (20) | Value Function | `PACAgent.critic()` |
| (21) | Pareto Optimality Condition | `PACAgent.compute_virtual_joint_policy()` |
| (22) | Policy Optimization | `PACAgent.update_actor()` |
| (23) | Virtual Joint Policy | `PACAgent.compute_virtual_joint_policy()` |
| (24) | TD Error | `PACAgent.update_critic()` |
| (25) | Policy Gradient | `PACAgent.update_actor()` |
| (26) | Critic Update | `PACAgent.update_critic()` |
| (27) | Actor Update | `PACAgent.update_actor()` |
| Alg. 1 | PAC-MCoFL Training | `PACMCoFLTrainer.train()` |

## 🚧 Future Extensions

- [ ] Advanced MARL algorithms (MADDPG, QMIX, COMA)
- [ ] Hierarchical federated learning
- [ ] Real wireless channel simulation with fading
- [ ] Differential privacy integration
- [ ] Secure aggregation protocols
- [ ] Online learning and adaptation
- [ ] Fairness constraints and mechanisms
- [ ] Heterogeneous device modeling

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **FLSim**: Federated learning simulation
- **Hydra/OmegaConf**: Configuration management
- **NumPy**: Numerical computations
- **SciPy**: Optimization algorithms
- **Gym**: RL environment interface
- **Matplotlib**: Visualization (optional)

## 🤝 Contributing

This implementation serves as a comprehensive foundation for reproducing and extending the research paper. The modular design supports:

1. **New Optimization Algorithms**: Extend `ServiceProviderOptimizer`
2. **Additional MARL Methods**: Implement new agents in `marl_interface.py`
3. **Custom Game Strategies**: Extend `BestResponseCalculator`
4. **Enhanced Communication Models**: Modify `CommunicationModel`

## 📄 Research Implementation Status

✅ **COMPLETE IMPLEMENTATION**
- **Section 1**: System Model (Equations 1-13)
- **Section 2**: Problem Modeling (Equation 14)  
- **Section 3**: MDP Framework (Equations 15-18)
- **Section 4**: PAC-MCoFL Algorithm (Equations 19-27, Algorithm 1)
- **Game Theory**: Non-cooperative multi-agent strategies
- **MARL**: Multi-agent reinforcement learning training

🎯 **VALIDATION STATUS**
- All mathematical equations (1-27) implemented and tested
- Algorithm 1 completely implemented and verified
- Pareto optimal equilibrium vs Nash equilibrium comparison
- Comprehensive examples with multiple scenarios
- Performance metrics and convergence analysis
- Multi-service coordination validation

---

**Note**: This framework successfully implements the complete paper system model with advanced optimization, game theory, and multi-agent learning capabilities, providing a solid foundation for federated learning research with quantization and communication optimization.