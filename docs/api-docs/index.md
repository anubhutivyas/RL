# API Documentation

Welcome to the NeMo RL API documentation! This section provides comprehensive reference documentation for all the APIs, interfaces, and components that make up the NeMo RL framework.

## Overview

NeMo RL provides a modular and extensible API for reinforcement learning with large language models. The framework is built around several core abstractions that enable scalable, distributed training and inference across multiple backends.

## Core Architecture

NeMo RL is designed around four key capabilities that every RL system needs:

1. **Resource Management**: Allocate and manage compute resources (GPUs/CPUs)
2. **Isolation**: Provide isolated process environments for different components
3. **Coordination**: Control and orchestrate distributed components
4. **Communication**: Enable data flow between components

These capabilities are implemented through a set of composable abstractions that scale from single GPU to thousands of GPUs.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Computing
:link: distributed
:link-type: doc

Core distributed computing abstractions including VirtualCluster and WorkerGroup.

+++
{bdg-primary}`Core`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Models & Policies
:link: models
:link-type: doc

Model interfaces, policy implementations, and generation backends.

+++
{bdg-info}`Models`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Algorithms
:link: algorithms
:link-type: doc

RL algorithms including DPO, GRPO, SFT, and custom loss functions.

+++
{bdg-warning}`Training`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data & Environments
:link: data
:link-type: doc

Data processing, dataset interfaces, and environment implementations.

+++
{bdg-secondary}`Data`
:::

::::

## Key Components

### Distributed Computing

The distributed computing layer provides abstractions for managing compute resources and coordinating distributed processes:

- **VirtualCluster**: Manages resource allocation and placement groups
- **WorkerGroup**: Coordinates groups of distributed worker processes
- **BatchedDataDict**: Efficient data structures for distributed communication

### Models & Policies

The model layer provides interfaces for different model backends and policy implementations:

- **PolicyInterface**: Abstract interface for RL policies
- **GenerationInterface**: Unified interface for text generation backends
- **Model Backends**: Support for Hugging Face, Megatron, and custom backends

### Algorithms

The algorithms layer implements various RL algorithms and training methods:

- **DPO**: Direct Preference Optimization
- **GRPO**: Group Relative Policy Optimization  
- **SFT**: Supervised Fine-Tuning
- **Custom Loss Functions**: Extensible loss function framework

### Data & Environments

The data layer handles data processing, dataset management, and environment interactions:

- **Dataset Interfaces**: Standardized dataset loading and processing
- **Environment Interfaces**: RL environment abstractions
- **Data Processing**: Tokenization, batching, and preprocessing utilities

## Design Philosophy

NeMo RL follows several key design principles:

### Modular Abstractions

Each component is designed as a modular abstraction that can be composed and extended:

```python
# Example: Composing a training pipeline
policy = HuggingFacePolicy(model_name="llama2-7b")
generator = VllmGeneration(cluster, config)
environment = MathEnvironment()
dataloader = BatchedDataLoader(dataset)

# All components work together through interfaces
for batch in dataloader:
    generations = generator.generate(batch)
    rewards = environment.step(generations)
    policy.train(generations, rewards)
```

### Backend Independence

The framework is designed to be backend-agnostic, allowing easy switching between different implementations:

```python
# Same interface, different backends
policy_hf = HuggingFacePolicy(model_name="llama2-7b")
policy_megatron = MegatronPolicy(model_name="llama2-7b")

# Both implement the same PolicyInterface
generations = policy_hf.generate(batch)  # Same API
generations = policy_megatron.generate(batch)  # Same API
```

### Scalability

The abstractions scale seamlessly from single GPU to thousands of GPUs:

```python
# Single GPU
cluster = RayVirtualCluster([1])  # 1 GPU

# Multi-GPU
cluster = RayVirtualCluster([8, 8])  # 2 nodes, 8 GPUs each

# Same code works at any scale
worker_group = RayWorkerGroup(cluster, policy_class)
```

## Getting Started

### Basic Usage

```python
from nemo_rl.distributed import RayVirtualCluster, RayWorkerGroup
from nemo_rl.models import HuggingFacePolicy
from nemo_rl.algorithms import DPOTrainer

# Set up distributed environment
cluster = RayVirtualCluster([4])  # 4 GPUs
policy = HuggingFacePolicy("llama2-7b")

# Create trainer
trainer = DPOTrainer(policy, cluster)

# Start training
trainer.train(dataset)
```

### Custom Components

Extending NeMo RL with custom components is straightforward:

```python
from nemo_rl.models.interfaces import PolicyInterface

class CustomPolicy(PolicyInterface):
    def generate(self, batch):
        # Custom generation logic
        pass
    
    def train(self, batch, loss_fn):
        # Custom training logic
        pass
```

## API Reference

The following sections provide detailed API documentation for each component:

- [Distributed Computing](distributed): VirtualCluster, WorkerGroup, and distributed utilities
- [Models & Policies](models): Policy interfaces, generation backends, and model implementations
- [Algorithms](algorithms): RL algorithms, loss functions, and training utilities
- [Data & Environments](data): Dataset interfaces, environment abstractions, and data processing
- [Utilities](utils): Logging, configuration, and utility functions
- [Auto-Generated Reference](auto-generated): Complete API reference with all functions, classes, and parameters
- [Complete API Reference](nemo_rl/nemo_rl): Full auto-generated documentation for all modules

## Examples

See the [Guides](../guides/index) section for practical examples and tutorials covering:

- [DPO Training](../guides/training-algorithms/dpo)
- [GRPO Training](../guides/training-algorithms/grpo) 
- [SFT Training](../guides/training-algorithms/sft)
- [Model Evaluation](../guides/training-algorithms/eval)
- [Adding New Models](../guides/model-development/adding-new-models)

## Contributing

When contributing to NeMo RL, please follow these guidelines:

1. **Implement Interfaces**: New components should implement the appropriate abstract interfaces
2. **Add Tests**: Include comprehensive tests for new functionality
3. **Update Documentation**: Keep API documentation current with code changes
4. **Follow Patterns**: Use established patterns for consistency

For more information on contributing, see the [Development Guide](../guides/production-support/testing).

```{toctree}
:maxdepth: 2
:caption: API Documentation

distributed
models
auto-generated
nemo_rl/nemo_rl

# Auto-generated API Reference
nemo_rl/nemo_rl.algorithms
nemo_rl/nemo_rl.distributed
nemo_rl/nemo_rl.models
nemo_rl/nemo_rl.data
nemo_rl/nemo_rl.utils
``` 