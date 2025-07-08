# API Reference

This page provides access to the complete NeMo RL API documentation.

## Core Modules

### Algorithms
- **GRPO**: Group Relative Policy Optimization implementation
- **DPO**: Direct Preference Optimization implementation  
- **SFT**: Supervised Fine-Tuning implementation
- **Loss Functions**: Specialized loss functions for RL training

### Distributed Training
- **Virtual Cluster**: Ray-based resource management
- **Worker Groups**: Process isolation and management
- **Collectives**: High-performance communication utilities

### Models
- **Policy**: Language model policy implementations
- **Generation**: Text generation backends (vLLM, etc.)
- **Converters**: Model format conversion utilities

### Data
- **Datasets**: Dataset loading and processing
- **Message Utils**: LLM message formatting utilities
- **Interfaces**: Data processing interfaces

### Environments
- **Math Environment**: Mathematical problem solving environment
- **Games**: Game environments (sliding puzzle, etc.)
- **Metrics**: Environment evaluation metrics

### Utils
- **Configuration**: Configuration management utilities
- **Logging**: Logging and monitoring utilities
- **Checkpointing**: Model checkpoint management

## Auto-Generated Documentation

The complete API documentation is automatically generated from the source code and available at:

- **[Module Index](../genindex.html)** - Alphabetical index of all modules
- **[Search](../search.html)** - Search across all documentation

## Key Classes and Functions

### Core Training Classes

```python
from nemo_rl.algorithms import GRPO, DPO, SFT
from nemo_rl.models.policy import Policy
from nemo_rl.distributed import RayVirtualCluster, RayWorkerGroup
```

### Configuration

```python
from nemo_rl.utils.config import Config
from nemo_rl.utils.logger import setup_logger
```

### Data Processing

```python
from nemo_rl.data.datasets import load_dataset
from nemo_rl.data.llm_message_utils import format_messages
```

### Model Generation

```python
from nemo_rl.models.generation import VllmGeneration
from nemo_rl.models.generation.interfaces import GenerationInterface
```

## Examples

### Basic Training Setup

```python
from nemo_rl.distributed import RayVirtualCluster
from nemo_rl.models.policy import Policy
from nemo_rl.algorithms import SFT

# Initialize cluster
cluster = RayVirtualCluster(gpus_per_node=4)

# Create policy
policy = Policy(cluster, config, tokenizer)

# Run training
sft = SFT(policy, config)
sft.train()
```

### Custom Dataset

```python
from nemo_rl.data.datasets import DatasetInterface

class CustomDataset(DatasetInterface):
    def __init__(self, config):
        super().__init__(config)
    
    def __getitem__(self, idx):
        # Return formatted data
        return {
            "input_ids": input_tokens,
            "attention_mask": attention_mask,
            "labels": label_tokens
        }
```

### Custom Environment

```python
from nemo_rl.environments.interfaces import EnvironmentInterface

class CustomEnvironment(EnvironmentInterface):
    def __init__(self, config):
        super().__init__(config)
    
    def step(self, action):
        # Implement environment step
        return observation, reward, done, info
```

## Type Hints

NeMo RL uses comprehensive type hints throughout the codebase. Key type definitions:

```python
from typing import Dict, List, Optional, Union
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# Common types
BatchedDataDict = Dict[str, Tensor]
ConfigDict = Dict[str, Any]
```

## Error Handling

### Common Exceptions

```python
from nemo_rl.utils.exceptions import (
    ConfigurationError,
    TrainingError,
    ResourceError
)

try:
    # Training code
    sft.train()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except TrainingError as e:
    print(f"Training error: {e}")
```

### Debugging

Enable debug mode for detailed error information:

```python
import logging
logging.getLogger("nemo_rl").setLevel(logging.DEBUG)
```

## Performance Considerations

### Memory Management

```python
# Enable gradient checkpointing
policy.gradient_checkpointing = True

# Use mixed precision
policy.mixed_precision = True

# Clear cache periodically
torch.cuda.empty_cache()
```

### Distributed Training

```python
# Configure tensor parallelism
policy.dtensor_cfg.enabled = True
policy.dtensor_cfg.tensor_parallel_size = 2

# Configure pipeline parallelism
policy.megatron_cfg.pipeline_model_parallel_size = 2
```

## Contributing to the API

When contributing to NeMo RL:

1. **Follow Type Hints**: Always include type annotations
2. **Documentation**: Add docstrings for all public APIs
3. **Testing**: Include unit tests for new functionality
4. **Interfaces**: Use abstract base classes for extensibility

### Adding New Algorithms

```python
from nemo_rl.algorithms.interfaces import AlgorithmInterface

class CustomAlgorithm(AlgorithmInterface):
    def __init__(self, policy, config):
        super().__init__(policy, config)
    
    def train(self):
        # Implement training logic
        pass
    
    def evaluate(self):
        # Implement evaluation logic
        pass
```

## Version Compatibility

NeMo RL API compatibility:

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **Ray**: 2.6+

Check the [changelog](https://github.com/NVIDIA-NeMo/RL/releases) for detailed version information.

## Getting Help

- **API Documentation**: [Complete API Reference](../genindex.html)
- **Issues**: Report bugs on [GitHub](https://github.com/NVIDIA-NeMo/RL/issues)
- **Community**: Join the [NeMo Discord](https://discord.gg/nvidia-nemo) 