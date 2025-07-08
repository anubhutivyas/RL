(about-key-features)=
# Key Features

NeMo RL provides a comprehensive suite of features for scalable reinforcement learning and supervised fine-tuning of large language models.

## Core Capabilities

### Scalable Training
- **1 GPU to 64+ GPUs**: Scale seamlessly from single GPU to multi-node clusters
- **Tiny to 70B+ Parameters**: Support for models ranging from small to large scale
- **Advanced Parallelism**: FSDP2, Tensor Parallelism, Pipeline Parallelism, and Context Parallelism

### Multiple Training Backends
- **Hugging Face Integration**: Easy integration with popular pre-trained models
- **Megatron-LM**: High-performance training for large models with pipeline parallelism
- **DTensor (FSDP2)**: Next-generation distributed training with improved memory efficiency

### RL and Alignment Algorithms
- **GRPO (Group Relative Policy Optimization)**: Advanced RL algorithm for preference learning
- **DPO (Direct Preference Optimization)**: RL-free alignment algorithm
- **SFT (Supervised Fine-Tuning)**: Traditional supervised learning approach
- **Multi-Turn RL**: Support for complex multi-turn interactions and tool use

### Fast Inference
- **vLLM Backend**: Optimized inference with high throughput
- **Dynamic Weight Updates**: Real-time model updates during training
- **Efficient Resource Utilization**: Smart worker management and load balancing

## Architecture Features

### Modular Design
- **RL Actors**: Composable components for policy, environment, reward, etc.
- **Process Isolation**: Isolated environments to prevent dependency conflicts
- **Worker Groups**: Efficient resource allocation and management

### Distributed Infrastructure
- **Ray-Based**: Scalable and flexible deployment using Ray
- **Virtual Clusters**: Dynamic resource allocation and management
- **NCCL Collectives**: High-performance communication between workers

### Configuration System
- **YAML Configuration**: Human-readable configuration files
- **CLI Overrides**: Flexible parameter overrides via command line
- **Type-Safe**: Comprehensive type checking and validation

## Development Features

### Environment Management
- **Dependency Isolation**: Separate environments for different components
- **Virtual Environment Support**: Full uv and conda integration
- **Container Support**: Docker and Kubernetes deployment options

### Monitoring and Logging
- **WandB Integration**: Comprehensive experiment tracking
- **Performance Profiling**: NSight Systems integration for performance analysis
- **Debugging Tools**: Advanced debugging and diagnostic capabilities

### Code Quality
- **Type Hints**: Full type annotation support
- **Testing Framework**: Comprehensive unit and integration tests
- **Documentation**: Auto-generated API documentation

## Performance Features

### Memory Efficiency
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Mixed Precision**: Automatic mixed precision training
- **Optimized Loss Functions**: Carefully designed for correct gradient accumulation

### Training Optimization
- **Batch Normalization**: Proper handling of variable batch sizes
- **Right Padding**: Consistent padding strategy for LLM compatibility
- **Dynamic Batching**: Efficient handling of variable sequence lengths

## Ecosystem Integration

### Model Support
- **Hugging Face Models**: 0.6B-70B parameter models (Qwen, Llama, Gemma, etc.)
- **Custom Models**: Easy integration of custom model architectures
- **Model Conversion**: Tools for converting between different formats

### Dataset Support
- **Hugging Face Datasets**: Direct integration with HF datasets
- **Custom Datasets**: Flexible dataset loading and processing
- **Multi-Environment**: Support for complex multi-environment scenarios

### Deployment Options
- **Local Development**: Easy local setup and development
- **Cluster Deployment**: Slurm and Kubernetes support
- **Cloud Integration**: AWS, GCP, and Azure deployment options

