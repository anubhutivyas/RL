# Why NeMo RL?

NeMo RL is a comprehensive, open-source framework designed for reinforcement learning and supervised fine-tuning of large language models. Built for scalability, efficiency, and ease of use, it provides researchers and practitioners with the tools needed to train, evaluate, and deploy RL-enhanced language models at scale.

## Key Benefits

### **Scalable Training**
- **Multi-GPU & Multi-Node Support**: Scale from single GPU to 64+ GPUs across multiple nodes
- **Ray-Based Distributed Computing**: Leverage Ray's proven distributed computing capabilities
- **Memory-Efficient Training**: Advanced memory management for large model training

### **Unified Algorithm Support**
- **SFT (Supervised Fine-Tuning)**: Traditional supervised learning for language models
- **GRPO (Group Relative Policy Optimization)**: Advanced RL algorithm for policy optimization
- **DPO (Direct Preference Optimization)**: RL-free alignment using preference data

### **Flexible Model Integration**
- **Hugging Face Models**: Seamless integration with the Hugging Face ecosystem
- **Megatron-LM Support**: High-performance training backend for large models with pipeline parallelism
- **DTensor Support**: Advanced distributed tensor capabilities

### **Built-in Environments**
- **Math Environment**: Mathematical problem-solving tasks
- **Game Environments**: Sliding puzzle and other interactive games
- **Custom Environment Framework**: Easy creation of new RL environments

## Use Cases

### **Research & Development**
- **RLHF Research**: Explore reinforcement learning from human feedback
- **LLM Alignment**: Align language models with human preferences
- **Algorithm Development**: Test and benchmark new RL algorithms
- **Model Evaluation**: Comprehensive evaluation and benchmarking tools

### **Enterprise Applications**
- **Domain Adaptation**: Fine-tune models for specific business domains
- **Large-Scale Training**: Production-ready distributed training pipelines
- **Custom Task Development**: Build specialized RL environments for business needs
- **Model Optimization**: Optimize models for specific use cases

### **Cloud & On-Premises Deployment**
- **Cloud Training**: Scale training across cloud infrastructure
- **On-Premises Clusters**: Deploy on existing HPC infrastructure
- **Hybrid Environments**: Mix cloud and on-premises resources
- **Containerized Deployment**: Docker support for consistent environments

## Competitive Advantages

### **Open Source Excellence**
- **Fully Open Source**: MIT license for maximum flexibility
- **Active Development**: Backed by NVIDIA and the open-source community
- **Extensible Architecture**: Modular design for easy customization
- **Community Driven**: Regular updates and community contributions

### **Developer Experience**
- **Fast Onboarding**: Get started in minutes with comprehensive documentation
- **Intuitive APIs**: Clean, well-documented interfaces
- **Rich Ecosystem**: Extensive examples and tutorials
- **Production Ready**: Built for both research and production use

### **Research Focus**
- **State-of-the-Art Algorithms**: Latest RL and alignment techniques
- **Reproducible Results**: Comprehensive logging and evaluation
- **Academic Friendly**: Designed with research workflows in mind
- **Benchmarking Tools**: Built-in evaluation and comparison capabilities

## Technical Advantages

### **Modern Architecture**
- **PyTorch 2.0+**: Latest PyTorch features and optimizations
- **Transformers**: Cutting-edge transformer model support
- **Ray**: Proven distributed computing framework
- **CUDA**: Optimized GPU acceleration

### **Advanced Features**
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Mixed Precision**: Automatic mixed precision for faster training
- **Advanced Logging**: Comprehensive training monitoring and visualization
- **Checkpoint Management**: Robust checkpointing and recovery

### **Developer Tools**
- **Type Hints**: Full type annotation for better development experience
- **Comprehensive Testing**: Extensive unit and functional test suites
- **Debugging Support**: Advanced debugging tools for distributed training
- **Performance Profiling**: Built-in profiling and optimization tools

## Get Started

Ready to begin your NeMo RL journey? Here's how to get started:

### **Installation**
- [Installation Guide](../get-started/installation.md) - Complete setup instructions
- [Local Workstation](../get-started/local-workstation.md) - Development environment setup
- [Docker Setup](../get-started/docker.md) - Containerized installation

### **Quick Start**
- [Quickstart Tutorial](../get-started/quickstart.md) - Run your first training job
- [Cluster Setup](../get-started/cluster.md) - Distributed training configuration

### **Training Guides**
- See the {ref}`guides/algorithms/sft.md` for SFT training.
- See the {ref}`guides/algorithms/grpo.md` for GRPO training.
- See the {ref}`guides/algorithms/dpo.md` for DPO training.
- See the {ref}`guides/algorithms/eval.md` for evaluation.
- See the {ref}`guides/development/adding-new-models.md` for adding new models.
- See the {ref}`guides/development/environment-development.md` for custom environments.
- See the {ref}`guides/development/distributed-training.md` for distributed training.
- See the {ref}`guides/development/packaging.md` for packaging and deployment.

### **Reference Documentation**
- [API Reference](../reference/api.md) - Complete API documentation
- [Configuration](../reference/configuration.md) - Configuration options
- [CLI Reference](../reference/cli.md) - Command-line interface
- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions

### **Advanced Topics**
- [Adding New Models](../guides/development/adding-new-models.md) - Custom model integration
- [Create Custom RL Environments](../guides/development/environment-development.md) - Custom RL environments
- [Set Up Distributed Training with Ray](../guides/development/distributed-training.md) - Multi-node training
- [Packaging](../guides/development/packaging.md) - Production deployment

---

**Join the NeMo RL Community**
- [GitHub Repository](https://github.com/NVIDIA-NeMo/RL) - Source code and issues
- [Discord Community](https://discord.gg/nvidia-nemo) - Get help and share ideas
- [Documentation](https://docs.nvidia.com/nemo-rl/) - Complete documentation

Start building the future of reinforcement learning with large language models today! 