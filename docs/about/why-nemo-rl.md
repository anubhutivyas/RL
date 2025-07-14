# Why NeMo RL?

**NeMo RL** is an open-source, scalable post-training library developed by NVIDIA for applying **reinforcement learning (RL)** to large language models (LLMs). It supports models ranging from compact architectures to 100B+ parameters and scales from single-GPU setups to thousand-GPU clusters. NeMo RL is part of the broader [NVIDIA NeMo framework](https://docs.nvidia.com/nemo/rl/latest/index.html), which focuses on training and aligning generative AI.

Built for scalability, efficiency, and ease of use, NeMo RL provides researchers and practitioners with the tools needed to train, evaluate, and deploy RL-enhanced language models at scale.

## Architecture & Design Principles

- **Modular and Backend-Agnostic**  
  Separation between algorithm logic and backend execution—supports PyTorch Native, Megatron Core, and vLLM
- **Ray-Based Orchestration**  
  Rollouts and training components (actors, critics, reward models) run in isolated Ray processes
- **Environment Isolation**  
  Avoids global state conflicts by isolating training components in their own containers/processes
- **Unified Interfaces**  
  Training, rollout, and generation backends conform to standardized APIs for plug-and-play extensibility

## Supported Algorithms

| Algorithm | Purpose                    | Highlights                                |
|----------|-----------------------------|-------------------------------------------|
| **GRPO** | Reasoning-focused RL        | No critic needed, stable updates, memory-efficient |
| **DPO**  | Human preference alignment  | Uses pairwise comparisons, no reward model |
| **SFT**  | Initial supervised alignment| Prepares models for RL fine-tuning         |

NeMo RL also supports **multi-turn RL**, enabling training for agentic tasks like games or tool use.

## Backend Support

| Backend         | Best Use Case              | Key Features                                   |
|----------------|----------------------------|------------------------------------------------|
| **PyTorch Native** | Broad model support         | FSDP2, TP, SP, activation checkpointing        |
| **Megatron Core**  | Extremely large models      | Supports long context, MoE, deep parallelism   |
| **vLLM**           | High-speed generation       | Optimized inference, tensor parallelism        |

Backends are configurable and interchangeable without altering core algorithm logic.

## Integration Ecosystem

- **Hugging Face**: For model loading, datasets, tokenizer management
- **Ray**: Distributed rollout generation, process isolation, fault tolerance
- **uv**: Python environment management for reproducibility
- **Weights & Biases**: Logging and training visualization support

## Scalability & Performance

- **Rollout Generation**: Distributed across Ray actors using either Hugging Face or vLLM
- **Training Execution**: Supports various forms of parallelism including FSDP2 and DeepSpeed-style optimization
- **Multi-node Compatibility**: Works seamlessly with Slurm, Kubernetes, and bare-metal clusters
- **Checkpoint Flexibility**: Converts between Hugging Face and Torch-native formats

Benchmark highlights include GRPO achieving **3.2× faster convergence** compared to PPO on math-based tasks.

## Use Cases

NeMo RL supports a wide range of real-world applications for reinforcement learning with large language models. Each use case includes architectural patterns, implementation details, and production considerations.

### **Code Generation**
Train models to generate, debug, and optimize code across multiple programming languages. Includes syntax validation, multi-language support, and production deployment strategies.

• **Key Applications**: Code completion, bug fixing, code optimization, documentation generation  
• **Techniques**: Supervised fine-tuning on code datasets, preference learning for code quality  
• **Production**: Integration with IDEs, code review systems, automated testing

### **Mathematical Reasoning**
Build models that can solve complex mathematical problems with step-by-step reasoning, proof generation, and advanced mathematical concepts.

• **Key Applications**: Math tutoring, problem solving, theorem proving, educational AI  
• **Techniques**: Step-by-step reasoning, symbolic manipulation, proof generation  
• **Production**: Educational platforms, research tools, automated grading systems

### **Human Preference Alignment**
Use DPO or RLHF to align LLM output with human feedback and preferences.

• **Key Applications**: Content moderation, style adaptation, safety alignment  
• **Techniques**: Direct preference optimization, reinforcement learning from human feedback  
• **Production**: Content generation systems, conversational AI, safety filters

### **Multi-Turn RL**
Train models for agentic behavior over multiple interactions, enabling complex task completion and tool use.

• **Key Applications**: Game playing, tool use, multi-step problem solving  
• **Techniques**: Multi-turn reinforcement learning, environment interaction  
• **Production**: Gaming AI, automation systems, interactive assistants

For detailed implementation guides, architectural patterns, and production considerations, see the comprehensive [Use Cases](../tutorials-examples/use-cases/index) documentation.

**Example**: DeepScaleR (Qwen1.5B) trained with GRPO beat OpenAI's O1 on the AIME24 benchmark for mathematical reasoning tasks.

## Get Started

Ready to begin your NeMo RL journey? Here's how to get started:

### **Installation**
• [Installation Guide](../get-started/installation) - Complete setup instructions
• [Local Workstation](../get-started/local-workstation) - Development environment setup
• [Docker Setup](../get-started/docker) - Containerized installation
• [Cluster Setup](../get-started/cluster) - Multi-node environment configuration

### **Quick Start**
• [Quickstart Tutorial](../get-started/quickstart) - Run your first training job
• [Configuration Reference](../configuration-cli/configuration-reference) - YAML configuration options
• [CLI Reference](../configuration-cli/cli-reference) - Command-line interface guide
• [Examples](../tutorials-examples/index) - Ready-to-run training examples

### **Core Training Algorithms**
• [SFT Training](../guides/training-algorithms/sft) - Supervised fine-tuning for initial model alignment
• [GRPO Training](../guides/training-algorithms/grpo) - Group Relative Policy Optimization for reasoning tasks
• [DPO Training](../guides/training-algorithms/dpo) - Direct Preference Optimization for human alignment

### **Evaluation & Development**
• [Evaluation](../guides/training-algorithms/eval) - Model evaluation and benchmarking
• [Add New Models](../guides/model-development/adding-new-models) - Custom model integration
• [Debugging](../guides/environment-data/debugging) - Training debugging techniques
• [Environment Development](../guides/environment-data/environment-development) - Custom environment creation

### **Production & Scaling**
• [Distributed Training](../advanced/performance/distributed-training) - Multi-node training setup
• [Production Support](../guides/production-support/index) - Production deployment and monitoring
• [Performance Benchmarking](../advanced/performance/benchmarking) - Training performance optimization
• [Testing](../guides/production-support/testing) - Production testing strategies

### **Reference Documentation**
• [API Documentation](../api-docs/index) - Complete API documentation
• [Configuration & CLI](../configuration-cli/index) - Configuration options and command-line interface
• [Troubleshooting](../configuration-cli/troubleshooting) - Common issues and solutions
• [Architecture](../about/architecture) - System architecture and design principles

### **Advanced Topics**
• [Add New Models](../guides/model-development/adding-new-models) - Custom model integration
• [Debugging](../guides/environment-data/debugging) - Debugging techniques
• [Set Up Distributed Training with Ray](../advanced/performance/distributed-training) - Multi-node training
• [Testing](../guides/production-support/testing) - Production testing
• [Custom Loss Functions](../advanced/research/custom-algorithms) - Algorithm customization
• [Experimental Design](../advanced/research/experimental-design) - Research methodology

---

**Join the NeMo RL Community**
- [GitHub Repository](https://github.com/NVIDIA/NeMo-RL) - Source code and issues
- [Discord Community](https://discord.gg/nvidia-nemo) - Get help and share ideas
- [Documentation](https://docs.nvidia.com/nemo-rl/) - Complete documentation

Start building the future of reinforcement learning with large language models today! 