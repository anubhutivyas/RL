(about-key-features)=
# Key Features

NeMo RL provides a robust, modular toolkit for scalable reinforcement learning and supervised fine-tuning of large language models. This summary highlights the core features most relevant to AI developers.

## 1. Scalable & Efficient Training
- **Single-GPU to Multi-Node**: Seamless scaling from 1 GPU to 64+ GPUs and multi-node clusters.
- **Model Size Flexibility**: Supports models from small (0.6B) to 70B+ parameters.
- **Advanced Parallelism**: FSDP2, Tensor Parallelism, Pipeline Parallelism, Context Parallelism.
- **Distributed with Ray**: Ray-based orchestration for distributed rollouts and training.
- **Memory Optimization**: Gradient checkpointing, mixed precision, and efficient batching.

## 2. Supported Algorithms
- **GRPO (Group Relative Policy Optimization)**: RL for reasoning and preference learning, no critic required.
- **DPO (Direct Preference Optimization)**: RL-free alignment using pairwise preference data.
- **SFT (Supervised Fine-Tuning)**: Standard supervised learning for initial alignment.
- **Multi-Turn RL**: Support for agentic, multi-step, and tool-use tasks.

## 3. Backend & Model Support
- **Backends**: PyTorch Native (FSDP2), Megatron Core (pipeline, tensor parallelism), vLLM (fast inference).
- **Model Integration**: Hugging Face, Megatron, vLLM, and custom architectures.
- **Conversion Tools**: Utilities for converting between Hugging Face and Torch-native formats.

## 4. Data & Environment Integration
- **Dataset Support**: Direct integration with Hugging Face datasets and custom data loaders.
- **Environment Abstractions**: Standardized RL environment interfaces for custom or built-in tasks.
- **Multi-Environment**: Support for complex, multi-environment RL scenarios.

## 5. Configuration & Extensibility
- **YAML Configs**: Human-readable configuration files for all training and rollout parameters.
- **CLI Overrides**: Command-line parameter overrides for rapid experimentation.
- **Unified APIs**: Standardized interfaces for training, rollout, and generation backends.
- **Modular Components**: Plug-and-play actors, environments, reward models, and policies.

## 6. Developer Experience
- **Comprehensive Documentation**: Detailed guides, API docs, and examples.
- **Type Hints & Testing**: Full type annotation and extensive test coverage.
- **Debugging & Profiling**: Integrated tools for debugging distributed training and profiling performance.
- **Logging & Monitoring**: Weights & Biases integration, advanced logging, and experiment tracking.

## 7. Deployment & Reproducibility
- **Local & Cluster Deployment**: Easy setup for local development, Slurm, Kubernetes, and cloud.
- **Container Support**: Docker-based workflows for reproducibility.
- **Python Environment Management**: uv and conda integration for isolated, reproducible environments.
- **Checkpointing**: Robust checkpointing and recovery for long-running jobs.

---

For more details, see the [API Documentation](../api-docs/index), [Guides](../guides/index), and [Design Docs](../design-docs/index).

