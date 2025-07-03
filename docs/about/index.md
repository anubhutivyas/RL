---
description: "Learn about NeMo RL's core concepts, key features, and fundamental architecture for reinforcement learning with large language models."
tags: ["overview", "concepts", "architecture", "features", "reinforcement learning", "distributed training"]
categories: ["concepts"]
---

(about-overview)=
# About

NeMo RL is an open-source, comprehensive framework for reinforcement learning and supervised fine-tuning of large language models. Built for scalability and efficiency, NeMo RL enables researchers and practitioners to train, evaluate, and deploy RL-enhanced language models at scale.

## Target Users

- **Researchers**: Explore state-of-the-art RL algorithms (GRPO, DPO, SFT) with large language models
- **Machine Learning Engineers**: Deploy scalable RL training pipelines with distributed computing
- **DevOps Engineers**: Manage multi-node training clusters and production deployments
- **Data Scientists**: Fine-tune language models for specific domains and applications

## How NeMo RL Works

NeMo RL provides a unified framework for reinforcement learning with language models through:

- **Multiple RL Algorithms**: Support for SFT (Supervised Fine-Tuning), GRPO (Group Relative Policy Optimization), and DPO (Direct Preference Optimization)
- **Distributed Training**: Ray-based virtual clusters for scalable multi-GPU and multi-node training
- **Model Flexibility**: Support for Hugging Face models and Megatron-LM backends
- **Environment Integration**: Built-in environments for math problems, games, and custom tasks

### Key Technologies

- **Ray**: Distributed computing framework for scalable RL training
- **PyTorch**: Deep learning framework with advanced distributed training capabilities
- **Transformers**: Hugging Face integration for state-of-the-art language models
- **Megatron-LM**: High-performance training backend for large models

## Core Concepts

Explore the foundational concepts and organizational patterns used in NeMo RL.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` RL Algorithms
:link: key-features

Explore key RL algorithms including SFT, GRPO, and DPO with mathematical foundations and implementation details.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: architecture

Learn about Ray-based virtual clusters, worker groups, and multi-node training strategies for scalable RL.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Model Backends
:link: architecture

Understand Hugging Face integration, Megatron-LM support, and DTensor capabilities for different model architectures.
:::

:::{grid-item-card} {octicon}`game;1.5em;sd-mr-1` Environments
:link: key-features

Discover built-in environments for math problems, games, and custom environment development for RL tasks.
:::

::::

## Getting Started

- [Installation](../get-started/installation.md) - Set up NeMo RL on your system
- [Quickstart](../get-started/quickstart.md) - Run your first RL training job
- [Key Features](key-features.md) - Explore NeMo RL's capabilities
- [Architecture](architecture.md) - Understand the system design

```{toctree}
:maxdepth: 2
:caption: About

key-features
architecture
why-nemo-rl
```
