---
description: "Learn about NeMo RL's core concepts, key features, and fundamental architecture for reinforcement learning with large language models"
categories: ["concepts-architecture"]
tags: ["overview", "concepts", "architecture", "features", "reinforcement-learning", "distributed", "large-language-models"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

(about-overview)=
# About NeMo RL

NeMo RL is an open-source, comprehensive framework for reinforcement learning and supervised fine-tuning of large language models. Built for scalability and efficiency, NeMo RL enables researchers and practitioners to train, evaluate, and deploy RL-enhanced language models at scale.

## What is NeMo RL?

NeMo RL is a production-ready framework that combines the power of reinforcement learning with large language models. It provides a unified platform for training, fine-tuning, and deploying language models using state-of-the-art RL algorithms.

The framework is designed to handle the complexities of distributed training across multiple GPUs and nodes, making it suitable for both research and production environments. NeMo RL supports multiple training backends, including Hugging Face Transformers and Megatron-LM, providing flexibility for different model architectures and training requirements.

## Why Use NeMo RL?

NeMo RL offers several key advantages for reinforcement learning with language models:

- **Scalability**: Built on Ray for distributed computing, enabling training across multiple nodes and GPUs
- **Flexibility**: Support for multiple model backends and RL algorithms
- **Production Ready**: Comprehensive tooling for deployment and monitoring
- **Research Friendly**: Easy experimentation with different algorithms and environments
- **Performance**: Optimized for high-throughput training with advanced parallelization strategies

## Target Users

- **Researchers**: Explore state-of-the-art RL algorithms (GRPO, DPO, SFT) with large language models
- **Machine Learning Engineers**: Deploy scalable RL training pipelines with distributed computing
- **DevOps Engineers**: Manage multi-node training clusters and production deployments
- **Data Scientists**: Fine-tune language models for specific domains and applications

## Key Technologies

NeMo RL is built on a robust technology stack designed for performance and scalability:

- **Ray**: Distributed computing framework for scalable RL training and resource management
- **PyTorch**: Deep learning framework with advanced distributed training capabilities
- **Transformers**: Hugging Face integration for state-of-the-art language models
- **Megatron-LM**: High-performance training backend for large models with pipeline parallelism
- **DTensor**: Distributed tensor operations for efficient model parallelism

## Core Architecture

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` RL Algorithms
:link: key-features
:link-type: doc

Explore key RL algorithms including SFT, GRPO, and DPO with mathematical foundations and implementation details.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: architecture
:link-type: doc

Learn about Ray-based virtual clusters, worker groups, and multi-node training strategies for scalable RL.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Model Backends
:link: architecture
:link-type: doc

Understand Hugging Face integration, Megatron-LM support, and DTensor capabilities for different model architectures.

+++
{bdg-info}`Implementation`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Environments
:link: key-features
:link-type: doc

Discover built-in environments for math problems, games, and custom environment development for RL tasks.

+++
{bdg-secondary}`Development`
:::

::::



