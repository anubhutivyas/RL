---
description: "Get started quickly with NeMo RL by following these essential setup steps and choosing the right training approach for your reinforcement learning needs."
tags: ["quickstart", "setup", "beginner", "onboarding", "reinforcement learning", "distributed training"]
categories: ["getting-started"]
---

(gs-overview)=
# Get Started

Welcome to NeMo RL! This guide will help you set up your environment and run your first reinforcement learning training job with large language models.

## Before You Start

- **System Requirements**: Ensure you have CUDA-compatible GPUs and sufficient memory
- **Python Environment**: Python 3.9+ with PyTorch 2.0+ and Ray 2.6+
- **Model Access**: Access to Hugging Face models (Qwen, Llama, etc.)
- **Cluster Setup**: For distributed training, prepare your Ray cluster

---

## Quickstart Options

Choose the training approach that best fits your needs:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` SFT Quickstart
:link: quickstart
:link-alt: Supervised Fine-Tuning quickstart guide

Get started with supervised fine-tuning for language models. Perfect for beginners and domain adaptation tasks.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` GRPO Quickstart
:link: ../guides/grpo
:link-alt: Group Relative Policy Optimization guide

Learn Group Relative Policy Optimization for reinforcement learning with language models. Advanced RL training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Quickstart
:link: ../guides/dpo
:link-alt: Direct Preference Optimization guide

Explore Direct Preference Optimization for preference learning and alignment. Preference-based training.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: cluster
:link-alt: Cluster setup and distributed training guide

Set up multi-GPU and multi-node training clusters for large-scale RL experiments.

+++
{bdg-secondary}`Advanced`
:::
::::

## Essential Setup

### 1. Installation
- [Installation Guide](installation.md) - Complete setup instructions for all environments
- [Local Workstation](local-workstation.md) - Configure your development environment
- [Docker Setup](docker.md) - Use NeMo RL with Docker containers

### 2. Environment Configuration
- [Cluster Setup](cluster.md) - Configure Ray clusters for distributed training
- [Hardware Requirements](installation.md) - Understand GPU memory and compute needs
- [Authentication Setup](installation.md) - Set up Hugging Face authentication

### 3. First Training Run
- [Quickstart Tutorial](quickstart.md) - Run your first SFT training job
- [Configuration Guide](../reference/configuration.md) - Understand training parameters
- [Evaluation Setup](../guides/eval.md) - Set up model evaluation and benchmarking

## Next Steps

After completing your first training run:

1. **Explore Algorithms**: Try [GRPO](../guides/grpo.md) and [DPO](../guides/dpo.md) training
2. **Scale Up**: Set up [distributed training](cluster.md) for larger models
3. **Customize**: Learn to [add new models](../guides/adding-new-models.md) and environments
4. **Optimize**: Use [profiling tools](../guides/nsys-profiling.md) for performance tuning
5. **Deploy**: Follow the [packaging guide](../guides/packaging.md) for production deployment

## Getting Help

- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions
- [API Reference](../reference/api.md) - Complete API documentation
- [Community Support](https://github.com/NVIDIA-NeMo/RL/issues) - GitHub issues and discussions

```{toctree}
:maxdepth: 2
:caption: **Get Started**

installation
quickstart
local-workstation
docker
cluster
```
