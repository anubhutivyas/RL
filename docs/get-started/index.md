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
- **Python Environment**: Python 3.9+ with PyTorch 2.0+ and Ray
- **Model Access**: Access to Hugging Face models (Qwen, Llama, etc.)
- **Cluster Setup**: For distributed training, prepare your Ray cluster

---

## Essential Setup

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="gear" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Installation
:link: installation
:link-type: doc

Complete setup instructions for all environments and platforms.

+++
{bdg-success}`Essential`
:::

:::{grid-item-card} <span class="octicon" data-icon="rocket" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Quickstart
:link: quickstart
:link-type: doc

Run your first RL training job in minutes with our step-by-step guide.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} <span class="octicon" data-icon="computer" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Local Workstation
:link: local-workstation
:link-type: doc

Configure your development environment for optimal performance.

+++
{bdg-secondary}`Setup`
:::

:::{grid-item-card} <span class="octicon" data-icon="package" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Docker Setup
:link: docker
:link-type: doc

Use NeMo RL with Docker containers for consistent environments.

+++
{bdg-secondary}`Container`
:::

::::

## Choose Your Training Approach

Select the training method that best fits your needs and experience level:

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="play" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> SFT Training
:link: quickstart
:link-alt: Supervised Fine-Tuning quickstart guide

Get started with supervised fine-tuning for language models. Perfect for beginners and domain adaptation tasks.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> GRPO Training
:link: ../guides/algorithms/grpo
:link-alt: Group Relative Policy Optimization guide

Learn Group Relative Policy Optimization for reinforcement learning with language models. Advanced RL training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} <span class="octicon" data-icon="heart" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> DPO Training
:link: ../guides/algorithms/dpo
:link-alt: Direct Preference Optimization guide

Explore Direct Preference Optimization for preference learning and alignment. Preference-based training.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} <span class="octicon" data-icon="server" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Distributed Training
:link: cluster
:link-alt: Cluster setup and distributed training guide

Set up multi-GPU and multi-node training clusters for large-scale RL experiments.

+++
{bdg-secondary}`Advanced`
:::

::::

## Recommended Learning Path

If you're new to NeMo RL, we recommend following this progression:

1. **Start with SFT Training** - Build your foundation with supervised fine-tuning
2. **Explore DPO Training** - Learn preference-based learning techniques  
3. **Try an End-to-End Example** - Run a complete training workflow
4. **Dive into GRPO** - Master advanced reinforcement learning
5. **Scale with Performance Tools** - Optimize with distributed training and profiling
6. **Customize with Development Tools** - Extend and deploy your solutions

## Environment Configuration

### Setup Options
- **[Installation Guide](installation.md)** - Complete setup instructions for all environments
- **[Local Workstation](local-workstation.md)** - Configure your development environment
- **[Docker Setup](docker.md)** - Use NeMo RL with Docker containers
- **[Cluster Setup](cluster.md)** - Configure Ray clusters for distributed training

### System Requirements
- **Hardware Requirements** - Understand GPU memory and compute needs
- **Authentication Setup** - Set up Hugging Face authentication
- **Environment Variables** - Configure necessary environment variables

## Learning Paths

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="book" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Tutorials & Learning
:link: tutorials
:link-alt: Comprehensive tutorials and learning resources

Access structured learning paths, tutorial series, and advanced techniques for mastering NeMo RL.

+++
{bdg-primary}`Learning`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Evaluation Setup
:link: ../guides/algorithms/eval
:link-alt: Model evaluation and benchmarking

Set up model evaluation and benchmarking for your trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Next Steps

After completing your first training run:

1. **Explore Algorithms**: Try [GRPO](../guides/algorithms/grpo.md) and [DPO](../guides/algorithms/dpo.md) training
2. **Scale Up**: Set up [distributed training](cluster.md) for larger models
3. **Optimize Performance**: Use [profiling tools](../guides/development/nsys-profiling.md) for training optimization
4. **Customize**: Learn to [add new models](../guides/development/adding-new-models.md) and environments
5. **Deploy**: Follow the [packaging guide](../guides/development/packaging.md) for production deployment

## Get Help

- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions
- [API Reference](../reference/api.md) - Complete API documentation
- [Configuration Guide](../reference/configuration.md) - Understand training parameters
- [Community Support](https://github.com/NVIDIA-NeMo/RL/issues) - GitHub issues and discussions

---

```{toctree}
:caption: Get Started
:maxdepth: 2
:expanded:

installation
quickstart
local-workstation
docker
cluster
tutorials
feature-set-a
feature-set-b
```






