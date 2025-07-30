---
description: "Comprehensive collection of guides for mastering reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["guides", "training-algorithms", "model-development", "environment-data", "production-support", "reference"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "universal"
---

# About NeMo RL Guides

Welcome to the NeMo RL Guides! This comprehensive collection provides everything you need to master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our guides are organized into four core areas that cover the essential needs of NeMo RL practitioners:

### **Training Algorithms**
Master the fundamental training techniques for reinforcement learning with language models. Learn supervised fine-tuning, preference optimization, and advanced RL algorithms like DPO and GRPO. This section provides the foundation for training high-quality language models with human feedback.

### **Model Development**
Integrate custom models and architectures into NeMo RL training pipelines. Handle model-specific behaviors, special cases, and learn how to extend the framework for new model types. This section is essential for researchers and developers working with custom architectures.

### **Environment and Data**
Set up robust development environments and optimize your training infrastructure. Learn debugging techniques, performance profiling, and data management strategies. This section helps you build reliable, efficient training workflows.

### **Production and Support**
Deploy and maintain NeMo RL models in production environments. Learn testing strategies, debugging techniques, packaging, and deployment best practices. This section ensures your models are ready for real-world applications.





## Training Algorithms

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Training
:link: training-algorithms/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: training-algorithms/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: training-algorithms/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation
:link: training-algorithms/eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Model Development

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Add New Models
:link: model-development/adding-new-models
:link-type: doc

Learn how to integrate custom models and architectures into NeMo RL training pipelines.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Model Quirks and Special Cases
:link: model-development/model-quirks
:link-type: doc

Handle model-specific behaviors and special cases in NeMo RL.

+++
{bdg-warning}`Advanced`
:::

::::

## Environment and Data

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: environment-data/debugging
:link-type: doc

Ray distributed debugging with VS Code/Cursor integration and SLURM debugging techniques.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` NSYS Profiling
:link: environment-data/nsys-profiling
:link-type: doc

NSYS-specific profiling for RL training performance.

+++
{bdg-secondary}`Performance`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Advanced Performance
:link: ../advanced/performance/index
:link-type: doc

Comprehensive performance optimization and profiling techniques.

+++
{bdg-warning}`Advanced`
:::

::::

## Production and Support

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Testing
:link: production-support/testing
:link-type: doc

Comprehensive testing strategies including unit tests, functional tests, and metrics tracking.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: production-support/packaging
:link-type: doc

Package models and training code for deployment in production environments.

+++
{bdg-secondary}`Production`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation Development
:link: production-support/documentation-development
:link-type: doc

Guidelines and best practices for developing and maintaining NeMo RL documentation.

+++
{bdg-info}`Docs`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: production-support/troubleshooting
:link-type: doc

Resolve common problems and errors in production environments.

+++
{bdg-warning}`Support`
:::

::::

For additional learning resources, visit the main [Guides](../index) page.


