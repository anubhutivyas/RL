# Guides

Welcome to the NeMo RL Guides! This comprehensive collection provides everything you need to master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our guides are organized into five main categories to help you navigate from basic concepts to practical implementations:

### **Training Algorithms** 
Learn the core RL algorithms that power NeMo RL. Start with Supervised Fine-Tuning (SFT) as your foundation, then explore advanced techniques like Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO). Each algorithm guide includes detailed explanations, mathematical foundations, and practical implementation examples.

### **Examples & Tutorials**
Jump into complete, working examples that demonstrate real-world applications. These tutorials walk you through entire training workflows, from data preparation to model deployment, using specific datasets and environments.

### **Model Development**
Extend NeMo RL with custom model architectures and backends. Learn to add new models, handle model-specific quirks, and implement custom model diagnostics for specialized use cases.

### **Environment & Data**
Create custom RL environments and optimize data processing pipelines. Learn distributed training strategies, custom environment development, and performance optimization techniques.

### **Production & Support**
Deploy your models to production and maintain reliable training pipelines. Learn testing strategies, performance profiling, packaging solutions, and troubleshooting common issues.

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

## Examples & Tutorials

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Train GRPO Models on DeepScaleR
:link: examples-tutorials/grpo-deepscaler
:link-type: doc

DeepScaleR integration for large-scale distributed training.

+++
{bdg-secondary}`Cloud`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Fine-tune Models on OpenMathInstruct-2
:link: examples-tutorials/sft-openmathinstruct2
:link-type: doc

Math instruction fine-tuning example with OpenMathInstruct-2 dataset.

+++
{bdg-primary}`Example`
:::

::::

## Model Development

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Adding New Models
:link: model-development/adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Model Quirks & Special Cases
:link: model-development/model-quirks
:link-type: doc

Handle model-specific behaviors and special cases in NeMo RL.

+++
{bdg-warning}`Advanced`
:::

::::

## Environment & Data

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: environment-data/debugging
:link-type: doc

Debugging techniques and tools for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Performance Profiling
:link: environment-data/nsys-profiling
:link-type: doc

Profile and optimize training performance with NVIDIA Nsight Systems.

+++
{bdg-secondary}`Performance`
:::

::::

## Production & Support

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Testing & Debugging
:link: production-support/testing
:link-type: doc

Testing strategies and debugging techniques for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: production-support/packaging
:link-type: doc

Deployment and packaging strategies for production environments.

+++
{bdg-secondary}`Production`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: production-support/troubleshooting
:link-type: doc

Common issues, error messages, and solutions for NeMo RL.

+++
{bdg-warning}`Support`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation
:link: model-development/documentation
:link-type: doc

Build and maintain NeMo RL documentation.

+++
{bdg-info}`Development`
:::

::::

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

training-algorithms/index
examples-tutorials/index
model-development/index
environment-data/index
production-support/index
```
