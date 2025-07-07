# Guides

Welcome to the NeMo RL Guides! This comprehensive collection provides everything you need to master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our guides are organized into four main categories to help you navigate from basic concepts to advanced implementations:

### 🎯 **Training Algorithms** 
Learn the core RL algorithms that power NeMo RL. Start with Supervised Fine-Tuning (SFT) as your foundation, then explore advanced techniques like Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO). Each algorithm guide includes detailed explanations, mathematical foundations, and practical implementation examples.

### 🚀 **Examples & Tutorials**
Jump into complete, working examples that demonstrate real-world applications. These tutorials walk you through entire training workflows, from data preparation to model deployment, using specific datasets and environments.

### ⚡ **Performance & Scaling**
Master the performance optimization and scaling aspects of NeMo RL. Learn distributed training strategies, performance profiling techniques, and how to handle model-specific considerations for optimal results.

### 🛠️ **Development & Deployment**
Extend NeMo RL with custom components and deploy your models to production. Learn to add new models, create custom environments, implement testing strategies, and package your solutions for deployment.

## Training Algorithms

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="play" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> SFT Training
:link: algorithms/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> GRPO Training
:link: algorithms/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} <span class="octicon" data-icon="heart" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> DPO Training
:link: algorithms/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Evaluation
:link: algorithms/eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Examples & Tutorials

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="cloud" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Train GRPO Models on DeepScaleR
:link: examples/grpo-deepscaler
:link-type: doc

DeepScaleR integration for large-scale distributed training.

+++
{bdg-secondary}`Cloud`
:::

:::{grid-item-card} <span class="octicon" data-icon="calculator" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Fine-tune Models on OpenMathInstruct-2
:link: examples/sft-openmathinstruct2
:link-type: doc

Math instruction fine-tuning example with OpenMathInstruct-2 dataset.

+++
{bdg-primary}`Example`
:::

::::

## Performance & Scaling

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="server" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Distributed Training
:link: development/distributed-training
:link-type: doc

Scale RL training across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> NSYS Profiling
:link: development/nsys-profiling
:link-type: doc

Performance profiling with NSYS for training optimization.

+++
{bdg-secondary}`Performance`
:::

:::{grid-item-card} <span class="octicon" data-icon="alert" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Model Quirks
:link: development/model-quirks
:link-type: doc

Model-specific considerations and workarounds for special cases.

+++
{bdg-warning}`Advanced`
:::

::::

## Development & Deployment

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="plus" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Adding New Models
:link: development/adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} <span class="octicon" data-icon="game" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Custom Environments
:link: development/environment-development
:link-type: doc

Create custom RL environments for specialized training tasks.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} <span class="octicon" data-icon="checklist" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Testing & Debugging
:link: development/testing
:link-type: doc

Testing strategies and debugging techniques for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} <span class="octicon" data-icon="package" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Packaging
:link: development/packaging
:link-type: doc

Deployment and packaging strategies for production environments.

+++
{bdg-secondary}`Production`
:::

:::: 

```{toctree}
:caption: Guides
:maxdepth: 2
:expanded:

algorithms/index
examples/index
development/index
```

 