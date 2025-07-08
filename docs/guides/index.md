# Guides

Welcome to the NeMo RL Guides! This comprehensive collection provides everything you need to master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our guides are organized into three main categories to help you navigate from basic concepts to practical implementations:

### **Training Algorithms** 
Learn the core RL algorithms that power NeMo RL. Start with Supervised Fine-Tuning (SFT) as your foundation, then explore advanced techniques like Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO). Each algorithm guide includes detailed explanations, mathematical foundations, and practical implementation examples.

### **Examples & Tutorials**
Jump into complete, working examples that demonstrate real-world applications. These tutorials walk you through entire training workflows, from data preparation to model deployment, using specific datasets and environments.

### **Development & Deployment**
Extend NeMo RL with custom components and deploy your models to production. Learn to add new models, create custom environments, implement testing strategies, and package your solutions for deployment.

## Training Algorithms

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Training
:link: algorithms/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: algorithms/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: algorithms/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation
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

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Train GRPO Models on DeepScaleR
:link: examples/grpo-deepscaler
:link-type: doc

DeepScaleR integration for large-scale distributed training.

+++
{bdg-secondary}`Cloud`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Fine-tune Models on OpenMathInstruct-2
:link: examples/sft-openmathinstruct2
:link-type: doc

Math instruction fine-tuning example with OpenMathInstruct-2 dataset.

+++
{bdg-primary}`Example`
:::

::::

## Development & Deployment

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Adding New Models
:link: development/adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Custom Environments
:link: development/environment-development
:link-type: doc

Create custom RL environments for specialized training tasks.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Testing & Debugging
:link: development/testing
:link-type: doc

Testing strategies and debugging techniques for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: development/packaging
:link-type: doc

Deployment and packaging strategies for production environments.

+++
{bdg-secondary}`Production`
:::

::::

```{toctree}
:maxdepth: 1
:caption: Training Algorithms

algorithms/sft
algorithms/grpo
algorithms/dpo
algorithms/eval
```

```{toctree}
:maxdepth: 1
:caption: Examples & Tutorials

examples/grpo-deepscaler
examples/sft-openmathinstruct2
```

```{toctree}
:maxdepth: 1
:caption: Development & Deployment

development/adding-new-models
development/environment-development
development/testing
development/packaging
```
