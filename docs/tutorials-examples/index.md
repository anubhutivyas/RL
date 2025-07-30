---
description: "Comprehensive collection of tutorials and examples for learning and mastering reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["tutorials", "examples", "sft", "dpo", "grpo", "reinforcement-learning", "training-execution"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# About NeMo RL Tutorials and Examples

Welcome to the NeMo RL Tutorials and Examples! This comprehensive collection provides everything you need to learn and master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our tutorials and examples are organized into three main categories to help you progress from basic concepts to real-world applications:

### **Tutorials** 
Step-by-step learning guides that teach you NeMo RL concepts and techniques. Start with beginner tutorials to build your foundation, then progress to advanced topics like distributed training and performance optimization.

### **Examples**
Complete, working examples that demonstrate real-world applications. These examples walk you through entire training workflows using specific datasets and configurations, providing hands-on experience with NeMo RL.

### **Use Cases**
Real-world applications and production patterns for NeMo RL. Learn how to apply reinforcement learning to solve practical problems across different domains including code generation, mathematical reasoning, and conversational AI.

## Learning Path

Follow this structured learning path to master NeMo RL:

### **Beginner Path** (0-2 weeks)
1. **Installation and Setup** → [Installation Guide](../get-started/installation)
2. **First Training Run** → [Quickstart Guide](../get-started/quickstart)
3. **SFT Tutorial** → [Supervised Fine-Tuning Tutorial](tutorials/sft-tutorial)
4. **Basic Examples** → [SFT on OpenMathInstruct-2](examples/sft-openmathinstruct2)

### **Intermediate Path** (2-4 weeks)
1. **DPO Tutorial** → [Direct Preference Optimization Tutorial](tutorials/dpo-tutorial)
2. **Evaluation Tutorial** → [Model Evaluation Tutorial](tutorials/evaluation-tutorial)
3. **Advanced Examples** → [GRPO on DeepScaleR](examples/grpo-deepscaler)
4. **Use Cases** → [Code Generation](use-cases/code-generation) and [Mathematical Reasoning](use-cases/mathematical-reasoning)

### **Advanced Path** (4+ weeks)
1. **GRPO Tutorial** → [Group Relative Policy Optimization Tutorial](tutorials/grpo-tutorial)
2. **Advanced Performance** → [Performance and Optimization](../advanced/performance/index)
3. **Distributed Training** → [Distributed Training Guide](../advanced/performance/distributed-training)
4. **Production Deployment** → [Production and Support](../guides/production-support/index)

## Tutorials

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Tutorial
:link: tutorials/sft-tutorial
:link-type: doc

Learn supervised fine-tuning fundamentals with step-by-step guidance.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Tutorial
:link: tutorials/dpo-tutorial
:link-type: doc

Master Direct Preference Optimization for preference learning and alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Tutorial
:link: tutorials/grpo-tutorial
:link-type: doc

Advanced reinforcement learning with Group Relative Policy Optimization.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation Tutorial
:link: tutorials/evaluation-tutorial
:link-type: doc

Learn model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-info}`Intermediate`
:::

::::

## Examples

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` SFT on OpenMathInstruct-2
:link: examples/sft-openmathinstruct2
:link-type: doc

Complete example of supervised fine-tuning on math instruction dataset.

+++
{bdg-primary}`Example`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` GRPO on DeepScaleR
:link: examples/grpo-deepscaler
:link-type: doc

Large-scale distributed training example with DeepScaleR integration.

+++
{bdg-secondary}`Cloud`
:::

::::

## Use Cases

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: use-cases/code-generation
:link-type: doc

Train models to generate, debug, and optimize code across multiple programming languages.

+++
{bdg-primary}`Development`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Mathematical Reasoning
:link: use-cases/mathematical-reasoning
:link-type: doc

Build models that can solve complex mathematical problems with step-by-step reasoning.

+++
{bdg-warning}`Advanced`
:::

::::





## Next Steps

After completing the tutorials:

- **Explore Advanced Features**: Dive into [distributed computing](../advanced/performance/distributed-training) and [performance optimization](../advanced/performance/memory-optimization)
- **Build Your Workflows**: Create custom experiment pipelines with [algorithm customization](../advanced/algorithm-customization/index)
- **Optimize Performance**: Learn best practices for [production deployment](../advanced/production-deployment/index)
- **Research and Validation**: Design rigorous experiments with [research methodologies](../advanced/research-validation/index)
- **Contribute**: Share your experiences and contribute to the community

For additional learning resources and community support, visit the NeMo RL GitHub repository and documentation.

```{note}
The tutorial files referenced in this guide are available in the NeMo RL examples repository. Clone the repository to access the complete tutorial notebooks and scripts.
```

Start with the learning path that matches your experience level, and gradually work your way through the tutorial series. The advanced tutorials will help you master complex scenarios and production deployments.
