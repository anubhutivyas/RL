---
description: "Comprehensive documentation for NeMo RL - a scalable, modular, and efficient post-training library for reinforcement learning and supervised fine-tuning of large language models"
categories: ["getting-started"]
tags: ["reinforcement-learning", "supervised-fine-tuning", "large-language-models", "documentation", "overview"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# NeMo RL Documentation

Welcome to the NeMo RL documentation! NeMo RL is a scalable, modular, and efficient post-training 
library for reinforcement learning and supervised fine-tuning of large language models.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: get-started/index
:link-type: doc
:class-body: text-center

Set up your environment and run your first reinforcement learning training job with large language models.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Tutorials and Examples
:link: tutorials-examples/index
:link-type: doc
:class-body: text-center

Learn NeMo RL with step-by-step tutorials, working examples, and real-world use cases for reinforcement learning with language models.

+++
{bdg-primary}`Learning` {bdg-secondary}`Hands-on`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
:link: guides/index
:link-type: doc
:class-body: text-center

Master NeMo RL with organized guides covering training algorithms, model development, infrastructure, and production workflows.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Reference`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Core Design and Architecture
:link: design-docs/index
:link-type: doc
:class-body: text-center

Explore NeMo RL's core architecture, data management, computational systems, and development infrastructure.

+++
{bdg-secondary}`Architecture` {bdg-secondary}`Design`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Advanced Topics
:link: advanced/index
:link-type: doc
:class-body: text-center

Deep technical content for researchers and advanced practitioners including mathematical foundations, research methodologies, and optimization techniques.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Research`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Documentation
:link: api-docs/index
:link-type: doc
:class-body: text-center

Complete API documentation including high-level guides and detailed auto-generated reference for all NeMo RL components.

+++
{bdg-primary}`Technical` {bdg-secondary}`API`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Configuration and CLI
:link: configuration-cli/index
:link-type: doc
:class-body: text-center

Access comprehensive reference documentation including configuration options and CLI tools.

+++
{bdg-info}`Technical` {bdg-secondary}`Reference`
:::

::::

## Learning Path

Follow this structured progression to master NeMo RL:

### **Beginner Path** (0-2 weeks)
1. **Installation and Setup** → [Installation Guide](get-started/installation)
2. **First Training Run** → [Quickstart Guide](get-started/quickstart)
3. **SFT Tutorial** → [Supervised Fine-Tuning Tutorial](tutorials-examples/tutorials/sft-tutorial)
4. **Basic Examples** → [SFT on OpenMathInstruct-2](tutorials-examples/examples/sft-openmathinstruct2)

### **Intermediate Path** (2-4 weeks)
1. **DPO Tutorial** → [Direct Preference Optimization Tutorial](tutorials-examples/tutorials/dpo-tutorial)
2. **Evaluation Tutorial** → [Model Evaluation Tutorial](tutorials-examples/tutorials/evaluation-tutorial)
3. **Advanced Examples** → [GRPO on DeepScaleR](tutorials-examples/examples/grpo-deepscaler)
4. **Use Cases** → [Code Generation](tutorials-examples/use-cases/code-generation) and [Mathematical Reasoning](tutorials-examples/use-cases/mathematical-reasoning)

### **Advanced Path** (4+ weeks)
1. **GRPO Tutorial** → [Group Relative Policy Optimization Tutorial](tutorials-examples/tutorials/grpo-tutorial)
2. **Advanced Performance** → [Performance and Optimization](advanced/performance/index)
3. **Distributed Training** → [Distributed Training Guide](advanced/performance/distributed-training)
4. **Production Deployment** → [Production and Support](guides/production-support/index)

## Key Resources

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: advanced/performance/distributed-training
:link-type: doc

Scale RL training across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Add New Models
:link: guides/model-development/adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Performance and Optimization
:link: advanced/performance/index
:link-type: doc

Optimize training performance, benchmarking, and profiling techniques.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Production and Support
:link: guides/production-support/index
:link-type: doc

Testing, debugging, packaging, and deployment strategies.

+++
{bdg-secondary}`Production`
:::

::::

---

```{include} ../README.md
:relative-docs: docs/
```

```{toctree}
:maxdepth: 3
:hidden:
about/index
get-started/index
tutorials-examples/index
guides/index
design-docs/index
advanced/index
api-docs/index
configuration-cli/index
```
