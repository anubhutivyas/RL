---
description: "Explore comprehensive documentation for NeMo RL, including tutorials, guides, and deployment instructions for reinforcement learning with large language models."
tags: ["overview", "quickstart", "getting-started", "reinforcement-learning", "large-language-models"]
categories: ["getting-started"]
---

# NeMo RL Documentation

Welcome to the NeMo RL documentation. NeMo RL is an open-source, comprehensive framework for reinforcement learning and supervised fine-tuning of large language models.

## What is NeMo RL?

NeMo RL is a production-ready framework that combines the power of reinforcement learning with large language models. It provides a unified platform for training, fine-tuning, and deploying language models using state-of-the-art RL algorithms like DPO, GRPO, and SFT.

## Quick Start

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: get-started/quickstart
:link-type: doc

Get up and running with your first RL training job in minutes.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Installation
:link: get-started/installation
:link-type: doc

Complete setup instructions for all environments and platforms.

+++
{bdg-success}`Essential`
:::

::::

## Core Learning Path

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo RL
:link: about/index
:link-type: doc

Learn about NeMo RL's core concepts, key features, and fundamental architecture.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Training Algorithms
:link: guides/training-algorithms/index
:link-type: doc

Master SFT, DPO, and GRPO training algorithms with comprehensive guides.

+++
{bdg-info}`Training`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Tutorials & Examples
:link: tutorials-examples/index
:link-type: doc

Step-by-step tutorials and working examples for hands-on learning.

+++
{bdg-secondary}`Learning`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Advanced Topics
:link: advanced/index
:link-type: doc

Extend and customize NeMo RL for production deployment and research.

+++
{bdg-warning}`Advanced`
:::

::::

## Key Features

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: about/architecture-overview
:link-type: doc

Scale training across multiple GPUs and nodes with Ray-based distributed computing.

+++
{bdg-info}`Scalability`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Multiple Backends
:link: core-design/design-principles/index
:link-type: doc

Support for Hugging Face Transformers, Megatron-LM, and custom model backends.

+++
{bdg-secondary}`Flexibility`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` RL Algorithms
:link: guides/training-algorithms/index
:link-type: doc

State-of-the-art algorithms: DPO, GRPO, SFT with custom loss functions.

+++
{bdg-primary}`Algorithms`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Environments
:link: guides/environment-data/index
:link-type: doc

Built-in environments for math problems, games, and custom environment development.

+++
{bdg-info}`Environments`
:::

::::

## Reference Documentation

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Reference
:link: references/configuration-reference
:link-type: doc

Complete reference for all configuration options and parameters.

+++
{bdg-primary}`Reference`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: references/cli-reference
:link-type: doc

Command-line interface commands and usage patterns.

+++
{bdg-info}`CLI`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Documentation
:link: api-docs/index
:link-type: doc

Comprehensive API documentation for all NeMo RL components.

+++
{bdg-warning}`Development`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Core Design & Architecture
:link: core-design/index
:link-type: doc

Architectural decisions and technical specifications for framework internals.

+++
{bdg-warning}`Advanced`
:::

::::

## Getting Help

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: guides/troubleshooting
:link-type: doc

Common issues, error messages, and solutions.

+++
{bdg-warning}`Support`
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` Production Support
:link: guides/production-support/index
:link-type: doc

Deployment guides, monitoring, and production best practices.

+++
{bdg-info}`Production`
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About 
:maxdepth: 2
about/index
about/key-features
about/architecture-overview
:::: 

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2
get-started/index
get-started/quickstart
get-started/installation
get-started/docker
get-started/cluster
:::: 

::::{toctree}
:hidden:
:caption: Guides
:maxdepth: 2
guides/index
guides/training-algorithms/index
guides/model-development/index
guides/environment-data/index
guides/production-support/index
:::: 

::::{toctree}
:hidden:
:caption: Tutorials & Examples
:maxdepth: 3
tutorials-examples/index
tutorials-examples/tutorials/index
tutorials-examples/examples/index
tutorials-examples/use-cases/index
:::: 

::::{toctree}
:hidden:
:caption: Core Design & Architecture
:maxdepth: 3
core-design/index
core-design/design-principles/index
core-design/data-management/index
core-design/computational-systems/index
core-design/development-infrastructure/index
:::: 

::::{toctree}
:hidden:
:caption: Advanced Topics
:maxdepth: 3
advanced/index
advanced/algorithm-customization/index
advanced/performance-scaling/index
advanced/research-validation/index
advanced/production-deployment/index
:::: 

::::{toctree}
:hidden:
:caption: API Documentation
:maxdepth: 2
api-docs/index
:::: 

::::{toctree}
:hidden:
:caption: References
:maxdepth: 2
references/index
references/configuration-reference
references/cli-reference
::::
