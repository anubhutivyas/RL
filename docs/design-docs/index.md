---
description: "Explore NeMo RL's design documents, architectural decisions, and technical specifications for understanding the framework's internals."
tags: ["design", "architecture", "technical", "specifications", "internals"]
categories: ["design"]
---

(design-overview)=
# Core Design & Architecture

This section contains detailed design documents, architectural decisions, and technical specifications for NeMo RL. These documents provide insights into the framework's internals and design philosophy.

## What You'll Find Here

Our design documents are organized into four main categories to help you understand the framework's architecture and implementation:

### 🏗️ **Core Architecture** 
Explore the fundamental architectural decisions and design principles that shape NeMo RL. Understand the core abstractions, training backends, and parallelization strategies that enable scalable reinforcement learning.

### 📊 **Data Management**
Learn about data handling, processing, and persistence mechanisms. Discover how NeMo RL manages different data formats, implements efficient padding strategies, and maintains model state through checkpointing.

### 🧮 **Computational Systems**
Dive into the mathematical foundations and computational components that power NeMo RL. Understand loss function implementations, text generation systems, and observability features.

### 🛠️ **Development Infrastructure**
Master the development tools and infrastructure that support NeMo RL development. Learn about package management, logging systems, and other development utilities.

## Core Architecture

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} <span class="octicon" data-icon="light-bulb" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Design Philosophy
:link: core-architecture/design-and-philosophy

Understand the core design principles and philosophical approach behind NeMo RL's architecture.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} <span class="octicon" data-icon="gear" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Training Backends
:link: computational-systems/training-backends

Explore the different training backends supported by NeMo RL and their capabilities.

+++
{bdg-info}`Implementation`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> FSDP2 Parallel Plan
:link: core-architecture/fsdp2-parallel-plan

Learn about the FSDP2 parallelization strategy and implementation details.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} <span class="octicon" data-icon="play" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Generation
:link: core-architecture/generation

Explore the text generation pipeline and inference mechanisms.

+++
{bdg-info}`Computation`
:::

::::

## Data Management

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} <span class="octicon" data-icon="database" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Chat Datasets
:link: data-management/chat-datasets

Explore the design of chat dataset processing and conversation handling.

+++
{bdg-info}`Data`
:::

:::{grid-item-card} <span class="octicon" data-icon="gear" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Padding
:link: data-management/padding

Learn about padding strategies and their impact on training efficiency.

+++
{bdg-secondary}`Optimization`
:::

::::

## Computational Systems

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} <span class="octicon" data-icon="gear" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Training Backends
:link: computational-systems/training-backends

Explore the different training backends supported by NeMo RL and their capabilities.

+++
{bdg-info}`Implementation`
:::

:::{grid-item-card} <span class="octicon" data-icon="graph" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Logger
:link: computational-systems/logger

Understand the logging system and observability features in NeMo RL.

+++
{bdg-secondary}`Monitoring`
:::

:::{grid-item-card} <span class="octicon" data-icon="package" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> UV Package Manager
:link: computational-systems/uv

Learn about UV integration and package management strategies.

+++
{bdg-info}`Development`
:::

::::

## Development Infrastructure

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} <span class="octicon" data-icon="function" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Loss Functions
:link: development-infrastructure/loss-functions

Understand the mathematical foundations and implementation of RL loss functions.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} <span class="octicon" data-icon="save" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Checkpointing
:link: development-infrastructure/checkpointing

Understand the checkpointing mechanisms and strategies for model state persistence.

+++
{bdg-success}`Reliability`
:::

::::


```{toctree}
:caption: Core Design & Architecture
:maxdepth: 2
:expanded:

core-architecture/design-and-philosophy
core-architecture/fsdp2-parallel-plan
core-architecture/generation
computational-systems/index
data-management/index
development-infrastructure/index
```

