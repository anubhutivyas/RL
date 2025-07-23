---
description: "Explore NeMo RL's design documents, architectural decisions, and technical specifications for understanding the framework's internals"
categories: ["concepts-architecture"]
tags: ["design", "architecture", "technical", "specifications", "internals", "core-architecture"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "advanced"
content_type: "concept"
modality: "universal"
---

(design-overview)=
# Core Design and Architecture

This section contains detailed design documents, architectural decisions, and technical specifications for NeMo RL. These documents provide insights into the framework's internals and design philosophy.

## What You'll Find Here

Our design documents are organized into four main categories to help you understand the framework's architecture and implementation:

### **Core Architecture** 
Explore the fundamental architectural decisions and design principles that shape NeMo RL. Understand the core abstractions, training backends, and parallelization strategies that enable scalable reinforcement learning.

### **Data Management**
Learn about data handling, processing, and persistence mechanisms. Discover how NeMo RL manages different data formats, implements efficient padding strategies, and maintains model state through checkpointing.

### **Computational Systems**
Dive into the mathematical foundations and computational components that power NeMo RL. Understand loss function implementations, text generation systems, and observability features.

### **Development Infrastructure**
Master the development tools and infrastructure that support NeMo RL development. Learn about package management, logging systems, and other development utilities.

## Core Architecture

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Design Philosophy
:link: core-architecture/design-and-philosophy
:link-type: doc

Understand the core design principles and philosophical approach behind NeMo RL's architecture.

+++
{bdg-primary}`Foundation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` FSDP2 Parallel Plan
:link: core-architecture/fsdp2-parallel-plan
:link-type: doc

Learn about the FSDP2 parallelization strategy and implementation details.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Generation
:link: core-architecture/generation
:link-type: doc

Explore the text generation pipeline and inference mechanisms.

+++
{bdg-info}`Computation`
:::

::::

## Data Management

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Chat Datasets
:link: data-management/chat-datasets
:link-type: doc

Explore the design of chat dataset processing and conversation handling.

+++
{bdg-info}`Data`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Padding
:link: data-management/padding
:link-type: doc

Learn about padding strategies and their impact on training efficiency.

+++
{bdg-secondary}`Optimization`
:::

::::

## Computational Systems

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Training Backends
:link: computational-systems/training-backends
:link-type: doc

Explore the different training backends supported by NeMo RL and their capabilities.

+++
{bdg-info}`Implementation`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Logger
:link: computational-systems/logger
:link-type: doc

Understand the logging system and observability features in NeMo RL.

+++
{bdg-secondary}`Monitoring`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` UV Package Manager
:link: computational-systems/uv
:link-type: doc

Learn about UV integration and package management strategies.

+++
{bdg-info}`Development`
:::

::::

## Development Infrastructure

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Loss Functions
:link: development-infrastructure/loss-functions
:link-type: doc

Understand the mathematical foundations and implementation of RL loss functions.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Checkpointing
:link: development-infrastructure/checkpointing
:link-type: doc

Understand the checkpointing mechanisms and strategies for model state persistence.

+++
{bdg-success}`Reliability`
:::

::::

```{toctree}
:caption: Core Design and Architecture
:maxdepth: 2
:hidden:

core-architecture/index
data-management/index
computational-systems/index
development-infrastructure/index
```

