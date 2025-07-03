---
description: "Explore NeMo RL's design documents, architectural decisions, and technical specifications for understanding the framework's internals."
tags: ["design", "architecture", "technical", "specifications", "internals"]
categories: ["design"]
---

(design-overview)=
# Design Documents

This section contains detailed design documents, architectural decisions, and technical specifications for NeMo RL. These documents provide insights into the framework's internals and design philosophy.

## Core Design

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Design Philosophy
:link: design-and-philosophy

Understand the core design principles and philosophical approach behind NeMo RL's architecture.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Training Backends
:link: training-backends

Explore the different training backends supported by NeMo RL and their capabilities.
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` FSDP2 Parallel Plan
:link: fsdp2-parallel-plan

Learn about the FSDP2 parallelization strategy and implementation details.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Checkpointing
:link: checkpointing

Understand the checkpointing mechanisms and strategies for model state persistence.
:::

::::

## Data & Processing

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`chat;1.5em;sd-mr-1` Chat Datasets
:link: chat-datasets

Explore the design of chat dataset processing and conversation handling.
:::

:::{grid-item-card} {octicon}`text-size;1.5em;sd-mr-1` Padding
:link: padding

Learn about padding strategies and their impact on training efficiency.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Loss Functions
:link: loss-functions

Understand the mathematical foundations and implementation of RL loss functions.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Generation
:link: generation

Explore the text generation pipeline and inference mechanisms.
:::

::::

## Development & Tooling

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`log;1.5em;sd-mr-1` Logger
:link: logger

Understand the logging system and observability features in NeMo RL.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` UV Package Manager
:link: uv

Learn about UV integration and package management strategies.
:::

::::

```{toctree}
:maxdepth: 2
:caption: Design Documents

design-and-philosophy
training-backends
fsdp2-parallel-plan
checkpointing
chat-datasets
padding
loss-functions
generation
logger
uv
``` 