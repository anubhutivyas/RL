---
description: "Development tools, testing, debugging, and deployment guides for NeMo RL."
tags: ["development", "testing", "debugging", "profiling", "deployment"]
categories: ["development"]
---

# Development & Tools

Tools and guides for developing, testing, debugging, and deploying NeMo RL applications.

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Adding New Models
:link: adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`check;1.5em;sd-mr-1` Testing
:link: testing
:link-type: doc

Testing strategies and best practices for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging
:link: debugging
:link-type: doc

Debugging techniques and tools for distributed training environments.

+++
{bdg-warning}`Troubleshooting`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NSYS Profiling
:link: nsys-profiling
:link-type: doc

Performance profiling with NSYS for training optimization.

+++
{bdg-secondary}`Performance`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation
:link: documentation
:link-type: doc

Contributing to NeMo RL documentation and guides.

+++
{bdg-info}`Contributing`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: distributed-training
:link-type: doc

Scale RL training across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Custom Environments
:link: environment-development
:link-type: doc

Create custom RL environments for specialized training tasks.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: packaging
:link-type: doc

Deployment and packaging strategies for production environments.

+++
{bdg-secondary}`Production`
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Model Quirks
:link: model-quirks
:link-type: doc

Model-specific considerations and workarounds for special cases.

+++
{bdg-warning}`Advanced`
:::

::::

```{toctree}
:maxdepth: 1
:caption: Development
:expanded:

adding-new-models
testing
debugging
nsys-profiling
documentation
distributed-training
environment-development
packaging
model-quirks
``` 