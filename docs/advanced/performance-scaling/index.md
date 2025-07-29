---
description: "Scale NeMo RL from single GPU to production clusters with memory optimization and distributed training"
tags: ["performance", "scaling", "distributed", "memory", "optimization"]
categories: ["performance-scaling"]
---

# Performance & Scaling

This section covers how to scale NeMo RL from single GPU to production clusters. Learn to optimize memory usage, implement distributed training, and achieve maximum performance for large-scale models.

## What You'll Find Here

- **Distributed Training**: Scale from single GPU to multi-node clusters
- **Memory Optimization**: Optimize memory usage for large models

## Performance & Scaling

::::{grid} 1 1
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Distributed Training
:link: distributed-training
:link-type: doc

Scale from single GPU to multi-node clusters. Implement efficient distributed training strategies for large-scale models.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`memory;1.5em;sd-mr-1` Memory Optimization
:link: memory-optimization
:link-type: doc

Optimize memory usage for large models. Implement gradient checkpointing, mixed precision, and memory-efficient techniques.

+++
{bdg-warning}`Advanced`
:::

::::

---

::::{toctree}
:hidden:
:caption: Performance & Scaling
:maxdepth: 2
distributed-training
memory-optimization
::::