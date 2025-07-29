---
description: "Customize and extend NeMo RL algorithms for specific use cases and domains"
tags: ["algorithms", "customization", "dpo", "grpo", "loss-functions"]
categories: ["algorithm-customization"]
---

# Algorithm Customization

This section covers how to customize and extend NeMo RL algorithms for your specific use cases and domains. Learn to implement custom DPO and GRPO variants, design novel loss functions, and adapt algorithms for new applications.

## What You'll Find Here

- **Custom DPO Implementation**: Extend DPO for specific use cases and domains
- **Custom GRPO Implementation**: Adapt GRPO for new domains and use cases  
- **Custom Loss Functions**: Design and implement novel training objectives

## Algorithm Customization

::::{grid} 1 1 1
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Custom DPO Implementation
:link: custom-dpo
:link-type: doc

Extend DPO for specific use cases and domains. Implement custom DPO variants and adapt the algorithm for new applications.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Custom GRPO Implementation
:link: custom-grpo
:link-type: doc

Adapt GRPO for new domains and use cases. Implement custom GRPO variants and extend the algorithm for specific requirements.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Loss Functions
:link: custom-loss-functions
:link-type: doc

Design and implement novel training objectives. Create custom loss functions for specific domains and multi-objective training.

+++
{bdg-warning}`Advanced`
:::

::::

---

::::{toctree}
:hidden:
:caption: Algorithm Customization
:maxdepth: 2
custom-dpo
custom-grpo
custom-loss-functions
::::