---
description: "Training algorithms for NeMo RL including SFT, GRPO, DPO, and evaluation."
tags: ["algorithms", "training", "sft", "grpo", "dpo", "evaluation"]
categories: ["algorithms"]
---

# Training Algorithms

Learn about the core training algorithms supported by NeMo RL.

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` SFT Training
:link: sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` GRPO Training
:link: grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation
:link: eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

```{toctree}
:maxdepth: 1
:caption: Algorithms
:hidden:


sft
grpo
dpo
eval
``` 