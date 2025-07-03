# NeMo RL Documentation

Welcome to the NeMo RL documentation! NeMo RL is a scalable, modular, and efficient post-training 
library for reinforcement learning and supervised fine-tuning of large language models.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` **About**
:link: about/index
:link-type: doc
:class-body: text-center

Learn about NeMo RL's core concepts, key features, and fundamental architecture for reinforcement learning with large language models.

+++
{bdg-primary}`Concepts` {bdg-secondary}`Overview`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` **Get Started**
:link: get-started/index
:link-type: doc
:class-body: text-center

Set up your environment and run your first reinforcement learning training job with large language models.

+++
{bdg-success}`Beginner` {bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` **Guides**
:link: guides/index
:link-type: doc
:class-body: text-center

Explore detailed guides for NeMo RL algorithms, training, evaluation, and development workflows.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Tutorials`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` **Core Design & Architecture**
:link: design-docs/index
:link-type: doc
:class-body: text-center

Explore NeMo RL's core design principles and architectural foundations for understanding the framework's internals.

+++
{bdg-secondary}`Architecture` {bdg-secondary}`Design`
:::

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` **Reference**
:link: reference/index
:link-type: doc
:class-body: text-center

Access comprehensive reference documentation including API specifications, configuration options, and technical details.

+++
{bdg-info}`Technical` {bdg-secondary}`API`
:::

::::

## Popular Guides

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` SFT Training
:link: guides/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: guides/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: guides/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`chart-line;1.5em;sd-mr-1` Evaluation
:link: guides/eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Getting Started Fast

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: get-started/installation
:link-type: doc

Complete setup instructions for all environments and platforms.

+++
{bdg-success}`Essential`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Quickstart
:link: get-started/quickstart
:link-type: doc

Run your first RL training job in minutes with our step-by-step guide.

+++
{bdg-primary}`Beginner`
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
guides/index
reference/index
design-docs/index
```