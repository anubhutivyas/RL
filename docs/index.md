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

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Guides
:link: guides/index
:link-type: doc
:class-body: text-center

Master NeMo RL with organized guides covering training algorithms, examples, model development, infrastructure, and production workflows.

+++
{bdg-warning}`Advanced` {bdg-secondary}`Tutorials`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Core Design & Architecture
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

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Configuration & CLI
:link: configuration-cli/index
:link-type: doc
:class-body: text-center

Access comprehensive reference documentation including configuration options and CLI tools.

+++
{bdg-info}`Technical` {bdg-secondary}`Reference`
:::

::::

## Popular Guides

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Training
:link: guides/training-algorithms/sft
:link-type: doc

Supervised Fine-Tuning for language models - the foundation of RL training.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Training
:link: guides/training-algorithms/grpo
:link-type: doc

Group Relative Policy Optimization for advanced reinforcement learning training.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Training
:link: guides/training-algorithms/dpo
:link-type: doc

Direct Preference Optimization for preference learning and model alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Evaluation
:link: guides/training-algorithms/eval
:link-type: doc

Model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-secondary}`Analysis`
:::

::::

## Training Examples

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Train GRPO Models on DeepScaleR
:link: guides/examples-tutorials/grpo-deepscaler
:link-type: doc

DeepScaleR integration for large-scale distributed training.

+++
{bdg-secondary}`Cloud`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Fine-tune Models on OpenMathInstruct-2
:link: guides/examples-tutorials/sft-openmathinstruct2
:link-type: doc

Math instruction fine-tuning example with OpenMathInstruct-2 dataset.

+++
{bdg-primary}`Example`
:::

::::

## Performance & Development

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: guides/environment-data/distributed-training
:link-type: doc

Scale RL training across multiple GPUs and nodes with Ray clusters.

+++
{bdg-warning}`Advanced`
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Adding New Models
:link: guides/model-development/adding-new-models
:link-type: doc

Extend NeMo RL with custom model architectures and backends.

+++
{bdg-info}`Development`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Testing & Debugging
:link: guides/production-support/testing
:link-type: doc

Testing strategies and debugging techniques for RL training pipelines.

+++
{bdg-success}`Quality`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: guides/production-support/packaging
:link-type: doc

Deployment and packaging strategies for production environments.

+++
{bdg-secondary}`Production`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Performance & Optimization
:link: advanced/performance/index
:link-type: doc

Optimize training performance, benchmarking, and profiling techniques.

+++
{bdg-warning}`Advanced`
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
design-docs/index
advanced/index
api-docs/index
configuration-cli/index
```
