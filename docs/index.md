# NeMo RL Documentation

Welcome to the NeMo RL documentation! NeMo RL is a scalable, modular, and efficient post-training 
library for reinforcement learning and supervised fine-tuning of large language models.

## 📖 About
- [Overview & Philosophy](about/index.md)
- [Key Features](about/key-features.md)
- [Architecture](about/architecture.md)
- [Why NeMo RL](about/why-nemo-rl.md)

## 🚀 Get Started
- [Installation](get-started/installation.md)
- [Quickstart](get-started/quickstart.md)
- [Local Workstation](get-started/local-workstation.md)
- [Docker](get-started/docker.md)
- [Cluster Setup](get-started/cluster.md)

## 📚 NeMo RL Guides
- [SFT Training](guides/sft.md)
- [GRPO Training](guides/grpo.md)
- [DPO Training](guides/dpo.md)
- [Evaluation](guides/eval.md)
- [Adding New Models](guides/adding-new-models.md)
- [Testing](guides/testing.md)
- [Debugging](guides/debugging.md)
- [NSYS Profiling](guides/nsys-profiling.md)
- [Documentation](guides/documentation.md)
- [Packaging](guides/packaging.md)
- [Create Custom RL Environments](guides/environment-development.md)
- [Set Up Distributed Training with Ray](guides/distributed-training.md)
- [Model Quirks](guides/model-quirks.md)
- [GRPO DeepScaler](guides/grpo-deepscaler.md)
- [SFT OpenMathInstruct2](guides/sft-openmathinstruct2.md)

## 📋 Reference
- [API Reference](reference/api.md)
- [CLI Reference](reference/cli.md)
- [Configuration](reference/configuration.md)
- [Glossary](reference/glossary.md)
- [Troubleshooting](reference/troubleshooting.md)

---

_Navigate using the sidebar for more!_

```{include} ../README.md
:relative-docs: docs/
```

```{toctree}
:maxdepth: 3

about/index
get-started/index
guides/index
reference/index
design-docs/index
```