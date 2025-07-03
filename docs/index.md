# NeMo RL Documentation

Welcome to the NeMo RL documentation! NeMo RL is a scalable, modular, and efficient post-training library for reinforcement learning and supervised fine-tuning of large language models.

## 📖 About
- [Overview & Philosophy](about/index.md)
- [Key Features](about/key-features.md)
- [Architecture](about/architecture.md)

## 🚀 Get Started
- [Installation](get-started/installation.md)
- [Quickstart](get-started/quickstart.md)
- [Local Workstation](get-started/local-workstation.md)
- [Docker](get-started/docker.md)
- [Cluster Setup](get-started/cluster.md)

## 📚 Guides
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
- [Environment Development](guides/environment-development.md)
- [Distributed Training](guides/distributed-training.md)

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
:caption: 📖 About
:hidden:

about/index.md
about/key-features.md
about/architecture.md
about/concepts/index.md
about/release-notes/index.md

```

```{toctree}
:caption: 🚀 Get Started
:hidden:

get-started/index.md
get-started/installation.md
get-started/quickstart.md
get-started/local-workstation.md
get-started/docker.md
get-started/cluster.md

```

```{toctree}
:caption: 📚 Guides
:hidden:

guides/index.md
guides/sft.md
guides/grpo.md
guides/dpo.md
guides/eval.md
guides/adding-new-models.md
guides/testing.md
guides/debugging.md
guides/nsys-profiling.md
guides/documentation.md
guides/packaging.md
guides/model-quirks.md
guides/environment-development.md
guides/distributed-training.md
guides/grpo-deepscaler.md
guides/sft-openmathinstruct2.md

```

```{toctree}
:caption: 📋 Reference
:hidden:

reference/index.md
reference/api.md
reference/cli.md
reference/configuration.md
reference/glossary.md
reference/troubleshooting.md

```

```{toctree}
:caption: 📐 Design Docs
:hidden:

design-docs/design-and-philosophy.md
design-docs/padding.md
design-docs/logger.md
design-docs/uv.md
design-docs/chat-datasets.md
design-docs/generation.md
design-docs/checkpointing.md
design-docs/loss-functions.md
design-docs/fsdp2-parallel-plan.md
design-docs/training-backends.md

```

```{toctree}
:caption: 🎯 Feature Sets
:hidden:

feature-set-a/index.md
feature-set-a/category-a/index.md
feature-set-a/tutorials/index.md
feature-set-b/index.md
feature-set-b/category-a/index.md
feature-set-b/tutorials/index.md

```

```{toctree}
:caption: ⚙️ Admin
:hidden:

admin/index.md
admin/cicd/index.md
admin/deployment/index.md
admin/integrations/index.md

```
