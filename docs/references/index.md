---
description: "Comprehensive guide to NeMo RL configuration options, CLI commands, and usage patterns for training, evaluation, and deployment"
categories: ["reference"]
tags: ["configuration", "cli", "reference", "yaml", "command-line", "python-api"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

# About NeMo RL References

Welcome to the NeMo RL References guide! This section covers all configuration options, CLI commands, and usage patterns for NeMo RL.

## Reference Areas

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Reference
:link: configuration-reference
:link-type: doc

Complete parameter documentation and configuration options for all NeMo RL components.

+++
{bdg-info}`Reference`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: cli-reference
:link-type: doc

Command-line interface documentation with usage examples and parameter guides.

+++
{bdg-secondary}`CLI`
:::

::::

## Configuration System

NeMo RL uses a hierarchical configuration system that supports:

- **YAML Configuration Files**: Structured configuration with validation
- **Environment Variables**: Override configuration via environment
- **Command Line Arguments**: Direct parameter specification
- **Default Values**: Sensible defaults for all parameters

### Configuration Hierarchy

1. **Default Values**: Built-in defaults for all parameters
2. **Configuration File**: YAML file specified with `--config`
3. **Environment Variables**: Override specific values
4. **Command Line**: Final override for any parameter

### Example Configuration

```yaml
# training.yaml
algorithm:
  name: "dpo"
  learning_rate: 1e-5
  batch_size: 4

model:
  name: "llama2-7b"
  backend: "huggingface"
  max_length: 2048

data:
  dataset: "your-dataset"
  train_split: 0.9
  validation_split: 0.1

distributed:
  num_workers: 4
  backend: "ray"
```

## CLI Commands

### Basic Usage

```bash
# Train with configuration file
python -m nemo_rl.train --config training.yaml

# Override configuration parameters
python -m nemo_rl.train --config training.yaml --algorithm.learning_rate 2e-5

# Use environment variables
export NEMO_RL_ALGORITHM_LEARNING_RATE=2e-5
python -m nemo_rl.train --config training.yaml
```

### Common Commands

- **Training**: `nemo_rl.train` - Start training with specified algorithm
- **Evaluation**: `nemo_rl.eval` - Evaluate trained models
- **Inference**: `nemo_rl.inference` - Run inference on trained models
- **Export**: `nemo_rl.export` - Export models for deployment

## Key Concepts

### Configuration Parameters

- **Algorithm Parameters**: Learning rates, batch sizes, training steps
- **Model Parameters**: Model architecture, backend, generation settings
- **Data Parameters**: Dataset paths, preprocessing options
- **Distributed Parameters**: Worker configuration, cluster settings

### CLI Patterns

- **Subcommands**: Different operations (train, eval, inference)
- **Global Options**: Apply to all subcommands
- **Subcommand Options**: Specific to each operation
- **Configuration Override**: Mix config files and CLI arguments

## Getting Help

```bash
# General help
python -m nemo_rl --help

# Command-specific help
python -m nemo_rl.train --help

# Configuration help
python -m nemo_rl.config --help
```

## Next Steps

- Read the [Configuration Reference](configuration-reference) for detailed parameter documentation
- Explore the [CLI Reference](cli-reference) for command usage patterns
- Check the [Troubleshooting](../guides/troubleshooting) guide for common issues
- Review the [Guides](../guides/index) for practical examples

---