# Command Line Interface Reference

This document provides a comprehensive reference for the NeMo RL command-line interface (CLI).

## Overview

The NeMo RL CLI provides a unified interface for running training, evaluation, and other operations. It supports both YAML configuration files and command-line arguments.

## Basic Usage

```bash
# Run training with a configuration file
python -m nemo_rl.train --config configs/dpo.yaml

# Run evaluation
python -m nemo_rl.eval --config configs/eval.yaml

# Run with command-line overrides
python -m nemo_rl.train --config configs/dpo.yaml --training.batch_size 8
```

## Command Structure

### Training Command

```bash
python -m nemo_rl.train [OPTIONS]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--output_dir`: Output directory for logs and checkpoints
- `--resume`: Resume from checkpoint
- `--debug`: Enable debug mode
- `--dry_run`: Validate configuration without running

**Examples:**
```bash
# Basic training
python -m nemo_rl.train --config configs/dpo.yaml

# Resume training
python -m nemo_rl.train --config configs/dpo.yaml --resume checkpoints/step_1000

# Override configuration
python -m nemo_rl.train --config configs/dpo.yaml --training.batch_size 8 --training.learning_rate 1e-4
```

### Evaluation Command

```bash
python -m nemo_rl.eval [OPTIONS]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--checkpoint`: Path to model checkpoint
- `--output_file`: Output file for results
- `--metrics`: Comma-separated list of metrics to compute

**Examples:**
```bash
# Evaluate model
python -m nemo_rl.eval --config configs/eval.yaml --checkpoint checkpoints/best_model

# Custom metrics
python -m nemo_rl.eval --config configs/eval.yaml --metrics accuracy,bleu,rouge
```

### Configuration Command

```bash
python -m nemo_rl.config [OPTIONS]
```

**Options:**
- `--validate`: Validate configuration file
- `--print`: Print configuration
- `--diff`: Show differences between configs

**Examples:**
```bash
# Validate configuration
python -m nemo_rl.config --validate configs/dpo.yaml

# Print configuration
python -m nemo_rl.config --print configs/dpo.yaml

# Compare configurations
python -m nemo_rl.config --diff configs/dpo.yaml configs/grpo.yaml
```

## Configuration Overrides

You can override any configuration value using dot notation:

```bash
# Override nested configuration
python -m nemo_rl.train --config configs/dpo.yaml \
  --training.batch_size 8 \
  --training.learning_rate 1e-4 \
  --model.max_length 1024 \
  --data.train_file "path/to/data.json"
```

### Supported Override Types

- **Strings**: `--model.name "llama2-7b"`
- **Numbers**: `--training.batch_size 8`
- **Booleans**: `--training.gradient_checkpointing true`
- **Lists**: `--data.metrics ["accuracy", "f1"]`
- **Dictionaries**: `--training.optimizer '{"name": "adamw", "lr": 1e-4}'`

## Environment Variables

The CLI respects several environment variables:

```bash
# Set log level
export NEMO_RL_LOG_LEVEL=DEBUG

# Set cache directory
export NEMO_RL_CACHE_DIR=/path/to/cache

# Set wandb project
export NEMO_RL_WANDB_PROJECT=my_project

# Disable wandb
export NEMO_RL_DISABLE_WANDB=1
```

## Distributed Training

### Ray Cluster

```bash
# Start Ray cluster
ray start --head --port=6379

# Run distributed training
python -m nemo_rl.train --config configs/dpo.yaml --cluster.name ray
```

### FSDP Training

```bash
# Run FSDP training
python -m nemo_rl.train --config configs/dpo.yaml --cluster.name fsdp
```

## Logging and Monitoring

### Log Levels

```bash
# Set log level
python -m nemo_rl.train --config configs/dpo.yaml --log_level DEBUG
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

### TensorBoard

```bash
# Enable TensorBoard logging
python -m nemo_rl.train --config configs/dpo.yaml --logging.tensorboard true

# View logs
tensorboard --logdir logs/
```

### Weights & Biases

```bash
# Enable W&B logging
python -m nemo_rl.train --config configs/dpo.yaml --logging.wandb true

# Set project and entity
python -m nemo_rl.train --config configs/dpo.yaml \
  --logging.wandb true \
  --logging.wandb_project my_project \
  --logging.wandb_entity my_entity
```

## Checkpointing

### Save Checkpoints

```bash
# Save checkpoints every 1000 steps
python -m nemo_rl.train --config configs/dpo.yaml \
  --training.checkpointing.save_steps 1000 \
  --training.checkpointing.save_total_limit 3
```

### Resume Training

```bash
# Resume from checkpoint
python -m nemo_rl.train --config configs/dpo.yaml --resume checkpoints/step_1000

# Resume from latest checkpoint
python -m nemo_rl.train --config configs/dpo.yaml --resume latest
```

## Debugging

### Dry Run

```bash
# Validate configuration without running
python -m nemo_rl.train --config configs/dpo.yaml --dry_run
```

### Debug Mode

```bash
# Enable debug mode
python -m nemo_rl.train --config configs/dpo.yaml --debug
```

### Verbose Logging

```bash
# Enable verbose logging
python -m nemo_rl.train --config configs/dpo.yaml --log_level DEBUG
```

## Examples

### DPO Training

```bash
# Basic DPO training
python -m nemo_rl.train --config configs/dpo.yaml

# DPO with custom parameters
python -m nemo_rl.train --config configs/dpo.yaml \
  --algorithm.beta 0.2 \
  --training.batch_size 8 \
  --training.learning_rate 1e-4
```

### GRPO Training

```bash
# Basic GRPO training
python -m nemo_rl.train --config configs/grpo.yaml

# GRPO with custom parameters
python -m nemo_rl.train --config configs/grpo.yaml \
  --algorithm.gamma 0.99 \
  --algorithm.lambda_ 0.95 \
  --training.batch_size 16
```

### Evaluation

```bash
# Evaluate model
python -m nemo_rl.eval --config configs/eval.yaml --checkpoint checkpoints/best_model

# Evaluate with custom metrics
python -m nemo_rl.eval --config configs/eval.yaml \
  --checkpoint checkpoints/best_model \
  --evaluation.metrics accuracy,bleu,rouge
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Use `--dry_run` to validate configuration
2. **Memory Issues**: Reduce batch size or enable gradient checkpointing
3. **Distributed Issues**: Check Ray cluster status and resource allocation
4. **Checkpoint Issues**: Verify checkpoint paths and permissions

### Getting Help

```bash
# Show help
python -m nemo_rl.train --help

# Show configuration help
python -m nemo_rl.config --help
```

For more detailed troubleshooting, see the [Troubleshooting Guide](../guides/production-support/troubleshooting.md). 