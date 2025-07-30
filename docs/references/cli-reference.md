# CLI Reference

This document provides comprehensive reference documentation for all NeMo RL command-line interface tools and utilities.

## Overview

NeMo RL provides a rich set of CLI tools for training, evaluation, and model management. All commands follow a consistent pattern and provide detailed help information.

## Command Structure

```bash
nemo-rl <command> [subcommand] [options]
```

## Main Commands

### Training Commands

#### `nemo-rl train`

Start a training job with the specified configuration.

```bash
nemo-rl train --config path/to/config.yaml [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--checkpoint`: Path to checkpoint for resuming training
- `--local`: Run in local mode (single node)
- `--cluster`: Run on Ray cluster
- `--debug`: Enable debug mode with verbose logging

**Examples:**
```bash
# Basic training
nemo-rl train --config examples/configs/dpo.yaml

# Resume from checkpoint
nemo-rl train --config examples/configs/dpo.yaml --checkpoint checkpoints/model_1000.pt

# Local development
nemo-rl train --config examples/configs/dpo.yaml --local --debug
```

#### `nemo-rl eval`

Evaluate a trained model.

```bash
nemo-rl eval --config path/to/eval_config.yaml [options]
```

**Options:**
- `--config`: Path to evaluation configuration file (required)
- `--model`: Path to model checkpoint
- `--output`: Output directory for results
- `--metrics`: Comma-separated list of metrics to compute

**Examples:**
```bash
# Basic evaluation
nemo-rl eval --config examples/configs/eval.yaml

# Custom metrics
nemo-rl eval --config examples/configs/eval.yaml --metrics accuracy,bleu,rouge
```

### Model Management

#### `nemo-rl convert`

Convert models between different formats.

```bash
nemo-rl convert --input path/to/model --output path/to/output --format [hf|megatron|vllm]
```

**Options:**
- `--input`: Input model path (required)
- `--output`: Output path (required)
- `--format`: Target format (required)
- `--config`: Conversion configuration file

**Examples:**
```bash
# Convert to HuggingFace format
nemo-rl convert --input checkpoints/model.pt --output hf_model --format hf

# Convert to VLLM format
nemo-rl convert --input checkpoints/model.pt --output vllm_model --format vllm
```

#### `nemo-rl export`

Export models for deployment.

```bash
nemo-rl export --model path/to/model --output path/to/output [options]
```

**Options:**
- `--model`: Model checkpoint path (required)
- `--output`: Output directory (required)
- `--format`: Export format (onnx, torchscript, etc.)
- `--quantize`: Enable quantization

### Utility Commands

#### `nemo-rl validate`

Validate configuration files.

```bash
nemo-rl validate --config path/to/config.yaml
```

**Options:**
- `--config`: Configuration file to validate (required)
- `--strict`: Enable strict validation

#### `nemo-rl info`

Display information about models and configurations.

```bash
nemo-rl info --model path/to/model
nemo-rl info --config path/to/config.yaml
```

**Options:**
- `--model`: Model path to inspect
- `--config`: Configuration file to inspect
- `--detailed`: Show detailed information

## Configuration Options

### Global Options

All commands support these global options:

- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Log to file instead of console
- `--quiet`: Suppress output
- `--version`: Show version information
- `--help`: Show help information

### Environment Variables

NeMo RL respects these environment variables:

- `NEMO_RL_LOG_LEVEL`: Default logging level
- `NEMO_RL_CONFIG_PATH`: Default configuration directory
- `NEMO_RL_CACHE_DIR`: Cache directory for models and data
- `RAY_ADDRESS`: Ray cluster address

## Examples

### Complete Training Workflow

```bash
# 1. Validate configuration
nemo-rl validate --config examples/configs/dpo.yaml

# 2. Start training
nemo-rl train --config examples/configs/dpo.yaml --cluster

# 3. Evaluate model
nemo-rl eval --config examples/configs/eval.yaml --model checkpoints/best.pt

# 4. Export for deployment
nemo-rl export --model checkpoints/best.pt --output deployed_model
```

### Development Workflow

```bash
# Local development with debug
nemo-rl train --config examples/configs/dpo.yaml --local --debug

# Quick evaluation
nemo-rl eval --config examples/configs/eval.yaml --model checkpoints/latest.pt

# Model inspection
nemo-rl info --model checkpoints/model.pt --detailed
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Validate configuration first
   nemo-rl validate --config your_config.yaml
   ```

2. **Memory Issues**
   ```bash
   # Use local mode for debugging
   nemo-rl train --config config.yaml --local
   ```

3. **Ray Connection Issues**
   ```bash
   # Check Ray status
   ray status
   
   # Start local Ray cluster
   ray start --head
   ```

### Getting Help

```bash
# General help
nemo-rl --help

# Command-specific help
nemo-rl train --help
nemo-rl eval --help

# Configuration help
nemo-rl validate --config config.yaml
```

## Advanced Usage

### Custom Scripts

You can integrate NeMo RL CLI into custom scripts:

```bash
#!/bin/bash
# Training script example

CONFIG="examples/configs/dpo.yaml"
CHECKPOINT_DIR="checkpoints"

# Validate config
nemo-rl validate --config $CONFIG || exit 1

# Start training
nemo-rl train --config $CONFIG --cluster

# Wait for completion and evaluate
if [ -f "$CHECKPOINT_DIR/best.pt" ]; then
    nemo-rl eval --config examples/configs/eval.yaml --model $CHECKPOINT_DIR/best.pt
fi
```

### Batch Processing

```bash
# Process multiple configurations
for config in configs/*.yaml; do
    echo "Processing $config"
    nemo-rl train --config "$config" --local
done
```

## Integration with Other Tools

### Ray Dashboard

When using Ray clusters, access the dashboard at `http://localhost:8265` to monitor training progress.

### TensorBoard

Training logs are automatically saved and can be viewed with TensorBoard:

```bash
tensorboard --logdir logs/
```

### MLflow

NeMo RL integrates with MLflow for experiment tracking:

```bash
# Enable MLflow logging in configuration
nemo-rl train --config config_with_mlflow.yaml
```

## Best Practices

1. **Always validate configurations** before training
2. **Use local mode** for development and debugging
3. **Monitor resources** when using clusters
4. **Save checkpoints regularly** for recovery
5. **Use descriptive configuration names** for organization
6. **Log all experiments** for reproducibility

## Reference

- [Configuration Reference](configuration-reference)
- [Troubleshoot NeMo RL](troubleshooting)
- [API Documentation](../api-docs/index) 