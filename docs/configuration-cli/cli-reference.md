# CLI Reference

This document provides a comprehensive reference for all command-line interface (CLI) commands and options available in NeMo RL.

## Overview

NeMo RL provides a command-line interface for training, evaluation, and utility operations. The CLI is built on top of the configuration system and provides convenient access to all NeMo RL functionality.

## Basic Usage

```bash
# Basic command structure
python -m nemo_rl <command> [options] <config_file>

# Example: Run DPO training
python -m nemo_rl train configs/dpo.yaml

# Example: Run evaluation
python -m nemo_rl eval configs/eval.yaml
```

## Commands

### Training Commands

#### `train`

Run training with the specified algorithm and configuration.

```bash
python -m nemo_rl train <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--checkpoint`: Path to checkpoint to resume from
- `--output_dir`: Output directory for logs and checkpoints
- `--num_gpus`: Number of GPUs to use (overrides config)
- `--num_nodes`: Number of nodes to use (overrides config)
- `--local_rank`: Local rank for distributed training
- `--world_size`: World size for distributed training
- `--master_addr`: Master address for distributed training
- `--master_port`: Master port for distributed training
- `--debug`: Enable debug mode
- `--dry_run`: Validate configuration without running training

**Examples:**
```bash
# Basic DPO training
python -m nemo_rl train configs/dpo.yaml

# Resume from checkpoint
python -m nemo_rl train configs/dpo.yaml --checkpoint checkpoints/step_1000

# Override GPU count
python -m nemo_rl train configs/dpo.yaml --num_gpus 4

# Debug mode
python -m nemo_rl train configs/dpo.yaml --debug
```

#### `run_dpo`

Direct DPO training command with simplified interface.

```bash
python -m nemo_rl run_dpo <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--model`: Model name or path
- `--dataset`: Dataset name
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--max_steps`: Maximum training steps
- `--beta`: DPO beta parameter
- `--output_dir`: Output directory

**Examples:**
```bash
# Quick DPO training
python -m nemo_rl run_dpo configs/dpo.yaml

# Custom parameters
python -m nemo_rl run_dpo configs/dpo.yaml --batch_size 8 --learning_rate 2e-5
```

#### `run_grpo`

Direct GRPO training command.

```bash
python -m nemo_rl run_grpo <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--model`: Model name or path
- `--environment`: Environment type (math, sliding_puzzle)
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--max_steps`: Maximum training steps
- `--gamma`: Discount factor
- `--lambda_`: GAE parameter
- `--output_dir`: Output directory

**Examples:**
```bash
# GRPO math training
python -m nemo_rl run_grpo configs/grpo_math.yaml

# Custom environment
python -m nemo_rl run_grpo configs/grpo_sliding_puzzle.yaml
```

#### `run_sft`

Direct SFT training command.

```bash
python -m nemo_rl run_sft <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--model`: Model name or path
- `--dataset`: Dataset name
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--max_steps`: Maximum training steps
- `--output_dir`: Output directory

**Examples:**
```bash
# SFT training
python -m nemo_rl run_sft configs/sft.yaml

# Custom dataset
python -m nemo_rl run_sft configs/sft.yaml --dataset custom_dataset
```

### Evaluation Commands

#### `eval`

Run evaluation on a trained model.

```bash
python -m nemo_rl eval <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--checkpoint`: Path to model checkpoint
- `--dataset`: Evaluation dataset
- `--batch_size`: Evaluation batch size
- `--metrics`: Comma-separated list of metrics
- `--output_file`: Output file for predictions
- `--num_samples`: Number of samples to evaluate
- `--seed`: Random seed for reproducibility

**Examples:**
```bash
# Basic evaluation
python -m nemo_rl eval configs/eval.yaml --checkpoint checkpoints/best_model

# Custom metrics
python -m nemo_rl eval configs/eval.yaml --metrics accuracy,bleu,rouge

# Save predictions
python -m nemo_rl eval configs/eval.yaml --output_file predictions.json
```

#### `run_eval`

Direct evaluation command.

```bash
python -m nemo_rl run_eval <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--model`: Model name or path
- `--dataset`: Dataset name
- `--batch_size`: Batch size
- `--metrics`: Comma-separated metrics
- `--output_file`: Output file

**Examples:**
```bash
# Quick evaluation
python -m nemo_rl run_eval configs/eval.yaml

# Custom metrics
python -m nemo_rl run_eval configs/eval.yaml --metrics accuracy,bleu
```

### Utility Commands

#### `convert`

Convert models between different formats.

```bash
python -m nemo_rl convert <source> <target> [options]
```

**Options:**
- `--source`: Source model path or format
- `--target`: Target model path or format
- `--output_dir`: Output directory
- `--config`: Conversion configuration file

**Examples:**
```bash
# Convert Hugging Face to Megatron
python -m nemo_rl convert huggingface megatron --source model_path --target output_path

# Convert Megatron to Hugging Face
python -m nemo_rl convert megatron huggingface --source model_path --target output_path
```

#### `validate`

Validate configuration files.

```bash
python -m nemo_rl validate <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--strict`: Strict validation mode
- `--output`: Output format (json, yaml, text)

**Examples:**
```bash
# Validate configuration
python -m nemo_rl validate configs/dpo.yaml

# Strict validation
python -m nemo_rl validate configs/dpo.yaml --strict
```

#### `info`

Display information about models, datasets, or configurations.

```bash
python -m nemo_rl info <type> <name> [options]
```

**Types:**
- `model`: Model information
- `dataset`: Dataset information
- `config`: Configuration information

**Options:**
- `--type`: Information type (model, dataset, config)
- `--name`: Name or path to inspect
- `--format`: Output format (json, yaml, text)

**Examples:**
```bash
# Model information
python -m nemo_rl info model meta-llama/Llama-2-7b-hf

# Dataset information
python -m nemo_rl info dataset helpsteer3

# Configuration information
python -m nemo_rl info config configs/dpo.yaml
```

## Distributed Training Commands

### Ray-based Training

```bash
# Launch Ray cluster
ray start --head

# Submit training job
python -m nemo_rl train configs/dpo.yaml --distributed ray

# Stop Ray cluster
ray stop
```

### Multi-node Training

```bash
# Node 0 (master)
python -m nemo_rl train configs/dpo.yaml --num_nodes 2 --node_rank 0

# Node 1
python -m nemo_rl train configs/dpo.yaml --num_nodes 2 --node_rank 1
```

## Environment Variables

NeMo RL supports configuration through environment variables:

```bash
# Model configuration
export NEMO_RL_MODEL_BACKEND=huggingface
export NEMO_RL_MODEL_NAME=meta-llama/Llama-2-7b-hf

# Training configuration
export NEMO_RL_TRAINING_ALGORITHM=dpo
export NEMO_RL_BATCH_SIZE=4
export NEMO_RL_LEARNING_RATE=1e-5

# Distributed configuration
export NEMO_RL_DISTRIBUTED_STRATEGY=fsdp2
export NEMO_RL_NUM_GPUS=8
export NEMO_RL_NUM_NODES=1

# Logging configuration
export NEMO_RL_LOG_LEVEL=INFO
export NEMO_RL_TENSORBOARD=true
export NEMO_RL_WANDB=false
```

## Configuration Overrides

You can override configuration values directly from the command line:

```bash
# Override model settings
python -m nemo_rl train configs/dpo.yaml model.name=meta-llama/Llama-2-13b-hf

# Override training settings
python -m nemo_rl train configs/dpo.yaml training.batch_size=8 training.learning_rate=2e-5

# Override distributed settings
python -m nemo_rl train configs/dpo.yaml distributed.num_gpus=4 distributed.strategy=fsdp1

# Multiple overrides
python -m nemo_rl train configs/dpo.yaml model.name=meta-llama/Llama-2-13b-hf training.batch_size=8 distributed.num_gpus=4
```

## Debugging Commands

### `debug`

Debug configuration and setup.

```bash
python -m nemo_rl debug <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--check_model`: Check model loading
- `--check_data`: Check data loading
- `--check_distributed`: Check distributed setup
- `--verbose`: Verbose output

**Examples:**
```bash
# Debug configuration
python -m nemo_rl debug configs/dpo.yaml

# Check specific components
python -m nemo_rl debug configs/dpo.yaml --check_model --check_data
```

### `profile`

Profile training performance.

```bash
python -m nemo_rl profile <config_file> [options]
```

**Options:**
- `--config`: Path to configuration file (required)
- `--steps`: Number of steps to profile
- `--output`: Output file for profile data
- `--memory`: Profile memory usage
- `--gpu`: Profile GPU usage

**Examples:**
```bash
# Basic profiling
python -m nemo_rl profile configs/dpo.yaml --steps 100

# Memory profiling
python -m nemo_rl profile configs/dpo.yaml --memory --output profile.json
```

## Example Workflows

### Complete DPO Training Workflow

```bash
# 1. Validate configuration
python -m nemo_rl validate configs/dpo.yaml

# 2. Debug setup
python -m nemo_rl debug configs/dpo.yaml

# 3. Run training
python -m nemo_rl train configs/dpo.yaml

# 4. Evaluate model
python -m nemo_rl eval configs/eval.yaml --checkpoint checkpoints/best_model
```

### GRPO Training Workflow

```bash
# 1. Start Ray cluster
ray start --head

# 2. Run GRPO training
python -m nemo_rl run_grpo configs/grpo_math.yaml

# 3. Stop Ray cluster
ray stop
```

### Model Conversion Workflow

```bash
# 1. Convert Hugging Face to Megatron
python -m nemo_rl convert huggingface megatron --source model_path --target megatron_path

# 2. Train with Megatron backend
python -m nemo_rl train configs/dpo_megatron.yaml

# 3. Convert back to Hugging Face
python -m nemo_rl convert megatron huggingface --source megatron_path --target hf_path
```

## Error Handling

### Common Errors and Solutions

1. **Configuration Errors**
   ```bash
   # Validate configuration
   python -m nemo_rl validate configs/dpo.yaml
   ```

2. **Model Loading Errors**
   ```bash
   # Check model loading
   python -m nemo_rl debug configs/dpo.yaml --check_model
   ```

3. **Distributed Training Errors**
   ```bash
   # Check distributed setup
   python -m nemo_rl debug configs/dpo.yaml --check_distributed
   ```

4. **Memory Errors**
   ```bash
   # Reduce batch size
   python -m nemo_rl train configs/dpo.yaml training.batch_size=2
   
   # Use gradient accumulation
   python -m nemo_rl train configs/dpo.yaml training.gradient_accumulation_steps=8
   ```

## Best Practices

1. **Always validate configurations before training**
   ```bash
   python -m nemo_rl validate configs/dpo.yaml
   ```

2. **Use debug mode for troubleshooting**
   ```bash
   python -m nemo_rl train configs/dpo.yaml --debug
   ```

3. **Profile performance for optimization**
   ```bash
   python -m nemo_rl profile configs/dpo.yaml --steps 100
   ```

4. **Use environment variables for sensitive settings**
   ```bash
   export NEMO_RL_WANDB_API_KEY=your_key
   ```

5. **Keep configuration files in version control**
   ```bash
   git add configs/
   git commit -m "Add training configurations"
   ```

## Command Reference Summary

| Command | Description | Usage |
|---------|-------------|-------|
| `train` | Run training | `python -m nemo_rl train <config>` |
| `run_dpo` | Run DPO training | `python -m nemo_rl run_dpo <config>` |
| `run_grpo` | Run GRPO training | `python -m nemo_rl run_grpo <config>` |
| `run_sft` | Run SFT training | `python -m nemo_rl run_sft <config>` |
| `eval` | Run evaluation | `python -m nemo_rl eval <config>` |
| `run_eval` | Run evaluation | `python -m nemo_rl run_eval <config>` |
| `convert` | Convert models | `python -m nemo_rl convert <source> <target>` |
| `validate` | Validate config | `python -m nemo_rl validate <config>` |
| `info` | Show information | `python -m nemo_rl info <type> <name>` |
| `debug` | Debug setup | `python -m nemo_rl debug <config>` |
| `profile` | Profile performance | `python -m nemo_rl profile <config>` | 