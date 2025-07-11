# Configuration Reference

This document provides a comprehensive reference for all configuration options available in NeMo RL.

## Overview

NeMo RL uses a hierarchical configuration system that allows you to customize every aspect of your training setup. Configurations are defined using YAML files and can be extended through command-line arguments.

## Configuration Structure

### Master Configuration

The main configuration object is `MasterConfig`, which contains all the settings for your training run:

```python
from nemo_rl.utils.config import MasterConfig

config = MasterConfig(
    # Training algorithm configuration
    algorithm=AlgorithmConfig(...),
    
    # Model configuration
    model=ModelConfig(...),
    
    # Data configuration
    data=DataConfig(...),
    
    # Cluster configuration
    cluster=ClusterConfig(...),
    
    # Training configuration
    training=TrainingConfig(...),
    
    # Evaluation configuration
    evaluation=EvaluationConfig(...),
)
```

### Algorithm Configuration

#### DPO Configuration

```yaml
algorithm:
  name: "dpo"
  beta: 0.1
  loss_type: "sigmoid"  # or "hinge"
  reference_free: false
  scale_by_temperature: true
  temperature: 1.0
```

#### GRPO Configuration

```yaml
algorithm:
  name: "grpo"
  gamma: 0.99
  lambda_: 0.95
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
```

#### SFT Configuration

```yaml
algorithm:
  name: "sft"
  loss_type: "cross_entropy"
  label_smoothing: 0.0
```

### Model Configuration

#### HuggingFace Models

```yaml
model:
  name: "huggingface"
  model_name: "microsoft/DialoGPT-medium"
  tokenizer_name: "microsoft/DialoGPT-medium"
  max_length: 512
  trust_remote_code: false
```

#### Megatron Models

```yaml
model:
  name: "megatron"
  model_name: "llama2-7b"
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  sequence_parallel: false
  use_flash_attention: true
```

### Data Configuration

#### Dataset Configuration

```yaml
data:
  train_file: "path/to/train.json"
  validation_file: "path/to/validation.json"
  test_file: "path/to/test.json"
  max_length: 512
  truncation: true
  padding: "max_length"
```

#### Data Processing

```yaml
data:
  preprocessing:
    remove_duplicates: true
    filter_by_length: true
    min_length: 10
    max_length: 512
  augmentation:
    enabled: false
    techniques: []
```

### Cluster Configuration

#### Ray Configuration

```yaml
cluster:
  name: "ray"
  num_workers: 4
  resources_per_worker:
    CPU: 1
    GPU: 0.25
  placement_group:
    strategy: "PACK"
```

#### FSDP Configuration

```yaml
cluster:
  name: "fsdp"
  fsdp_config:
    mixed_precision: true
    activation_checkpointing: true
    sharding_strategy: "FULL_SHARD"
```

### Training Configuration

#### Basic Training Settings

```yaml
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_steps: 10000
```

#### Optimizer Configuration

```yaml
training:
  optimizer:
    name: "adamw"
    lr: 5e-5
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.01
  scheduler:
    name: "cosine"
    warmup_steps: 100
    num_training_steps: 10000
```

#### Checkpointing

```yaml
training:
  checkpointing:
    save_steps: 1000
    save_total_limit: 3
    load_best_model_at_end: true
    metric_for_best_model: "eval_loss"
    greater_is_better: false
```

### Evaluation Configuration

```yaml
evaluation:
  eval_steps: 500
  eval_delay: 0
  eval_strategy: "steps"
  metrics:
    - "accuracy"
    - "f1"
    - "bleu"
  generation:
    max_new_tokens: 128
    do_sample: true
    temperature: 0.7
    top_p: 0.9
```

## Environment Variables

You can also configure NeMo RL using environment variables:

```bash
export NEMO_RL_LOG_LEVEL=INFO
export NEMO_RL_CACHE_DIR=/path/to/cache
export NEMO_RL_WANDB_PROJECT=my_project
export NEMO_RL_WANDB_ENTITY=my_entity
```

## Configuration Validation

NeMo RL automatically validates your configuration and will raise helpful errors if there are issues:

```python
from nemo_rl.utils.config import validate_config

# Validate configuration
errors = validate_config(config)
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
```

## Best Practices

1. **Use Configuration Files**: Store your configurations in YAML files for reproducibility
2. **Version Control**: Include configuration files in your version control system
3. **Environment-Specific Configs**: Create separate configs for different environments (dev, staging, prod)
4. **Documentation**: Document any custom configuration options
5. **Validation**: Always validate configurations before running training

## Troubleshooting

### Common Configuration Issues

1. **Memory Issues**: Reduce batch size or use gradient accumulation
2. **Training Instability**: Adjust learning rate and warmup steps
3. **Poor Performance**: Check data preprocessing and model configuration
4. **Distributed Training Issues**: Verify cluster configuration and resource allocation

For more help with configuration, see the [CLI Reference](cli.md) and [Troubleshooting Guide](../guides/production-support/troubleshooting.md). 