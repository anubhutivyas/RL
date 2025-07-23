# Configuration Reference

This document provides a comprehensive reference for all configuration options available in NeMo RL.

## Overview

NeMo RL uses YAML configuration files to define training parameters, model settings, and runtime behavior. All configuration files follow a hierarchical structure that allows for flexible and powerful configuration management.

## Configuration File Structure

### Basic Structure

```yaml
# Top-level configuration sections
model:
  # Model configuration
  name: "llama3.1-8b"
  backend: "huggingface"
  
training:
  # Training algorithm and parameters
  algorithm: "dpo"
  batch_size: 4
  learning_rate: 1e-5
  
data:
  # Dataset configuration
  train_dataset: "helpsteer3"
  eval_dataset: "helpsteer3"
  
distributed:
  # Distributed training settings
  strategy: "fsdp2"
  num_nodes: 1
  num_gpus: 8
```

## UV Configuration

NeMo RL uses `uv` for dependency management. Key configuration options:

```toml
# pyproject.toml
[tool.uv]
python = "3.9"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "ray>=2.5.0",
    "datasets>=2.10.0",
    "accelerate>=0.20.0"
]

[tool.uv.scripts]
grpo = "examples.run_grpo_math:main"
dpo = "examples.run_dpo:main"
sft = "examples.run_sft:main"
eval = "examples.run_eval:main"

[tool.uv.dev-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.0.0"
]
```

## Model Configuration

### Backend Selection

NeMo RL supports multiple model backends:

```yaml
model:
  backend: "huggingface"  # Options: huggingface, megatron
  name: "meta-llama/Llama-2-7b-hf"
```

### HuggingFace Backend

```yaml
model:
  backend: "huggingface"
  name: "meta-llama/Llama-2-7b-hf"
  trust_remote_code: true
  torch_dtype: "bfloat16"
  device_map: "auto"
  load_in_8bit: false
  load_in_4bit: false
```

### Megatron Backend

```yaml
model:
  backend: "megatron"
  name: "llama3.1-8b"
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  sequence_parallel: false
  use_flash_attention: true
```

## Training Configuration

### Algorithm Selection

```yaml
training:
  algorithm: "dpo"  # Options: dpo, grpo, sft, eval
```

### DPO Training

```yaml
training:
  algorithm: "dpo"
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  lr_scheduler: "cosine"
  warmup_steps: 100
  max_steps: 1000
  beta: 0.1  # DPO temperature parameter
  loss_type: "sigmoid"  # Options: sigmoid, hinge
```

### GRPO Training

```yaml
training:
  algorithm: "grpo"
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  max_steps: 1000
  gamma: 0.99  # Discount factor
  lambda_: 0.95  # GAE parameter
  clip_ratio: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
```

### SFT Training

```yaml
training:
  algorithm: "sft"
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  max_steps: 1000
  weight_decay: 0.01
  warmup_ratio: 0.1
```

## Data Configuration

### Dataset Selection

```yaml
data:
  train_dataset: "helpsteer3"
  eval_dataset: "helpsteer3"
  max_length: 2048
  truncation: true
  padding: "max_length"
```

### Custom Dataset Configuration

```yaml
data:
  train_dataset:
    name: "custom_dataset"
    path: "/path/to/dataset"
    split: "train"
    format: "json"  # Options: json, parquet, arrow
  eval_dataset:
    name: "custom_dataset"
    path: "/path/to/dataset"
    split: "validation"
```

### Chat Template Configuration

```yaml
data:
  chat_template: "llama3"  # Options: llama3, chatml, alpaca
  system_prompt: "You are a helpful assistant."
  max_length: 2048
  truncation: true
  padding: "max_length"
```

## Distributed Training Configuration

For comprehensive distributed training configuration and best practices, see the [Distributed Training Guide](../advanced/performance/distributed-training.md).

## Environment Configuration

### Math Environment (GRPO)

```yaml
environment:
  type: "math"
  difficulty: "medium"  # Options: easy, medium, hard
  max_steps: 100
  reward_scale: 1.0
  use_curriculum: true
  curriculum_config:
    start_difficulty: "easy"
    end_difficulty: "hard"
    steps_per_level: 1000
```

### Sliding Puzzle Environment

```yaml
environment:
  type: "sliding_puzzle"
  board_size: 3  # 3x3 puzzle
  max_steps: 50
  reward_scale: 1.0
```

## Generation Configuration

### VLLM Backend

```yaml
generation:
  backend: "vllm"
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
  pad_token_id: 0
  eos_token_id: 2
```

### HuggingFace Backend

```yaml
generation:
  backend: "huggingface"
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
  pad_token_id: 0
  eos_token_id: 2
```

## Logging Configuration

```yaml
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "file"
      filename: "training.log"
      level: "INFO"
    - type: "console"
      level: "INFO"
  tensorboard: true
  wandb: false
  wandb_project: "nemo-rl"
```

## Checkpointing Configuration

```yaml
checkpointing:
  save_dir: "./checkpoints"
  save_steps: 1000
  save_total_limit: 3
  load_checkpoint: null  # Path to checkpoint to load
  resume_from_checkpoint: false
  save_optimizer: true
  save_scheduler: true
```

## Evaluation Configuration

```yaml
evaluation:
  eval_steps: 100
  eval_batch_size: 4
  metrics:
    - "accuracy"
    - "bleu"
    - "rouge"
  save_predictions: true
  predictions_file: "predictions.json"
```

## Advanced Configuration

### Custom Loss Functions

```yaml
training:
  custom_loss:
    type: "weighted_dpo"
    weights:
      dpo: 1.0
      kl: 0.1
    temperature: 0.1
```

### Gradient Clipping

```yaml
training:
  max_grad_norm: 1.0
  gradient_clipping: true
```

### Mixed Precision

```yaml
training:
  mixed_precision: "bf16"  # Options: fp16, bf16, fp32
  autocast: true
```

## Environment Variables

NeMo RL supports configuration through environment variables:

```bash
export NEMO_RL_MODEL_BACKEND=huggingface
export NEMO_RL_TRAINING_ALGORITHM=dpo
export NEMO_RL_DISTRIBUTED_STRATEGY=fsdp2
export NEMO_RL_NUM_GPUS=8
```

## Configuration Validation

NeMo RL validates configuration files and provides helpful error messages for:

- Missing required fields
- Invalid field values
- Conflicting configurations
- Unsupported combinations

## Best Practices

1. **Use templates**: Start with provided example configurations
2. **Version control**: Keep configuration files in version control
3. **Environment-specific configs**: Use different configs for different environments
4. **Documentation**: Comment your configuration files
5. **Validation**: Test configurations before running long training jobs

## Example Complete Configuration

```yaml
# Complete DPO training configuration
model:
  backend: "huggingface"
  name: "meta-llama/Llama-2-7b-hf"
  trust_remote_code: true
  torch_dtype: "bfloat16"

training:
  algorithm: "dpo"
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  lr_scheduler: "cosine"
  warmup_steps: 100
  max_steps: 1000
  beta: 0.1
  max_grad_norm: 1.0
  mixed_precision: "bf16"

data:
  train_dataset: "helpsteer3"
  eval_dataset: "helpsteer3"
  max_length: 2048
  truncation: true
  padding: "max_length"
  chat_template: "llama3"

distributed:
  strategy: "fsdp2"
  num_nodes: 1
  num_gpus: 8
  fsdp_config:
    mixed_precision: "bf16"
    activation_checkpointing: true
    sharding_strategy: "FULL_SHARD"

logging:
  level: "INFO"
  tensorboard: true
  wandb: false

checkpointing:
  save_dir: "./checkpoints"
  save_steps: 1000
  save_total_limit: 3

evaluation:
  eval_steps: 100
  eval_batch_size: 4
  metrics:
    - "accuracy"
    - "bleu"
``` 