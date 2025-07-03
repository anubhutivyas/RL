# Configuration Reference

This page documents all configuration options available in NeMo RL.

## Configuration Structure

NeMo RL uses a hierarchical YAML configuration system with the following main sections:

```yaml
# Cluster configuration
cluster:
  gpus_per_node: 4
  num_nodes: 1

# Policy/model configuration
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 2

# Data configuration
data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  dataset_key: "train_sft"

# Algorithm-specific configuration
sft:
  max_num_epochs: 3
  learning_rate: 1e-4

# Logging configuration
logger:
  wandb:
    enabled: true
    name: "my-experiment"
```

## Cluster Configuration

### Basic Settings

```yaml
cluster:
  # Number of GPUs per node
  gpus_per_node: 4
  
  # Number of nodes in cluster
  num_nodes: 1
  
  # Number of workers per node
  workers_per_node: 4
  
  # Ray cluster configuration
  ray:
    # Ray dashboard port
    dashboard_port: 8265
    
    # Ray object store memory (GB)
    object_store_memory: 10
```

### Advanced Settings

```yaml
cluster:
  # Resource allocation
  resources:
    # CPU cores per worker
    cpu_per_worker: 2
    
    # Memory per worker (GB)
    memory_per_worker: 8
    
    # GPU memory fraction per worker
    gpu_memory_fraction: 0.9
  
  # Network configuration
  network:
    # Interface to use for communication
    interface: "eth0"
    
    # Port range for Ray
    port_range: [10000, 20000]
```

## Policy Configuration

### Model Settings

```yaml
policy:
  # Hugging Face model name or path
  model_name: "Qwen/Qwen2.5-1.5B"
  
  # Tokenizer name (defaults to model_name if not specified)
  tokenizer_name: "Qwen/Qwen2.5-1.5B"
  
  # Maximum sequence length
  max_seq_len: 4096
  
  # Model precision
  precision: "bf16"  # "fp32", "fp16", "bf16"
  
  # Mixed precision training
  mixed_precision: true
  
  # Gradient checkpointing
  gradient_checkpointing: true
```

### DTensor Configuration

```yaml
policy:
  dtensor_cfg:
    # Enable DTensor backend
    enabled: true
    
    # Tensor parallelism size
    tensor_parallel_size: 2
    
    # Context parallelism size
    context_parallel_size: 1
    
    # Data parallelism size (auto-calculated)
    data_parallel_size: 4
    
    # Pipeline parallelism size
    pipeline_parallel_size: 1
```

### Megatron Configuration

```yaml
policy:
  megatron_cfg:
    # Enable Megatron backend
    enabled: true
    
    # Tensor model parallel size
    tensor_model_parallel_size: 2
    
    # Pipeline model parallel size
    pipeline_model_parallel_size: 1
    
    # Context parallel size
    context_parallel_size: 1
    
    # Optimizer configuration
    optimizer:
      type: "adamw"
      lr: 1e-4
      weight_decay: 0.01
    
    # Scheduler configuration
    scheduler:
      type: "cosine"
      warmup_steps: 1000
      total_steps: 10000
```

## Data Configuration

### Dataset Settings

```yaml
data:
  # Dataset name or path
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  
  # Dataset split/key
  dataset_key: "train_sft"
  
  # Dataset configuration
  dataset_cfg:
    # Maximum number of samples
    max_samples: 10000
    
    # Random seed for sampling
    seed: 42
    
    # Filter conditions
    filter:
      min_length: 10
      max_length: 2048
```

### Data Loading

```yaml
data:
  # Number of data loading workers
  num_workers: 4
  
  # Prefetch factor
  prefetch_factor: 2
  
  # Pin memory
  pin_memory: true
  
  # Drop last incomplete batch
  drop_last: true
  
  # Shuffle data
  shuffle: true
```

### Tokenization

```yaml
data:
  # Tokenization settings
  tokenization:
    # Add special tokens
    add_special_tokens: true
    
    # Padding side
    padding_side: "right"
    
    # Truncation side
    truncation_side: "right"
    
    # Maximum length for truncation
    max_length: 4096
```

## Algorithm Configuration

### SFT Configuration

```yaml
sft:
  # Training epochs
  max_num_epochs: 3
  
  # Maximum training steps
  max_num_steps: 10000
  
  # Learning rate
  learning_rate: 1e-4
  
  # Weight decay
  weight_decay: 0.01
  
  # Warmup steps
  warmup_steps: 1000
  
  # Micro batch size
  micro_batch_size: 4
  
  # Gradient accumulation steps
  gradient_accumulation_steps: 4
  
  # Global batch size (auto-calculated)
  global_batch_size: 64
  
  # Gradient clipping
  max_grad_norm: 1.0
  
  # Validation settings
  val_period: 100
  val_batches: 10
  val_at_start: true
```

### GRPO Configuration

```yaml
grpo:
  # Training epochs
  max_num_epochs: 3
  
  # Maximum training steps
  max_num_steps: 10000
  
  # Policy learning rate
  learning_rate: 1e-4
  
  # Value function learning rate
  value_learning_rate: 1e-4
  
  # Entropy coefficient
  entropy_coeff: 0.01
  
  # PPO clipping ratio
  clip_ratio: 0.2
  
  # Value function clipping
  value_clip_ratio: 0.2
  
  # Micro batch size
  micro_batch_size: 4
  
  # Gradient accumulation steps
  gradient_accumulation_steps: 4
  
  # Global batch size
  global_batch_size: 64
  
  # Gradient clipping
  max_grad_norm: 1.0
  
  # Validation settings
  val_period: 100
  val_batches: 10
  val_at_start: true
  
  # Environment settings
  env:
    # Number of environment workers
    num_workers: 4
    
    # Maximum episode length
    max_episode_length: 1000
    
    # Reward scaling
    reward_scale: 1.0
```

### DPO Configuration

```yaml
dpo:
  # Training epochs
  max_num_epochs: 3
  
  # Maximum training steps
  max_num_steps: 10000
  
  # Learning rate
  learning_rate: 1e-4
  
  # DPO beta parameter
  beta: 0.1
  
  # Preference loss weight
  preference_loss_weight: 1.0
  
  # SFT loss weight
  sft_loss_weight: 0.1
  
  # Reference policy KL penalty
  reference_policy_kl_penalty: 0.0
  
  # Average log probs for preferences
  preference_average_log_probs: true
  
  # Average log probs for SFT
  sft_average_log_probs: true
  
  # Micro batch size
  micro_batch_size: 4
  
  # Gradient accumulation steps
  gradient_accumulation_steps: 4
  
  # Global batch size
  global_batch_size: 64
  
  # Gradient clipping
  max_grad_norm: 1.0
  
  # Validation settings
  val_period: 100
  val_batches: 10
  val_at_start: true
```

## Generation Configuration

### vLLM Settings

```yaml
generation:
  # Model name or path
  model_name: "/path/to/checkpoint"
  
  # vLLM configuration
  vllm_cfg:
    # Maximum model length
    max_model_len: 4096
    
    # Tensor parallel size
    tensor_parallel_size: 1
    
    # Trust remote code
    trust_remote_code: true
    
    # Download directory
    download_dir: "/path/to/downloads"
    
    # Load format
    load_format: "auto"
    
    # Dtype
    dtype: "bfloat16"
    
    # Seed
    seed: 42
```

### Sampling Parameters

```yaml
generation:
  # Temperature for sampling
  temperature: 0.7
  
  # Top-p sampling
  top_p: 0.9
  
  # Top-k sampling
  top_k: 50
  
  # Maximum new tokens
  max_new_tokens: 512
  
  # Minimum new tokens
  min_new_tokens: 1
  
  # Repetition penalty
  repetition_penalty: 1.1
  
  # Length penalty
  length_penalty: 1.0
  
  # Early stopping
  early_stopping: true
  
  # Pad token ID
  pad_token_id: 0
  
  # EOS token ID
  eos_token_id: 2
```

## Logging Configuration

### WandB Settings

```yaml
logger:
  # Weights & Biases configuration
  wandb:
    # Enable WandB logging
    enabled: true
    
    # Project name
    project: "nemo-rl"
    
    # Run name
    name: "my-experiment"
    
    # Entity/team
    entity: "my-team"
    
    # Tags
    tags: ["sft", "qwen"]
    
    # Notes
    notes: "Training run description"
    
    # Configuration
    config:
      model_size: "1.5B"
      dataset: "ultrachat"
```

### Local Logging

```yaml
logger:
  # Log level
  level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
  
  # Log directory
  log_dir: "./logs"
  
  # Log file name
  log_file: "training.log"
  
  # Console logging
  console: true
  
  # File logging
  file: true
  
  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Evaluation Configuration

### Basic Settings

```yaml
eval:
  # Number of tests per prompt
  num_tests_per_prompt: 16
  
  # Evaluation dataset
  dataset_name: "HuggingFaceH4/MATH-500"
  
  # Dataset split
  dataset_key: "test"
  
  # Maximum samples to evaluate
  max_samples: 1000
  
  # Random seed
  seed: 42
```

### Metrics

```yaml
eval:
  # Metrics to compute
  metrics:
    - "accuracy"
    - "exact_match"
    - "bleu"
    - "rouge"
  
  # Custom metric configuration
  custom_metrics:
    math_accuracy:
      type: "exact_match"
      normalize: true
```

## Checkpoint Configuration

### Saving

```yaml
checkpoint:
  # Save directory
  save_dir: "./checkpoints"
  
  # Save frequency (steps)
  save_freq: 1000
  
  # Save best model
  save_best: true
  
  # Save latest model
  save_latest: true
  
  # Number of checkpoints to keep
  keep_last: 3
  
  # Save optimizer state
  save_optimizer: true
  
  # Save scheduler state
  save_scheduler: true
```

### Loading

```yaml
checkpoint:
  # Load path
  load_path: "/path/to/checkpoint"
  
  # Load optimizer state
  load_optimizer: true
  
  # Load scheduler state
  load_scheduler: true
  
  # Strict loading
  strict: true
  
  # Map location
  map_location: "auto"
```

## Environment Variables

### Required

```bash
# Hugging Face
export HF_HOME="/path/to/huggingface/cache"
export HF_DATASETS_CACHE="/path/to/datasets/cache"

# Weights & Biases
export WANDB_API_KEY="your_api_key"
```

### Optional

```bash
# CUDA
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Debug
export NEMO_RL_DEBUG=1

# Force rebuild
export NRL_FORCE_REBUILD_VENVS=true

# Ray
export RAY_DISABLE_IMPORT_WARNING=1
```

## Configuration Validation

### Type Checking

NeMo RL automatically validates configuration types:

```yaml
# These will be automatically converted
cluster:
  gpus_per_node: "4"  # Converted to int
  num_nodes: 1.0      # Converted to int

sft:
  learning_rate: "1e-4"  # Converted to float
  mixed_precision: "true"  # Converted to bool
```

### Required Fields

Some configuration fields are required:

```yaml
# Required fields
policy:
  model_name: "Qwen/Qwen2.5-1.5B"  # Required

data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"  # Required

cluster:
  gpus_per_node: 4  # Required
```

### Default Values

Many fields have sensible defaults:

```yaml
# Default values (can be overridden)
sft:
  learning_rate: 1e-4  # Default
  max_grad_norm: 1.0   # Default
  warmup_steps: 0      # Default

logger:
  level: "INFO"        # Default
  console: true        # Default
```

## Configuration Examples

### Minimal Configuration

```yaml
cluster:
  gpus_per_node: 1

policy:
  model_name: "Qwen/Qwen2.5-0.5B"

data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  dataset_key: "train_sft"

sft:
  max_num_epochs: 1
```

### Production Configuration

```yaml
cluster:
  gpus_per_node: 8
  num_nodes: 2

policy:
  model_name: "Qwen/Qwen2.5-7B"
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 2
    context_parallel_size: 1
  mixed_precision: true
  gradient_checkpointing: true

data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  dataset_key: "train_sft"
  num_workers: 8

sft:
  max_num_epochs: 3
  learning_rate: 1e-4
  micro_batch_size: 2
  gradient_accumulation_steps: 8

logger:
  wandb:
    enabled: true
    name: "production-sft-7b"
    project: "nemo-rl-production"
```

### Research Configuration

```yaml
cluster:
  gpus_per_node: 4

policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_seq_len: 8192

data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  dataset_key: "train_sft"
  dataset_cfg:
    max_samples: 50000

sft:
  max_num_epochs: 5
  learning_rate: 5e-5
  warmup_steps: 1000
  micro_batch_size: 4

logger:
  wandb:
    enabled: true
    name: "research-sft-1.5b"
    tags: ["research", "sft", "1.5b"]
```

## CLI Overrides

All configuration values can be overridden via command line:

```bash
uv run python examples/run_sft.py \
    --config examples/configs/sft.yaml \
    cluster.gpus_per_node=4 \
    policy.model_name=Qwen/Qwen2.5-7B \
    sft.learning_rate=1e-4 \
    sft.max_num_epochs=3 \
    logger.wandb.name="my-experiment"
```

## Getting Help

- [CLI Reference](cli.md) - Command-line interface documentation
- [Troubleshooting Guide](troubleshooting.md) - Common configuration issues
- [API Reference](api.md) - Programmatic configuration 