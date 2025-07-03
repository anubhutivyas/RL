# CLI Reference

This page documents all command-line interfaces available in NeMo RL.

## Training Scripts

### SFT (Supervised Fine-Tuning)

```bash
uv run python examples/run_sft.py [OPTIONS]
```

**Basic Usage**:
```bash
uv run python examples/run_sft.py \
    --config examples/configs/sft.yaml \
    cluster.gpus_per_node=4 \
    policy.model_name=Qwen/Qwen2.5-1.5B
```

**Key Parameters**:
- `--config`: Path to YAML configuration file
- `cluster.gpus_per_node`: Number of GPUs per node
- `policy.model_name`: Hugging Face model name
- `data.dataset_name`: Dataset name or path
- `sft.max_num_epochs`: Maximum number of training epochs
- `sft.learning_rate`: Learning rate for training
- `logger.wandb.name`: Experiment name for Weights & Biases

### GRPO (Group Relative Policy Optimization)

```bash
uv run python examples/run_grpo_math.py [OPTIONS]
```

**Basic Usage**:
```bash
uv run python examples/run_grpo_math.py \
    --config examples/configs/grpo_math_1B.yaml \
    cluster.gpus_per_node=8 \
    policy.model_name=Qwen/Qwen2.5-1.5B
```

**Key Parameters**:
- `grpo.max_num_epochs`: Maximum number of training epochs
- `grpo.learning_rate`: Learning rate for policy training
- `grpo.value_learning_rate`: Learning rate for value function
- `grpo.entropy_coeff`: Entropy coefficient for exploration
- `grpo.clip_ratio`: PPO clipping ratio

### DPO (Direct Preference Optimization)

```bash
uv run python examples/run_dpo.py [OPTIONS]
```

**Basic Usage**:
```bash
uv run python examples/run_dpo.py \
    --config examples/configs/dpo.yaml \
    cluster.gpus_per_node=4 \
    policy.model_name=Qwen/Qwen2.5-1.5B
```

**Key Parameters**:
- `dpo.max_num_epochs`: Maximum number of training epochs
- `dpo.learning_rate`: Learning rate for training
- `dpo.beta`: DPO beta parameter
- `dpo.preference_loss_weight`: Weight for preference loss
- `dpo.sft_loss_weight`: Weight for SFT loss

### GRPO on Sliding Puzzle

```bash
uv run python examples/run_grpo_sliding_puzzle.py [OPTIONS]
```

**Basic Usage**:
```bash
uv run python examples/run_grpo_sliding_puzzle.py \
    --config examples/configs/grpo_sliding_puzzle.yaml \
    cluster.gpus_per_node=4
```

## Evaluation Scripts

### Model Evaluation

```bash
uv run python examples/run_eval.py [OPTIONS]
```

**Basic Usage**:
```bash
uv run python examples/run_eval.py \
    generation.model_name=/path/to/checkpoint \
    data.dataset_name=HuggingFaceH4/MATH-500 \
    data.dataset_key=test \
    eval.num_tests_per_prompt=16
```

**Key Parameters**:
- `generation.model_name`: Path to model checkpoint or HF model name
- `data.dataset_name`: Evaluation dataset name
- `data.dataset_key`: Dataset split to evaluate on
- `eval.num_tests_per_prompt`: Number of samples per prompt
- `generation.temperature`: Sampling temperature
- `generation.top_p`: Top-p sampling parameter
- `generation.max_new_tokens`: Maximum tokens to generate

## Configuration Overrides

### YAML Configuration

All scripts support YAML configuration files with CLI overrides:

```bash
uv run python examples/run_sft.py \
    --config examples/configs/sft.yaml \
    sft.learning_rate=1e-4 \
    sft.max_num_epochs=3 \
    logger.wandb.name="my-experiment"
```

### Dot Notation

Use dot notation to override nested configuration parameters:

```bash
uv run python examples/run_sft.py \
    policy.model_name=Qwen/Qwen2.5-7B \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg.tensor_parallel_size=2 \
    cluster.gpus_per_node=8
```

### Type Conversion

CLI overrides automatically convert types:

```bash
# Boolean
policy.mixed_precision=true

# Integer
cluster.gpus_per_node=4

# Float
sft.learning_rate=1e-4

# String
logger.wandb.name="experiment-name"
```

## Cluster Configuration

### Single Node

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=4 \
    cluster.num_nodes=1
```

### Multi-Node (Slurm)

```bash
srun --nodes=2 --gpus-per-node=8 \
    uv run python examples/run_sft.py \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=2
```

### Kubernetes

```bash
kubectl apply -f k8s/nemo-rl-job.yaml
```

## Environment Variables

### Required Variables

```bash
# Hugging Face
export HF_HOME="/path/to/huggingface/cache"
export HF_DATASETS_CACHE="/path/to/datasets/cache"

# Authentication
export WANDB_API_KEY="your_wandb_api_key"
```

### Optional Variables

```bash
# CUDA
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Debug
export NEMO_RL_DEBUG=1

# Force rebuild
export NRL_FORCE_REBUILD_VENVS=true
```

## Model Conversion Scripts

### Convert to Hugging Face Format

```bash
python examples/converters/convert_megatron_to_hf.py \
    --input_dir /path/to/megatron/checkpoint \
    --output_dir /path/to/hf/checkpoint \
    --model_name Qwen/Qwen2.5-1.5B
```

### Convert from DCP Format

```bash
python examples/converters/convert_dcp_to_hf.py \
    --input_dir /path/to/dcp/checkpoint \
    --output_dir /path/to/hf/checkpoint \
    --model_name Qwen/Qwen2.5-1.5B
```

## Utility Scripts

### Custom Parallel Training

```bash
python examples/custom_parallel.py \
    --config examples/configs/custom_parallel.yaml
```

## Common Patterns

### Development Workflow

```bash
# Quick test run
uv run python examples/run_sft.py \
    cluster.gpus_per_node=1 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    sft.max_num_epochs=1 \
    logger.wandb.name="dev-test"

# Full training run
uv run python examples/run_sft.py \
    cluster.gpus_per_node=8 \
    policy.model_name=Qwen/Qwen2.5-7B \
    sft.max_num_epochs=3 \
    policy.dtensor_cfg.enabled=true \
    logger.wandb.name="production-run"
```

### Hyperparameter Tuning

```bash
# Learning rate sweep
for lr in 1e-5 5e-5 1e-4 5e-4; do
    uv run python examples/run_sft.py \
        cluster.gpus_per_node=1 \
        sft.learning_rate=$lr \
        logger.wandb.name="lr-sweep-$lr"
done
```

### Model Comparison

```bash
# Compare different models
for model in Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B; do
    uv run python examples/run_sft.py \
        cluster.gpus_per_node=4 \
        policy.model_name=$model \
        sft.max_num_epochs=1 \
        logger.wandb.name="model-comparison-${model//\//-}"
done
```

## Error Handling

### Debug Mode

Enable debug logging:

```bash
export NEMO_RL_DEBUG=1
uv run python examples/run_sft.py ...
```

### Verbose Output

Increase log verbosity:

```bash
uv run python examples/run_sft.py \
    logger.level=DEBUG \
    ...
```

### Dry Run

Test configuration without training:

```bash
uv run python examples/run_sft.py \
    sft.max_num_epochs=0 \
    sft.max_num_steps=1 \
    ...
```

## Performance Tuning

### Memory Optimization

```bash
uv run python examples/run_sft.py \
    policy.gradient_checkpointing=true \
    policy.mixed_precision=true \
    sft.micro_batch_size=1 \
    sft.gradient_accumulation_steps=4 \
    ...
```

### Speed Optimization

```bash
uv run python examples/run_sft.py \
    policy.mixed_precision=true \
    sft.micro_batch_size=8 \
    data.num_workers=4 \
    ...
```

### Large Model Training

```bash
uv run python examples/run_sft.py \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg.tensor_parallel_size=2 \
    policy.dtensor_cfg.context_parallel_size=1 \
    cluster.gpus_per_node=8 \
    ...
```

## Troubleshooting Commands

### Check Installation

```bash
python -c "import nemo_rl; print('NeMo RL installed successfully!')"
```

### Check GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Check Ray Status

```bash
ray status
```

### Monitor Resources

```bash
# GPU usage
watch -n 1 nvidia-smi

# Memory usage
watch -n 1 free -h

# Disk usage
df -h
```

## Getting Help

### Command Help

```bash
# Get help for any script
python examples/run_sft.py --help
```

### Configuration Help

```bash
# View default configuration
cat examples/configs/sft.yaml
```

### Documentation

- [Configuration Reference](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)
- [API Reference](api.md) 