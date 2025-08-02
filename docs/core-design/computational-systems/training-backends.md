# Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and hardware configurations.

## Available Backends

- **DTensor (FSDP2)** - PyTorch's next-generation distributed training with improved memory efficiency
- **Megatron** - NVIDIA's high-performance training framework for scaling to large models (>100B parameters)

## Backend Selection

The training backend is automatically determined based on your YAML configuration settings. Here's how to configure each backend.

### Megatron Backend
To enable Megatron-based training:

1. Add the `megatron_cfg` key to your policy configuration.
2. Set `policy.megatron_cfg.enabled=True`.
3. Refer to the [examples directory](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) for complete configuration examples.

_Note_: When using Megatron, the optimizer and learning rate schedule are configured through `policy.megatron_cfg.optimizer` and `policy.megatron_cfg.scheduler`, respectively.

### DTensor Backend
To enable DTensor (FSDP2) training:

1. Set `policy.dtensor_config.enabled=True`.
2. Refer to the [examples directory](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) for configuration examples.

## Backend Priority

**Megatron takes precedence over DTensor.** If both backends are enabled simultaneously (`policy.megatron_cfg.enabled=True` and `policy.dtensor_config.enabled=True`), the Megatron backend will be used.

## Configuration Examples

### Basic Configuration

```yaml
# Basic training configuration
cluster:
  gpus_per_node: 1
  num_nodes: 1

policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_seq_length: 1024

algorithm:
  batch_size: 2
  gradient_accumulation_steps: 2
```

### Advanced Configuration

```yaml
# Advanced training configuration
cluster:
  gpus_per_node: 4
  num_nodes: 2

policy:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_seq_length: 2048

algorithm:
  batch_size: 1
  gradient_accumulation_steps: 8
```

For comprehensive examples, see the [examples directory](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) in the repository.
