# Distributed Training

This guide covers distributed training strategies for NeMo RL, including multi-GPU and multi-node configurations.

## Overview

Distributed training in NeMo RL leverages Ray's distributed computing capabilities to scale training across multiple GPUs and nodes. This enables training larger models and processing more data efficiently.

## Key Concepts

### Ray Virtual Cluster

NeMo RL uses Ray Virtual Clusters to manage distributed resources:

```python
from nemo_rl.distributed import RayVirtualCluster

# Create a virtual cluster with 4 GPUs per node
cluster = RayVirtualCluster(
    bundle_ct_per_node_list=[4],  # 4 GPUs per node
    num_nodes=2  # 2 nodes total
)
```

### Worker Groups

Worker groups manage distributed processes:

```python
from nemo_rl.distributed import RayWorkerGroup

# Create worker group for policy training
policy_workers = RayWorkerGroup(
    cluster=cluster,
    worker_class=PolicyWorker,
    num_workers=4
)
```

## Configuration

### Single Node Multi-GPU

```yaml
# Single node with 8 GPUs
cluster:
  num_nodes: 1
  gpus_per_node: 8
  
policy:
  workers_per_node: 4
  batch_size: 32
```

### Multi-Node Training

```yaml
# Multi-node configuration
cluster:
  num_nodes: 4
  gpus_per_node: 8
  
policy:
  workers_per_node: 2
  batch_size: 128
  
communication:
  backend: "nccl"
  timeout: 300
```

## Communication Strategies

### NCCL Backend

NCCL (NVIDIA Collective Communications Library) is the recommended backend for GPU communication:

```python
# Configure NCCL
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=world_size,
    rank=rank
)
```

### Gradient Synchronization

Gradients are synchronized across workers using all-reduce operations:

```python
def sync_gradients(model):
    """Synchronize gradients across all workers."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()
```

## Memory Management

### Gradient Checkpointing

Enable gradient checkpointing to reduce memory usage:

```python
# Enable gradient checkpointing
policy.gradient_checkpointing = True

# Configure checkpointing frequency
policy.checkpoint_every = 4  # Checkpoint every 4 layers
```

### Mixed Precision

Use mixed precision training for memory efficiency:

```python
# Enable mixed precision
policy.mixed_precision = True
policy.precision = "bfloat16"  # or "float16"
```

## Performance Optimization

### Batch Size Scaling

Scale batch size with the number of GPUs:

```python
# Calculate effective batch size
effective_batch_size = local_batch_size * num_gpus * gradient_accumulation_steps
```

### Communication Overlap

Overlap computation and communication:

```python
# Use gradient accumulation to overlap communication
policy.gradient_accumulation_steps = 4
```

## Monitoring

### Resource Monitoring

Monitor cluster resources during training:

```python
def monitor_resources():
    """Monitor cluster resource usage."""
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    return {
        "cluster_resources": cluster_resources,
        "available_resources": available_resources
    }
```

### Performance Metrics

Track key performance metrics:

```python
# Training throughput
samples_per_second = num_samples / training_time

# GPU utilization
gpu_utilization = gpu_time / total_time

# Communication overhead
comm_overhead = comm_time / total_time
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Communication Timeouts**
   - Increase timeout values
   - Check network connectivity
   - Verify NCCL configuration

3. **Load Balancing**
   - Ensure even data distribution
   - Monitor worker utilization
   - Adjust worker allocation

### Debug Commands

```bash
# Check Ray cluster status
ray status

# Monitor GPU usage
nvidia-smi

# Check network connectivity
ping worker_node_ip

# View Ray logs
ray logs
```

## Best Practices

### Resource Planning

1. **Calculate Requirements**
   - Model size and memory requirements
   - Batch size and gradient accumulation
   - Communication overhead

2. **Optimize Configuration**
   - Balance computation and communication
   - Minimize idle time
   - Maximize GPU utilization

3. **Monitor Performance**
   - Track training throughput
   - Monitor resource usage
   - Identify bottlenecks

### Scaling Strategies

1. **Data Parallelism**
   - Replicate model across GPUs
   - Synchronize gradients
   - Scale batch size

2. **Model Parallelism**
   - Split model across GPUs
   - Use tensor parallelism
   - Pipeline parallelism for large models

3. **Hybrid Approaches**
   - Combine data and model parallelism
   - Optimize for specific workloads
   - Balance memory and computation

## Example Configurations

### Small Scale (4 GPUs)

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 4
  
policy:
  workers_per_node: 2
  batch_size: 16
  learning_rate: 1e-4
```

### Medium Scale (32 GPUs)

```yaml
cluster:
  num_nodes: 4
  gpus_per_node: 8
  
policy:
  workers_per_node: 2
  batch_size: 64
  learning_rate: 5e-5
  
communication:
  backend: "nccl"
  timeout: 300
```

### Large Scale (128 GPUs)

```yaml
cluster:
  num_nodes: 16
  gpus_per_node: 8
  
policy:
  workers_per_node: 4
  batch_size: 256
  learning_rate: 2e-5
  
advanced:
  gradient_accumulation_steps: 4
  mixed_precision: true
  gradient_checkpointing: true
```

## Next Steps

After setting up distributed training:

1. **Optimize Performance**: Monitor and tune your configuration
2. **Scale Up**: Add more nodes as needed
3. **Monitor Resources**: Keep track of utilization and costs
4. **Debug Issues**: Use the troubleshooting guide for common problems

For more advanced topics, see:
- [Memory Optimization](memory-optimization.md) - Memory management techniques
- [Mixed Precision](mixed-precision.md) - Mixed precision training
- [Performance Profiling](../profiling.md) - Profiling tools and techniques 