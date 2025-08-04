---
description: "Scale from single GPU to multi-node clusters with Ray-based distributed training strategies"
tags: ["distributed", "training", "scaling", "multi-gpu", "clusters", "ray"]
categories: ["performance"]
---

# Distributed Training

This guide covers how to scale NeMo RL from single GPU to multi-node clusters with efficient Ray-based distributed training strategies.

## Overview

NeMo RL provides robust distributed training capabilities using Ray for distributed computing, allowing you to scale your training across multiple GPUs and nodes. This is essential for training large models efficiently and reducing training time.

## Key Components

### Ray-Based Distributed Computing

NeMo RL uses Ray for distributed computing, providing seamless scaling:

```python
import ray
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.distributed.worker_groups import RayWorkerGroup, RayWorkerBuilder

# Initialize Ray
init_ray()

# Create virtual cluster
cluster = RayVirtualCluster(
    bundle_ct_per_node_list=[8, 8, 8, 8],  # 4 nodes with 8 GPUs each
    num_gpus_per_node=8,
    use_gpus=True
)

# Create worker builder
builder = RayWorkerBuilder("nemo_rl.models.policy.DTensorPolicyWorker")

# Create worker group
worker_group = RayWorkerGroup(
    cluster=cluster,
    remote_worker_builder=builder,
    workers_per_node=2,  # 2 workers per node
    name_prefix="policy_worker"
)
```

### Single Node Multi-GPU Training

Start with single node multi-GPU training:

```python
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

def setup_single_node_training():
    """
    Setup single node multi-GPU training
    """
    # Create single node cluster
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[8],  # Single node with 8 GPUs
        num_gpus_per_node=8,
        use_gpus=True
    )
    
    # Create worker builder
    builder = RayWorkerBuilder("nemo_rl.models.policy.DTensorPolicyWorker")
    
    # Create worker group
    worker_group = RayWorkerGroup(
        cluster=cluster,
        remote_worker_builder=builder,
        workers_per_node=4,  # 4 workers on single node
        name_prefix="single_node_worker"
    )
    
    return worker_group
```

### Multi-Node Training

Scale to multiple nodes:

```python
def setup_multi_node_training():
    """
    Setup multi-node distributed training
    """
    # Create multi-node cluster
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[8, 8, 8, 8],  # 4 nodes
        num_gpus_per_node=8,
        use_gpus=True
    )
    
    # Create worker builder
    builder = RayWorkerBuilder("nemo_rl.models.policy.DTensorPolicyWorker")
    
    # Create worker group across nodes
    worker_group = RayWorkerGroup(
        cluster=cluster,
        remote_worker_builder=builder,
        workers_per_node=2,  # 2 workers per node = 8 total workers
        name_prefix="multi_node_worker"
    )
    
    return worker_group
```

## Configuration

### Distributed Training Configuration

Configure distributed training in your YAML config:

```yaml
# Real NeMo RL distributed training config
cluster:
  gpus_per_node: 8
  num_nodes: 4  # Total nodes in cluster

policy:
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 4     # Distribute across 4 GPUs
    context_parallel_size: 1    # No context parallelism
    cpu_offload: true           # Enable CPU offloading
    sequence_parallel: false    # Disable for simplicity
    
  # Distributed training settings
  train_global_batch_size: 64   # Global batch size across all workers
  train_micro_batch_size: 1     # Micro batch size per worker
  generation_batch_size: 32      # Generation batch size
  logprob_batch_size: 4         # Logprob batch size
```

### Advanced Distributed Configuration

```yaml
# Advanced distributed training
cluster:
  gpus_per_node: 8
  num_nodes: 4

policy:
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 8     # Full node parallelism
    context_parallel_size: 1
    cpu_offload: true
    sequence_parallel: true      # Enable for large models
    custom_parallel_plan: null
    
  # Advanced distributed settings
  train_global_batch_size: 128
  train_micro_batch_size: 1
  max_total_sequence_length: 8192
  
  # Optimizer settings for distributed training
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      foreach: False  # Required for DTensor
      fused: False    # Required for DTensor
      weight_decay: 0.01
```

## Training Patterns

### Distributed Training Loop

```python
def distributed_training_loop(worker_group, data, loss_fn):
    """
    Distributed training loop using NeMo RL patterns
    """
    # Run training across all workers
    futures = worker_group.run_all_workers_single_data(
        method_name="train",
        data=data,
        loss_fn=loss_fn
    )
    
    # Get results from all workers
    results = worker_group.get_all_worker_results(futures)
    
    return results
```

### Distributed Inference

```python
def distributed_inference(worker_group, data):
    """
    Distributed inference using NeMo RL patterns
    """
    # Run inference across all workers
    futures = worker_group.run_all_workers_single_data(
        method_name="get_logprobs",
        data=data
    )
    
    # Get results from all workers
    logprobs = worker_group.get_all_worker_results(futures)
    
    return logprobs
```

### Sharded Data Training

```python
def sharded_data_training(worker_group, data):
    """
    Training with sharded data across workers
    """
    # Run training with sharded data
    futures = worker_group.run_all_workers_sharded_data(
        method_name="train",
        data=data,
        in_sharded_axes=["batch"],  # Shard along batch dimension
        replicate_on_axes=["model"], # Replicate model across workers
        output_is_replicated=["loss"] # Loss is replicated
    )
    
    # Get results
    results = worker_group.get_all_worker_results(futures)
    
    return results
```

## Best Practices

### 1. Resource Management

```python
def manage_distributed_resources(worker_group):
    """
    Manage distributed resources effectively
    """
    # Get cluster information
    cluster = worker_group.cluster
    world_size = cluster.world_size()
    node_count = cluster.node_count()
    
    print(f"Cluster: {node_count} nodes, {world_size} total GPUs")
    
    # Monitor worker health
    for i, worker in enumerate(worker_group.workers):
        if worker.is_alive():
            print(f"Worker {i}: Alive")
        else:
            print(f"Worker {i}: Dead")
    
    return {
        'world_size': world_size,
        'node_count': node_count,
        'active_workers': len([w for w in worker_group.workers if w.is_alive()])
    }
```

### 2. Communication Optimization

```python
def optimize_communication(worker_group):
    """
    Optimize communication between workers
    """
    # Use efficient data transfer
    futures = worker_group.run_all_workers_single_data(
        method_name="prepare_for_training",
        run_rank_0_only_axes=["setup"]  # Only rank 0 does setup
    )
    
    # Synchronize workers
    worker_group.get_all_worker_results(futures)
    
    return "Communication optimized"
```

### 3. Load Balancing

```python
def balance_workload(worker_group, data_batches):
    """
    Balance workload across workers
    """
    # Distribute data evenly across workers
    num_workers = len(worker_group.workers)
    batch_size = len(data_batches) // num_workers
    
    balanced_batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        balanced_batches.append(data_batches[start_idx:end_idx])
    
    # Run training with balanced batches
    futures = []
    for i, batch in enumerate(balanced_batches):
        future = worker_group.run_single_worker_single_data(
            method_name="train",
            worker_idx=i,
            data=batch
        )
        futures.append(future)
    
    # Get results
    results = worker_group.get_all_worker_results(
        MultiWorkerFuture(futures=futures)
    )
    
    return results
```

## Troubleshooting

### Common Distributed Issues

1. **Worker Communication Errors**
   ```python
   # Solution: Check network connectivity and Ray status
   import ray
   print(f"Ray status: {ray.is_initialized()}")
   print(f"Available resources: {ray.available_resources()}")
   ```

2. **Memory Issues Across Workers**
   ```python
   # Solution: Monitor memory on all workers
   futures = worker_group.run_all_workers_single_data(
       method_name="get_gpu_info"
   )
   gpu_infos = worker_group.get_all_worker_results(futures)
   
   for i, info in enumerate(gpu_infos):
       print(f"Worker {i} GPU info: {info}")
   ```

3. **Load Balancing Issues**
   ```python
   # Solution: Check worker distribution
   worker_metadata = worker_group.worker_metadata
   for i, metadata in enumerate(worker_metadata):
       print(f"Worker {i} metadata: {metadata}")
   ```

### Debugging Distributed Training

```python
def debug_distributed_training(worker_group):
    """
    Debug distributed training issues
    """
    print("=== Distributed Training Debug ===")
    
    # Check cluster status
    cluster = worker_group.cluster
    print(f"Cluster world size: {cluster.world_size()}")
    print(f"Cluster node count: {cluster.node_count()}")
    
    # Check worker status
    active_workers = 0
    for i, worker in enumerate(worker_group.workers):
        if worker.is_alive():
            active_workers += 1
            print(f"Worker {i}: Active")
        else:
            print(f"Worker {i}: Inactive")
    
    print(f"Active workers: {active_workers}/{len(worker_group.workers)}")
    
    # Check data parallel size
    print(f"Data parallel size: {worker_group.dp_size}")
    
    print("================================")
    
    return {
        'active_workers': active_workers,
        'total_workers': len(worker_group.workers),
        'dp_size': worker_group.dp_size
    }
```

## Next Steps

After setting up distributed training:

1. **Monitor Performance**: Track training speed and resource utilization
2. **Optimize Communication**: Minimize communication overhead
3. **Scale Further**: Add more nodes for larger models
4. **Profile Performance**: Use profiling tools to identify bottlenecks

For more advanced topics, see:
- [Memory Optimization](memory-optimization.md) - Optimize memory usage
- [Performance Profiling](profiling.md) - Profile distributed performance
- [Monitoring](monitoring.md) - Monitor distributed training 