---
description: "Scale NeMo RL to production with distributed training, memory optimization, and performance tuning techniques"
categories: ["advanced"]
tags: ["performance", "scaling", "distributed", "optimization", "production", "memory"]
personas: ["mle-focused", "admin-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "universal"
---

# Performance and Scaling

This guide covers how to scale NeMo RL to production with distributed training, memory optimization, and performance tuning techniques.

## Overview

NeMo RL provides comprehensive tools for scaling from single GPU to multi-node clusters. This guide focuses on production-ready scaling strategies.

## Distributed Training

### Ray Cluster Setup

```python
from nemo_rl.distributed import RayVirtualCluster, RayWorkerGroup

# Single node, multiple GPUs
cluster = RayVirtualCluster([8])  # 8 GPUs on one node

# Multi-node cluster
cluster = RayVirtualCluster([8, 8, 8])  # 3 nodes, 8 GPUs each

# Heterogeneous cluster
cluster = RayVirtualCluster([4, 8, 16])  # Different GPU counts per node
```

### Worker Group Configuration

```python
from nemo_rl.algorithms import DPOTrainer

# Configure worker group
worker_group = RayWorkerGroup(
    cluster=cluster,
    trainer_class=DPOTrainer,
    num_workers=4,
    resources_per_worker={"GPU": 2}
)

# Start training
worker_group.start_training(config)
```

## Memory Optimization

### Gradient Checkpointing

```python
# Enable gradient checkpointing
config = {
    "model": {
        "gradient_checkpointing": True,
        "checkpoint_ratio": 0.5  # Checkpoint every other layer
    }
}
```

### Mixed Precision Training

```python
# Configure mixed precision
config = {
    "training": {
        "mixed_precision": True,
        "fp16": True,
        "bf16": False,
        "loss_scaling": "dynamic"
    }
}
```

### Memory-Efficient Attention

```python
# Use memory-efficient attention
config = {
    "model": {
        "attention": {
            "type": "flash_attention",
            "memory_efficient": True
        }
    }
}
```

## Performance Profiling

### NSight Systems Profiling

```bash
# Profile training run
nsys profile --stats=true python train.py --config config.yaml
```

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("training_step"):
        loss = model.train_step(batch)
        
# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

```python
import torch

def profile_memory():
    """Profile GPU memory usage"""
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Detailed memory breakdown
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
```

## Optimization Strategies

### Batch Size Optimization

```python
def find_optimal_batch_size(model, dataset, max_memory=0.9):
    """Find optimal batch size for given memory constraint"""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        try:
            # Test batch size
            batch = dataset.get_batch(batch_size)
            loss = model(batch)
            loss.backward()
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if memory_used < max_memory:
                return batch_size
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_sizes[batch_sizes.index(batch_size) - 1]
    
    return 1
```

### Learning Rate Scaling

```python
def scale_learning_rate(base_lr, batch_size, base_batch_size=32):
    """Scale learning rate with batch size"""
    return base_lr * (batch_size / base_batch_size) ** 0.5
```

### Gradient Accumulation

```python
def gradient_accumulation_step(model, batch, accumulation_steps=4):
    """Implement gradient accumulation for large effective batch sizes"""
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (model.global_step + 1) % accumulation_steps == 0:
        model.optimizer.step()
        model.optimizer.zero_grad()
```

## Production Scaling Patterns

### Multi-Node Training

```python
# Multi-node configuration
config = {
    "distributed": {
        "backend": "ray",
        "num_nodes": 3,
        "gpus_per_node": 8,
        "world_size": 24
    },
    "training": {
        "batch_size": 192,  # 8 * 24 GPUs
        "gradient_accumulation_steps": 4
    }
}
```

### Model Parallelism

```python
# Configure model parallelism
config = {
    "model": {
        "parallelism": {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 2,
            "sequence_parallel": True
        }
    }
}
```

### Data Parallelism

```python
# Data parallel configuration
config = {
    "distributed": {
        "data_parallel_size": 8,
        "shard_strategy": "ddp"
    }
}
```

## Monitoring and Debugging

### Training Monitoring

```python
import wandb
from nemo_rl.utils.logging import setup_logging

def setup_monitoring():
    """Setup comprehensive training monitoring"""
    wandb.init(project="nemo-rl-training")
    
    # Log performance metrics
    wandb.log({
        "gpu_memory_used": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_cached": torch.cuda.memory_reserved() / 1e9,
        "gpu_utilization": get_gpu_utilization(),
        "training_loss": loss.item()
    })
```

### Performance Debugging

```python
def debug_performance_issues():
    """Debug common performance issues"""
    
    # Check for memory leaks
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Run training step
    loss = model.train_step(batch)
    
    # Check memory after training
    final_memory = torch.cuda.memory_allocated()
    memory_increase = final_memory - initial_memory
    
    if memory_increase > 1e9:  # More than 1GB increase
        print(f"Warning: Large memory increase: {memory_increase / 1e9:.2f} GB")
```

## Best Practices

### 1. Start Small, Scale Up
- Begin with single GPU training
- Gradually increase batch size and model size
- Monitor memory usage carefully

### 2. Profile Early and Often
- Use profiling tools to identify bottlenecks
- Monitor GPU utilization and memory usage
- Profile both training and inference

### 3. Optimize Data Pipeline
- Use efficient data loading
- Implement proper batching strategies
- Minimize CPU-GPU transfers

### 4. Monitor Resource Usage
- Track GPU memory usage
- Monitor network bandwidth in distributed training
- Watch for memory leaks

## Common Issues and Solutions

### Out of Memory Errors

```python
def handle_oom_error():
    """Handle out of memory errors"""
    try:
        # Attempt training step
        loss = model.train_step(batch)
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Reduce batch size
            model.config.batch_size //= 2
            torch.cuda.empty_cache()
            return handle_oom_error()
        else:
            raise e
```

### Slow Training

```python
def optimize_training_speed():
    """Optimize training speed"""
    
    # Use mixed precision
    model = model.half()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Use efficient optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
```

## Next Steps

- Read [Custom Loss Functions](custom-loss-functions) for advanced loss design
- Explore [Model Validation](model-validation) for evaluation frameworks
- Check [Production Deployment](production-deployment) for deployment strategies
- Review [Algorithm Implementation](algorithm-implementation) for custom algorithms 