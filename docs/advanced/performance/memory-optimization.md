---
description: "Optimize memory usage for large models with CPU offloading, model management, and memory-efficient techniques"
tags: ["memory", "optimization", "cpu-offload", "model-management", "large-models"]
categories: ["performance"]
---

# Memory Optimization

This guide covers how to optimize memory usage for large models in NeMo RL using CPU offloading, model management, and other memory-efficient techniques.

## Overview

Training large language models requires significant memory resources. NeMo RL provides several techniques to optimize memory usage while maintaining training efficiency and model quality.

## Key Components

### CPU Offloading

NeMo RL uses CPU offloading to manage memory during training:

```python
# Real NeMo RL memory management patterns
from nemo_rl.models.policy import DTensorPolicyWorker

class MemoryOptimizedTraining:
    def __init__(self, worker: DTensorPolicyWorker):
        self.worker = worker
    
    def optimize_memory_usage(self):
        """
        Optimize memory usage using NeMo RL patterns
        """
        # Offload optimizer state to CPU
        self.worker.offload_before_refit()
        
        # Offload model to CPU after training
        self.worker.offload_after_refit()
        
        # Move model to CPU for memory efficiency
        self.worker.move_to_cpu(self.worker.model)
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

### Model Device Management

Manage model placement for optimal memory usage:

```python
def manage_model_memory(worker: DTensorPolicyWorker):
    """
    Manage model memory using NeMo RL patterns
    """
    # Move model to CPU to free GPU memory
    worker.model = worker.move_to_cpu(worker.model)
    
    # Move model back to GPU when needed
    worker.model = worker.move_to_cuda(worker.model)
    
    # Move buffers to device (required for FSDP)
    worker.model = worker.move_buffer_to_device(worker.model, "cuda")
```

### Memory Monitoring

Monitor memory usage during training:

```python
def monitor_memory_usage(worker: DTensorPolicyWorker):
    """
    Monitor memory usage using NeMo RL patterns
    """
    # Get GPU information
    gpu_info = worker.get_gpu_info()
    
    # Reset peak memory stats
    worker.reset_peak_memory_stats()
    
    # Get current memory usage
    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
    
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'gpu_info': gpu_info
    }
```

## Configuration

### Memory Optimization Configuration

Configure memory optimization in your YAML config:

```yaml
# Real NeMo RL memory optimization config
policy:
  precision: "bfloat16"  # Use lower precision
  
  dtensor_cfg:
    enabled: true
    cpu_offload: true           # Enable CPU offloading
    activation_checkpointing: false  # Disable for memory
    tensor_parallel_size: 4     # Distribute across GPUs
    context_parallel_size: 1
    
  # Memory-efficient settings
  train_micro_batch_size: 1    # Small batch size
  logprob_batch_size: 4        # Small inference batch
  max_total_sequence_length: 8192  # Limit sequence length
```

### Advanced Memory Configuration

```yaml
# Advanced memory optimization
policy:
  dtensor_cfg:
    enabled: true
    cpu_offload: true
    sequence_parallel: false    # Disable for memory
    activation_checkpointing: false
    tensor_parallel_size: 4
    context_parallel_size: 1
    
  # Optimizer settings for memory efficiency
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      foreach: False  # Required for DTensor
      fused: False    # Required for DTensor
      weight_decay: 0.01
```

## Best Practices

### 1. CPU Offloading Strategy

```python
def implement_cpu_offloading(worker: DTensorPolicyWorker):
    """
    Implement CPU offloading strategy
    """
    # Before training step
    worker.offload_before_refit()
    
    # Perform training
    results = worker.train(data, loss_fn)
    
    # After training step
    worker.offload_after_refit()
    
    return results
```

### 2. Model State Management

```python
def manage_model_state(worker: DTensorPolicyWorker):
    """
    Manage model state for memory efficiency
    """
    # Prepare for inference
    worker.prepare_for_lp_inference()
    
    # Perform inference
    logprobs = worker.get_logprobs(data)
    
    # Prepare for training
    worker.prepare_for_training()
    
    return logprobs
```

### 3. Memory-Efficient Training Loop

```python
def memory_efficient_training(worker: DTensorPolicyWorker, data, loss_fn):
    """
    Memory-efficient training loop
    """
    # Monitor initial memory
    initial_memory = monitor_memory_usage(worker)
    
    # Offload before training
    worker.offload_before_refit()
    
    # Training step
    results = worker.train(data, loss_fn)
    
    # Offload after training
    worker.offload_after_refit()
    
    # Monitor final memory
    final_memory = monitor_memory_usage(worker)
    
    return results, initial_memory, final_memory
```

## Troubleshooting

### Common Memory Issues

1. **Out of Memory Errors**
   ```python
   # Solution: Reduce batch size and enable CPU offload
   policy:
     train_micro_batch_size: 1
     dtensor_cfg:
       cpu_offload: true
   ```

2. **Slow Training Due to CPU Offloading**
   ```python
   # Solution: Balance CPU offload with performance
   policy:
     dtensor_cfg:
       cpu_offload: false  # Disable if too slow
       tensor_parallel_size: 4  # Use more GPUs instead
   ```

3. **Memory Fragmentation**
   ```python
   # Solution: Regular memory cleanup
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

### Debugging Memory Issues

```python
def debug_memory_issues(worker: DTensorPolicyWorker):
    """
    Debug memory issues
    """
    # Get detailed GPU info
    gpu_info = worker.get_gpu_info()
    print(f"GPU Info: {gpu_info}")
    
    # Monitor memory at different stages
    print("Memory before training:")
    monitor_memory_usage(worker)
    
    # Perform training
    worker.offload_before_refit()
    results = worker.train(data, loss_fn)
    worker.offload_after_refit()
    
    print("Memory after training:")
    monitor_memory_usage(worker)
    
    return results
```

## Next Steps

After implementing memory optimization:

1. **Monitor Performance**: Track memory usage and training speed
2. **Optimize Configuration**: Adjust settings based on your hardware
3. **Scale Up**: Use distributed training for larger models
4. **Profile Memory**: Use profiling tools to identify bottlenecks

For more advanced topics, see:
- [Distributed Training](distributed-training.md) - Scale across multiple GPUs
- [Performance Profiling](profiling.md) - Profile memory usage
- [Mixed Precision](mixed-precision.md) - Use lower precision training 