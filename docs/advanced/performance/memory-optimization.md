# Memory Optimization

This guide covers memory optimization techniques for NeMo RL training, including gradient checkpointing, activation checkpointing, and memory-efficient training strategies.

## Overview

Memory optimization is crucial for training large language models efficiently. NeMo RL provides several techniques to reduce memory usage while maintaining training quality.

## Key Techniques

### Gradient Checkpointing

Gradient checkpointing trades compute for memory by recomputing activations during the backward pass:

```python
# Enable gradient checkpointing
policy.gradient_checkpointing = True

# Configure checkpointing frequency
policy.checkpoint_every = 4  # Checkpoint every 4 layers
```

**Benefits:**
- Reduces memory usage by ~50%
- Enables training larger models
- Maintains training quality

**Trade-offs:**
- Increases computation time by ~20%
- Requires more forward passes

### Activation Checkpointing

Activation checkpointing saves memory by storing only essential activations:

```python
# Enable activation checkpointing
policy.activation_checkpointing = True

# Configure which layers to checkpoint
policy.checkpoint_layers = [0, 4, 8, 12]  # Checkpoint specific layers
```

### Mixed Precision Training

Use lower precision to reduce memory usage:

```python
# Enable mixed precision
policy.mixed_precision = True
policy.precision = "bfloat16"  # or "float16"

# Configure loss scaling
policy.loss_scaling = 2**15
```

## Memory Management Strategies

### Dynamic Memory Allocation

Optimize memory allocation patterns:

```python
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use memory-efficient data types
dtype = torch.bfloat16 if policy.mixed_precision else torch.float32
```

### Memory Pinning

Optimize CPU-GPU memory transfers:

```python
# Pin memory for faster transfers
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=4
)
```

### Gradient Accumulation

Use gradient accumulation to simulate larger batch sizes:

```python
# Configure gradient accumulation
policy.gradient_accumulation_steps = 4

# Effective batch size = batch_size * accumulation_steps
effective_batch_size = batch_size * gradient_accumulation_steps
```

## Model-Specific Optimizations

### Large Model Training

For models with billions of parameters:

```python
# Use model parallelism
policy.model_parallel = True
policy.tensor_parallel_size = 2

# Enable pipeline parallelism
policy.pipeline_parallel_size = 4

# Use CPU offloading for optimizer states
policy.offload_optimizer = True
```

### Attention Optimization

Optimize attention mechanisms for memory efficiency:

```python
# Use flash attention
policy.use_flash_attention = True

# Configure attention heads
policy.attention_heads = 16
policy.attention_head_dim = 64
```

## Memory Monitoring

### Memory Profiling

Monitor memory usage during training:

```python
import torch

def monitor_memory():
    """Monitor GPU memory usage."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    
    return {
        "allocated_mb": allocated / 1024**2,
        "reserved_mb": reserved / 1024**2,
        "max_allocated_mb": max_allocated / 1024**2
    }
```

### Memory Leaks

Detect and fix memory leaks:

```python
# Track memory usage over time
memory_history = []

def track_memory():
    """Track memory usage over time."""
    memory = monitor_memory()
    memory_history.append(memory)
    
    # Check for memory leaks
    if len(memory_history) > 10:
        recent_avg = sum(m['allocated_mb'] for m in memory_history[-10:]) / 10
        if recent_avg > memory_history[0]['allocated_mb'] * 1.5:
            print("Potential memory leak detected!")
```

## Configuration Examples

### Memory-Efficient Training

```yaml
policy:
  # Memory optimization
  gradient_checkpointing: true
  activation_checkpointing: true
  mixed_precision: true
  precision: "bfloat16"
  
  # Gradient accumulation
  gradient_accumulation_steps: 4
  
  # Model parallelism
  model_parallel: true
  tensor_parallel_size: 2
  
  # Memory management
  offload_optimizer: true
  pin_memory: true
```

### Large Model Configuration

```yaml
policy:
  # Advanced memory optimization
  gradient_checkpointing: true
  checkpoint_every: 2
  
  # Mixed precision
  mixed_precision: true
  precision: "bfloat16"
  loss_scaling: 32768
  
  # Model parallelism
  model_parallel: true
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  
  # Memory offloading
  offload_optimizer: true
  offload_activations: true
```

## Best Practices

### Memory Planning

1. **Calculate Requirements**
   - Model parameters and gradients
   - Activations and intermediate states
   - Optimizer states and buffers

2. **Optimize Configuration**
   - Balance memory and computation
   - Use appropriate precision
   - Enable relevant optimizations

3. **Monitor Usage**
   - Track memory over time
   - Identify bottlenecks
   - Prevent memory leaks

### Optimization Strategies

1. **Gradient Checkpointing**
   - Enable for large models
   - Configure appropriate frequency
   - Monitor performance impact

2. **Mixed Precision**
   - Use bfloat16 for stability
   - Configure loss scaling
   - Monitor training stability

3. **Model Parallelism**
   - Split large models
   - Optimize communication
   - Balance load across devices

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Instability**
   - Adjust loss scaling
   - Check gradient clipping
   - Monitor loss values

3. **Performance Degradation**
   - Balance memory and computation
   - Profile memory usage
   - Optimize data loading

### Debug Commands

```bash
# Monitor GPU memory
nvidia-smi

# Profile memory usage
python -m memory_profiler script.py

# Check for memory leaks
python -m tracemalloc script.py
```

## Performance Impact

### Memory vs. Speed Trade-offs

| Technique | Memory Reduction | Speed Impact |
|-----------|-----------------|--------------|
| Gradient Checkpointing | ~50% | ~20% slower |
| Mixed Precision | ~50% | ~10% faster |
| Activation Checkpointing | ~30% | ~15% slower |
| Model Parallelism | Variable | Communication overhead |

### Optimization Guidelines

1. **Start Conservative**
   - Begin with basic optimizations
   - Monitor performance impact
   - Gradually enable advanced features

2. **Profile First**
   - Measure baseline performance
   - Identify bottlenecks
   - Target specific optimizations

3. **Iterate and Optimize**
   - Test different configurations
   - Monitor training stability
   - Balance memory and speed

## Next Steps

After implementing memory optimizations:

1. **Monitor Performance**: Track memory usage and training speed
2. **Tune Configuration**: Adjust settings based on results
3. **Scale Up**: Apply optimizations to larger models
4. **Profile Regularly**: Monitor for memory leaks and inefficiencies

For more advanced topics, see:
- [Distributed Training](distributed-training.md) - Multi-GPU training strategies
- [Mixed Precision](mixed-precision.md) - Mixed precision training techniques
- [Performance Profiling](profiling.md) - Profiling tools and techniques 