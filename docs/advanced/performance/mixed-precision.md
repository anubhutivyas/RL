# Mixed Precision Training

This guide covers mixed precision training techniques for NeMo RL, including FP16 and BF16 training strategies.

## Overview

Mixed precision training uses lower precision data types (FP16/BF16) to reduce memory usage and increase training speed while maintaining numerical stability.

## Key Concepts

### Precision Types

NeMo RL supports multiple precision types:

```python
# FP32 (Full Precision)
dtype = torch.float32

# FP16 (Half Precision)
dtype = torch.float16

# BF16 (Brain Float 16)
dtype = torch.bfloat16
```

### Precision Comparison

| Precision | Bits | Range | Precision | Memory |
|-----------|------|-------|-----------|--------|
| FP32 | 32 | ±3.4e38 | 7 digits | 100% |
| FP16 | 16 | ±65,504 | 3 digits | 50% |
| BF16 | 16 | ±3.4e38 | 3 digits | 50% |

## Configuration

### Basic Mixed Precision

Enable mixed precision training in your configuration:

```yaml
policy:
  precision: "bfloat16"  # or "float16" or "float32"
```

### Loss Scaling

Loss scaling is automatically handled by PyTorch's autocast and GradScaler:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop with automatic loss scaling
for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Training Strategies

### Automatic Mixed Precision (AMP)

Use PyTorch's AMP for automatic precision management:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop with AMP
for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Manual Precision Control

Manually control precision for specific operations:

```python
# Cast specific tensors to lower precision
input_ids = input_ids.to(torch.bfloat16)
attention_mask = attention_mask.to(torch.bfloat16)

# Keep sensitive operations in FP32
logits = model(input_ids, attention_mask)
loss = loss_fn(logits, labels)  # Keep in FP32
```

## Stability Techniques

### Gradient Clipping

Prevent gradient explosion in mixed precision:

```python
# Configure gradient clipping
policy.gradient_clipping = True
policy.max_grad_norm = 1.0

# Clip gradients before optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Loss Scaling

Prevent loss underflow in FP16:

```python
# Scale loss before backward pass
scaled_loss = loss * loss_scale

# Unscale gradients after backward pass
scaled_loss.backward()
for param in model.parameters():
    if param.grad is not None:
        param.grad.data /= loss_scale
```

## Configuration Examples

### BF16 Training (Recommended)

```yaml
policy:
  # Mixed precision configuration
  precision: "bfloat16"
  
  # Stability settings
  max_grad_norm: 1.0
```

### FP16 Training

```yaml
policy:
  # Mixed precision configuration
  precision: "float16"
  
  # Enhanced stability for FP16
  max_grad_norm: 0.5
```

## Performance Benefits

### Memory Reduction

Mixed precision reduces memory usage for model parameters and activations:

```python
# Memory comparison for parameters
fp32_memory = model_size * 4  # 4 bytes per parameter
fp16_memory = model_size * 2  # 2 bytes per parameter
bf16_memory = model_size * 2  # 2 bytes per parameter

# Memory reduction (varies by model and batch size)
memory_reduction = (fp32_memory - fp16_memory) / fp32_memory * 100
# Result: ~50% memory reduction for parameters
```

### Speed Improvements

Mixed precision can improve training speed (actual improvements depend on hardware and model size):

```python
# Typical speed improvements (varies by hardware)
fp32_time = baseline_training_time
fp16_time = fp32_time * 0.8  # ~20% faster on supported hardware
bf16_time = fp32_time * 0.85  # ~15% faster on supported hardware
```

## Monitoring and Debugging

### Loss Scaling Monitoring

Monitor loss scaling during training:

```python
def monitor_loss_scaling():
    """Monitor loss scaling during training."""
    current_scale = scaler.get_scale()
    scale_history.append(current_scale)
    
    # Check for scale changes
    if len(scale_history) > 1:
        if scale_history[-1] != scale_history[-2]:
            print(f"Loss scale changed to: {current_scale}")
```

### Numerical Stability

Monitor for numerical issues:

```python
def check_numerical_stability():
    """Check for numerical stability issues."""
    # Check for NaN or Inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name}")
        if torch.isinf(param).any():
            print(f"Inf detected in {name}")
    
    # Check loss values
    if torch.isnan(loss) or torch.isinf(loss):
        print("Loss is NaN or Inf!")
```

## Best Practices

### Choosing Precision

1. **BF16 (Recommended)**
   - Better numerical stability
   - Wider dynamic range
   - Good for most models

2. **FP16**
   - Faster on some hardware
   - Requires careful tuning
   - May need more loss scaling

3. **FP32**
   - Maximum stability
   - Higher memory usage
   - Slower training

### Configuration Guidelines

1. **Start Conservative**
   - Begin with BF16
   - Use moderate loss scaling
   - Monitor training stability

2. **Gradual Optimization**
   - Test different configurations
   - Monitor performance impact
   - Balance speed and stability

3. **Monitor Closely**
   - Watch for NaN/Inf values
   - Track loss scaling changes
   - Monitor training metrics

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Increase loss scaling
   - Use BF16 instead of FP16
   - Reduce learning rate

2. **Loss Explosion**
   - Decrease loss scaling
   - Enable gradient clipping
   - Check for NaN values

3. **Poor Convergence**
   - Disable mixed precision temporarily
   - Check numerical stability
   - Verify loss scaling

### Debug Commands

```bash
# Monitor GPU utilization
nvidia-smi

# Check for NaN values
python -c "import torch; print(torch.isnan(tensor).any())"

# Profile memory usage
python -m memory_profiler script.py
```

## Advanced Techniques

### Selective Precision

Use different precision for different parts:

```python
# Keep embeddings in FP32 for stability
model.embeddings = model.embeddings.to(torch.float32)

# Use mixed precision for transformer layers
for layer in model.transformer.layers:
    layer = layer.to(torch.bfloat16)
```

### Dynamic Precision

Adjust precision based on training progress:

```python
def adjust_precision(epoch):
    """Adjust precision based on training epoch."""
    if epoch < 10:
        # Start with FP32 for stability
        return torch.float32
    elif epoch < 50:
        # Switch to BF16
        return torch.bfloat16
    else:
        # Use FP16 for speed
        return torch.float16
```

## Performance Comparison

### Training Speed

| Configuration | Speed | Memory | Stability |
|---------------|-------|--------|-----------|
| FP32 | 1.0x | 100% | Excellent |
| BF16 | 1.15x | ~50% | Good |
| FP16 | 1.2x | ~50% | Fair |

*Note: Actual performance varies by hardware and model size*

### Memory Usage

| Component | FP32 | BF16 | FP16 |
|-----------|------|------|------|
| Model Parameters | 100% | ~50% | ~50% |
| Gradients | 100% | ~50% | ~50% |
| Optimizer States | 100% | ~50% | ~50% |
| Activations | 100% | ~50% | ~50% |

*Note: Actual memory usage depends on model architecture and batch size*

## Performance Considerations

**Note**: The performance improvements and memory reductions mentioned in this guide are estimates based on typical usage patterns. Actual results may vary significantly depending on:
- Hardware architecture (GPU model, tensor cores, memory bandwidth)
- Model size and architecture
- Batch size and sequence length
- Training workload characteristics
- Software stack versions

## Next Steps

After implementing mixed precision training:

1. **Monitor Stability**: Watch for numerical issues
2. **Optimize Configuration**: Tune loss scaling and clipping
3. **Scale Up**: Apply to larger models
4. **Profile Performance**: Measure speed and memory improvements

For more advanced topics, see:
- [Memory Optimization](memory-optimization.md) - Memory management techniques
- [Distributed Training](distributed-training.md) - Multi-GPU training strategies
- [Performance Profiling](profiling.md) - Profiling tools and techniques 