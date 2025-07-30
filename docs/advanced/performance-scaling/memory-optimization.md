---
description: "Optimize memory usage for large models with gradient checkpointing, mixed precision, and memory-efficient techniques"
tags: ["memory", "optimization", "gradient-checkpointing", "mixed-precision", "large-models"]
categories: ["performance-scaling"]
---

# Memory Optimization

This guide covers how to optimize memory usage for large models in NeMo RL using gradient checkpointing, mixed precision, and other memory-efficient techniques.

## Overview

Training large language models requires significant memory resources. NeMo RL provides several techniques to optimize memory usage while maintaining training efficiency and model quality.

## Key Components

### Gradient Checkpointing

Gradient checkpointing trades computation for memory by recomputing intermediate activations during the backward pass:

```python
import torch
from nemo_rl.models import LargeLanguageModel

class MemoryOptimizedModel(LargeLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Enable gradient checkpointing
        self.enable_gradient_checkpointing()
    
    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory optimization
        """
        # Enable for transformer layers
        for layer in self.transformer_layers:
            if hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = True
        
        # Enable for attention layers
        for layer in self.attention_layers:
            if hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = True
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with gradient checkpointing
        """
        # Use gradient checkpointing for transformer layers
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.transformer_layers:
            if self.training and layer.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        return self.output_layer(hidden_states)
```

### Mixed Precision Training

Use mixed precision training to reduce memory usage and speed up training:

```python
import torch.cuda.amp as amp
from nemo_rl.trainers import BaseTrainer

class MixedPrecisionTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Initialize mixed precision training
        self.setup_mixed_precision()
    
    def setup_mixed_precision(self):
        """
        Setup mixed precision training
        """
        # Initialize gradient scaler
        self.scaler = amp.GradScaler()
        
        # Enable automatic mixed precision
        self.autocast = amp.autocast
        
        # Configure mixed precision settings
        self.mixed_precision_config = {
            'enabled': True,
            'dtype': torch.float16,
            'loss_scale': 2**16,
            'initial_scale': 2**16,
            'growth_factor': 2,
            'backoff_factor': 0.5
        }
    
    def train_step(self, batch):
        """
        Training step with mixed precision
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with self.autocast():
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and optimizer step
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss
    
    def compute_loss(self, outputs, batch):
        """
        Compute loss with mixed precision
        """
        # Ensure loss computation is in float32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(outputs, batch['labels'])
        
        return loss
```

## Configuration

### Memory Optimization Configuration

```yaml
# configs/memory_optimization.yaml
memory_optimization:
  enabled: true
  
  # Gradient checkpointing
  gradient_checkpointing:
    enabled: true
    checkpoint_every_n_layers: 1
    checkpoint_attention: true
    checkpoint_mlp: true
  
  # Mixed precision
  mixed_precision:
    enabled: true
    dtype: float16
    loss_scale: 65536
    initial_scale: 65536
    growth_factor: 2
    backoff_factor: 0.5
  
  # Activation checkpointing
  activation_checkpointing:
    enabled: true
    checkpoint_layers: ["transformer", "attention"]
    
  # Memory efficient attention
  memory_efficient_attention:
    enabled: true
    attention_type: "flash_attention"  # or "xformers"
    
  # CPU offloading
  cpu_offloading:
    enabled: false
    offload_optimizer: false
    offload_parameters: false
```

### Advanced Memory Configuration

```yaml
# configs/advanced_memory.yaml
memory_optimization:
  # Model parallelism
  model_parallelism:
    enabled: true
    tensor_parallel_size: 2
    pipeline_parallel_size: 1
    
  # ZeRO optimization
  zero_optimization:
    enabled: true
    stage: 2  # 1, 2, or 3
    offload_optimizer: false
    offload_parameters: false
    
  # Memory profiling
  memory_profiling:
    enabled: true
    profile_peak_memory: true
    profile_activation_memory: true
```

## Advanced Techniques

### Activation Checkpointing

Implement selective activation checkpointing:

```python
class SelectiveCheckpointing:
    def __init__(self, config):
        self.config = config
        self.checkpoint_layers = config.get('checkpoint_layers', [])
    
    def should_checkpoint(self, layer_name):
        """
        Determine if a layer should use checkpointing
        """
        return any(checkpoint_layer in layer_name for checkpoint_layer in self.checkpoint_layers)
    
    def checkpoint_forward(self, layer, *args, **kwargs):
        """
        Forward pass with selective checkpointing
        """
        if self.should_checkpoint(layer.__class__.__name__):
            return torch.utils.checkpoint.checkpoint(
                layer,
                *args,
                use_reentrant=False,
                **kwargs
            )
        else:
            return layer(*args, **kwargs)
```

### Memory-Efficient Attention

Implement memory-efficient attention mechanisms:

```python
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class MemoryEfficientAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_flash_attention = config.get('use_flash_attention', True)
        self.use_xformers = config.get('use_xformers', False)
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Memory-efficient attention forward pass
        """
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's built-in flash attention
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )
        elif self.use_xformers:
            # Use xFormers for memory efficiency
            import xformers.ops as xops
            return xops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask
            )
        else:
            # Fallback to standard attention
            return self.standard_attention(query, key, value, attention_mask)
    
    def standard_attention(self, query, key, value, attention_mask=None):
        """
        Standard attention implementation
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.size(-1))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output
```

### CPU Offloading

Implement CPU offloading for extreme memory constraints:

```python
class CPUOffloadingTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Setup CPU offloading
        self.setup_cpu_offloading()
    
    def setup_cpu_offloading(self):
        """
        Setup CPU offloading for memory optimization
        """
        self.offload_optimizer = self.config.get('offload_optimizer', False)
        self.offload_parameters = self.config.get('offload_parameters', False)
        
        if self.offload_optimizer:
            # Offload optimizer states to CPU
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            
            # Move optimizer states to CPU
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad.data = param.grad.data.cpu()
    
    def train_step(self, batch):
        """
        Training step with CPU offloading
        """
        # Move batch to GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Move gradients to CPU if offloading
        if self.offload_optimizer:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss
```

## Memory Monitoring

### Memory Profiling

Implement comprehensive memory profiling:

```python
import psutil
import GPUtil
from memory_profiler import profile

class MemoryProfiler:
    def __init__(self):
        self.memory_stats = {}
    
    def profile_memory_usage(self):
        """
        Profile current memory usage
        """
        # GPU memory
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            gpu_memory[f'gpu_{i}_allocated'] = torch.cuda.memory_allocated(i) / 1024**3
            gpu_memory[f'gpu_{i}_cached'] = torch.cuda.memory_reserved(i) / 1024**3
        
        # CPU memory
        cpu_memory = {
            'cpu_used_gb': psutil.virtual_memory().used / 1024**3,
            'cpu_available_gb': psutil.virtual_memory().available / 1024**3,
            'cpu_percent': psutil.virtual_memory().percent
        }
        
        # Model memory
        model_memory = self.get_model_memory_usage()
        
        self.memory_stats = {
            'gpu': gpu_memory,
            'cpu': cpu_memory,
            'model': model_memory
        }
        
        return self.memory_stats
    
    def get_model_memory_usage(self):
        """
        Get model-specific memory usage
        """
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            param_size = param.numel() * param.element_size()
            total_params += param_size
            
            if param.requires_grad:
                trainable_params += param_size
        
        return {
            'total_params_mb': total_params / 1024**2,
            'trainable_params_mb': trainable_params / 1024**2,
            'model_size_mb': total_params / 1024**2
        }
    
    def log_memory_usage(self):
        """
        Log memory usage statistics
        """
        stats = self.profile_memory_usage()
        
        print("=== Memory Usage Statistics ===")
        print(f"GPU Memory: {stats['gpu']}")
        print(f"CPU Memory: {stats['cpu']}")
        print(f"Model Memory: {stats['model']}")
        print("===============================")
```

### Memory Optimization Monitoring

Monitor the effectiveness of memory optimizations:

```python
class MemoryOptimizationMonitor:
    def __init__(self, config):
        self.config = config
        self.baseline_memory = None
        self.optimized_memory = None
    
    def measure_optimization_effectiveness(self):
        """
        Measure the effectiveness of memory optimizations
        """
        if self.baseline_memory is None:
            print("Warning: Baseline memory not set")
            return
        
        if self.optimized_memory is None:
            print("Warning: Optimized memory not measured")
            return
        
        # Calculate memory savings
        gpu_savings = {}
        for gpu_id in range(torch.cuda.device_count()):
            baseline = self.baseline_memory['gpu'][f'gpu_{gpu_id}_allocated']
            optimized = self.optimized_memory['gpu'][f'gpu_{gpu_id}_allocated']
            savings = (baseline - optimized) / baseline * 100
            gpu_savings[f'gpu_{gpu_id}_savings_percent'] = savings
        
        # Calculate model memory savings
        model_savings = {}
        baseline_model = self.baseline_memory['model']['model_size_mb']
        optimized_model = self.optimized_memory['model']['model_size_mb']
        model_savings['model_size_savings_percent'] = (baseline_model - optimized_model) / baseline_model * 100
        
        return {
            'gpu_savings': gpu_savings,
            'model_savings': model_savings
        }
```

## Best Practices

### 1. Gradual Memory Optimization

Start with basic optimizations and gradually add more advanced techniques:

```python
def apply_memory_optimizations(self, model, config):
    """
    Apply memory optimizations gradually
    """
    optimizations = []
    
    # Level 1: Basic optimizations
    if config.get('enable_gradient_checkpointing', False):
        self.enable_gradient_checkpointing(model)
        optimizations.append('gradient_checkpointing')
    
    # Level 2: Mixed precision
    if config.get('enable_mixed_precision', False):
        self.enable_mixed_precision(model)
        optimizations.append('mixed_precision')
    
    # Level 3: Advanced optimizations
    if config.get('enable_memory_efficient_attention', False):
        self.enable_memory_efficient_attention(model)
        optimizations.append('memory_efficient_attention')
    
    # Level 4: Extreme optimizations
    if config.get('enable_cpu_offloading', False):
        self.enable_cpu_offloading(model)
        optimizations.append('cpu_offloading')
    
    return optimizations
```

### 2. Memory Budget Planning

Plan memory usage based on model size and available resources:

```python
def calculate_memory_budget(self, model_size_gb, available_gpu_memory_gb):
    """
    Calculate memory budget for training
    """
    # Model parameters (typically 4 bytes per parameter)
    model_memory = model_size_gb * 4
    
    # Optimizer states (typically 8 bytes per parameter for Adam)
    optimizer_memory = model_size_gb * 8
    
    # Activations (depends on batch size and sequence length)
    activation_memory = model_size_gb * 2  # Conservative estimate
    
    # Gradient memory (typically 4 bytes per parameter)
    gradient_memory = model_size_gb * 4
    
    # Total estimated memory
    total_memory = model_memory + optimizer_memory + activation_memory + gradient_memory
    
    # Check if we need optimizations
    if total_memory > available_gpu_memory_gb:
        required_optimizations = self.calculate_required_optimizations(
            total_memory, available_gpu_memory_gb
        )
        return required_optimizations
    else:
        return ['no_optimization_needed']
```

### 3. Dynamic Memory Management

Implement dynamic memory management based on training progress:

```python
class DynamicMemoryManager:
    def __init__(self, config):
        self.config = config
        self.memory_threshold = config.get('memory_threshold', 0.9)
        self.optimization_level = 0
    
    def check_memory_usage(self):
        """
        Check current memory usage and apply optimizations if needed
        """
        current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        if current_memory > self.memory_threshold:
            self.increase_optimization_level()
        elif current_memory < self.memory_threshold * 0.7:
            self.decrease_optimization_level()
    
    def increase_optimization_level(self):
        """
        Increase memory optimization level
        """
        if self.optimization_level == 0:
            # Enable gradient checkpointing
            self.enable_gradient_checkpointing()
            self.optimization_level = 1
        elif self.optimization_level == 1:
            # Enable mixed precision
            self.enable_mixed_precision()
            self.optimization_level = 2
        elif self.optimization_level == 2:
            # Enable CPU offloading
            self.enable_cpu_offloading()
            self.optimization_level = 3
```

## Troubleshooting

### Common Memory Issues

1. **Out of Memory (OOM)**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Balance memory optimization with training speed
3. **Numerical Instability**: Use mixed precision carefully and monitor loss values

### Debugging Tips

```python
# Add debugging to memory optimization
def debug_memory_optimization(self):
    """
    Debug memory optimization issues
    """
    print("=== Memory Optimization Debug ===")
    
    # Check if optimizations are enabled
    print(f"Gradient checkpointing: {self.model.gradient_checkpointing}")
    print(f"Mixed precision: {hasattr(self, 'scaler')}")
    print(f"CPU offloading: {hasattr(self, 'offload_optimizer')}")
    
    # Check memory usage
    memory_usage = torch.cuda.memory_allocated() / 1024**3
    print(f"Current GPU memory usage: {memory_usage:.2f} GB")
    
    # Check for memory leaks
    torch.cuda.empty_cache()
    memory_after_clear = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after cache clear: {memory_after_clear:.2f} GB")
    
    print("================================")
```

## Next Steps

- Learn about [Distributed Training](distributed-training) for scaling across multiple GPUs
- Review [Production Deployment](../production-deployment/index) for deployment strategies
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 