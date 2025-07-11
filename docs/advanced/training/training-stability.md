# Training Stability

This guide covers techniques for ensuring stable training in NeMo RL, including gradient management, loss scaling, and monitoring strategies to prevent training divergence and improve convergence.

## Overview

Training stability is crucial for successful reinforcement learning with large language models. Unstable training can lead to gradient explosion, loss divergence, and poor model performance. This guide provides comprehensive strategies for maintaining stable training throughout the learning process.

## Common Stability Issues

### 1. Gradient Explosion

Gradient explosion occurs when gradients become extremely large, causing parameter updates to be unstable:

```python
import torch
import torch.nn.functional as F

class GradientExplosionDetector:
    def __init__(self, model, max_grad_norm=1.0):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.gradient_history = []
    
    def check_gradients(self):
        """Check for gradient explosion."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        self.gradient_history.append(total_norm)
        
        if total_norm > self.max_grad_norm * 10:
            print(f"Warning: Gradient explosion detected! Norm: {total_norm}")
            return True
        return False
    
    def clip_gradients(self):
        """Clip gradients to prevent explosion."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```

### 2. Loss Divergence

Loss divergence occurs when the loss function becomes unstable:

```python
class LossStabilityMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.loss_history = []
        self.divergence_threshold = 10.0
    
    def check_loss_stability(self, loss):
        """Check if loss is diverging."""
        self.loss_history.append(loss.item())
        
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_std = torch.std(torch.tensor(recent_losses))
            loss_mean = torch.mean(torch.tensor(recent_losses))
            
            # Check for divergence
            if loss_std > self.divergence_threshold or torch.isnan(loss_mean):
                print(f"Warning: Loss divergence detected! Mean: {loss_mean}, Std: {loss_std}")
                return True
        
        return False
```

### 3. Learning Rate Issues

Improper learning rates can cause training instability:

```python
class LearningRateMonitor:
    def __init__(self, initial_lr, min_lr=1e-8, max_lr=1e-2):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
    
    def check_lr_stability(self, optimizer):
        """Check learning rate stability."""
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr < self.min_lr:
            print(f"Warning: Learning rate too small: {current_lr}")
            return False
        elif current_lr > self.max_lr:
            print(f"Warning: Learning rate too large: {current_lr}")
            return False
        
        return True
    
    def adjust_lr(self, optimizer, factor=0.5):
        """Adjust learning rate for stability."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * factor, self.min_lr)
```

## Stability Techniques

### 1. Gradient Clipping

Implement gradient clipping to prevent gradient explosion:

```python
class StableTrainer:
    def __init__(self, model, optimizer, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.gradient_monitor = GradientExplosionDetector(model, max_grad_norm)
    
    def training_step(self, batch):
        """Perform a stable training step."""
        # Forward pass
        outputs = self.model(batch['inputs'])
        loss = F.cross_entropy(outputs, batch['targets'])
        
        # Backward pass
        loss.backward()
        
        # Check for gradient explosion
        if self.gradient_monitor.check_gradients():
            # Clip gradients
            self.gradient_monitor.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### 2. Loss Scaling

Use loss scaling for mixed precision training:

```python
class LossScaler:
    def __init__(self, initial_scale=2**16, growth_factor=2, backoff_factor=0.5):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_tracker = 0
        self.backoff_tracker = 0
    
    def scale_loss(self, loss):
        """Scale loss for mixed precision training."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients after backward pass."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def update_scale(self, overflow):
        """Update loss scale based on gradient overflow."""
        if overflow:
            self.scale *= self.backoff_factor
            self.backoff_tracker += 1
            self.growth_tracker = 0
        else:
            self.backoff_tracker = 0
            self.growth_tracker += 1
            
            if self.growth_tracker >= 2000:
                self.scale *= self.growth_factor
                self.growth_tracker = 0
```

### 3. Learning Rate Scheduling

Implement stable learning rate scheduling:

```python
class StableLearningRateScheduler:
    def __init__(self, optimizer, initial_lr, warmup_steps=1000, decay_steps=10000):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            lr = self.initial_lr * 0.5 * (1 + torch.cos(torch.pi * progress))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 4. Weight Initialization

Use stable weight initialization:

```python
def stable_weight_init(model):
    """Initialize model weights for stability."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
```

## Monitoring and Early Warning

### 1. Training Monitor

```python
class TrainingStabilityMonitor:
    def __init__(self):
        self.loss_monitor = LossStabilityMonitor()
        self.lr_monitor = LearningRateMonitor(1e-5)
        self.gradient_monitor = GradientExplosionDetector(None, 1.0)
        self.warnings = []
    
    def check_stability(self, loss, model, optimizer):
        """Check overall training stability."""
        stability_issues = []
        
        # Check loss stability
        if self.loss_monitor.check_loss_stability(loss):
            stability_issues.append("Loss divergence detected")
        
        # Check learning rate
        if not self.lr_monitor.check_lr_stability(optimizer):
            stability_issues.append("Learning rate instability")
        
        # Check gradients
        if self.gradient_monitor.check_gradients():
            stability_issues.append("Gradient explosion detected")
        
        # Log warnings
        for issue in stability_issues:
            if issue not in self.warnings:
                print(f"Stability Warning: {issue}")
                self.warnings.append(issue)
        
        return len(stability_issues) == 0
```

### 2. Automatic Recovery

```python
class AutoRecoveryTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.monitor = TrainingStabilityMonitor()
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def training_step(self, batch):
        """Training step with automatic recovery."""
        try:
            # Normal training step
            loss = self._compute_loss(batch)
            loss.backward()
            
            # Check stability
            if not self.monitor.check_stability(loss, self.model, self.optimizer):
                self._attempt_recovery()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            print(f"Training error: {e}")
            self._attempt_recovery()
            return None
    
    def _attempt_recovery(self):
        """Attempt to recover from instability."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            raise RuntimeError("Max recovery attempts exceeded")
        
        self.recovery_attempts += 1
        print(f"Attempting recovery {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Reset optimizer
        self.optimizer.zero_grad()
```

## Advanced Stability Techniques

### 1. Gradient Accumulation

Use gradient accumulation for stable training with large models:

```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def training_step(self, batch):
        """Training step with gradient accumulation."""
        # Forward pass
        outputs = self.model(batch['inputs'])
        loss = F.cross_entropy(outputs, batch['targets'])
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Update weights every accumulation_steps
        if self.current_step % self.accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
```

### 2. Mixed Precision Training

Implement stable mixed precision training:

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def training_step(self, batch):
        """Mixed precision training step."""
        # Forward pass with autocast
        with autocast():
            outputs = self.model(batch['inputs'])
            loss = F.cross_entropy(outputs, batch['targets'])
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### 3. Layer-wise Learning Rates

Use different learning rates for different layers:

```python
class LayerWiseOptimizer:
    def __init__(self, model, base_lr=1e-5):
        self.model = model
        self.base_lr = base_lr
        
        # Group parameters by layer type
        param_groups = []
        
        # Embedding layers
        embedding_params = []
        for name, param in model.named_parameters():
            if 'embed' in name:
                embedding_params.append(param)
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': base_lr * 0.1  # Lower LR for embeddings
            })
        
        # Attention layers
        attention_params = []
        for name, param in model.named_parameters():
            if 'attention' in name:
                attention_params.append(param)
        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': base_lr * 2.0  # Higher LR for attention
            })
        
        # Other layers
        other_params = []
        for name, param in model.named_parameters():
            if 'embed' not in name and 'attention' not in name:
                other_params.append(param)
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr
            })
        
        self.optimizer = torch.optim.AdamW(param_groups)
```

## Configuration for Stability

### 1. Stable Configuration

```yaml
# stable_training.yaml
training:
  # Gradient management
  max_grad_norm: 1.0
  gradient_clipping: true
  gradient_accumulation_steps: 4
  
  # Learning rate
  learning_rate: 1e-5
  warmup_steps: 1000
  lr_scheduler: "cosine"
  
  # Mixed precision
  mixed_precision: "bf16"
  autocast: true
  
  # Stability monitoring
  stability_monitoring: true
  early_stopping: true
  patience: 10
  
  # Recovery
  auto_recovery: true
  max_recovery_attempts: 3
```

### 2. Monitoring Configuration

```yaml
monitoring:
  # Loss monitoring
  loss_window_size: 100
  divergence_threshold: 10.0
  
  # Gradient monitoring
  gradient_history_size: 1000
  explosion_threshold: 10.0
  
  # Learning rate monitoring
  lr_min: 1e-8
  lr_max: 1e-2
  
  # Logging
  log_frequency: 100
  save_checkpoints: true
  checkpoint_frequency: 1000
```

## Best Practices

### 1. Initialization

1. **Use stable weight initialization** for all layers
2. **Start with conservative learning rates** and increase gradually
3. **Use proper data normalization** to prevent input instability
4. **Initialize bias terms to zero** for most layers

### 2. Training Process

1. **Monitor gradients** continuously during training
2. **Use gradient clipping** to prevent explosion
3. **Implement learning rate scheduling** for stable convergence
4. **Use mixed precision** carefully with proper scaling

### 3. Monitoring

1. **Track loss trends** and detect divergence early
2. **Monitor gradient norms** and distributions
3. **Check learning rate values** throughout training
4. **Log stability metrics** for analysis

### 4. Recovery

1. **Implement automatic recovery** mechanisms
2. **Use checkpointing** to resume from stable states
3. **Adjust hyperparameters** when instability is detected
4. **Have fallback strategies** for critical failures

## Troubleshooting

### Common Issues

1. **Loss Explosion**
   ```python
   # Reduce learning rate
   for param_group in optimizer.param_groups:
       param_group['lr'] *= 0.1
   
   # Increase gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
   ```

2. **Gradient Vanishing**
   ```python
   # Use gradient accumulation
   accumulation_steps = 8
   
   # Use layer-wise learning rates
   param_groups = [
       {'params': model.embeddings.parameters(), 'lr': 1e-4},
       {'params': model.transformer.parameters(), 'lr': 1e-5}
   ]
   ```

3. **Learning Rate Issues**
   ```python
   # Implement warmup
   warmup_steps = 1000
   if step < warmup_steps:
       lr = base_lr * (step / warmup_steps)
   
   # Use cosine scheduling
   lr = base_lr * 0.5 * (1 + cos(pi * step / total_steps))
   ```

4. **Memory Issues**
   ```python
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Reduce batch size
   batch_size = 2
   
   # Use mixed precision
   with autocast():
       outputs = model(inputs)
   ```

## Next Steps

- [Learning Rate Scheduling](learning-rate-scheduling) - Learn advanced scheduling techniques
- [Custom Loss Functions](custom-loss-functions) - Design stable loss functions
- [Multi-Objective Training](multi-objective-training) - Balance multiple objectives
- [Advanced Performance](../performance/index) - Optimize performance 