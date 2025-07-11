# Learning Rate Scheduling

This guide covers advanced learning rate scheduling techniques for NeMo RL, including warmup strategies, decay schedules, and adaptive learning rate methods to optimize training convergence and stability.

## Overview

Learning rate scheduling is crucial for successful training of large language models. The right learning rate schedule can significantly improve convergence speed, final model performance, and training stability. This guide covers various scheduling strategies and their implementation in NeMo RL.

## Basic Scheduling Concepts

### 1. Warmup Period

A warmup period gradually increases the learning rate from a small value to the target learning rate:

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr=1e-8):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate with warmup."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
        else:
            lr = self.target_lr
            
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 2. Decay Schedules

Implement various decay schedules after warmup:

```python
class CosineDecayScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate with cosine decay."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.pi * progress))
            
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## Advanced Scheduling Strategies

### 1. Multi-Step Decay

Implement step-wise learning rate decay:

```python
class MultiStepScheduler:
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate with multi-step decay."""
        self.current_step += 1
        
        # Find current milestone
        current_lr = self.base_lr
        for milestone in self.milestones:
            if self.current_step >= milestone:
                current_lr *= self.gamma
                
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
```

### 2. Exponential Decay

Implement exponential learning rate decay:

```python
class ExponentialDecayScheduler:
    def __init__(self, optimizer, decay_rate=0.95, decay_steps=1000):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate with exponential decay."""
        self.current_step += 1
        
        # Exponential decay
        lr = self.base_lr * (self.decay_rate ** (self.current_step / self.decay_steps))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 3. Cyclical Learning Rates

Implement cyclical learning rate scheduling:

```python
class CyclicalScheduler:
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular'):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.current_step = 0
        
    def step(self):
        """Update learning rate with cyclical scheduling."""
        self.current_step += 1
        
        # Calculate cycle
        cycle = 1 + self.current_step // (2 * self.step_size)
        x = abs(self.current_step / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (0.999 ** self.current_step)
            
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## Adaptive Learning Rate Methods

### 1. ReduceLROnPlateau

Reduce learning rate when validation loss plateaus:

```python
class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=0):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        
    def step(self, metrics):
        """Update learning rate based on metrics."""
        if self.best is None:
            self.best = metrics
        elif self.mode == 'min' and metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        elif self.mode == 'max' and metrics > self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate from {old_lr} to {new_lr}")
```

### 2. OneCycleLR

Implement one-cycle learning rate policy:

```python
class OneCycleScheduler:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos'):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate with one-cycle policy."""
        self.current_step += 1
        
        # Calculate progress
        progress = self.current_step / self.total_steps
        
        if progress <= self.pct_start:
            # Warmup phase
            lr = self.base_lr + (self.max_lr - self.base_lr) * (progress / self.pct_start)
        else:
            # Annealing phase
            anneal_progress = (progress - self.pct_start) / (1 - self.pct_start)
            if self.anneal_strategy == 'cos':
                lr = self.max_lr + (self.base_lr - self.max_lr) * 0.5 * (1 + torch.cos(torch.pi * anneal_progress))
            else:  # linear
                lr = self.max_lr + (self.base_lr - self.max_lr) * anneal_progress
                
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## Layer-wise Learning Rate Scheduling

### 1. Different Learning Rates for Different Layers

```python
class LayerWiseScheduler:
    def __init__(self, model, base_lr, layer_multipliers):
        self.model = model
        self.base_lr = base_lr
        self.layer_multipliers = layer_multipliers
        self.param_groups = self._create_param_groups()
        
    def _create_param_groups(self):
        """Create parameter groups with different learning rates."""
        param_groups = []
        
        for name, param in self.model.named_parameters():
            multiplier = 1.0
            for layer_name, mult in self.layer_multipliers.items():
                if layer_name in name:
                    multiplier = mult
                    break
                    
            param_groups.append({
                'params': [param],
                'lr': self.base_lr * multiplier,
                'name': name
            })
            
        return param_groups
        
    def step(self):
        """Update learning rates for all parameter groups."""
        for group in self.param_groups:
            # Apply scheduling logic to each group
            current_lr = group['lr']
            # Add your scheduling logic here
            group['lr'] = current_lr
```

### 2. Transformer-specific Scheduling

```python
class TransformerScheduler:
    def __init__(self, model, base_lr, warmup_steps=1000, total_steps=100000):
        self.model = model
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
        # Different learning rates for different components
        self.param_groups = [
            {'params': [], 'lr': base_lr * 0.1, 'name': 'embeddings'},  # Lower LR for embeddings
            {'params': [], 'lr': base_lr * 2.0, 'name': 'attention'},    # Higher LR for attention
            {'params': [], 'lr': base_lr, 'name': 'other'}               # Standard LR for others
        ]
        
        self._group_parameters()
        
    def _group_parameters(self):
        """Group parameters by component type."""
        for name, param in self.model.named_parameters():
            if 'embed' in name:
                self.param_groups[0]['params'].append(param)
            elif 'attention' in name or 'attn' in name:
                self.param_groups[1]['params'].append(param)
            else:
                self.param_groups[2]['params'].append(param)
                
    def step(self):
        """Update learning rates with transformer-specific scheduling."""
        self.current_step += 1
        
        for group in self.param_groups:
            if self.current_step <= self.warmup_steps:
                # Linear warmup
                lr = group['lr'] * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = group['lr'] * 0.5 * (1 + torch.cos(torch.pi * progress))
                
            group['lr'] = lr
```

## Custom Scheduling Strategies

### 1. Custom Scheduler Base Class

```python
class CustomScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.current_step = 0
        
    def step(self):
        """Update learning rate - override in subclasses."""
        raise NotImplementedError
        
    def get_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self):
        """Save scheduler state."""
        return {
            'current_step': self.current_step,
            'optimizer_state': self.optimizer.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
```

### 2. Custom Warmup with Custom Decay

```python
class CustomWarmupDecayScheduler(CustomScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, decay_function='cosine'):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_function = decay_function
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        """Update learning rate with custom warmup and decay."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Custom warmup (e.g., exponential)
            progress = self.current_step / self.warmup_steps
            lr = self.base_lr * (progress ** 2)  # Quadratic warmup
        else:
            # Custom decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.decay_function == 'cosine':
                lr = self.base_lr * 0.5 * (1 + torch.cos(torch.pi * progress))
            elif self.decay_function == 'exponential':
                lr = self.base_lr * (0.95 ** (progress * 100))
            elif self.decay_function == 'linear':
                lr = self.base_lr * (1 - progress)
            else:
                lr = self.base_lr
                
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## Integration with NeMo RL

### 1. Configuration-based Scheduling

```yaml
# training_config.yaml
training:
  learning_rate: 1e-5
  scheduler:
    type: "cosine_warmup"
    warmup_steps: 1000
    total_steps: 100000
    min_lr: 1e-7
    
  # Layer-wise scheduling
  layer_scheduling:
    embeddings: 0.1
    attention: 2.0
    other: 1.0
```

### 2. Scheduler Factory

```python
class SchedulerFactory:
    @staticmethod
    def create_scheduler(optimizer, config):
        """Create scheduler based on configuration."""
        scheduler_type = config.get('type', 'cosine_warmup')
        
        if scheduler_type == 'cosine_warmup':
            return CosineDecayScheduler(
                optimizer,
                total_steps=config.get('total_steps', 100000),
                warmup_steps=config.get('warmup_steps', 1000),
                min_lr=config.get('min_lr', 0)
            )
        elif scheduler_type == 'onecycle':
            return OneCycleScheduler(
                optimizer,
                max_lr=config.get('max_lr', 1e-4),
                total_steps=config.get('total_steps', 100000),
                pct_start=config.get('pct_start', 0.3)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'min'),
                factor=config.get('factor', 0.1),
                patience=config.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
```

### 3. Training Integration

```python
class ScheduledTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = SchedulerFactory.create_scheduler(optimizer, config['scheduler'])
        self.config = config
        
    def training_step(self, batch):
        """Training step with scheduler update."""
        # Forward pass
        loss = self.model(batch)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Scheduler step
        self.scheduler.step()
        
        return loss.item()
        
    def validation_step(self, batch):
        """Validation step for plateau-based scheduling."""
        loss = self.model(batch)
        return loss.item()
        
    def on_validation_end(self, val_loss):
        """Update scheduler based on validation loss."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
```

## Best Practices

### 1. Warmup Strategies

1. **Start with linear warmup** for most cases
2. **Use longer warmup** for larger models
3. **Consider exponential warmup** for very large models
4. **Monitor warmup phase** to ensure stability

### 2. Decay Strategies

1. **Use cosine decay** for most training scenarios
2. **Consider step decay** for specific use cases
3. **Implement exponential decay** for fine-tuning
4. **Set appropriate minimum learning rates**

### 3. Adaptive Methods

1. **Use ReduceLROnPlateau** when validation metrics are available
2. **Implement OneCycleLR** for faster convergence
3. **Consider cyclical learning rates** for exploration
4. **Monitor learning rate changes** carefully

### 4. Layer-wise Scheduling

1. **Use lower learning rates** for embeddings
2. **Use higher learning rates** for attention layers
3. **Group parameters** logically by function
4. **Monitor layer-wise gradients** for stability

## Monitoring and Debugging

### 1. Learning Rate Monitoring

```python
class LRSchedulerMonitor:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.lr_history = []
        
    def log_lr(self):
        """Log current learning rate."""
        current_lr = self.scheduler.get_lr()
        self.lr_history.append(current_lr)
        
        if len(self.lr_history) % 100 == 0:
            print(f"Step {len(self.lr_history)}, LR: {current_lr}")
            
    def plot_lr_schedule(self):
        """Plot learning rate schedule."""
        import matplotlib.pyplot as plt
        
        steps = range(len(self.lr_history))
        plt.plot(steps, self.lr_history)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.show()
```

### 2. Scheduler Debugging

```python
def debug_scheduler(scheduler, num_steps=1000):
    """Debug scheduler behavior."""
    lr_history = []
    
    for step in range(num_steps):
        scheduler.step()
        lr_history.append(scheduler.get_lr())
        
    # Check for issues
    if any(lr <= 0 for lr in lr_history):
        print("Warning: Negative or zero learning rate detected")
        
    if any(torch.isnan(torch.tensor(lr)) for lr in lr_history):
        print("Warning: NaN learning rate detected")
        
    return lr_history
```

## Troubleshooting

### Common Issues

1. **Learning Rate Too High**
   ```python
   # Reduce base learning rate
   base_lr = 1e-6  # Reduce from 1e-5
   
   # Increase warmup steps
   warmup_steps = 2000  # Increase from 1000
   ```

2. **Learning Rate Too Low**
   ```python
   # Increase base learning rate
   base_lr = 1e-4  # Increase from 1e-5
   
   # Reduce warmup steps
   warmup_steps = 500  # Reduce from 1000
   ```

3. **Unstable Training**
   ```python
   # Use more conservative scheduling
   scheduler = CosineDecayScheduler(
       optimizer,
       total_steps=total_steps,
       warmup_steps=warmup_steps,
       min_lr=base_lr * 0.01  # Set minimum LR
   )
   ```

4. **Slow Convergence**
   ```python
   # Use OneCycleLR for faster convergence
   scheduler = OneCycleScheduler(
       optimizer,
       max_lr=1e-4,
       total_steps=total_steps,
       pct_start=0.3
   )
   ```

## Next Steps

- [Training Stability](training-stability) - Learn stability techniques
- [Custom Loss Functions](custom-loss-functions) - Design custom loss functions
- [Hyperparameter Optimization](hyperparameter-optimization) - Optimize hyperparameters
- [Advanced Performance](../performance/index) - Performance optimization 