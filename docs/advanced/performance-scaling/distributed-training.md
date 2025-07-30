---
description: "Scale from single GPU to multi-node clusters with efficient distributed training strategies"
tags: ["distributed", "training", "scaling", "multi-gpu", "clusters"]
categories: ["performance-scaling"]
---

# Distributed Training

This guide covers how to scale NeMo RL from single GPU to multi-node clusters with efficient distributed training strategies.

## Overview

NeMo RL provides robust distributed training capabilities that allow you to scale your training across multiple GPUs and nodes. This is essential for training large models efficiently and reducing training time.

## Key Components

### Ray-Based Distributed Computing

NeMo RL uses Ray for distributed computing, providing seamless scaling:

```python
import ray
from nemo_rl.distributed import RayTrainer

# Initialize Ray
ray.init()

# Create distributed trainer
trainer = RayTrainer(
    model=model,
    config=config,
    num_workers=4  # Number of worker processes
)
```

### Single Node Multi-GPU Training

Start with single node multi-GPU training:

```python
from nemo_rl.distributed import DistributedTrainer

class MultiGPUTrainer(DistributedTrainer):
    def __init__(self, model, config, num_gpus=4):
        super().__init__(model, config)
        self.num_gpus = num_gpus
        
        # Initialize distributed training
        self.setup_distributed()
    
    def setup_distributed(self):
        """
        Setup distributed training environment
        """
        # Initialize process group
        torch.distributed.init_process_group(
            backend='nccl',  # Use NCCL for GPU communication
            init_method='env://'
        )
        
        # Wrap model with DistributedDataParallel
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank]
        )
    
    def train_step(self, batch):
        """
        Distributed training step
        """
        # Move batch to correct device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(batch)
        
        # Compute loss
        loss = self.compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Synchronize gradients across processes
        self.synchronize_gradients()
        
        return loss
```

## Configuration

### Distributed Training Configuration

```yaml
# configs/distributed_training.yaml
distributed:
  enabled: true
  backend: nccl  # or gloo for CPU
  num_gpus: 4
  num_nodes: 2
  
  # Communication settings
  communication:
    timeout: 1800  # 30 minutes
    bucket_cap_mb: 25
    
  # Gradient synchronization
  gradient_sync:
    enabled: true
    bucket_size: 25
    
  # Mixed precision
  mixed_precision:
    enabled: true
    dtype: float16
    
  # Memory optimization
  memory_optimization:
    gradient_checkpointing: true
    activation_checkpointing: true
```

### Multi-Node Configuration

```yaml
# configs/multi_node.yaml
distributed:
  enabled: true
  num_nodes: 4
  gpus_per_node: 8
  
  # Node configuration
  nodes:
    - address: "192.168.1.10"
      gpus: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: "192.168.1.11"
      gpus: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: "192.168.1.12"
      gpus: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: "192.168.1.13"
      gpus: [0, 1, 2, 3, 4, 5, 6, 7]
```

## Advanced Strategies

### Pipeline Parallelism

For very large models, implement pipeline parallelism:

```python
class PipelineParallelTrainer(DistributedTrainer):
    def __init__(self, model, config, num_stages=4):
        super().__init__(model, config)
        self.num_stages = num_stages
        self.stage_id = self.rank // (self.world_size // num_stages)
        
        # Split model into stages
        self.model_stages = self.split_model_into_stages()
    
    def split_model_into_stages(self):
        """
        Split model into pipeline stages
        """
        stages = []
        layers_per_stage = len(self.model.layers) // self.num_stages
        
        for i in range(self.num_stages):
            start_idx = i * layers_per_stage
            end_idx = (i + 1) * layers_per_stage if i < self.num_stages - 1 else len(self.model.layers)
            
            stage = self.model.layers[start_idx:end_idx]
            stages.append(stage)
        
        return stages
    
    def forward_pipeline(self, batch):
        """
        Forward pass through pipeline stages
        """
        # Split batch into micro-batches
        micro_batches = self.split_batch(batch)
        
        # Process through pipeline stages
        outputs = []
        for micro_batch in micro_batches:
            stage_output = self.process_through_stages(micro_batch)
            outputs.append(stage_output)
        
        return self.combine_outputs(outputs)
```

### Data Parallelism with Gradient Accumulation

Implement efficient data parallelism with gradient accumulation:

```python
class GradientAccumulationTrainer(DistributedTrainer):
    def __init__(self, model, config, accumulation_steps=4):
        super().__init__(model, config)
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
    
    def train_step(self, batch):
        """
        Training step with gradient accumulation
        """
        # Forward pass
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        self.accumulation_counter += 1
        
        # Update weights after accumulation steps
        if self.accumulation_counter % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Synchronize gradients across processes
            self.synchronize_gradients()
        
        return loss
```

### Sharded Data Parallelism

For memory-efficient training with large models:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class ShardedDataParallelTrainer(DistributedTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Wrap model with FSDP
        self.model = FSDP(
            self.model,
            mixed_precision=True,
            cpu_offload=True,
            auto_wrap_policy=self.get_auto_wrap_policy()
        )
    
    def get_auto_wrap_policy(self):
        """
        Define auto-wrap policy for FSDP
        """
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        return transformer_auto_wrap_policy(
            self.model,
            transformer_layer_cls={torch.nn.TransformerEncoderLayer}
        )
    
    def train_step(self, batch):
        """
        Training step with sharded data parallelism
        """
        # Forward pass
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (handled by FSDP)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss
```

## Performance Optimization

### Communication Optimization

Optimize inter-process communication:

```python
class OptimizedCommunicationTrainer(DistributedTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Optimize communication
        self.setup_communication_optimization()
    
    def setup_communication_optimization(self):
        """
        Setup communication optimization
        """
        # Use gradient bucketing
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(seconds=1800)
        )
        
        # Enable gradient compression
        self.gradient_compression = True
        
        # Setup all-reduce fusion
        self.all_reduce_fusion = True
    
    def synchronize_gradients(self):
        """
        Optimized gradient synchronization
        """
        if self.gradient_compression:
            # Compress gradients before communication
            compressed_gradients = self.compress_gradients()
            torch.distributed.all_reduce(compressed_gradients)
            self.decompress_gradients(compressed_gradients)
        else:
            # Standard all-reduce
            for param in self.model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
```

### Memory Optimization

Implement memory-efficient distributed training:

```python
class MemoryOptimizedTrainer(DistributedTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        
        # Enable memory optimizations
        self.setup_memory_optimization()
    
    def setup_memory_optimization(self):
        """
        Setup memory optimization techniques
        """
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Enable activation checkpointing
        self.activation_checkpointing = True
        
        # Use mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch):
        """
        Memory-optimized training step
        """
        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and optimizer step
        self.scaler.unscale_(self.optimizer)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss
```

## Monitoring and Debugging

### Performance Monitoring

Monitor distributed training performance:

```python
class MonitoredDistributedTrainer(DistributedTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.metrics = {}
    
    def log_distributed_metrics(self):
        """
        Log distributed training metrics
        """
        # Communication metrics
        comm_time = self.measure_communication_time()
        self.metrics['communication_time'] = comm_time
        
        # Memory usage
        memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
        self.metrics['memory_usage_gb'] = memory_usage
        
        # GPU utilization
        gpu_utilization = self.measure_gpu_utilization()
        self.metrics['gpu_utilization'] = gpu_utilization
        
        # Throughput
        throughput = self.measure_throughput()
        self.metrics['samples_per_second'] = throughput
        
        # Log metrics
        if self.rank == 0:  # Only log from main process
            self.logger.log(self.metrics)
    
    def measure_communication_time(self):
        """
        Measure communication overhead
        """
        start_time = time.time()
        
        # Synchronize gradients
        self.synchronize_gradients()
        
        end_time = time.time()
        return end_time - start_time
```

### Debugging Distributed Training

Common issues and solutions:

```python
class DebugDistributedTrainer(DistributedTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.debug_mode = config.get('debug_mode', False)
    
    def debug_distributed_training(self):
        """
        Debug distributed training issues
        """
        if not self.debug_mode:
            return
        
        # Check process group
        if not torch.distributed.is_initialized():
            print(f"Rank {self.rank}: Process group not initialized")
            return
        
        # Check model parameters consistency
        self.check_parameter_consistency()
        
        # Check gradient synchronization
        self.check_gradient_synchronization()
        
        # Check memory usage
        self.check_memory_usage()
    
    def check_parameter_consistency(self):
        """
        Check if parameters are consistent across processes
        """
        for name, param in self.model.named_parameters():
            # Gather parameters from all processes
            gathered_params = [torch.zeros_like(param) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered_params, param.data)
            
            # Check consistency
            for i, gathered_param in enumerate(gathered_params):
                if not torch.allclose(param.data, gathered_param):
                    print(f"Rank {self.rank}: Parameter {name} inconsistent with rank {i}")
```

## Best Practices

### 1. Efficient Data Loading

Optimize data loading for distributed training:

```python
def setup_distributed_dataloader(self, dataset, batch_size):
    """
    Setup distributed data loader
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=self.world_size,
        rank=self.rank,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader, sampler
```

### 2. Gradient Synchronization

Ensure proper gradient synchronization:

```python
def synchronize_gradients(self):
    """
    Synchronize gradients across all processes
    """
    for param in self.model.parameters():
        if param.grad is not None:
            # All-reduce gradients
            torch.distributed.all_reduce(
                param.grad.data,
                op=torch.distributed.ReduceOp.SUM
            )
            
            # Average gradients
            param.grad.data /= self.world_size
```

### 3. Checkpointing

Implement distributed checkpointing:

```python
def save_distributed_checkpoint(self, path):
    """
    Save distributed training checkpoint
    """
    if self.rank == 0:
        # Save model state
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        
        # Save training state
        training_state = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_loss': self.best_loss
        }
        torch.save(training_state, f"{path}/training_state.pt")
    
    # Synchronize all processes
    torch.distributed.barrier()
```

## Troubleshooting

### Common Issues

1. **Deadlocks**: Ensure all processes participate in collective operations
2. **Memory Issues**: Use gradient accumulation and mixed precision
3. **Communication Errors**: Check network connectivity and timeout settings

### Debugging Tips

```python
# Add debugging to distributed training
def debug_distributed_setup(self):
    """
    Debug distributed training setup
    """
    print(f"Rank {self.rank}: World size = {self.world_size}")
    print(f"Rank {self.rank}: Local rank = {self.local_rank}")
    print(f"Rank {self.rank}: Device = {self.device}")
    
    # Check process group
    if torch.distributed.is_initialized():
        print(f"Rank {self.rank}: Process group initialized")
    else:
        print(f"Rank {self.rank}: Process group not initialized")
```

## Next Steps

- Learn about [Memory Optimization](memory-optimization) for large model training
- Review [Production Deployment](../production-deployment/index) for deployment strategies
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 