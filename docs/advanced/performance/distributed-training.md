# Distributed Training

This guide covers distributed training techniques for scaling NeMo RL across multiple GPUs and nodes.

## Overview

Distributed training allows you to scale RL training across multiple GPUs and nodes, significantly reducing training time and enabling training of larger models. This guide covers the various distributed training strategies available in NeMo RL.

## Distributed Training Strategies

### Data Parallelism

Distribute data across multiple GPUs while keeping the model replicated:

```yaml
# Data parallel configuration
cluster:
  name: "ray"
  num_workers: 4
  resources_per_worker:
    GPU: 0.25  # Each worker gets 1/4 of a GPU
```

### Model Parallelism

Split the model across multiple GPUs:

```yaml
# Model parallel configuration
cluster:
  name: "fsdp"
  fsdp_config:
    mixed_precision: true
    activation_checkpointing: true
    sharding_strategy: "FULL_SHARD"
    cpu_offload: false
```

### Pipeline Parallelism

Split model layers across different GPUs:

```yaml
# Pipeline parallel configuration
cluster:
  name: "pipeline"
  pipeline_config:
    stages: 4  # Split model into 4 stages
    micro_batch_size: 4
    gradient_accumulation_steps: 8
```

## Ray-Based Distributed Training

### Ray Cluster Setup

```python
import ray
from nemo_rl.distributed import RayCluster

# Initialize Ray
ray.init(address="auto")

# Create cluster
cluster = RayCluster(
    num_workers=4,
    resources_per_worker={"CPU": 1, "GPU": 0.25}
)
```

### Ray Training Configuration

```yaml
# Ray training configuration
cluster:
  name: "ray"
  num_workers: 4
  resources_per_worker:
    CPU: 1
    GPU: 0.25
  placement_group:
    strategy: "PACK"
  timeout: 300
```

### Ray Worker Implementation

```python
import ray
from nemo_rl.distributed import RayWorker

@ray.remote(num_gpus=0.25)
class TrainingWorker(RayWorker):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()
    
    def train_step(self, batch):
        """Execute one training step."""
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_model_state(self):
        """Get current model state."""
        return self.model.state_dict()
    
    def set_model_state(self, state_dict):
        """Set model state."""
        self.model.load_state_dict(state_dict)
```

## FSDP (Fully Sharded Data Parallel)

### FSDP Configuration

```yaml
# FSDP configuration
cluster:
  name: "fsdp"
  fsdp_config:
    mixed_precision: true
    activation_checkpointing: true
    sharding_strategy: "FULL_SHARD"
    cpu_offload: false
    state_dict_type: "FULL_STATE_DICT"
    auto_wrap_policy: "transformer_layer"
```

### FSDP Implementation

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

def setup_fsdp_model(model, rank, world_size):
    """Setup FSDP model."""
    
    # Define mixed precision policy
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Define auto wrap policy
    auto_wrap_policy = transformer_auto_wrap_policy(
        model,
        transformer_layer_cls={TransformerBlock},
    )
    
    # Create FSDP model
    fsdp_model = FSDP(
        model,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )
    
    return fsdp_model
```

### FSDP Training Loop

```python
def train_with_fsdp(model, dataloader, optimizer, rank, world_size):
    """Train with FSDP."""
    
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient synchronization happens automatically
        optimizer.step()
        
        if rank == 0:
            print(f"Loss: {loss.item():.4f}")
```

## Communication Optimization

### Gradient Compression

```python
from torch.distributed.algorithms.model_averaging import averagers

# Use gradient compression
compression = averagers.CompressionAverager(
    comm_backend="nccl",
    compression="fp16"
)

# Apply compression during training
for param in model.parameters():
    if param.grad is not None:
        compressed_grad = compression.compress(param.grad)
        param.grad = compressed_grad
```

### Overlap Communication

```python
from torch.distributed import all_reduce
import torch.distributed as dist

def overlap_allreduce(gradients):
    """Overlap gradient allreduce with computation."""
    handles = []
    
    for grad in gradients:
        if grad is not None:
            handle = all_reduce(grad, async_op=True)
            handles.append(handle)
    
    # Continue with computation while communication happens
    # ...
    
    # Wait for communication to complete
    for handle in handles:
        handle.wait()
```

### Bucketing

```python
from torch.distributed import all_reduce_coalesced

def bucketed_allreduce(parameters):
    """Use bucketed allreduce for efficiency."""
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # Coalesce gradients
    coalesced_grads = all_reduce_coalesced(grads, group=dist.group.WORLD)
    
    # Update gradients
    idx = 0
    for param in parameters:
        if param.grad is not None:
            param.grad = coalesced_grads[idx]
            idx += 1
```

## Load Balancing

### Dynamic Batching

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.current_batch = []
    
    def add_sample(self, sample):
        """Add sample to current batch."""
        self.current_batch.append(sample)
        
        if len(self.current_batch) >= self.max_batch_size:
            return self.get_batch()
        return None
    
    def get_batch(self):
        """Get current batch and reset."""
        batch = self.current_batch
        self.current_batch = []
        return batch
```

### Workload Distribution

```python
def distribute_workload(dataset, world_size, rank):
    """Distribute dataset across workers."""
    # Split dataset
    dataset_size = len(dataset)
    samples_per_worker = dataset_size // world_size
    
    start_idx = rank * samples_per_worker
    end_idx = start_idx + samples_per_worker
    
    if rank == world_size - 1:
        end_idx = dataset_size
    
    return dataset[start_idx:end_idx]
```

## Fault Tolerance

### Checkpointing

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def save_checkpoint(model, optimizer, epoch, rank):
    """Save distributed checkpoint."""
    if rank == 0:
        # Save on rank 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    
    # Synchronize all processes
    dist.barrier()
    
    # Load checkpoint on all processes
    if rank != 0:
        checkpoint = torch.load(f'checkpoint_epoch_{epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Error Handling

```python
import torch.distributed as dist

def train_with_fault_tolerance(model, dataloader, optimizer):
    """Train with fault tolerance."""
    try:
        for batch in dataloader:
            try:
                # Training step
                loss = model(batch)
                loss.backward()
                optimizer.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                    
    except Exception as e:
        # Synchronize error across processes
        dist.barrier()
        raise e
```

## Performance Monitoring

### Communication Metrics

```python
import time
import torch.distributed as dist

class CommunicationMonitor:
    def __init__(self):
        self.comm_times = []
    
    def measure_allreduce(self, tensor):
        """Measure allreduce time."""
        start_time = time.time()
        dist.all_reduce(tensor)
        end_time = time.time()
        
        self.comm_times.append(end_time - start_time)
        return tensor
    
    def get_stats(self):
        """Get communication statistics."""
        if self.comm_times:
            return {
                'mean_time': sum(self.comm_times) / len(self.comm_times),
                'max_time': max(self.comm_times),
                'min_time': min(self.comm_times),
                'total_time': sum(self.comm_times)
            }
        return {}
```

### Scaling Efficiency

```python
def calculate_scaling_efficiency(times, world_size):
    """Calculate scaling efficiency."""
    baseline_time = times[1]  # Single GPU time
    scaled_time = times[world_size]
    
    ideal_time = baseline_time / world_size
    efficiency = ideal_time / scaled_time
    
    return efficiency
```

## Configuration Examples

### Multi-GPU Training

```yaml
# Multi-GPU configuration
cluster:
  name: "fsdp"
  num_gpus: 4
  
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  
model:
  name: "llama2-7b"
  max_length: 2048
```

### Multi-Node Training

```yaml
# Multi-node configuration
cluster:
  name: "ray"
  num_nodes: 4
  gpus_per_node: 8
  
training:
  batch_size: 128
  gradient_accumulation_steps: 2
  
model:
  name: "llama2-70b"
  max_length: 4096
```

### Heterogeneous Training

```yaml
# Heterogeneous configuration
cluster:
  name: "ray"
  resources:
    - {"CPU": 4, "GPU": 1, "memory": 32000}
    - {"CPU": 8, "GPU": 2, "memory": 64000}
    - {"CPU": 16, "GPU": 4, "memory": 128000}
```

## Best Practices

### 1. Start Small

- Begin with single GPU training
- Gradually scale to multiple GPUs
- Test with small models first

### 2. Monitor Performance

- Track communication overhead
- Monitor memory usage
- Measure scaling efficiency

### 3. Optimize Communication

- Use gradient compression
- Overlap communication with computation
- Use appropriate communication primitives

### 4. Handle Faults

- Implement checkpointing
- Handle node failures gracefully
- Monitor for OOM errors

### 5. Balance Load

- Distribute work evenly
- Use dynamic batching
- Monitor worker utilization

## Troubleshooting

### Common Issues

1. **Communication Deadlocks**
   - Check for mismatched allreduce calls
   - Ensure proper synchronization
   - Use timeout mechanisms

2. **Memory Issues**
   - Reduce batch size per GPU
   - Enable gradient checkpointing
   - Use CPU offloading

3. **Load Imbalance**
   - Monitor worker utilization
   - Use dynamic batching
   - Balance dataset distribution

4. **Network Issues**
   - Optimize network configuration
   - Use appropriate communication backend
   - Monitor network bandwidth

For more advanced distributed training techniques, see [Performance Optimization](index.md) and [Memory Optimization](memory-optimization.md). 