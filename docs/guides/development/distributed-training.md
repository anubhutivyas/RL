# Set Up Distributed Training with Ray

This guide covers distributed training strategies in NeMo RL, from single-GPU setups to large-scale multi-node clusters using Ray.

## Overview

NeMo RL leverages Ray for distributed computing to scale reinforcement learning training across multiple GPUs and nodes. This enables training large language models efficiently and handling complex RL environments that require significant computational resources.

## Ray Architecture

### Virtual Clusters

NeMo RL uses Ray virtual clusters to manage distributed training:

```python
import ray
from nemo_rl.distributed import RayClusterManager

# Initialize Ray cluster
ray.init(
    address="auto",  # Connect to existing cluster
    namespace="nemo_rl"
)

# Create virtual cluster for training
cluster_manager = RayClusterManager()
cluster = cluster_manager.create_cluster(
    num_nodes=4,
    gpus_per_node=8,
    memory_per_node="64GB"
)
```

### Worker Groups

Training is organized into worker groups that handle different aspects:

- **Policy Workers**: Run the language model and generate actions
- **Environment Workers**: Execute RL environments and calculate rewards
- **Data Workers**: Process training data and manage datasets
- **Evaluation Workers**: Run model evaluation and benchmarking

## Single-Node Multi-GPU Training

### Basic Setup

Start with a single node containing multiple GPUs:

```yaml
# config.yaml
grpo:
  # Single node, multiple GPUs
  num_nodes: 1
  gpus_per_node: 4
  
  # Worker configuration
  policy_workers: 2
  env_workers: 8
  data_workers: 2
  
  # Training parameters
  batch_size: 32
  learning_rate: 1e-4
  max_episode_length: 100
```

### GPU Memory Management

Optimize GPU memory usage for multi-GPU training:

```python
# Configure GPU memory allocation
import torch

# Set memory fraction per GPU
torch.cuda.set_per_process_memory_fraction(0.9)

# Enable gradient checkpointing for memory efficiency
model_config = {
    "gradient_checkpointing": True,
    "max_memory_MB": 40000,  # 40GB per GPU
    "offload_optimizer": True
}
```

## Multi-Node Training

### Cluster Setup

Set up a multi-node Ray cluster:

```bash
# Head node (start first)
ray start --head --port=6379 --dashboard-port=8265

# Worker nodes
ray start --address='head_node_ip:6379'
```

### Configuration

Configure multi-node training:

```yaml
# multi_node_config.yaml
grpo:
  # Multi-node configuration
  num_nodes: 4
  gpus_per_node: 8
  
  # Network configuration
  ray_address: "head_node_ip:6379"
  namespace: "nemo_rl_training"
  
  # Worker distribution
  policy_workers_per_node: 2
  env_workers_per_node: 16
  data_workers_per_node: 4
  
  # Communication settings
  backend: "nccl"  # NVIDIA Collective Communications Library
  timeout: 300  # 5 minutes timeout for operations
```

### Node Communication

Optimize inter-node communication:

```python
# Configure NCCL for optimal performance
import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Network interface
os.environ["NCCL_BLOCKING_WAIT"] = "1"

# Set up distributed training
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
```

## Worker Configuration

### Policy Workers

Configure policy workers for model inference:

```python
class PolicyWorker:
    def __init__(self, model_config, device_id):
        self.device = f"cuda:{device_id}"
        self.model = load_model(model_config).to(self.device)
        
    def generate_action(self, observation):
        """Generate action from observation."""
        with torch.no_grad():
            action = self.model(observation)
        return action
    
    def update_policy(self, gradients):
        """Update policy with gradients from training."""
        self.model.load_state_dict(gradients)
```

### Environment Workers

Set up environment workers for parallel execution:

```python
class EnvironmentWorker:
    def __init__(self, env_config, worker_id):
        self.env = create_environment(env_config)
        self.worker_id = worker_id
        
    def step(self, action):
        """Execute environment step."""
        observation, reward, done, info = self.env.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "worker_id": self.worker_id
        }
    
    def reset(self):
        """Reset environment."""
        return self.env.reset()
```

### Data Workers

Configure data workers for efficient data processing:

```python
class DataWorker:
    def __init__(self, dataset_config):
        self.dataset = load_dataset(dataset_config)
        self.preprocessor = DataPreprocessor()
        
    def get_batch(self, batch_size):
        """Get training batch."""
        batch = self.dataset.sample(batch_size)
        return self.preprocessor.process(batch)
    
    def update_dataset(self, new_data):
        """Update dataset with new experiences."""
        self.dataset.add(new_data)
```

## Performance Optimization

### Load Balancing

Distribute work evenly across workers:

```python
def balance_workload(num_workers, total_work):
    """Distribute work evenly across workers."""
    work_per_worker = total_work // num_workers
    remainder = total_work % num_workers
    
    distribution = []
    for i in range(num_workers):
        extra = 1 if i < remainder else 0
        distribution.append(work_per_worker + extra)
    
    return distribution
```

### Memory Management

Optimize memory usage across nodes:

```python
def optimize_memory_usage():
    """Configure memory optimization settings."""
    # Enable gradient accumulation
    accumulation_steps = 4
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Configure memory pool
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
```

### Communication Optimization

Optimize inter-worker communication:

```python
def optimize_communication():
    """Configure communication settings."""
    # Use asynchronous communication
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )
    
    # Batch communications
    batch_size = 32
    buffer_size = 1024
```

## Fault Tolerance

### Checkpointing

Implement robust checkpointing for long training runs:

```python
def save_checkpoint(model, optimizer, epoch, path):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": current_loss,
        "config": training_config
    }
    
    torch.save(checkpoint, path)
    
def load_checkpoint(model, optimizer, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
```

### Error Recovery

Handle worker failures gracefully:

```python
@ray.remote
class ResilientWorker:
    def __init__(self, config):
        self.config = config
        self.max_retries = 3
        
    def execute_with_retry(self, task):
        """Execute task with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return self.execute_task(task)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
```

## Monitoring and Debugging

### Resource Monitoring

Monitor cluster resources during training:

```python
def monitor_resources():
    """Monitor cluster resource usage."""
    # Get Ray cluster info
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    # Monitor GPU usage
    gpu_usage = {}
    for node_id in ray.nodes():
        gpu_info = ray.get_gpu_ids()
        gpu_usage[node_id] = len(gpu_info)
    
    return {
        "cluster_resources": cluster_resources,
        "available_resources": available_resources,
        "gpu_usage": gpu_usage
    }
```

### Performance Profiling

Profile training performance:

```python
def profile_training():
    """Profile training performance."""
    import torch.profiler as profiler
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        # Training loop
        for batch in dataloader:
            loss = training_step(batch)
            loss.backward()
            optimizer.step()
    
    # Save profile results
    prof.export_chrome_trace("training_profile.json")
```

## Example Configurations

### Small Scale (4 GPUs)

```yaml
# small_scale_config.yaml
grpo:
  num_nodes: 1
  gpus_per_node: 4
  
  policy_workers: 2
  env_workers: 8
  data_workers: 2
  
  batch_size: 16
  learning_rate: 1e-4
  max_episode_length: 50
```

### Medium Scale (32 GPUs)

```yaml
# medium_scale_config.yaml
grpo:
  num_nodes: 4
  gpus_per_node: 8
  
  policy_workers_per_node: 2
  env_workers_per_node: 16
  data_workers_per_node: 4
  
  batch_size: 64
  learning_rate: 5e-5
  max_episode_length: 100
  
  # Communication settings
  backend: "nccl"
  timeout: 300
```

### Large Scale (128 GPUs)

```yaml
# large_scale_config.yaml
grpo:
  num_nodes: 16
  gpus_per_node: 8
  
  policy_workers_per_node: 4
  env_workers_per_node: 32
  data_workers_per_node: 8
  
  batch_size: 256
  learning_rate: 2e-5
  max_episode_length: 200
  
  # Advanced settings
  gradient_accumulation_steps: 4
  mixed_precision: true
  gradient_checkpointing: true
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Communication Timeouts**
   - Increase timeout values
   - Check network connectivity
   - Verify NCCL configuration

3. **Worker Failures**
   - Implement retry logic
   - Check resource availability
   - Monitor worker health

### Debug Commands

```bash
# Check Ray cluster status
ray status

# Monitor GPU usage
nvidia-smi

# Check network connectivity
ping worker_node_ip

# View Ray logs
ray logs
```

## Getting Help

- [Cluster Setup](../get-started/cluster.md) - Detailed cluster configuration
- [Debugging](debugging.md) - Debugging distributed training issues
- [Performance Profiling](nsys-profiling.md) - Profiling tools and techniques
- [API Reference](../reference/api.md) - Complete distributed training API
- [Community Support](https://github.com/NVIDIA-NeMo/RL/issues) - GitHub issues and discussions 