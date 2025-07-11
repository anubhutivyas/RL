# Distributed Computing API

This section documents the distributed computing abstractions in NeMo RL, which provide the foundation for scalable RL training across multiple GPUs and nodes.

## Overview

NeMo RL's distributed computing layer is built around Ray and provides abstractions for resource management, process isolation, coordination, and communication. The key components are designed to scale seamlessly from single GPU to thousands of GPUs.

## Core Components

### RayVirtualCluster

The `RayVirtualCluster` class provides a basic abstraction on top of Ray Placement Groups that allows you to section off compute resources for WorkerGroups to run on as though they had their own cluster.

```python
class RayVirtualCluster:
    """
    Creates a virtual distributed cluster using Ray placement groups.

    This class simplifies distributed training setup by:
    - Creating placement groups that represent logical compute nodes.
    - Allocating GPU and CPU resources for distributed workers.
    - Managing communication between distributed processes.

    - Bundle: A resource allocation unit (ex: 4 GPUs on a single node).
    - Worker: A process that performs computation (model training/inference).
    - Node: A physical or virtual machine containing multiple bundles.
    """
    
    def __init__(self, bundle_ct_per_node_list: List[int], **kwargs):
        """
        Initialize a virtual cluster using Ray placement groups.

        Args:
            bundle_ct_per_node_list: List specifying GPU bundles per node
                                    (e.g., [2,2] creates 2 nodes with 2 GPU bundles each)
        """
        pass
    
    def get_placement_groups(self):
        """
        Returns a list of placement groups that have at least one bundle, filtering out empty nodes.
        This represents the "virtual cluster" - only nodes that are actually being used.

        Returns:
            List of placement groups that have at least one bundle.
        """
        pass
```

#### Usage Examples

```python
# Single GPU setup
cluster = RayVirtualCluster([1])

# Multi-GPU single node
cluster = RayVirtualCluster([8])  # 8 GPUs on one node

# Multi-node setup
cluster = RayVirtualCluster([8, 8])  # 2 nodes, 8 GPUs each

# Heterogeneous setup
cluster = RayVirtualCluster([4, 8, 2])  # 3 nodes with different GPU counts
```

### RayWorkerGroup

The `RayWorkerGroup` class manages groups of distributed Ray worker/actor processes that execute tasks in parallel.

```python
class RayWorkerGroup:
    """
    Manages a group of distributed Ray worker/actor processes that execute tasks in parallel.

    This class creates and manages Ray actor instances that run on resources
    allocated by a RayVirtualCluster. It handles:
    - Worker creation and placement on specific GPU resources.
    - Setting up distributed training environment variables (rank, world size, etc.).
    - Executing methods across all workers in parallel.
    - Collecting and aggregating results.
    - Support for tied worker groups where multiple workers process the same data.
    """
    
    def __init__(self, virtual_cluster: RayVirtualCluster, worker_class, **kwargs):
        """
        Initialize a worker group with the specified virtual cluster and worker class.
        
        Args:
            virtual_cluster: The virtual cluster providing resources
            worker_class: The class to instantiate for each worker
        """
        pass
    
    def run_all_workers_single_data(self, method_name: str, *args, **kwargs):
        """
        Execute a method on all workers with the same data.
        
        Args:
            method_name: Name of the method to execute
            *args, **kwargs: Arguments to pass to the method
            
        Returns:
            List of results from all workers
        """
        pass
    
    def run_all_workers_multiple_data(self, method_name: str, data_list, **kwargs):
        """
        Execute a method on all workers with different data for each worker.
        
        Args:
            method_name: Name of the method to execute
            data_list: List of data items, one per worker
            **kwargs: Additional arguments to pass to the method
            
        Returns:
            List of results from all workers
        """
        pass
```

#### Usage Examples

```python
# Create worker group for policy training
policy_workers = RayWorkerGroup(cluster, HuggingFacePolicyWorker)

# Execute training step on all workers
losses = policy_workers.run_all_workers_single_data("train_step", batch)

# Execute generation with different prompts per worker
generations = policy_workers.run_all_workers_multiple_data(
    "generate", 
    [prompt1, prompt2, prompt3, prompt4]
)
```

### BatchedDataDict

The `BatchedDataDict` class provides efficient data structures for distributed communication and batching.

```python
class BatchedDataDict:
    """
    Efficient data structure for distributed communication and batching.
    
    This class provides a standardized way to handle batched data across
    distributed workers, with support for masking, padding, and metadata.
    """
    
    def __init__(self, data: Dict[str, torch.Tensor], **kwargs):
        """
        Initialize a batched data dictionary.
        
        Args:
            data: Dictionary of tensors representing the batched data
        """
        pass
    
    def repeat_interleave(self, repeats: int):
        """
        Repeat each sample in the batch a specified number of times.
        
        Args:
            repeats: Number of times to repeat each sample
            
        Returns:
            New BatchedDataDict with repeated samples
        """
        pass
    
    def to_device(self, device: torch.device):
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device for the tensors
            
        Returns:
            New BatchedDataDict with tensors on the target device
        """
        pass
```

## Communication Patterns

### Single Controller Pattern

NeMo RL uses a single-process controller pattern where all coordination happens through a central controller:

```python
def grpo_train(
    policy: PolicyInterface,
    policy_generation: GenerationInterface,
    environment: EnvironmentInterface,
    dataloader: Iterable[BatchedDataDict[DatumSpec]],
):
    """Example of single controller pattern for GRPO training."""
    loss_fn = GRPOLossFn()
    
    for batch in dataloader:
        # Repeat batch for multiple generations per prompt
        batch.repeat_interleave(num_generations_per_prompt)
        
        # Generate responses using distributed workers
        generations = policy_generation.generate(batch)
        
        # Get rewards from environment
        rewards = environment.step(generations)
        
        # Get log probabilities for training
        logprobs = policy.get_logprobs(generations)
        reference_logprobs = policy.get_reference_logprobs(generations)
        
        # Calculate training data and update policy
        training_data = calculate_grpo_training_data(
            generations, logprobs, reference_logprobs, rewards
        )
        policy.train(generations, logprobs, reference_logprobs, GRPOLossFn)
```

### Colocation Support

VirtualCluster supports colocation where multiple WorkerGroups share resources:

```python
# Create cluster with colocation support
cluster = RayVirtualCluster([8], enable_colocation=True)

# Create separate worker groups for different components
policy_workers = RayWorkerGroup(cluster, PolicyWorker)
generation_workers = RayWorkerGroup(cluster, GenerationWorker)

# Both worker groups can run on the same GPUs in turn
policy_workers.run_all_workers_single_data("train_step", batch)
generation_workers.run_all_workers_single_data("generate", batch)
```

## Resource Management

### GPU Allocation

The distributed system provides fine-grained control over GPU allocation:

```python
# Allocate specific GPUs to workers
cluster = RayVirtualCluster([4], gpu_ids=[0, 1, 2, 3])

# Allocate GPUs across multiple nodes
cluster = RayVirtualCluster([4, 4], gpu_ids=[[0, 1, 2, 3], [4, 5, 6, 7]])

# Use all available GPUs
cluster = RayVirtualCluster([8], gpu_ids="auto")
```

### Memory Management

The system includes memory management utilities:

```python
# Monitor GPU memory usage
cluster = RayVirtualCluster([4], monitor_memory=True)

# Set memory limits per worker
worker_group = RayWorkerGroup(
    cluster, 
    PolicyWorker, 
    memory_limit="8GB"
)
```

## Error Handling

### Worker Failure Recovery

The system provides automatic recovery from worker failures:

```python
# Configure automatic restart on failure
worker_group = RayWorkerGroup(
    cluster,
    PolicyWorker,
    restart_on_failure=True,
    max_restarts=3
)

# Handle specific failure scenarios
try:
    result = worker_group.run_all_workers_single_data("train_step", batch)
except WorkerFailureError as e:
    # Handle worker failure
    worker_group.restart_failed_workers()
    result = worker_group.run_all_workers_single_data("train_step", batch)
```

### Health Monitoring

Monitor worker health and performance:

```python
# Get worker status
status = worker_group.get_worker_status()

# Monitor resource usage
usage = worker_group.get_resource_usage()

# Check for stuck workers
stuck_workers = worker_group.detect_stuck_workers()
```

## Performance Optimization

### Communication Optimization

Optimize communication patterns for better performance:

```python
# Use efficient communication backend
cluster = RayVirtualCluster([8], communication_backend="nccl")

# Enable gradient compression
worker_group = RayWorkerGroup(
    cluster,
    PolicyWorker,
    enable_gradient_compression=True
)

# Use asynchronous communication
worker_group = RayWorkerGroup(
    cluster,
    PolicyWorker,
    async_communication=True
)
```

### Load Balancing

Distribute work evenly across workers:

```python
# Enable dynamic load balancing
worker_group = RayWorkerGroup(
    cluster,
    PolicyWorker,
    enable_load_balancing=True
)

# Set custom load balancing strategy
worker_group = RayWorkerGroup(
    cluster,
    PolicyWorker,
    load_balancing_strategy="round_robin"
)
```

## Configuration

### Cluster Configuration

Configure the virtual cluster with various options:

```python
# Basic configuration
cluster_config = {
    "bundle_ct_per_node_list": [8],
    "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "enable_colocation": True,
    "communication_backend": "nccl",
    "monitor_memory": True
}

cluster = RayVirtualCluster(**cluster_config)
```

### Worker Group Configuration

Configure worker groups with specific parameters:

```python
# Worker group configuration
worker_config = {
    "memory_limit": "8GB",
    "restart_on_failure": True,
    "max_restarts": 3,
    "enable_gradient_compression": True,
    "async_communication": True
}

worker_group = RayWorkerGroup(cluster, PolicyWorker, **worker_config)
```

## Best Practices

### Resource Planning

1. **Plan GPU allocation carefully**: Consider model size, batch size, and memory requirements
2. **Use colocation when possible**: Share GPUs between different components to maximize utilization
3. **Monitor resource usage**: Use built-in monitoring to identify bottlenecks

### Error Handling

1. **Implement proper error handling**: Always handle worker failures gracefully
2. **Use health checks**: Monitor worker health and restart failed workers
3. **Log distributed operations**: Use structured logging for debugging distributed issues

### Performance

1. **Optimize communication**: Use efficient communication backends and patterns
2. **Balance load**: Distribute work evenly across workers
3. **Monitor performance**: Track metrics to identify optimization opportunities

## Examples

### Complete Training Setup

```python
from nemo_rl.distributed import RayVirtualCluster, RayWorkerGroup
from nemo_rl.models import HuggingFacePolicyWorker
from nemo_rl.algorithms import GRPOLossFn

# Set up distributed environment
cluster = RayVirtualCluster([4, 4])  # 2 nodes, 4 GPUs each

# Create worker groups for different components
policy_workers = RayWorkerGroup(cluster, HuggingFacePolicyWorker)
generation_workers = RayWorkerGroup(cluster, VllmGenerationWorker)

# Training loop
for batch in dataloader:
    # Generate responses
    generations = generation_workers.run_all_workers_single_data(
        "generate", batch
    )
    
    # Get rewards
    rewards = environment.step(generations)
    
    # Train policy
    losses = policy_workers.run_all_workers_single_data(
        "train_step", 
        batch, 
        generations, 
        rewards, 
        GRPOLossFn
    )
```

### Custom Worker Implementation

```python
from nemo_rl.distributed import RayWorkerGroup

class CustomPolicyWorker:
    def __init__(self, model_name: str):
        self.model = load_model(model_name)
    
    def train_step(self, batch, generations, rewards, loss_fn):
        """Custom training step implementation."""
        # Custom training logic
        loss = loss_fn(self.model, batch, generations, rewards)
        return {"loss": loss.item()}
    
    def generate(self, batch):
        """Custom generation implementation."""
        outputs = self.model.generate(batch)
        return outputs

# Use custom worker
worker_group = RayWorkerGroup(cluster, CustomPolicyWorker)
``` 