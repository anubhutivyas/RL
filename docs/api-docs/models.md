# Models and Policies API

This section documents the model interfaces, policy implementations, and generation backends in NeMo RL.

## Overview

NeMo RL provides a unified interface for different model backends and policy implementations. The framework supports multiple backends (Hugging Face, Megatron) while maintaining consistent APIs for training and inference.

## Core Interfaces

### PolicyInterface

The `PolicyInterface` is the main abstract interface that all policy implementations must implement:

```python
class PolicyInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        """Generate text responses for the given input data."""
        pass

    @abstractmethod
    def get_logprobs(
        self, data: BatchedDataDict["GenerationDatumSpec"]
    ) -> BatchedDataDict["LogprobsSpec"]:
        """Get log probabilities for the given input data."""
        pass

    @abstractmethod
    def get_reference_logprobs(
        self, data: BatchedDataDict["GenerationDatumSpec"]
    ) -> BatchedDataDict["LogprobsSpec"]:
        """Get reference log probabilities for the given input data."""
        pass

    @abstractmethod
    def train(
        self, 
        data: BatchedDataDict["TrainingDatumSpec"], 
        loss_fn: LossFunction
    ) -> Dict[str, Any]:
        """Train the policy using the given data and loss function."""
        pass
```

### GenerationInterface

The `GenerationInterface` provides a unified interface for text generation backends:

```python
class GenerationInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        """Generate text responses for the given input data."""
        pass

    @abstractmethod
    def prepare_for_generation(self, *args, **kwargs):
        """Prepare the generation backend for text generation."""
        pass

    @abstractmethod
    def finish_generation(self, *args, **kwargs):
        """Clean up after text generation."""
        pass
```

## Data Specifications

### GenerationDatumSpec

Defines the input data format for generation:

```python
class GenerationDatumSpec(TypedDict):
    """Input data specification for generation."""
    input_ids: torch.Tensor         # Input token IDs
    attention_mask: torch.Tensor    # Attention mask
    __extra__: Any                  # Additional data specific to the backend
```

### GenerationOutputSpec

Defines the output data format for generation:

```python
class GenerationOutputSpec(TypedDict):
    """Output data specification for generation."""
    output_ids: torch.Tensor
    generation_lengths: torch.Tensor  # Length of just the generated response part
    unpadded_sequence_lengths: torch.Tensor  # Length of full valid sequence (input + generated response)
    logprobs: torch.Tensor
    __extra__: Any                  # Additional output data specific to the backend
```

### GenerationConfig

Configuration for generation backends:

```python
class GenerationConfig(TypedDict):
    """Configuration for generation."""
    backend: str              # The backend to use (e.g., "vllm", "hf")
    max_new_tokens: int       # Maximum number of tokens to generate
    temperature: float        # Sampling temperature
    top_p: float              # Top-p sampling parameter
    top_k: int                # Top-k sampling parameter
    model_name: str           # Name or path of the model
```

## Model Backends

### Hugging Face Backend

The Hugging Face backend provides support for Hugging Face models:

```python
class HuggingFacePolicy(PolicyInterface):
    """Hugging Face-based policy implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize Hugging Face policy.
        
        Args:
            model_name: Name or path of the Hugging Face model
            **kwargs: Additional configuration parameters
        """
        pass
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using Hugging Face model."""
        pass
    
    def get_logprobs(self, data: BatchedDataDict["GenerationDatumSpec"]):
        """Get log probabilities using Hugging Face model."""
        pass
    
    def train(self, data: BatchedDataDict["TrainingDatumSpec"], loss_fn: LossFunction):
        """Train the Hugging Face model."""
        pass
```

#### Usage Example

```python
from nemo_rl.models import HuggingFacePolicy

# Initialize policy
policy = HuggingFacePolicy(
    model_name="meta-llama/Llama-2-7b-hf",
    torch_dtype="bfloat16",
    device_map="auto"
)

# Generate text
generations = policy.generate(batch, greedy=False)

# Get log probabilities
logprobs = policy.get_logprobs(batch)

# Train the model
loss = policy.train(batch, DPOLossFn())
```

### Megatron Backend

The Megatron backend provides support for Megatron-LM models:

```python
class MegatronPolicy(PolicyInterface):
    """Megatron-based policy implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize Megatron policy.
        
        Args:
            model_name: Name or path of the Megatron model
            **kwargs: Additional configuration parameters
        """
        pass
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using Megatron model."""
        pass
    
    def get_logprobs(self, data: BatchedDataDict["GenerationDatumSpec"]):
        """Get log probabilities using Megatron model."""
        pass
    
    def train(self, data: BatchedDataDict["TrainingDatumSpec"], loss_fn: LossFunction):
        """Train the Megatron model."""
        pass
```

#### Usage Example

```python
from nemo_rl.models import MegatronPolicy

# Initialize policy
policy = MegatronPolicy(
    model_name="llama3.1-8b",
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    sequence_parallel=True
)

# Generate text
generations = policy.generate(batch, greedy=False)

# Get log probabilities
logprobs = policy.get_logprobs(batch)

# Train the model
loss = policy.train(batch, GRPOLossFn())
```

## Generation Backends

### VLLM Backend

The VLLM backend provides efficient text generation using the VLLM library:

```python
class VllmGeneration(GenerationInterface):
    """VLLM-based generation implementation."""
    
    def __init__(self, cluster: RayVirtualCluster, config: GenerationConfig):
        """
        Initialize VLLM generation backend.
        
        Args:
            cluster: Virtual cluster for distributed generation
            config: Generation configuration
        """
        pass
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using VLLM backend."""
        pass
    
    def prepare_for_generation(self):
        """Prepare VLLM workers for generation."""
        pass
    
    def finish_generation(self):
        """Clean up VLLM workers after generation."""
        pass
```

#### VllmGenerationWorker

The `VllmGenerationWorker` is a Ray actor that manages VLLM model instances:

```python
class VllmGenerationWorker:
    """Ray actor for VLLM generation workers."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize VLLM generation worker.
        
        Args:
            model_name: Name or path of the model
            **kwargs: Additional VLLM configuration
        """
        pass
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate text for the given input tokens."""
        pass
    
    def update_weights(self, weights_handle):
        """Update model weights from IPC handle."""
        pass
```

#### Usage Example

```python
from nemo_rl.models.generation import VllmGeneration, VllmConfig
from nemo_rl.distributed import RayVirtualCluster

# Set up configuration
config = VllmConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    max_new_tokens=100,
    temperature=0.7,
    top_p=1,
    top_k=None,
    backend="vllm",
    vllm_cfg={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
    }
)

# Initialize cluster and generation backend
cluster = RayVirtualCluster([4])  # 4 GPUs
generator = VllmGeneration(cluster, config)

# Prepare input data
input_data = BatchedDataDict({
    "input_ids": torch.tensor([[1, 2, 3, 4]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]])
})

# Generate text
generator.prepare_for_generation()
output = generator.generate(input_data, greedy=False)
generator.finish_generation()
```

### Hugging Face Generation Backend

The Hugging Face generation backend provides generation using Hugging Face models:

```python
class HuggingFaceGeneration(GenerationInterface):
    """Hugging Face-based generation implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize Hugging Face generation backend.
        
        Args:
            model_name: Name or path of the Hugging Face model
            **kwargs: Additional configuration parameters
        """
        pass
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using Hugging Face model."""
        pass
    
    def prepare_for_generation(self):
        """Prepare Hugging Face model for generation."""
        pass
    
    def finish_generation(self):
        """Clean up Hugging Face model after generation."""
        pass
```

## Model Validation

### Log Probability Consistency

NeMo RL includes validation tools to ensure log probability consistency between training and inference backends:

```python
def validate_log_probability_consistency(
    training_backend: PolicyInterface,
    inference_backend: GenerationInterface,
    test_data: BatchedDataDict,
    threshold: float = 1.05
) -> bool:
    """
    Validate log probability consistency between training and inference backends.
    
    Args:
        training_backend: Training backend (e.g., Hugging Face)
        inference_backend: Inference backend (e.g., VLLM)
        test_data: Test data for validation
        threshold: Acceptable error threshold (default: 1.05)
    
    Returns:
        True if consistency is within threshold, False otherwise
    """
    pass
```

### Model Diagnostics

NeMo RL provides diagnostic scripts for model validation:

```python
# Test model length respect
def test_max_model_len_respected(model_name: str):
    """Test if model respects max_model_len parameter."""
    pass

# Test generation consistency
def test_generation_consistency(model_name: str):
    """Test consistency between decoding and prefill passes."""
    pass
```

## Custom Model Implementation

### Implementing Custom Policy

To implement a custom policy, inherit from `PolicyInterface`:

```python
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.distributed import BatchedDataDict

class CustomPolicy(PolicyInterface):
    """Custom policy implementation."""
    
    def __init__(self, model_path: str):
        """Initialize custom policy."""
        self.model = load_custom_model(model_path)
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using custom model."""
        input_ids = data["input_ids"]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=data.get("max_new_tokens", 100),
            do_sample=not greedy,
            temperature=data.get("temperature", 0.7)
        )
        return BatchedDataDict({
            "output_ids": outputs,
            "generation_lengths": torch.tensor([len(o) - len(i) for o, i in zip(outputs, input_ids)]),
            "unpadded_sequence_lengths": torch.tensor([len(o) for o in outputs])
        })
    
    def get_logprobs(self, data: BatchedDataDict["GenerationDatumSpec"]):
        """Get log probabilities using custom model."""
        input_ids = data["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids)
            logprobs = torch.log_softmax(outputs.logits, dim=-1)
        return BatchedDataDict({"logprobs": logprobs})
    
    def train(self, data: BatchedDataDict["TrainingDatumSpec"], loss_fn: LossFunction):
        """Train the custom model."""
        # Custom training logic
        loss = loss_fn(self.model, data)
        return {"loss": loss.item()}
```

### Implementing Custom Generation Backend

To implement a custom generation backend, inherit from `GenerationInterface`:

```python
from nemo_rl.models.generation.interfaces import GenerationInterface

class CustomGeneration(GenerationInterface):
    """Custom generation backend implementation."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize custom generation backend."""
        self.model = load_custom_model(model_name)
    
    def generate(self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool):
        """Generate text using custom backend."""
        # Custom generation logic
        pass
    
    def prepare_for_generation(self):
        """Prepare custom backend for generation."""
        # Custom preparation logic
        pass
    
    def finish_generation(self):
        """Clean up custom backend after generation."""
        # Custom cleanup logic
        pass
```

## Configuration

### Model Configuration

Configure models with various parameters:

```python
# Hugging Face configuration
hf_config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "load_in_8bit": False,
    "load_in_4bit": False
}

# Megatron configuration
megatron_config = {
    "model_name": "llama3.1-8b",
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 4,
    "sequence_parallel": True,
    "use_flash_attention": True
}

# VLLM configuration
vllm_config = {
    "model_name": "Qwen/Qwen2.5-1.5B",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 1,
    "top_k": None,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 2048
}
```

### Generation Configuration

Configure generation parameters:

```python
# Generation configuration
gen_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": 0,
    "eos_token_id": 2
}
```

## Best Practices

### Model Selection

1. **Choose appropriate backend**: Use Hugging Face for prototyping, Megatron for large-scale training
2. **Consider memory constraints**: Use quantization and model parallelism for large models
3. **Validate consistency**: Always validate log probability consistency between backends

### Performance Optimization

1. **Use appropriate precision**: Use mixed precision training when possible
2. **Optimize generation**: Use efficient generation backends like VLLM for inference
3. **Monitor memory usage**: Track GPU memory usage and optimize accordingly

### Customization

1. **Implement interfaces**: Always implement the appropriate abstract interfaces
2. **Add validation**: Include validation tests for custom implementations
3. **Document behavior**: Clearly document any deviations from standard behavior

## Examples

### Complete Training Pipeline

```python
from nemo_rl.models import HuggingFacePolicy
from nemo_rl.models.generation import VllmGeneration
from nemo_rl.algorithms import DPOLossFn

# Initialize components
policy = HuggingFacePolicy("meta-llama/Llama-2-7b-hf")
generator = VllmGeneration(cluster, vllm_config)

# Training loop
for batch in dataloader:
    # Generate responses
    generations = generator.generate(batch, greedy=False)
    
    # Get log probabilities
    logprobs = policy.get_logprobs(generations)
    ref_logprobs = policy.get_reference_logprobs(generations)
    
    # Train policy
    loss = policy.train(batch, DPOLossFn())
```

### Model Validation

```python
from nemo_rl.models.validation import validate_log_probability_consistency

# Validate consistency
is_consistent = validate_log_probability_consistency(
    training_backend=policy,
    inference_backend=generator,
    test_data=validation_batch,
    threshold=1.05
)

if not is_consistent:
    print("Warning: Log probability consistency issues detected")
``` 