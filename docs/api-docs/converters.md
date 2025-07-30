---
description: "Model conversion and export utilities in NeMo RL including HuggingFace to vLLM conversions and Megatron model support"
categories: ["reference"]
tags: ["converters", "model-export", "huggingface", "megatron", "vllm", "deployment"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

# Converters API

This section documents model conversion and export utilities in NeMo RL, enabling seamless model format conversions for deployment and production use.

## Overview

NeMo RL provides model conversion utilities to transform models between different formats and backends. These converters enable deployment flexibility and integration with various inference engines.

## Core Converters

### HuggingFace Converters

Convert HuggingFace models to other formats for deployment:

```python
from nemo_rl.converters.huggingface import convert_to_vllm

# Convert HuggingFace model to vLLM format
convert_to_vllm(
    model_name="llama2-7b",
    output_dir="./vllm_model",
    tensor_parallel_size=4
)
```

#### Supported Conversions

- **HuggingFace → vLLM**: Optimized for high-throughput inference
- **HuggingFace → ONNX**: For deployment in ONNX-compatible environments
- **HuggingFace → TensorRT**: For NVIDIA GPU optimization

### Megatron Converters

Convert Megatron models to other formats:

```python
from nemo_rl.converters.megatron import convert_megatron_to_hf

# Convert Megatron model to HuggingFace format
convert_megatron_to_hf(
    megatron_checkpoint_path="./megatron_checkpoint",
    output_dir="./hf_model",
    model_type="llama2"
)
```

#### Supported Conversions

- **Megatron → HuggingFace**: For standard HuggingFace workflows
- **Megatron → vLLM**: For high-performance inference
- **Megatron → ONNX**: For deployment flexibility

## Conversion Utilities

### Model Format Detection

```python
from nemo_rl.converters.utils import detect_model_format

# Detect model format automatically
format_info = detect_model_format("./model_path")
print(f"Model format: {format_info['format']}")
print(f"Model type: {format_info['model_type']}")
```

### Conversion Validation

```python
from nemo_rl.converters.utils import validate_conversion

# Validate converted model
validation_result = validate_conversion(
    original_model_path="./original_model",
    converted_model_path="./converted_model",
    test_inputs=["Hello, world!"]
)
```

## Deployment Integration

### vLLM Export

Export models for vLLM deployment:

```python
from nemo_rl.converters.huggingface.vllm_export import export_for_vllm

# Export model for vLLM deployment
export_for_vllm(
    model_name="llama2-7b",
    output_dir="./vllm_deployment",
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.9
)
```

### Production Deployment

Prepare models for production deployment:

```python
from nemo_rl.converters.utils import prepare_for_deployment

# Prepare model for production
deployment_config = prepare_for_deployment(
    model_path="./converted_model",
    deployment_type="vllm",
    optimization_level="high"
)
```

## Configuration

### Converter Configuration

```yaml
# converter_config.yaml
converter:
  source_format: "huggingface"
  target_format: "vllm"
  optimization:
    tensor_parallel_size: 4
    max_model_len: 8192
    gpu_memory_utilization: 0.9
  validation:
    run_tests: true
    test_inputs: ["Hello", "How are you?"]
```

### Environment Variables

```bash
# Converter environment variables
export NRL_CONVERTER_CACHE_DIR="./converter_cache"
export NRL_CONVERTER_LOG_LEVEL="INFO"
export NRL_CONVERTER_VALIDATE_OUTPUT=true
```

## Error Handling

### Common Conversion Errors

```python
from nemo_rl.converters.utils import handle_conversion_error

try:
    convert_model(source_path, target_path)
except ConversionError as e:
    # Handle conversion-specific errors
    handle_conversion_error(e)
except ModelNotFoundError as e:
    # Handle missing model errors
    print(f"Model not found: {e.model_path}")
except ValidationError as e:
    # Handle validation errors
    print(f"Validation failed: {e.details}")
```

### Troubleshooting

Common issues and solutions:

1. **Memory Errors**: Reduce tensor parallel size or use CPU offloading
2. **Format Errors**: Verify source model format and compatibility
3. **Validation Errors**: Check model weights and configuration
4. **Performance Issues**: Optimize conversion parameters

## Best Practices

### Model Preparation

1. **Verify Source Model**: Ensure source model is complete and valid
2. **Check Compatibility**: Verify target format supports model architecture
3. **Test Conversion**: Validate conversion with sample inputs
4. **Optimize Parameters**: Tune conversion parameters for your use case

### Deployment Considerations

1. **Hardware Requirements**: Ensure target hardware supports converted format
2. **Performance Testing**: Benchmark converted model performance
3. **Memory Optimization**: Configure memory usage for deployment environment
4. **Error Handling**: Implement robust error handling for production

## API Reference

### Core Functions

- `convert_to_vllm()`: Convert HuggingFace model to vLLM format
- `convert_megatron_to_hf()`: Convert Megatron model to HuggingFace format
- `detect_model_format()`: Automatically detect model format
- `validate_conversion()`: Validate converted model functionality
- `prepare_for_deployment()`: Prepare model for production deployment

### Configuration Options

- **tensor_parallel_size**: Number of GPUs for tensor parallelism
- **max_model_len**: Maximum sequence length for inference
- **gpu_memory_utilization**: GPU memory usage optimization
- **optimization_level**: Conversion optimization level (low/medium/high)

For detailed API reference, see the [auto-generated documentation](nemo_rl/nemo_rl.converters). 