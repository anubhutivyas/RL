# Troubleshooting

This guide covers common issues, error messages, and solutions for NeMo RL. If you encounter a problem not covered here, please check the [GitHub Issues](https://github.com/your-repo/issues) or create a new one.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`warning;1.5em;sd-mr-1` Common Errors
:link: #common-errors
:link-type: ref

Frequently encountered error messages and solutions.

+++
{bdg-warning}`Errors`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Issues
:link: #configuration-issues
:link-type: ref

Problems with configuration files and parameters.

+++
{bdg-info}`Config`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Training
:link: #distributed-training-issues
:link-type: ref

Issues with Ray and distributed training.

+++
{bdg-warning}`Distributed`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Model Issues
:link: #model-issues
:link-type: ref

Problems with model loading and training.

+++
{bdg-secondary}`Models`
:::

::::

## Common Errors

### CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   algorithm:
     batch_size: 2  # Reduce from 4 to 2
   ```

2. **Enable gradient accumulation:**
   ```yaml
   algorithm:
     gradient_accumulation_steps: 4
     batch_size: 1
   ```

3. **Reduce model size or use smaller model:**
   ```yaml
   model:
     name: "llama2-7b"  # Use smaller model
   ```

4. **Enable memory optimization:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
   ```

### Model Loading Errors

**Error Message:**
```
OSError: We couldn't connect to 'https://huggingface.co/...'
```

**Solutions:**

1. **Check internet connection**
2. **Use local model path:**
   ```yaml
   model:
     name: "/path/to/local/model"
   ```
3. **Set Hugging Face token:**
   ```bash
   export HF_TOKEN="your_token"
   ```

### Configuration Validation Errors

**Error Message:**
```
ValidationError: Invalid configuration parameter
```

**Solutions:**

1. **Validate configuration:**
   ```bash
   python -m nemo_rl.config validate --config training.yaml
   ```

2. **Check parameter names and types**
3. **Use configuration template:**
   ```bash
   python -m nemo_rl.config template --algorithm dpo --output dpo.yaml
   ```

## Configuration Issues

### Missing Required Parameters

**Error Message:**
```
Missing required parameter: algorithm.name
```

**Solutions:**

1. **Add missing parameter to configuration:**
   ```yaml
   algorithm:
     name: "dpo"  # Add this line
   ```

2. **Use configuration template:**
   ```bash
   python -m nemo_rl.config template --algorithm dpo --output dpo.yaml
   ```

### Invalid Parameter Values

**Error Message:**
```
Invalid value for learning_rate: must be positive
```

**Solutions:**

1. **Check parameter ranges:**
   - Learning rates: positive values
   - Batch sizes: positive integers
   - Model names: valid paths or names

2. **Use reasonable defaults:**
   ```yaml
   algorithm:
     learning_rate: 1e-5  # Use small positive value
     batch_size: 4        # Use reasonable batch size
   ```

### Environment Variable Issues

**Problem:** Environment variables not being recognized

**Solutions:**

1. **Check variable naming:**
   ```bash
   # Correct format
   export NEMO_RL_ALGORITHM_LEARNING_RATE=2e-5
   export NEMO_RL_MODEL_NAME="llama2-7b"
   ```

2. **Verify variable is set:**
   ```bash
   echo $NEMO_RL_ALGORITHM_LEARNING_RATE
   ```

3. **Restart terminal after setting variables**

## Distributed Training Issues

### Ray Connection Errors

**Error Message:**
```
RayConnectionError: Failed to connect to Ray cluster
```

**Solutions:**

1. **Start Ray cluster:**
   ```bash
   ray start --head
   ```

2. **Check Ray status:**
   ```bash
   ray status
   ```

3. **Use local mode for testing:**
   ```yaml
   distributed:
     backend: "ray"
     local_mode: true
   ```

### Worker Allocation Errors

**Error Message:**
```
Resource allocation failed: insufficient resources
```

**Solutions:**

1. **Reduce resource requirements:**
   ```yaml
   distributed:
     num_workers: 2        # Reduce from 4
     num_gpus_per_worker: 1
     memory_per_worker: "8GB"  # Reduce from 16GB
   ```

2. **Check available resources:**
   ```bash
   nvidia-smi  # Check GPU availability
   ray status  # Check Ray cluster resources
   ```

3. **Use single GPU mode:**
   ```yaml
   distributed:
     backend: "torch"  # Use PyTorch DDP instead of Ray
   ```

### Communication Errors

**Error Message:**
```
NCCL error: unhandled cuda error
```

**Solutions:**

1. **Check GPU connectivity:**
   ```bash
   nvidia-smi topo -m
   ```

2. **Use different communication backend:**
   ```yaml
   distributed:
     communication:
       backend: "gloo"  # Use gloo instead of nccl
   ```

3. **Reduce batch size and workers**

## Model Issues

### Model Loading Failures

**Error Message:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check model compatibility:**
   ```yaml
   model:
     backend: "huggingface"  # Ensure correct backend
     trust_remote_code: true  # For custom models
   ```

2. **Verify model path and format**
3. **Use compatible model version**

### Generation Errors

**Error Message:**
```
RuntimeError: Expected tensor to have size X but got size Y
```

**Solutions:**

1. **Check input format:**
   ```python
   # Ensure inputs are properly tokenized
   inputs = tokenizer(prompts, return_tensors="pt", padding=True)
   ```

2. **Verify model configuration:**
   ```yaml
   generation:
     max_new_tokens: 512
     pad_token_id: tokenizer.pad_token_id
     eos_token_id: tokenizer.eos_token_id
   ```

### Memory Issues During Training

**Error Message:**
```
RuntimeError: CUDA out of memory during training
```

**Solutions:**

1. **Enable memory optimization:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
     device_map: "auto"
   ```

2. **Use gradient checkpointing:**
   ```yaml
   algorithm:
     gradient_checkpointing: true
   ```

3. **Reduce model precision:**
   ```yaml
   model:
     torch_dtype: "bfloat16"  # Use bfloat16 for better memory efficiency
   ```

## Data Issues

### Dataset Loading Errors

**Error Message:**
```
DatasetNotFoundError: Dataset not found
```

**Solutions:**

1. **Check dataset path:**
   ```yaml
   data:
     dataset: "/correct/path/to/dataset"  # Use absolute path
   ```

2. **Verify dataset format:**
   - Ensure dataset is in expected format
   - Check required columns are present

3. **Use HuggingFace dataset:**
   ```yaml
   data:
     dataset: "huggingface-dataset-name"
   ```

### Data Preprocessing Errors

**Error Message:**
```
TokenizationError: Input too long
```

**Solutions:**

1. **Reduce sequence length:**
   ```yaml
   data:
     preprocessing:
       max_length: 1024  # Reduce from 2048
   ```

2. **Enable truncation:**
   ```yaml
   data:
     preprocessing:
       truncation: true
       padding: "max_length"
   ```

## Performance Issues

### Slow Training

**Problem:** Training is slower than expected

**Solutions:**

1. **Optimize data loading:**
   ```yaml
   data:
     num_workers: 8        # Increase from 4
     pin_memory: true
     prefetch_factor: 2
   ```

2. **Use mixed precision:**
   ```yaml
   algorithm:
     mixed_precision: true
     fp16: true
   ```

3. **Enable optimizations:**
   ```yaml
   model:
     use_flash_attention: true
     use_cache: false
   ```

### High Memory Usage

**Problem:** Excessive memory consumption

**Solutions:**

1. **Reduce batch size and use gradient accumulation:**
   ```yaml
   algorithm:
     batch_size: 1
     gradient_accumulation_steps: 4
   ```

2. **Enable memory optimizations:**
   ```yaml
   model:
     use_cache: false
     torch_dtype: "float16"
   ```

3. **Use smaller model or model parallelism**

## Debugging Tips

### Enable Debug Logging

```bash
python -m nemo_rl.train --config training.yaml --log-level DEBUG
```

### Validate Configuration

```bash
python -m nemo_rl.config validate --config training.yaml
```

### Check System Resources

```bash
# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check Ray cluster
ray status
```

### Dry Run Mode

```bash
python -m nemo_rl.train --config training.yaml --dry-run
```

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues on GitHub**
3. **Validate your configuration**
4. **Check system requirements**
5. **Try with minimal configuration**

### Providing Information

When reporting issues, include:

1. **Error message and stack trace**
2. **Configuration file (sanitized)**
3. **System information:**
   ```bash
   python --version
   nvidia-smi
   ray status
   ```
4. **Steps to reproduce**
5. **Expected vs actual behavior**

### Useful Commands

```bash
# Check NeMo RL version
python -c "import nemo_rl; print(nemo_rl.__version__)"

# Validate configuration
python -m nemo_rl.config validate --config training.yaml

# Test model loading
python -c "from nemo_rl.models import HuggingFaceModel; model = HuggingFaceModel.from_pretrained('llama2-7b')"

# Check Ray cluster
ray status
ray list nodes
```

## Common Workarounds

### Temporary Solutions

1. **Use smaller model for testing**
2. **Reduce batch size and sequence length**
3. **Use single GPU mode**
4. **Disable optimizations temporarily**

### Performance vs Memory Trade-offs

- **Higher batch size** → Better performance, more memory
- **Longer sequences** → More context, more memory
- **Mixed precision** → Less memory, potential precision loss
- **Gradient accumulation** → Same effective batch size, less memory

### Configuration Templates

Use provided templates as starting points:

```bash
# Generate templates
python -m nemo_rl.config template --algorithm dpo --output dpo.yaml
python -m nemo_rl.config template --algorithm grpo --output grpo.yaml
python -m nemo_rl.config template --algorithm sft --output sft.yaml
``` 