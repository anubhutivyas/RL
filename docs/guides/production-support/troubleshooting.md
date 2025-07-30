# Troubleshoot NeMo RL

This guide helps you diagnose and resolve common issues when using NeMo RL.

## Overview

This troubleshooting guide covers common problems you might encounter during training, evaluation, and deployment. Each section includes symptoms, causes, and solutions.

## Training Issues

### Memory Errors

**Symptoms:**
- CUDA out of memory errors
- OOM (Out of Memory) errors
- Training crashes with memory-related messages

**Causes:**
- Batch size too large
- Sequence length too long
- Model too large for available GPU memory
- Insufficient system RAM

**Solutions:**

1. **Reduce Batch Size:**
   ```yaml
   training:
     batch_size: 2  # Reduce from 4 or 8
   ```

2. **Enable Gradient Accumulation:**
   ```yaml
   training:
     gradient_accumulation_steps: 4
     batch_size: 2  # Effective batch size = 2 * 4 = 8
   ```

3. **Reduce Sequence Length:**
   ```yaml
   model:
     max_length: 1024  # Reduce from 2048
   ```

4. **Enable Gradient Checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true
   ```

5. **Use Mixed Precision:**
   ```yaml
   training:
     fp16: true
     # or
     bf16: true
   ```

### Training Instability

**Symptoms:**
- Loss spikes or divergence
- NaN values in loss
- Unstable training curves
- Poor convergence

**Causes:**
- Learning rate too high
- Poor initialization
- Data quality issues
- Algorithm-specific issues

**Solutions:**

1. **Reduce Learning Rate:**
   ```yaml
   training:
     learning_rate: 1e-5  # Reduce from 5e-5
   ```

2. **Add Warmup:**
   ```yaml
   training:
     warmup_steps: 100
     warmup_ratio: 0.1
   ```

3. **Adjust DPO Parameters:**
   ```yaml
   algorithm:
     beta: 0.05  # Reduce from 0.1
     temperature: 0.5  # Reduce from 1.0
   ```

4. **Check Data Quality:**
   ```python
   # Validate data
   from nemo_rl.data import validate_dataset
   
   errors = validate_dataset("path/to/data.json")
   if errors:
       print("Data issues found:", errors)
   ```

### Slow Training

**Symptoms:**
- Training takes much longer than expected
- Low GPU utilization
- Slow data loading

**Causes:**
- Inefficient data loading
- Suboptimal distributed setup
- Hardware bottlenecks
- Configuration issues

**Solutions:**

1. **Optimize Data Loading:**
   ```yaml
   data:
     num_workers: 4
     pin_memory: true
     prefetch_factor: 2
   ```

2. **Enable FSDP:**
   ```yaml
   cluster:
     name: "fsdp"
     fsdp_config:
       mixed_precision: true
       activation_checkpointing: true
   ```

3. **Use Ray for Distributed Training:**
   ```yaml
   cluster:
     name: "ray"
     num_workers: 4
     resources_per_worker:
       GPU: 0.25
   ```

## Evaluation Issues

### Poor Evaluation Results

**Symptoms:**
- Low accuracy scores
- Poor BLEU/ROUGE scores
- Inconsistent evaluation metrics

**Causes:**
- Overfitting to training data
- Evaluation dataset mismatch
- Metric calculation errors
- Model not converged

**Solutions:**

1. **Check for Overfitting:**
   ```python
   # Monitor training vs validation loss
   import matplotlib.pyplot as plt
   
   plt.plot(train_losses, label='Training')
   plt.plot(val_losses, label='Validation')
   plt.legend()
   plt.show()
   ```

2. **Validate Evaluation Data:**
   ```python
   # Check evaluation data format
   from nemo_rl.evals import validate_eval_data
   
   errors = validate_eval_data("eval_data.json")
   ```

3. **Adjust Evaluation Parameters:**
   ```yaml
   evaluation:
     generation:
       temperature: 0.7
       top_p: 0.9
       max_new_tokens: 128
   ```

### Evaluation Crashes

**Symptoms:**
- Evaluation process crashes
- Memory errors during evaluation
- Timeout errors

**Causes:**
- Insufficient memory
- Long sequences
- Batch size too large

**Solutions:**

1. **Reduce Evaluation Batch Size:**
   ```yaml
   evaluation:
     batch_size: 1  # Reduce from default
   ```

2. **Limit Sequence Length:**
   ```yaml
   evaluation:
     generation:
       max_new_tokens: 64  # Reduce from 128
   ```

3. **Use Gradient Checkpointing:**
   ```yaml
   evaluation:
     gradient_checkpointing: true
   ```

## Configuration Issues

### Configuration Validation Errors

**Symptoms:**
- Configuration validation fails
- Unknown configuration parameters
- Type errors in configuration

**Causes:**
- Invalid YAML syntax
- Unknown parameters
- Type mismatches
- Missing required fields

**Solutions:**

1. **Validate Configuration:**
   ```bash
   python -m nemo_rl.config --validate config.yaml
   ```

2. **Check Configuration Schema:**
   ```python
   from nemo_rl.utils.config import validate_config
   
   config = load_config("config.yaml")
   errors = validate_config(config)
   for error in errors:
       print(f"Error: {error}")
   ```

3. **Use Configuration Template:**
   ```bash
   # Generate template
   python -m nemo_rl.config --template dpo > config_template.yaml
   ```

### Missing Dependencies

**Symptoms:**
- Import errors
- Module not found errors
- Missing package errors

**Causes:**
- Incomplete installation
- Version conflicts
- Missing optional dependencies

**Solutions:**

1. **Install Dependencies:**
   ```bash
   pip install nemo-rl[all]
   # or
   pip install nemo-rl[training,evaluation]
   ```

2. **Check Installation:**
   ```bash
   python -c "import nemo_rl; print(nemo_rl.__version__)"
   ```

3. **Install Optional Dependencies:**
   ```bash
   pip install torch transformers datasets
   pip install ray[default]  # For distributed training
   ```

## Distributed Training Issues

### Ray Cluster Issues

**Symptoms:**
- Ray cluster connection failures
- Resource allocation errors
- Worker failures

**Causes:**
- Ray cluster not started
- Resource conflicts
- Network issues
- Configuration problems

**Solutions:**

1. **Start Ray Cluster:**
   ```bash
   # Start Ray head node
   ray start --head --port=6379
   
   # Check cluster status
   ray status
   ```

2. **Check Resources:**
   ```bash
   # Check available resources
   ray list resources
   
   # Check cluster nodes
   ray list nodes
   ```

3. **Adjust Resource Configuration:**
   ```yaml
   cluster:
     name: "ray"
     num_workers: 2  # Reduce if resources limited
     resources_per_worker:
       CPU: 1
       GPU: 0.5  # Reduce GPU allocation
   ```

### FSDP Issues

**Symptoms:**
- FSDP initialization errors
- Memory errors with FSDP
- Slow training with FSDP

**Causes:**
- Incorrect FSDP configuration
- Memory fragmentation
- Suboptimal sharding strategy

**Solutions:**

1. **Adjust FSDP Configuration:**
   ```yaml
   cluster:
     name: "fsdp"
     fsdp_config:
       mixed_precision: true
       activation_checkpointing: true
       sharding_strategy: "FULL_SHARD"
       cpu_offload: false
   ```

2. **Use CPU Offloading:**
   ```yaml
   cluster:
     name: "fsdp"
     fsdp_config:
       cpu_offload: true
       state_dict_type: "FULL_STATE_DICT"
   ```

## Model Issues

### Model Loading Errors

**Symptoms:**
- Model loading fails
- Checkpoint corruption errors
- Architecture mismatch errors

**Causes:**
- Corrupted checkpoint files
- Version incompatibilities
- Missing model files
- Architecture changes

**Solutions:**

1. **Check Checkpoint Integrity:**
   ```python
   from nemo_rl.utils.checkpoint import validate_checkpoint
   
   errors = validate_checkpoint("checkpoints/model.pt")
   if errors:
       print("Checkpoint issues:", errors)
   ```

2. **Load with Error Handling:**
   ```python
   try:
       model = load_model("checkpoints/model.pt")
   except Exception as e:
       print(f"Loading failed: {e}")
       # Try loading with different options
       model = load_model("checkpoints/model.pt", strict=False)
   ```

3. **Convert Checkpoint Format:**
   ```bash
   python -m nemo_rl.convert --input checkpoints/model.pt --output converted_model
   ```

### Generation Issues

**Symptoms:**
- Poor generation quality
- Repetitive outputs
- Inappropriate responses

**Causes:**
- Training data quality
- Generation parameters
- Model not converged
- Prompt engineering issues

**Solutions:**

1. **Adjust Generation Parameters:**
   ```yaml
   evaluation:
     generation:
       temperature: 0.7
       top_p: 0.9
       top_k: 50
       do_sample: true
       repetition_penalty: 1.1
   ```

2. **Improve Prompt Engineering:**
   ```python
   # Use better prompts
   prompts = [
       "Please solve the following math problem step by step:",
       "Given the equation, provide a detailed solution:",
       "Solve this mathematical problem with clear reasoning:"
   ]
   ```

3. **Check Training Data:**
   ```python
   # Analyze training data quality
   from nemo_rl.data import analyze_dataset
   
   analysis = analyze_dataset("training_data.json")
   print("Data quality metrics:", analysis)
   ```

## Environment Issues

### CUDA Issues

**Symptoms:**
- CUDA initialization errors
- GPU memory errors
- CUDA version conflicts

**Causes:**
- CUDA version mismatch
- Insufficient GPU memory
- Driver issues
- PyTorch-CUDA incompatibility

**Solutions:**

1. **Check CUDA Installation:**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Check PyTorch CUDA support
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install Compatible PyTorch:**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Set CUDA Environment Variables:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   export CUDA_LAUNCH_BLOCKING=1
   ```

### Environment Setup Issues

**Symptoms:**
- Package conflicts
- Path issues
- Environment variable problems

**Causes:**
- Conflicting package versions
- Incorrect environment setup
- Missing environment variables

**Solutions:**

1. **Create Clean Environment:**
   ```bash
   # Create new conda environment
   conda create -n nemo_rl python=3.9
   conda activate nemo_rl
   
   # Install NeMo RL
   pip install nemo-rl[all]
   ```

2. **Check Environment:**
   ```bash
   # Check Python version
   python --version
   
   # Check installed packages
   pip list | grep -E "(torch|transformers|nemo)"
   ```

3. **Set Environment Variables:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/nemo_rl"
   export NEMO_RL_CACHE_DIR="/path/to/cache"
   ```

## Debugging Tools

### Logging

Enable detailed logging for debugging:

```yaml
logging:
  level: DEBUG
  file: training.log
  tensorboard: true
  wandb: true
```

### Debug Mode

Run with debug mode for more information:

```bash
python -m nemo_rl.train --config config.yaml --debug
```

### Profiling

Profile training for performance issues:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Your training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Getting Help

### Self-Diagnosis

1. **Check Logs:** Look for error messages and warnings
2. **Validate Configuration:** Use configuration validation tools
3. **Test Components:** Test individual components separately
4. **Check Documentation:** Refer to relevant documentation sections

### Community Support

1. **GitHub Issues:** Report bugs and feature requests
2. **Discussions:** Ask questions in GitHub Discussions
3. **Documentation:** Check the comprehensive documentation
4. **Examples:** Review example configurations and scripts

### Reporting Issues

When reporting issues, include:

1. **Environment Information:**
   - OS and version
   - Python version
   - NeMo RL version
   - CUDA version (if applicable)

2. **Configuration:**
   - Relevant configuration files
   - Command line arguments
   - Environment variables

3. **Error Details:**
   - Complete error messages
   - Stack traces
   - Log files

4. **Reproduction Steps:**
   - Clear steps to reproduce the issue
   - Minimal example if possible

For more help, see the [Documentation Guide](documentation.md) and [Configuration Reference](../../configuration-cli/configuration-reference.md). 