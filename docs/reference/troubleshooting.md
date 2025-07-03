# Troubleshooting

This guide covers common issues and their solutions when using NeMo RL.

## Installation Issues

### ModuleNotFoundError: No module named 'megatron'

**Problem**: Missing Megatron submodule or virtual environment issue.

**Solution**:
```bash
# Initialize submodules
git submodule update --init --recursive

# Force rebuild virtual environments
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...
```

### CUDA Installation Issues

**Problem**: CUDA not found or incompatible version.

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Permission Denied Errors

**Problem**: Script execution permissions.

**Solution**:
```bash
chmod +x examples/*.py
```

## Training Issues

### Out of Memory (OOM)

**Problem**: GPU memory exceeded during training.

**Solutions**:

1. **Reduce batch size**:
   ```bash
   sft.micro_batch_size=1
   sft.gradient_accumulation_steps=4
   ```

2. **Enable gradient checkpointing**:
   ```bash
   policy.gradient_checkpointing=true
   ```

3. **Use mixed precision**:
   ```bash
   policy.mixed_precision=true
   ```

4. **Reduce model size**:
   ```bash
   policy.model_name=Qwen/Qwen2.5-0.5B  # Use smaller model
   ```

5. **Enable tensor parallelism**:
   ```bash
   policy.dtensor_cfg.enabled=true
   policy.dtensor_cfg.tensor_parallel_size=2
   ```

### Slow Training

**Problem**: Training is slower than expected.

**Solutions**:

1. **Enable mixed precision**:
   ```bash
   policy.mixed_precision=true
   ```

2. **Increase batch size** (if memory allows):
   ```bash
   sft.micro_batch_size=4
   ```

3. **Use more GPUs**:
   ```bash
   cluster.gpus_per_node=4
   ```

4. **Optimize data loading**:
   ```bash
   data.num_workers=4
   ```

### Loss Not Decreasing

**Problem**: Training loss is not improving.

**Solutions**:

1. **Check learning rate**:
   ```bash
   sft.learning_rate=1e-4  # Try different learning rates
   ```

2. **Verify data format**:
   - Ensure dataset is properly formatted
   - Check tokenization is correct

3. **Monitor gradients**:
   - Check for gradient clipping
   - Verify loss computation

4. **Reduce model complexity**:
   - Start with smaller model
   - Use simpler dataset

### Authentication Errors

**Problem**: Hugging Face authentication issues.

**Solution**:
```bash
huggingface-cli login
# Enter your token when prompted
```

## Distributed Training Issues

### NCCL Communication Errors

**Problem**: GPU communication failures in distributed training.

**Solutions**:

1. **Set NCCL environment variables**:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1
   ```

2. **Check network connectivity**:
   ```bash
   # Test inter-node communication
   nc -v node1 12345
   ```

3. **Use specific network interface**:
   ```bash
   export NCCL_SOCKET_IFNAME=eth0
   ```

### Ray Connection Issues

**Problem**: Ray cluster connection problems.

**Solutions**:

1. **Check Ray status**:
   ```bash
   ray status
   ```

2. **Restart Ray**:
   ```bash
   ray stop
   ray start --head
   ```

3. **Check firewall settings**:
   - Ensure ports 6379, 10001 are open
   - Configure firewall for Ray communication

### Load Balancing Issues

**Problem**: Uneven GPU utilization across workers.

**Solutions**:

1. **Check worker distribution**:
   ```bash
   # Monitor GPU usage
   nvidia-smi
   ```

2. **Adjust worker configuration**:
   ```bash
   cluster.workers_per_node=4  # Match GPU count
   ```

3. **Check data distribution**:
   - Ensure data is evenly distributed
   - Monitor batch sizes per worker

## Model Issues

### Model Loading Errors

**Problem**: Failed to load pre-trained model.

**Solutions**:

1. **Check model name**:
   ```bash
   # Verify model exists on Hugging Face
   policy.model_name=Qwen/Qwen2.5-0.5B
   ```

2. **Clear cache**:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

3. **Check disk space**:
   ```bash
   df -h
   ```

### Tokenization Errors

**Problem**: Tokenization mismatches or errors.

**Solutions**:

1. **Use model's tokenizer**:
   ```bash
   # Ensure tokenizer matches model
   policy.tokenizer_name=Qwen/Qwen2.5-0.5B
   ```

2. **Check sequence lengths**:
   ```bash
   # Reduce max sequence length
   policy.max_seq_len=2048
   ```

3. **Verify padding**:
   - Ensure right padding is used
   - Check padding token configuration

## Environment Issues

### Python Environment Problems

**Problem**: Package conflicts or missing dependencies.

**Solutions**:

1. **Use virtual environment**:
   ```bash
   uv sync  # Recommended
   # or
   conda create -n nemo-rl python=3.9
   conda activate nemo-rl
   ```

2. **Check package versions**:
   ```bash
   pip list | grep torch
   pip list | grep transformers
   ```

3. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### CUDA Version Mismatch

**Problem**: PyTorch and CUDA version incompatibility.

**Solution**:
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvidia-smi

# Install compatible PyTorch
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Configuration Issues

### YAML Configuration Errors

**Problem**: Invalid configuration file or parameters.

**Solutions**:

1. **Validate YAML syntax**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **Check parameter names**:
   - Verify parameter names match documentation
   - Check for typos in configuration

3. **Use CLI overrides**:
   ```bash
   # Override specific parameters
   uv run python run_sft.py sft.learning_rate=1e-4
   ```

### Missing Configuration Parameters

**Problem**: Required parameters not specified.

**Solution**:
```bash
# Check default configuration
cat examples/configs/sft.yaml

# Add missing parameters
uv run python run_sft.py \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    data.dataset_name=HuggingFaceH4/ultrachat_200k
```

## Performance Issues

### Low GPU Utilization

**Problem**: GPUs not being used efficiently.

**Solutions**:

1. **Increase batch size**:
   ```bash
   sft.micro_batch_size=8
   ```

2. **Optimize data pipeline**:
   ```bash
   data.num_workers=4
   data.prefetch_factor=2
   ```

3. **Check for bottlenecks**:
   - Monitor CPU usage
   - Check disk I/O
   - Verify network bandwidth

### Memory Leaks

**Problem**: Memory usage increasing over time.

**Solutions**:

1. **Enable gradient checkpointing**:
   ```bash
   policy.gradient_checkpointing=true
   ```

2. **Clear cache periodically**:
   ```bash
   # Add to training script
   torch.cuda.empty_cache()
   ```

3. **Monitor memory usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Logging and Monitoring Issues

### WandB Connection Problems

**Problem**: Cannot connect to Weights & Biases.

**Solutions**:

1. **Check API key**:
   ```bash
   export WANDB_API_KEY="your_api_key"
   ```

2. **Test connection**:
   ```bash
   wandb login
   ```

3. **Disable WandB** (if needed):
   ```bash
   logger.wandb.enabled=false
   ```

### Log File Issues

**Problem**: Logs not being written or corrupted.

**Solutions**:

1. **Check disk space**:
   ```bash
   df -h
   ```

2. **Verify permissions**:
   ```bash
   ls -la logs/
   ```

3. **Change log directory**:
   ```bash
   logger.log_dir=/path/to/logs
   ```

## Getting Help

### Debug Mode

Enable debug logging for more information:

```bash
export NEMO_RL_DEBUG=1
uv run python run_sft.py ...
```

### Collecting Information

When reporting issues, include:

1. **Environment information**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   python --version
   ```

2. **Configuration**:
   - YAML configuration file
   - CLI overrides used

3. **Error logs**:
   - Full error traceback
   - Console output
   - Log files

### Support Channels

- **GitHub Issues**: [NeMo RL Issues](https://github.com/NVIDIA-NeMo/RL/issues)
- **Documentation**: Check relevant guides and reference pages
- **Community**: Join the [NeMo Discord](https://discord.gg/nvidia-nemo)
- **Email**: Contact NVIDIA support for enterprise issues 