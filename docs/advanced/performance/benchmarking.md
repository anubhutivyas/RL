# Benchmarking

This guide covers benchmarking and performance testing for NeMo RL, including throughput measurement, memory profiling, and performance optimization.

## Overview

Benchmarking is essential for understanding and optimizing NeMo RL performance. This guide covers tools and techniques for measuring training speed, memory usage, and system efficiency.

## Key Metrics

### Training Throughput

Measure samples processed per second:

```python
import time

def measure_throughput(model, dataloader, num_steps=100):
    """Measure training throughput in samples per second."""
    model.train()
    start_time = time.time()
    samples_processed = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
            
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        samples_processed += batch['input_ids'].size(0)
    
    end_time = time.time()
    throughput = samples_processed / (end_time - start_time)
    
    return {
        "samples_per_second": throughput,
        "total_samples": samples_processed,
        "total_time": end_time - start_time
    }
```

### Memory Usage

Monitor GPU memory consumption:

```python
import torch

def measure_memory_usage():
    """Measure current GPU memory usage."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    
    return {
        "allocated_mb": allocated / 1024**2,
        "reserved_mb": reserved / 1024**2,
        "max_allocated_mb": max_allocated / 1024**2
    }
```

### GPU Utilization

Monitor GPU utilization during training:

```python
def monitor_gpu_utilization():
    """Monitor GPU utilization using nvidia-smi."""
    import subprocess
    
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )
    
    utilization, memory = result.stdout.strip().split(', ')
    return {
        "gpu_utilization": int(utilization),
        "memory_used_mb": int(memory)
    }
```

## Benchmarking Tools

### PyTorch Profiler

Use PyTorch's built-in profiler:

```python
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training(model, dataloader):
    """Profile training performance using PyTorch profiler."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        for batch in dataloader:
            with record_function("training_step"):
                outputs = model(batch)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return prof
```

### Memory Profiler

Profile memory usage over time:

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function to profile memory usage."""
    large_tensor = torch.randn(1000, 1000, device='cuda')
    result = large_tensor @ large_tensor.T
    return result
```

### Ray Dashboard

Monitor Ray cluster performance:

```python
import ray

def monitor_ray_cluster():
    """Monitor Ray cluster resources."""
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    
    return {
        "cluster_resources": cluster_resources,
        "available_resources": available_resources,
        "utilization": {
            "cpu": (cluster_resources.get("CPU", 0) - available_resources.get("CPU", 0)) / cluster_resources.get("CPU", 1),
            "gpu": (cluster_resources.get("GPU", 0) - available_resources.get("GPU", 0)) / cluster_resources.get("GPU", 1)
        }
    }
```

## Benchmarking Configurations

### Baseline Configuration

Measure baseline performance:

```yaml
# Baseline configuration
model:
  name: "llama-7b"
  precision: "float32"
  
training:
  batch_size: 8
  gradient_accumulation_steps: 1
  mixed_precision: false
  
hardware:
  num_gpus: 1
  gpu_memory: "24GB"
```

### Optimized Configuration

Test optimized settings:

```yaml
# Optimized configuration
model:
  name: "llama-7b"
  precision: "bfloat16"
  
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  mixed_precision: true
  gradient_checkpointing: true
  
hardware:
  num_gpus: 4
  gpu_memory: "24GB"
```

## Performance Testing

### Scalability Testing

Test performance scaling with resources:

```python
def scalability_test(model_configs, hardware_configs):
    """Test performance scaling across different configurations."""
    results = {}
    
    for model_config in model_configs:
        for hw_config in hardware_configs:
            # Setup configuration
            config = {**model_config, **hw_config}
            
            # Run benchmark
            result = run_benchmark(config)
            
            # Store results
            key = f"{model_config['name']}_{hw_config['num_gpus']}gpu"
            results[key] = result
    
    return results
```

### Memory Efficiency Testing

Test memory efficiency:

```python
def memory_efficiency_test():
    """Test memory efficiency of different configurations."""
    configs = [
        {"mixed_precision": False, "gradient_checkpointing": False},
        {"mixed_precision": True, "gradient_checkpointing": False},
        {"mixed_precision": True, "gradient_checkpointing": True}
    ]
    
    results = {}
    for config in configs:
        # Measure memory usage
        memory_before = measure_memory_usage()
        
        # Run training step
        train_step()
        
        memory_after = measure_memory_usage()
        
        results[str(config)] = {
            "memory_increase_mb": memory_after["allocated_mb"] - memory_before["allocated_mb"],
            "peak_memory_mb": memory_after["max_allocated_mb"]
        }
    
    return results
```

## Benchmarking Scripts

### Automated Benchmarking

Create automated benchmarking scripts:

```python
#!/usr/bin/env python3
"""Automated benchmarking script for NeMo RL."""

import argparse
import json
import time
from pathlib import Path

def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    benchmarks = {
        "throughput": measure_throughput,
        "memory": measure_memory_usage,
        "gpu_utilization": monitor_gpu_utilization,
        "scalability": scalability_test
    }
    
    results = {}
    for name, benchmark in benchmarks.items():
        print(f"Running {name} benchmark...")
        results[name] = benchmark()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo RL benchmarks")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    results = run_benchmark_suite()
```

### Continuous Benchmarking

Set up continuous benchmarking:

```python
def continuous_benchmarking():
    """Set up continuous benchmarking pipeline."""
    import schedule
    import time
    
    def run_daily_benchmarks():
        """Run daily benchmark suite."""
        results = run_benchmark_suite()
        
        # Compare with baseline
        compare_with_baseline(results)
        
        # Alert on regressions
        check_for_regressions(results)
    
    # Schedule daily benchmarks
    schedule.every().day.at("02:00").do(run_daily_benchmarks)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
```

## Performance Analysis

### Bottleneck Identification

Identify performance bottlenecks:

```python
def identify_bottlenecks(profiler_output):
    """Identify performance bottlenecks from profiler output."""
    bottlenecks = []
    
    # Analyze CPU bottlenecks
    cpu_ops = profiler_output.key_averages().filter(lambda x: x.cpu_time_total > 1000)
    for op in cpu_ops:
        bottlenecks.append({
            "type": "cpu",
            "operation": op.key,
            "time_ms": op.cpu_time_total / 1000,
            "percentage": op.cpu_time_total / profiler_output.total_cpu_time * 100
        })
    
    # Analyze GPU bottlenecks
    gpu_ops = profiler_output.key_averages().filter(lambda x: x.cuda_time_total > 1000)
    for op in gpu_ops:
        bottlenecks.append({
            "type": "gpu",
            "operation": op.key,
            "time_ms": op.cuda_time_total / 1000,
            "percentage": op.cuda_time_total / profiler_output.total_cuda_time * 100
        })
    
    return sorted(bottlenecks, key=lambda x: x["time_ms"], reverse=True)
```

### Performance Regression Detection

Detect performance regressions:

```python
def detect_regressions(current_results, baseline_results, threshold=0.1):
    """Detect performance regressions compared to baseline."""
    regressions = []
    
    for metric in current_results:
        if metric in baseline_results:
            current_value = current_results[metric]
            baseline_value = baseline_results[metric]
            
            # Calculate regression percentage
            regression_pct = (baseline_value - current_value) / baseline_value
            
            if regression_pct > threshold:
                regressions.append({
                    "metric": metric,
                    "current": current_value,
                    "baseline": baseline_value,
                    "regression_pct": regression_pct * 100
                })
    
    return regressions
```

## Reporting

### Benchmark Reports

Generate comprehensive benchmark reports:

```python
def generate_benchmark_report(results):
    """Generate comprehensive benchmark report."""
    report = {
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results.values() if r.get("status") == "passed"),
            "failed": sum(1 for r in results.values() if r.get("status") == "failed")
        },
        "performance_metrics": {
            "throughput": results.get("throughput", {}),
            "memory_usage": results.get("memory", {}),
            "gpu_utilization": results.get("gpu_utilization", {})
        },
        "recommendations": generate_recommendations(results)
    }
    
    return report

def generate_recommendations(results):
    """Generate performance optimization recommendations."""
    recommendations = []
    
    # Check throughput
    if results.get("throughput", {}).get("samples_per_second", 0) < 100:
        recommendations.append("Consider increasing batch size or using mixed precision")
    
    # Check memory usage
    memory_usage = results.get("memory", {}).get("max_allocated_mb", 0)
    if memory_usage > 20000:  # 20GB
        recommendations.append("Consider enabling gradient checkpointing or reducing batch size")
    
    # Check GPU utilization
    gpu_util = results.get("gpu_utilization", {}).get("gpu_utilization", 0)
    if gpu_util < 80:
        recommendations.append("GPU utilization is low - check for CPU bottlenecks")
    
    return recommendations
```

## Best Practices

### Benchmarking Guidelines

1. **Consistent Environment**
   - Use same hardware configuration
   - Control for system load
   - Run multiple iterations

2. **Comprehensive Testing**
   - Test different model sizes
   - Test different batch sizes
   - Test different precision settings

3. **Regular Monitoring**
   - Set up continuous benchmarking
   - Monitor for regressions
   - Track performance trends

### Optimization Workflow

1. **Baseline Measurement**
   - Establish performance baseline
   - Document configuration
   - Set performance targets

2. **Optimization Testing**
   - Test different configurations
   - Measure impact of changes
   - Validate improvements

3. **Production Deployment**
   - Monitor production performance
   - Track performance over time
   - Alert on regressions

## Next Steps

After setting up benchmarking:

1. **Establish Baselines**: Create performance baselines for your models
2. **Monitor Trends**: Track performance over time
3. **Optimize Continuously**: Use benchmarks to guide optimizations
4. **Scale Testing**: Extend benchmarks to larger models and clusters

For more advanced topics, see:
- [Performance Profiling](../profiling.md) - Detailed profiling techniques
- [Memory Optimization](memory-optimization.md) - Memory optimization strategies
- [Distributed Training](distributed-training.md) - Multi-GPU training benchmarks 