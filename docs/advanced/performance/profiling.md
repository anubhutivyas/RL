---
description: "Advanced profiling techniques with NSYS, PyTorch Profiler, and custom profiling tools."
tags: ["profiling", "performance", "optimization", "nsys"]
categories: ["performance"]
---

# Performance Profiling

This document provides comprehensive profiling techniques for optimizing NeMo RL training performance, including GPU profiling, memory analysis, and communication optimization.

## Overview

Performance profiling is essential for identifying bottlenecks and optimizing training efficiency. This guide covers advanced profiling tools and methodologies specific to NeMo RL training workflows.

## Profiling Tools

### NSYS Profiler

#### GPU Performance Analysis
NeMo RL provides integrated NSYS profiling through environment variables for Ray workers:

```bash
# Profile policy workers for steps 2-3
NRL_NSYS_PROFILE_STEP_RANGE=2:3 NRL_NSYS_WORKER_PATTERNS="*policy*" uv run examples/run_grpo.py

# Profile multiple worker types
NRL_NSYS_PROFILE_STEP_RANGE=1:2 NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" uv run examples/run_grpo.py

# Profile with detailed tracing
NRL_NSYS_PROFILE_STEP_RANGE=3:5 NRL_NSYS_WORKER_PATTERNS="dtensor_policy_worker" uv run examples/run_grpo.py
```

**Note**: For detailed NSYS profiling setup and usage, see the [NSYS Profiling Guide](../../guides/environment-data/nsys-profiling.md).

#### Custom NVTX Markers
Add custom markers to identify specific operations:

```python
import torch.cuda.nvtx as nvtx

class ProfiledTraining:
    def __init__(self):
        self.profiler = None
    
    def train_step(self, model, batch):
        """Profiled training step with NVTX markers"""
        
        # Mark forward pass
        nvtx.range_push("forward_pass")
        outputs = model(batch)
        nvtx.range_pop()
        
        # Mark loss computation
        nvtx.range_push("loss_computation")
        loss = self.compute_loss(outputs, batch)
        nvtx.range_push("backward_pass")
        loss.backward()
        nvtx.range_pop()
        nvtx.range_pop()
        
        # Mark optimizer step
        nvtx.range_push("optimizer_step")
        self.optimizer.step()
        nvtx.range_pop()
        
        return loss
    
    def profile_training(self, model, dataloader, num_steps=100):
        """Profile complete training loop"""
        
        # Start NSYS profiling
        torch.cuda.profiler.start()
        
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
            
            # Mark epoch boundary
            nvtx.range_push(f"epoch_{i}")
            
            loss = self.train_step(model, batch)
            
            nvtx.range_pop()
        
        # Stop profiling
        torch.cuda.profiler.stop()
```

#### Memory Profiling
Analyze GPU memory usage patterns during training:

```python
import torch.cuda.memory as memory

class MemoryProfiler:
    def __init__(self):
        self.memory_stats = []
    
    def log_memory_stats(self, step, description=""):
        """Log current memory statistics"""
        stats = {
            "step": step,
            "description": description,
            "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
            "cached": torch.cuda.memory_reserved() / 1e9,      # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
            "max_cached": torch.cuda.max_memory_reserved() / 1e9
        }
        
        self.memory_stats.append(stats)
        print(f"Step {step}: {stats['allocated']:.2f}GB allocated, {stats['cached']:.2f}GB cached")
    
    def analyze_memory_patterns(self):
        """Analyze memory usage patterns"""
        import numpy as np
        
        allocated = [s["allocated"] for s in self.memory_stats]
        cached = [s["cached"] for s in self.memory_stats]
        
        analysis = {
            "peak_allocated": max(allocated),
            "peak_cached": max(cached),
            "avg_allocated": np.mean(allocated),
            "avg_cached": np.mean(cached),
            "memory_efficiency": np.mean(allocated) / np.mean(cached) if np.mean(cached) > 0 else 0
        }
        
        return analysis
    
    def detect_memory_leaks(self):
        """Detect potential memory leaks"""
        allocated = [s["allocated"] for s in self.memory_stats]
        
        # Check for increasing trend
        if len(allocated) < 10:
            return False
        
        # Simple trend analysis
        trend = np.polyfit(range(len(allocated)), allocated, 1)[0]
        
        return trend > 0.001  # 1MB per step threshold
```

### PyTorch Profiler

#### Model Execution Profiling
Use PyTorch Profiler for detailed model analysis:

```python
from torch.profiler import profile, record_function, ProfilerActivity

class PyTorchProfiler:
    def __init__(self):
        self.profiler = None
    
    def profile_model(self, model, dataloader, num_steps=100):
        """Profile model execution with PyTorch Profiler"""
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            
            for i, batch in enumerate(dataloader):
                if i >= num_steps:
                    break
                
                with record_function("model_forward"):
                    outputs = model(batch)
                
                with record_function("loss_computation"):
                    loss = self.compute_loss(outputs, batch)
                
                with record_function("backward_pass"):
                    loss.backward()
                
                with record_function("optimizer_step"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
        return prof
    
    def analyze_profiler_results(self, prof):
        """Analyze PyTorch Profiler results"""
        
        # Get top operations by time
        top_ops = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print("Top operations by CUDA time:")
        print(top_ops)
        
        # Get memory analysis
        memory_stats = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
        print("\nTop operations by memory usage:")
        print(memory_stats)
        
        # Export to Chrome trace
        prof.export_chrome_trace("trace.json")
        
        return {
            "top_ops": top_ops,
            "memory_stats": memory_stats,
            "trace_file": "trace.json"
        }
```

#### Operator Performance Analysis
Analyze specific operator performance:

```python
def analyze_operator_performance(prof):
    """Analyze performance of specific operators"""
    
    # Group operations by type
    op_stats = {}
    
    for event in prof.function_events:
        op_name = event.name
        if op_name not in op_stats:
            op_stats[op_name] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "total_memory": 0
            }
        
        op_stats[op_name]["count"] += 1
        op_stats[op_name]["total_time"] += event.cuda_time_total
        op_stats[op_name]["total_memory"] += event.cuda_memory_usage
    
    # Calculate averages
    for op_name, stats in op_stats.items():
        if stats["count"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_memory"] = stats["total_memory"] / stats["count"]
    
    return op_stats

def identify_bottlenecks(op_stats):
    """Identify performance bottlenecks"""
    
    bottlenecks = []
    
    # Sort by total time
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_time"], reverse=True)
    
    for op_name, stats in sorted_ops[:10]:
        if stats["total_time"] > 1000:  # 1ms threshold
            bottlenecks.append({
                "operation": op_name,
                "total_time_ms": stats["total_time"] / 1000,
                "avg_time_ms": stats["avg_time"] / 1000,
                "count": stats["count"],
                "memory_mb": stats["total_memory"] / 1e6
            })
    
    return bottlenecks
```

### Custom Profiling

#### Application-Specific Metrics
Create custom profiling for NeMo RL specific operations:

```python
import time
import psutil
import GPUtil

class NeMoRLProfiler:
    def __init__(self):
        self.metrics = []
        self.start_time = None
    
    def start_profiling(self):
        """Start profiling session"""
        self.start_time = time.time()
        self.metrics = []
    
    def log_metrics(self, step, model, optimizer, dataloader):
        """Log comprehensive metrics for NeMo RL training"""
        
        # GPU metrics
        gpu_util = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Training metrics
        batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 0
        learning_rate = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0
        
        # Model metrics
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            "gpu_utilization": gpu_util,
            "gpu_memory_gb": gpu_memory,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def generate_report(self):
        """Generate comprehensive profiling report"""
        
        if not self.metrics:
            return "No metrics collected"
        
        import numpy as np
        
        # Calculate statistics
        gpu_utils = [m["gpu_utilization"] for m in self.metrics]
        gpu_memories = [m["gpu_memory_gb"] for m in self.metrics]
        cpu_percents = [m["cpu_percent"] for m in self.metrics]
        
        report = {
            "total_steps": len(self.metrics),
            "total_time": self.metrics[-1]["timestamp"],
            "avg_gpu_utilization": np.mean(gpu_utils),
            "max_gpu_utilization": np.max(gpu_utils),
            "avg_gpu_memory": np.mean(gpu_memories),
            "max_gpu_memory": np.max(gpu_memories),
            "avg_cpu_percent": np.mean(cpu_percents),
            "max_cpu_percent": np.max(cpu_percents)
        }
        
        return report
```

#### Performance Counters
Implement custom performance counters:

```python
import threading
import time

class PerformanceCounter:
    def __init__(self):
        self.counters = {}
        self.lock = threading.Lock()
    
    def start_counter(self, name):
        """Start a performance counter"""
        with self.lock:
            self.counters[name] = {
                "start_time": time.time(),
                "total_time": 0,
                "count": 0
            }
    
    def end_counter(self, name):
        """End a performance counter"""
        with self.lock:
            if name in self.counters:
                elapsed = time.time() - self.counters[name]["start_time"]
                self.counters[name]["total_time"] += elapsed
                self.counters[name]["count"] += 1
    
    def get_counter_stats(self, name):
        """Get statistics for a counter"""
        with self.lock:
            if name in self.counters:
                counter = self.counters[name]
                return {
                    "total_time": counter["total_time"],
                    "count": counter["count"],
                    "avg_time": counter["total_time"] / counter["count"] if counter["count"] > 0 else 0
                }
        return None
    
    def get_all_stats(self):
        """Get statistics for all counters"""
        with self.lock:
            return {name: self.get_counter_stats(name) for name in self.counters.keys()}

# Usage example
counter = PerformanceCounter()

def profiled_function():
    counter.start_counter("my_function")
    # ... function code ...
    counter.end_counter("my_function")
```

## Profiling Methodologies

### GPU Profiling

#### Kernel Execution Analysis
Analyze CUDA kernel performance:

```python
def analyze_cuda_kernels(prof):
    """Analyze CUDA kernel performance"""
    
    kernel_stats = {}
    
    for event in prof.function_events:
        if "cuda" in event.name.lower():
            kernel_name = event.name
            
            if kernel_name not in kernel_stats:
                kernel_stats[kernel_name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0
                }
            
            kernel_stats[kernel_name]["count"] += 1
            kernel_stats[kernel_name]["total_time"] += event.cuda_time_total
    
    # Calculate averages
    for kernel_name, stats in kernel_stats.items():
        if stats["count"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["count"]
    
    return kernel_stats

def identify_slow_kernels(kernel_stats, threshold_ms=1.0):
    """Identify slow CUDA kernels"""
    
    slow_kernels = []
    
    for kernel_name, stats in kernel_stats.items():
        avg_time_ms = stats["avg_time"] / 1000
        
        if avg_time_ms > threshold_ms:
            slow_kernels.append({
                "kernel": kernel_name,
                "avg_time_ms": avg_time_ms,
                "count": stats["count"],
                "total_time_ms": stats["total_time"] / 1000
            })
    
    return sorted(slow_kernels, key=lambda x: x["avg_time_ms"], reverse=True)
```

#### Memory Bandwidth Analysis
Analyze memory bandwidth utilization:

```python
def analyze_memory_bandwidth(prof):
    """Analyze memory bandwidth usage"""
    
    memory_ops = []
    
    for event in prof.function_events:
        if "memcpy" in event.name.lower() or "memset" in event.name.lower():
            memory_ops.append({
                "operation": event.name,
                "time": event.cuda_time_total,
                "memory_usage": event.cuda_memory_usage
            })
    
    # Calculate bandwidth
    total_memory_bytes = sum(op["memory_usage"] for op in memory_ops)
    total_time_ms = sum(op["time"] for op in memory_ops) / 1000
    
    if total_time_ms > 0:
        bandwidth_gbps = (total_memory_bytes / 1e9) / (total_time_ms / 1000)
    else:
        bandwidth_gbps = 0
    
    return {
        "total_memory_ops": len(memory_ops),
        "total_memory_bytes": total_memory_bytes,
        "total_time_ms": total_time_ms,
        "bandwidth_gbps": bandwidth_gbps,
        "operations": memory_ops
    }
```

### Memory Profiling

#### Memory Allocation Patterns
Analyze memory allocation patterns:

```python
def analyze_memory_patterns(prof):
    """Analyze memory allocation patterns"""
    
    allocation_events = []
    deallocation_events = []
    
    for event in prof.function_events:
        if "alloc" in event.name.lower():
            allocation_events.append({
                "operation": event.name,
                "memory_usage": event.cuda_memory_usage,
                "time": event.cuda_time_total
            })
        elif "free" in event.name.lower():
            deallocation_events.append({
                "operation": event.name,
                "memory_usage": event.cuda_memory_usage,
                "time": event.cuda_time_total
            })
    
    # Analyze patterns
    total_allocated = sum(op["memory_usage"] for op in allocation_events)
    total_freed = sum(op["memory_usage"] for op in deallocation_events)
    
    return {
        "total_allocations": len(allocation_events),
        "total_deallocations": len(deallocation_events),
        "total_allocated_bytes": total_allocated,
        "total_freed_bytes": total_freed,
        "memory_leak_bytes": total_allocated - total_freed,
        "allocation_events": allocation_events,
        "deallocation_events": deallocation_events
    }
```

#### Memory Leak Detection
Detect potential memory leaks:

```python
def detect_memory_leaks(memory_patterns):
    """Detect potential memory leaks"""
    
    leak_bytes = memory_patterns["memory_leak_bytes"]
    
    if leak_bytes > 0:
        leak_mb = leak_bytes / 1e6
        return {
            "leak_detected": True,
            "leak_size_mb": leak_mb,
            "severity": "high" if leak_mb > 100 else "medium" if leak_mb > 10 else "low"
        }
    else:
        return {
            "leak_detected": False,
            "leak_size_mb": 0,
            "severity": "none"
        }
```

### Communication Profiling

#### Inter-Node Communication Analysis
Analyze distributed training communication:

```python
def analyze_communication(prof):
    """Analyze distributed training communication"""
    
    communication_ops = []
    
    for event in prof.function_events:
        if any(op in event.name.lower() for op in ["allreduce", "broadcast", "gather", "scatter"]):
            communication_ops.append({
                "operation": event.name,
                "time": event.cuda_time_total,
                "memory_usage": event.cuda_memory_usage
            })
    
    # Calculate communication overhead
    total_comm_time = sum(op["time"] for op in communication_ops)
    total_time = sum(event.cuda_time_total for event in prof.function_events)
    
    if total_time > 0:
        comm_overhead_percent = (total_comm_time / total_time) * 100
    else:
        comm_overhead_percent = 0
    
    return {
        "communication_operations": len(communication_ops),
        "total_communication_time_ms": total_comm_time / 1000,
        "communication_overhead_percent": comm_overhead_percent,
        "operations": communication_ops
    }
```

#### Network Bandwidth Analysis
Analyze network bandwidth utilization:

```python
def analyze_network_bandwidth(communication_analysis):
    """Analyze network bandwidth usage"""
    
    total_comm_memory = sum(op["memory_usage"] for op in communication_analysis["operations"])
    total_comm_time_ms = communication_analysis["total_communication_time_ms"]
    
    if total_comm_time_ms > 0:
        bandwidth_gbps = (total_comm_memory / 1e9) / (total_comm_time_ms / 1000)
    else:
        bandwidth_gbps = 0
    
    return {
        "total_communication_memory_gb": total_comm_memory / 1e9,
        "total_communication_time_ms": total_comm_time_ms,
        "network_bandwidth_gbps": bandwidth_gbps
    }
```

## Best Practices

### Profiling Strategy

#### Identify Bottlenecks
Systematic approach to bottleneck identification:

```python
def identify_bottlenecks(prof):
    """Systematic bottleneck identification"""
    
    bottlenecks = []
    
    # 1. Check GPU utilization
    gpu_util = analyze_gpu_utilization(prof)
    if gpu_util < 80:
        bottlenecks.append({
            "type": "low_gpu_utilization",
            "value": gpu_util,
            "severity": "high" if gpu_util < 50 else "medium"
        })
    
    # 2. Check memory usage
    memory_analysis = analyze_memory_patterns(prof)
    if memory_analysis["memory_leak_bytes"] > 0:
        bottlenecks.append({
            "type": "memory_leak",
            "value": memory_analysis["memory_leak_bytes"] / 1e6,
            "severity": "high"
        })
    
    # 3. Check communication overhead
    comm_analysis = analyze_communication(prof)
    if comm_analysis["communication_overhead_percent"] > 20:
        bottlenecks.append({
            "type": "high_communication_overhead",
            "value": comm_analysis["communication_overhead_percent"],
            "severity": "medium"
        })
    
    # 4. Check slow kernels
    kernel_stats = analyze_cuda_kernels(prof)
    slow_kernels = identify_slow_kernels(kernel_stats)
    if slow_kernels:
        bottlenecks.append({
            "type": "slow_kernels",
            "value": slow_kernels,
            "severity": "medium"
        })
    
    return bottlenecks
```

#### Measure Impact
Measure the impact of optimizations:

```python
def measure_optimization_impact(baseline_prof, optimized_prof):
    """Measure the impact of optimizations"""
    
    baseline_time = sum(event.cuda_time_total for event in baseline_prof.function_events)
    optimized_time = sum(event.cuda_time_total for event in optimized_prof.function_events)
    
    if baseline_time > 0:
        speedup = baseline_time / optimized_time
        improvement_percent = ((baseline_time - optimized_time) / baseline_time) * 100
    else:
        speedup = 1.0
        improvement_percent = 0
    
    return {
        "baseline_time_ms": baseline_time / 1000,
        "optimized_time_ms": optimized_time / 1000,
        "speedup": speedup,
        "improvement_percent": improvement_percent
    }
```

### Data Collection

#### Comprehensive Metric Collection
Collect comprehensive profiling metrics:

```python
def collect_comprehensive_metrics(prof, model, dataloader):
    """Collect comprehensive profiling metrics"""
    
    metrics = {}
    
    # Basic profiling metrics
    metrics["total_time_ms"] = sum(event.cuda_time_total for event in prof.function_events) / 1000
    metrics["num_events"] = len(prof.function_events)
    
    # GPU metrics
    metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
    metrics["gpu_memory_cached_gb"] = torch.cuda.memory_reserved() / 1e9
    
    # Model metrics
    metrics["model_parameters"] = sum(p.numel() for p in model.parameters())
    metrics["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Data metrics
    metrics["batch_size"] = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 0
    metrics["dataset_size"] = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
    
    # Performance metrics
    metrics["throughput_samples_per_sec"] = (metrics["dataset_size"] / metrics["total_time_ms"]) * 1000
    
    return metrics
```

#### Statistical Analysis
Perform statistical analysis of profiling results:

```python
def analyze_profiling_statistics(metrics_list):
    """Analyze profiling statistics across multiple runs"""
    
    import numpy as np
    from scipy import stats
    
    # Extract key metrics
    total_times = [m["total_time_ms"] for m in metrics_list]
    gpu_memories = [m["gpu_memory_allocated_gb"] for m in metrics_list]
    throughputs = [m["throughput_samples_per_sec"] for m in metrics_list]
    
    statistics = {
        "total_time": {
            "mean": np.mean(total_times),
            "std": np.std(total_times),
            "min": np.min(total_times),
            "max": np.max(total_times),
            "confidence_interval": stats.t.interval(0.95, len(total_times)-1, 
                                                 loc=np.mean(total_times), 
                                                 scale=stats.sem(total_times))
        },
        "gpu_memory": {
            "mean": np.mean(gpu_memories),
            "std": np.std(gpu_memories),
            "min": np.min(gpu_memories),
            "max": np.max(gpu_memories)
        },
        "throughput": {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs),
            "min": np.min(throughputs),
            "max": np.max(throughputs)
        }
    }
    
    return statistics
```

### Optimization Workflow

#### Profile Before Optimization
Always profile before optimizing:

```python
def profile_before_optimization(model, dataloader, num_steps=100):
    """Profile before applying optimizations"""
    
    print("Profiling before optimization...")
    
    # Run profiling
    profiler = PyTorchProfiler()
    prof = profiler.profile_model(model, dataloader, num_steps)
    
    # Analyze results
    analysis = profiler.analyze_profiler_results(prof)
    
    # Identify bottlenecks
    bottlenecks = identify_bottlenecks(prof)
    
    print("Baseline profiling complete")
    print(f"Total time: {analysis['total_time_ms']:.2f} ms")
    print(f"Number of bottlenecks: {len(bottlenecks)}")
    
    return prof, analysis, bottlenecks
```

#### Measure Impact
Measure the impact of each optimization:

```python
def measure_optimization_impact(baseline_metrics, optimized_metrics):
    """Measure the impact of optimizations"""
    
    impact = {}
    
    for metric in ["total_time_ms", "gpu_memory_allocated_gb", "throughput_samples_per_sec"]:
        if metric in baseline_metrics and metric in optimized_metrics:
            baseline = baseline_metrics[metric]
            optimized = optimized_metrics[metric]
            
            if baseline > 0:
                if "time" in metric or "memory" in metric:
                    # Lower is better
                    improvement = ((baseline - optimized) / baseline) * 100
                else:
                    # Higher is better
                    improvement = ((optimized - baseline) / baseline) * 100
            else:
                improvement = 0
            
            impact[metric] = {
                "baseline": baseline,
                "optimized": optimized,
                "improvement_percent": improvement
            }
    
    return impact
```

## Next Steps

After understanding performance profiling:

1. **Profile Your Training**: Identify performance bottlenecks
2. **Analyze Memory Usage**: Optimize memory allocation patterns
3. **Optimize Communication**: Reduce distributed training overhead
4. **Measure Improvements**: Validate optimization impact
5. **Monitor Continuously**: Set up ongoing performance monitoring

## Performance Considerations

**Note**: Performance improvements and memory reductions mentioned in this guide are estimates based on typical usage patterns. Actual results may vary depending on:
- Hardware configuration (GPU model, memory, interconnect)
- Model architecture and size
- Batch size and sequence length
- Training workload characteristics

## References

- NVIDIA NSYS Documentation: https://docs.nvidia.com/nsight-systems/
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- CUDA Performance Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Memory Management in PyTorch: https://pytorch.org/docs/stable/notes/cuda.html 