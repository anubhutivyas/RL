# Ray Autoscaling Scripts

This directory contains Ray autoscaling scripts for testing dynamic cluster scaling during training workloads. Perfect for simulating how your training jobs can automatically scale as new GPUs become available.

The setup provides a complete solution for testing dynamic Ray cluster autoscaling during training workloads.

> **Note:** All commands in this guide should be run from the `auto_scaling/` directory.

## 🎯 What This Setup Provides

1. **Local Ray Cluster Management** - Start clusters with configurable number of workers
2. **Dynamic Worker Addition** - Add workers one at a time to test scaling
3. **Training Loop with Autoscaling** - Simulates real training that adapts to new resources
4. **GPU Resource Management** - Automatically assigns non-overlapping GPUs

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `start_ray_cluster.sh` | Start Ray cluster with N workers |
| `add_ray_worker.sh` | Add a single worker to existing cluster |
| `training_loop_with_autoscaling.py` | Training simulation with dynamic scaling |

## 🚀 Quick Start

### 1. Start a Ray Cluster

```bash
# Start with just head node (GPU 0) - blocks until Ctrl+C
./start_ray_cluster.sh

# Start with head + 2 workers (GPUs 0, 1, 2)
./start_ray_cluster.sh 2
```

### 2. Run Training Loop

```bash
# In another terminal, start the training loop
uv run python training_loop_with_autoscaling.py
```

### 3. Add Workers Dynamically

```bash
# In a third terminal, add workers during training
./add_ray_worker.sh  # Adds worker on next available GPU
./add_ray_worker.sh  # Adds another worker
```

## 🏗️ Detailed Usage

### Ray Cluster Management

**Start Ray Cluster:**
```bash
./start_ray_cluster.sh [num_workers]
```
- `num_workers`: Number of additional workers (default: 0)
- Head node always uses GPU 0
- Workers use GPUs 1, 2, 3, ...
- Dashboard available at: http://localhost:8265
- **Script blocks** until Ctrl+C (with proper cleanup)

**Add Single Worker:**
```bash
./add_ray_worker.sh
```
- Automatically determines next available GPU
- Blocks until worker is running
- Safe to run multiple times

**Stop Cluster:**
```bash
ray stop
```

### Training Loop Features

The training loop (`training_loop_with_autoscaling.py`) provides:

- **Automatic Worker Detection** - Checks for new workers at each training step
- **Dynamic Actor Scaling** - Creates new training actors when workers join
- **GPU Training Simulation** - Uses PyTorch for realistic GPU workloads
- **Real-time Monitoring** - Shows training progress and scaling events

**Training Output Example:**
```
🎯 DYNAMIC TRAINING LOOP STARTED
============================================================
💡 This training loop will:
   • Run training steps on all available actors
   • Monitor for new workers at each step
   • Automatically scale up when new workers join
   • Use 1 GPU per training actor

🔄 To add workers: ./add_ray_worker.sh
============================================================

🚀 Running training step 1 on 1 actors...
📊 Step 1 completed:
   - 1 actors trained
   - Average duration: 0.045s
   - Actor 0: GPU training: loss=0.2341

🆕 Detected new worker: a1b2c3d4e5... with 1.0 GPUs
✅ Created placement group 2
🎭 Spawned training actor 1 on node a1b2c3d4e5...

🚀 Running training step 2 on 2 actors...
📊 Step 2 completed:
   - 2 actors trained
   - Average duration: 0.038s
   - Actor 0: GPU training: loss=0.1987
   - Actor 1: GPU training: loss=0.2145
```

## 🔧 Configuration

### GPU Assignment Strategy

The setup uses this GPU assignment strategy:
- **Head Node:** Always GPU 0
- **Worker 1:** GPU 1  
- **Worker 2:** GPU 2
- **Worker N:** GPU N

This is handled automatically by the scripts using `CUDA_VISIBLE_DEVICES`.

### Port Configuration

Hard-coded ports (configurable in scripts):
- **Ray Head:** 6379
- **Dashboard:** 8265  
- **Node Manager:** 6380 + worker_id
- **Object Manager:** 6381 + worker_id

### Resource Allocation

Each node (head + workers) gets:
- **1 GPU** (via CUDA_VISIBLE_DEVICES)
- **1 CPU** (Ray resource)
- **1 worker_unit** (custom resource for tracking)

## 📊 Example Workflows

### Workflow 1: Start Small, Scale Up
```bash
# Terminal 1: Start minimal cluster (blocks)
./start_ray_cluster.sh 0

# Terminal 2: Start training
uv run python training_loop_with_autoscaling.py

# Terminal 3: Add workers during training
./add_ray_worker.sh
sleep 10
./add_ray_worker.sh
```

### Workflow 2: Start with Workers, Add More
```bash
# Terminal 1: Start with 2 workers (blocks)
./start_ray_cluster.sh 2

# Terminal 2: Start training (will use all 3 nodes)
uv run python training_loop_with_autoscaling.py

# Terminal 3: Add more workers
./add_ray_worker.sh  # Adds 4th worker
```

### Workflow 3: Testing Resource Limits
```bash
# Start cluster (blocks)
./start_ray_cluster.sh 1

# Start training (new terminal)
uv run python training_loop_with_autoscaling.py

# Add workers until you hit GPU limit (new terminal)
./add_ray_worker.sh  # GPU 2
./add_ray_worker.sh  # GPU 3 (if available)
./add_ray_worker.sh  # Will fail if no more GPUs
```

## 🐛 Troubleshooting

### Common Issues

**"No Ray cluster found"**
```bash
# Make sure cluster is started first
./start_ray_cluster.sh
```

**"GPU X not available"**
```bash
# Check available GPUs
nvidia-smi -L

# Or start with fewer workers
./start_ray_cluster.sh 1  # Instead of 2
```

**"Failed to create placement group"**
- Cluster may be at resource capacity
- Check `ray status` for available resources
- Try stopping and restarting cluster

**Training actors not scaling**
- Check that new workers have GPU resources
- Verify training loop is checking for new workers
- Look for error messages in training output

### Debug Commands

```bash
# Check cluster status
ray status

# List available GPUs
nvidia-smi -L

# View Ray dashboard
open http://localhost:8265

# Check which GPUs are in use
nvidia-smi
```

## 🎯 Integration with NeMo RL

This setup demonstrates the exact pattern you'd use for autoscaling in NeMo RL:

1. **Dynamic Worker Detection** - Monitor `ray.nodes()` for new workers
2. **Placement Group Creation** - Create groups when new resources available
3. **Actor Spawning** - Launch new training/inference actors automatically
4. **Resource Management** - Ensure proper GPU allocation across workers

### Adapting for NeMo RL

Replace `TrainingActor` with your specific actors:
- **Policy Training Actors** - For GRPO/PPO training
- **VLLM Generation Actors** - For inference scaling  
- **Environment Actors** - For evaluation environments

The core scaling logic remains the same!

## 🔮 Next Steps

To extend this for production:

1. **Replace with real training** - Swap `TrainingActor` with actual model training
2. **Add health monitoring** - Check actor health and restart failed ones
3. **Implement resource optimization** - Better placement group sizing
4. **Add metrics collection** - Export scaling metrics for monitoring
5. **Connect to external autoscalers** - Integrate with Kubernetes HPA, SLURM, etc.

---

**🎉 Happy Autoscaling!** This setup gives you a complete testing environment for dynamic Ray cluster scaling during training workloads. 