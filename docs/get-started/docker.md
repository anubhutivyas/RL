---
description: "Complete Docker setup guide for NeMo RL with three image types: release, hermetic, and base for different use cases"
categories: ["getting-started"]
tags: ["docker", "containerization", "gpu-accelerated", "deployment", "configuration"]
personas: ["devops-focused", "admin-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "universal"
---

# Build Docker Images

This guide provides three methods for building Docker images:

* **release**: Contains everything from the hermetic image, plus the NeMo RL source code and pre-fetched virtual environments for isolated workers.
* **hermetic**: Includes the base image plus pre-fetched NeMo RL Python packages in the `uv` cache.
* **base**: A minimal image with CUDA, `ray`, and `uv` installed, ideal for specifying Python dependencies at runtime.

## Image Types

### Release Image (Recommended)

Use the **release** image if you want to pre-fetch the NeMo RL [worker virtual environments](local-workstation) and copy in the project source code. This is the most complete image and is recommended for most use cases.

### Hermetic Image

Use the **hermetic** image if you want to pre-fetch NeMo RL Python packages into the `uv` cache. This provides a good balance between completeness and flexibility.

### Base Image

Use the **base** image if you want maximum flexibility and plan to specify Python dependencies at runtime. This is the most minimal image.

## Building Docker Images

### Prerequisites

Before building Docker images, ensure you have:

- **Docker**: Installed and running
- **NVIDIA Container Toolkit**: For GPU support
- **Git**: To clone the repository

### Build Commands

#### Release Image

```bash
# Clone the repository
git clone https://github.com/NVIDIA/NeMo-RL.git
cd RL

# Build the release image
docker build -f docker/Dockerfile -t nemo-rl:release .
```

#### Hermetic Image

```bash
# Build the hermetic image
docker build -f docker/Dockerfile.hermetic -t nemo-rl:hermetic .
```

#### Base Image

```bash
# Build the base image
docker build -f docker/Dockerfile.base -t nemo-rl:base .
```

## Running Docker Containers

### Basic Usage

```bash
# Run with GPU support
docker run --gpus all -it nemo-rl:release

# Run with specific GPU devices
docker run --gpus '"device=0,1"' -it nemo-rl:release

# Run with volume mounts for data
docker run --gpus all -v /path/to/data:/data -it nemo-rl:release
```

### Interactive Development

```bash
# Run with interactive shell
docker run --gpus all -it nemo-rl:release /bin/bash

# Mount the source code for development
docker run --gpus all -v $(pwd):/workspace -it nemo-rl:release /bin/bash
```

### Running Training Jobs

```bash
# Run a training job inside the container
docker run --gpus all -v /path/to/data:/data nemo-rl:release \
    uv run python examples/run_sft.py \
    --config examples/configs/sft.yaml \
    cluster.gpus_per_node=1
```

## Environment Variables

Set environment variables for the container:

```bash
# Set Hugging Face cache directory
docker run --gpus all \
    -e HF_HOME=/hf_cache \
    -v /path/to/hf_cache:/hf_cache \
    -it nemo-rl:release

# Set Weights and Biases API key
docker run --gpus all \
    -e WANDB_API_KEY=your_api_key \
    -it nemo-rl:release

# Set Hugging Face token
docker run --gpus all \
    -e HF_TOKEN=your_token \
    -it nemo-rl:release
```

## Multi-Node Docker Setup

For distributed training across multiple nodes:

### Head Node

```bash
# Start the head node
docker run --gpus all \
    -p 6379:6379 \
    -p 8265:8265 \
    -e RAY_HEAD_NODE=true \
    -it nemo-rl:release
```

### Worker Nodes

```bash
# Start worker nodes
docker run --gpus all \
    -e RAY_WORKER_NODE=true \
    -e RAY_HEAD_ADDRESS=head_node_ip:6379 \
    -it nemo-rl:release
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**:
   ```bash
   # Check if NVIDIA Container Toolkit is installed
   nvidia-docker run --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Permission Denied**:
   ```bash
   # Run with proper user permissions
   docker run --gpus all --user $(id -u):$(id -g) -it nemo-rl:release
   ```

3. **Out of Memory**:
   ```bash
   # Increase memory limits
   docker run --gpus all --memory=32g -it nemo-rl:release
   ```

### Getting Help

- Check the [troubleshooting guide](../configuration-cli/troubleshooting)
- Review [cluster setup](cluster.md) for multi-node configurations
- Join the [NeMo Discord](https://discord.gg/nvidia-nemo) for community support

## Next Steps

After setting up Docker:

1. **Run Your First Job**: Try the [quickstart guide](quickstart) inside the container
2. **Configure Clusters**: Set up [distributed training](cluster.md) with Docker
3. **Customize Images**: Modify Dockerfiles for your specific needs
4. **Production Deployment**: Use Docker for [production deployments](../guides/production-support/index)