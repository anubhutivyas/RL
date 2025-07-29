#!/bin/bash

# Add Ray Worker Script
# Usage: ./add_ray_worker.sh
# Adds a single worker to the existing Ray cluster

set -e

# Configuration (should match start_ray_cluster.sh)
HEAD_PORT=6379
NODE_MANAGER_PORT=6380
OBJECT_MANAGER_PORT=6381
WORKER_BASE_PORT=6400
LOCK_FILE="/tmp/ray_worker_add.lock"
MAX_RETRIES=10
RETRY_DELAY=2

echo "🔧 Adding new worker to Ray cluster..."

# Function to acquire lock
acquire_lock() {
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
            return 0
        fi
        echo "⏳ Another worker addition in progress, waiting... (attempt $((retry_count + 1))/$MAX_RETRIES)"
        sleep $RETRY_DELAY
        retry_count=$((retry_count + 1))
    done
    echo "❌ Failed to acquire lock after $MAX_RETRIES attempts"
    exit 1
}

# Function to release lock
release_lock() {
    rm -f "$LOCK_FILE"
}

# Trap to ensure lock is released on exit
trap release_lock EXIT

# Acquire lock before proceeding
acquire_lock

# Check if Ray cluster is running
if ! ray status >/dev/null 2>&1; then
    echo "❌ No Ray cluster found. Please start one first with ./start_ray_cluster.sh"
    exit 1
fi

# Get current cluster information
echo "📊 Checking current cluster status..."
cluster_info=$(ray status)

# Count active worker nodes (exclude head node)
# Look for nodes with GPU resources > 0
total_nodes=$(echo "$cluster_info" | grep "Active:" -A 10 | grep -c "node_" || echo "1")
existing_workers=$((total_nodes - 1))  # Subtract 1 for head node

# The next GPU ID equals the number of existing workers (workers use GPU 0, 1, 2, ...)
next_gpu_id=$existing_workers

echo "💡 Found $total_nodes total nodes (1 head + $existing_workers workers)"
echo "🎯 Next worker will use GPU $next_gpu_id"

# Check if GPU exists
if ! nvidia-smi -i $next_gpu_id >/dev/null 2>&1; then
    echo "❌ GPU $next_gpu_id not available. Available GPUs:"
    nvidia-smi -L
    exit 1
fi

# Calculate ports for the new worker (allocate 10 ports per worker to avoid conflicts)
port_block_size=10
worker_base_port=$((6400 + next_gpu_id * port_block_size))
worker_node_manager_port=$worker_base_port
worker_object_manager_port=$((worker_base_port + 1))

# Check if ports are already in use
if netstat -tuln 2>/dev/null | grep -q ":$worker_node_manager_port "; then
    echo "❌ Port $worker_node_manager_port already in use"
    exit 1
fi

if netstat -tuln 2>/dev/null | grep -q ":$worker_object_manager_port "; then
    echo "❌ Port $worker_object_manager_port already in use"
    exit 1
fi

echo "🚀 Starting worker on GPU $next_gpu_id..."
echo "   - Node manager port: $worker_node_manager_port"
echo "   - Object manager port: $worker_object_manager_port"

# Release lock before starting the long-running worker process
release_lock
trap '' EXIT  # Remove the exit trap since we manually released the lock

# Start the worker (blocking)
CUDA_VISIBLE_DEVICES=$next_gpu_id ray start \
    --address=127.0.0.1:$HEAD_PORT \
    --node-manager-port=$worker_node_manager_port \
    --object-manager-port=$worker_object_manager_port \
    --num-cpus=1 \
    --num-gpus=1 \
    --disable-usage-stats \
    --block 