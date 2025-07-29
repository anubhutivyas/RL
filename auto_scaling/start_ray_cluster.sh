#!/bin/bash

# Start Ray Cluster Script - Head Node Only (Unschedulable)
# Usage: ./start_ray_cluster.sh

set -e

# Trap handler for cleanup
cleanup() {
    echo ""
    echo "🛑 Received shutdown signal, stopping Ray cluster..."
    ray stop
    if [ -n "$HEAD_PID" ] && kill -0 $HEAD_PID 2>/dev/null; then
        echo "🔄 Terminating head node process ($HEAD_PID)..."
        kill $HEAD_PID 2>/dev/null || true
    fi
    echo "✅ Ray cluster stopped"
    exit 0
}

# Set up trap to catch SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Configuration
HEAD_PORT=6379
DASHBOARD_PORT=8265
NODE_MANAGER_PORT=6380
OBJECT_MANAGER_PORT=6381

echo "🚀 Starting Ray head node (unschedulable)..."

# Stop any existing Ray instances
echo "🛑 Stopping any existing Ray instances..."
ray stop 2>/dev/null || true
sleep 2

# Start unschedulable head node (no CPUs/GPUs available for scheduling)
echo "🌟 Starting Ray head node (unschedulable - no resources for scheduling)..."
ray start \
    --head \
    --port=$HEAD_PORT \
    --dashboard-port=$DASHBOARD_PORT \
    --node-manager-port=$NODE_MANAGER_PORT \
    --object-manager-port=$OBJECT_MANAGER_PORT \
    --num-cpus=0 \
    --num-gpus=0 \
    --disable-usage-stats \
    --block &

# Store the head node process ID
HEAD_PID=$!

# Give head node a moment to start
sleep 3

# Check if head node started successfully
if kill -0 $HEAD_PID 2>/dev/null; then
    echo "✅ Head node started successfully (unschedulable)"
    echo "📊 Dashboard: http://localhost:$DASHBOARD_PORT"
    echo "🔗 Cluster address: 127.0.0.1:$HEAD_PORT"
else
    echo "❌ Failed to start head node"
    exit 1
fi

# Show cluster status
echo ""
echo "📊 Cluster Status:"
ray status || echo "⚠️ Ray status not available yet"

echo ""
echo "🎉 Ray head node is ready!"
echo "💡 Head node configuration:"
echo "   - Unschedulable (0 CPUs, 0 GPUs)"
echo "   - Port: $HEAD_PORT"
echo "   - Dashboard: http://localhost:$DASHBOARD_PORT"
echo ""
echo "🔧 To add workers: ./add_ray_worker.sh (in another terminal)"
echo "🛑 To stop cluster: Press Ctrl+C"
echo ""
echo "⏳ Head node is running... (Press Ctrl+C to stop)"
echo "👀 Monitoring head node process (PID: $HEAD_PID)..."

# Wait for the head node process
wait "$HEAD_PID"

echo "🏁 Head node process has exited" 