#!/bin/bash

# SBATCH --job-name=interactive-notebook
# SBATCH --output=logs/notebook_job_%j.log
# SBATCH --account=llmservice_modelalignment_ppo
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=8
# SBATCH --mem=32G
# SBATCH --gpus=1
# SBATCH --partition=interactive
# SBATCH --container-image=/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh
# SBATCH --time=02:00:00


echo "===================================================================="
echo "Starting SLURM job for interactive Jupyter Notebook"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Log file: $(pwd)/logs/notebook_job_${SLURM_JOB_ID}.log"
echo "===================================================================="

# Step 1: Install dependencies from requirements.txt using uv
echo
echo "[1/3] Installing Python dependencies using uv..."
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Check requirements.txt and uv setup."
    exit 1
fi
echo "Dependencies installed successfully."
echo

# Step 2: Start Jupyter Lab server on a random port
echo "[2/3] Starting Jupyter Lab server..."
# Find a random available port
PORT=$(shuf -i 8000-9999 -n 1)
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Start jupyter lab in the background
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0 &

# Wait a few seconds for the server to start up
sleep 15

echo "Jupyter Lab server is starting in the background."
echo

# Step 3: Display connection instructions for VS Code
echo "[3/3] Connection Instructions for VS Code:"
echo "--------------------------------------------------------------------"
echo "1. Open a NEW terminal on your LOCAL machine and run this command to"
echo "   create an SSH tunnel. This command will seem to hang, which is normal."
echo
echo "   ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@your_cluster_login_node"
echo
echo "   - Replace 'your_cluster_login_node' with your cluster's SSH address."
echo "   - The job is running on compute node: $(hostname)"
echo "--------------------------------------------------------------------"
echo "2. In VS Code, connect to the Jupyter server:"
echo "   a. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P)."
echo "   b. Type and select 'Jupyter: Specify Jupyter server for connections'."
echo "   c. Select 'Existing'."
echo "   d. Paste one of the URLs below (it should start with http://127.0.0.1...)"
echo
echo "Available Jupyter Servers (copy a URL with the token):"
jupyter server list
echo "--------------------------------------------------------------------"

echo
echo "Job is now running. The allocation is reserved for the time you requested."
echo "To stop the job, run: scancel $SLURM_JOB_ID"

# Wait for the Jupyter Lab process to end.
# This keeps the SLURM job alive.
wait
