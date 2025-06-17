
# Set up virtual environment directory
VENV_DIR="$PWD/reinforcer_venv"
mkdir -p $VENV_DIR

# Set environment variables for UV and virtual environment
export UV_CACHE_DIR="$PWD/uv_cache"
export UV_LINK_MODE=copy
export VENV_DIR=$VENV_DIR

# Set vLLM port range to avoid conflicts
export VLLM_PORT_RANGE="20000-30000"

# check that $1 exists
if [ -z "$1" ]; then
    echo "Usage: $0 <output_name>"
    exit 1
fi

OUTFILE="nano_deepscaler_evals/nano_deepscaler_L1_data_context_20000_$1.jsonl"
echo OUTFILE=${OUTFILE}
JOB_NAME="nano_deepscaler_eval_$1"

source ~/secrets
echo WANDB_API_KEY=${WANDB_API_KEY}

CONTAINER='gitlab-master.nvidia.com/deci/research/lit-llama/rl_uv_amnon:latest' \
MOUNTS="/lustre:/lustre,$UV_CACHE_DIR:/home/ray/.cache/uv,$VENV_DIR:/opt/reinforcer_venv" \
COMMAND="uv run python examples/run_eval_with_planted_thinking.py data.shuffle_seed=$RANDOM debug.outfile=${OUTFILE} --config tests/configs/eval_nano_deepscaler.yaml" WANDB_API_KEY=${WANDB_API_KEY}  \
sbatch \
    --nodes=1 \
    --account='llmservice_deci_llm' \
    --job-name="${JOB_NAME}" \
    --partition='batch' \
    --time=04:00:00 \
    --gres=gpu:8 \
    ray.sub
