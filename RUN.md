
## Ask @terry for container

## One time setup
```
git submodule update --init --recursive


# inside container build and checkout custom vllm
tools/build-custom-vllm.sh
```

## Run
```
COMMAND="uv pip install setuptools_scm && NRL_FORCE_REBUILD_VENVS=true NRL_VLLM_USE_V1=0 uv run --env-file .env examples/run_grpo_math.py --config examples/configs/grpo_math_nm58B_megatron.yaml \
    policy.model_name=checkpoints/adis-12b-hf-ckpt-chunk256 \
    policy.tokenizer.name=nvidia/Nemotron-H-8B-Base-8K \
    cluster.num_nodes=4 \
    checkpointing.enabled=true \
    checkpointing.save_period=5 \
    grpo.val_period=5 \
    " \
LONG=1 bash launch.sh 4 4:0:0

