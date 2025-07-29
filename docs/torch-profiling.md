# Profiling with Torch Profiler

NeMo RL supports [Pytorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for profiling GRPO algorithm from the controller and workers. User can select the steps and workers to be profiled via environment variables.

## Configure the Environment Variables

Set the `NRL_TORCH_PROFILE_STEP_RANGE` environment variable to control which training steps the profiler captures. Its
format is colon separated integers representing `start:stop`, where `start` is inclusive and `stop` is exclusive
(same as slice syntax `arr[start:stop]`). Note that the `start` is 1-index, so `NRL_TORCH_PROFILE_STEP_RANGE=0:10` would error.

```bash
export NRL_TORCH_PROFILE_STEP_RANGE=3:5
```

Set the `NRL_TORCH_PROFILE_DIR` enviroment variable to the directory you want profiling traces to be saved, default is `"torch_profiler_trace"`. 

```bash
export NRL_TORCH_PROFILE_DIR=./torch_profiler_trace
```

The profiling trace will be saved in the following file structure

```bash
$NRL_TORCH_PROFILE_DIR/
    <controller trace json files>
    dtensor_policy_worker/
        0/
            <dtensor policy worker trace json files for rank 0>
        ...
    megatron_policy_worker/
        0/
            <megatron policy worker trace json files for rank 0>
        ...
    vllm_generation_worker/
        <vllm generation worker trace json files>
```

Set the `NRL_TORCH_PROFILE_WORKER_PATTERNS` environment variable with a comma-separated list of patterns to match worker names. Supported worker names: `dtensor_policy_worker, megatron_policy_worker, vllm_generation_worker`. Example:

```bash
export NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" # profile all

export NRL_NSYS_WORKER_PATTERNS="*vllm*" # profile only vllm

export NRL_NSYS_WORKER_PATTERNS="megatron_policy_worker" # profile only megatron policy worker. Nothing is profiled if you are using dtensor as policy.

```

### Pattern format
See [nsys-profiling.md - Pattern Format](./nsys-profiling.md#pattern-format) for details on the pattern format used for worker name matching.

## Analyze Profiling Traces
The traces can be visualized with [Perfetto UI](https://ui.perfetto.dev/).

## Known Issues
The Vllm profiler cannot profile Cuda events. To profile Vllm cuda events, you should follow this [vllm tutorial](https://docs.vllm.ai/en/v0.8.0/contributing/profiling/profiling_index.html), and not set any of the above environment variables because there is known bugs when the two profiling tools are used simultaneously. 