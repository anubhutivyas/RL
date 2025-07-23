---
description: "Comprehensive testing strategies for NeMo RL including unit tests, functional tests, and performance validation in local and distributed environments"
categories: ["deployment-operations"]
tags: ["testing", "unit-tests", "functional-tests", "performance", "validation", "quality-assurance"]
personas: ["mle-focused", "admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "universal"
---

# Test NeMo RL

This guide outlines how to test NeMo RL using unit and functional tests, detailing steps for local or Docker-based execution, dependency setup, and metric tracking to ensure effective and reliable testing.

## Unit Tests

:::{important}
Unit tests require 2 GPUs to test the full suite.
:::

```sh
# Run the unit tests using local GPUs
uv run --group test bash tests/run_unit.sh
```

:::{note}
Tests can also be run on Slurm with `ray.sub`, but note that some tests will be skipped
due to no GPUs being located on the head node. To run the full suite of tests, please
launch on a regular GPU allocation.
:::

### Run Unit Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in Docker with this script:

```sh
CONTAINER=... bash tests/run_unit_in_docker.sh
```

For containerized testing, see the [Docker Setup](../../get-started/docker.md) guide.

### Track Metrics in Unit Tests

Unit tests may also log metrics to a fixture. The fixture is called `tracker` and has the following API:

```python
# Track an arbitrary metric (must be json serializable)
tracker.track(metric_name, metric_value)
# Log the maximum memory across the entire cluster. Okay for tests since they are run serially.
tracker.log_max_mem(metric_name)
# Returns the maximum memory. Useful if you are measuring changes in memory.
tracker.get_max_mem()
```

Including the `tracker` fixture also tracks the elapsed time for the test implicitly.

Here is an example test:

```python
def test_exponentiate(tracker):
    starting_mem = tracker.get_max_mem()
    base = 2
    exponent = 4
    result = base ** exponent
    tracker.track("result", result)
    tracker.log_max_mem("memory_after_exponentiating")
    change_in_mem = tracker.get_max_mem() - starting_mem
    tracker.track("change_in_mem", change_in_mem)
    assert result == 16
```

Which would produce this file in `tests/unit/unit_results.json`:

```json
{
  "exit_status": 0,
  "git_commit": "f1062bd3fd95fc64443e2d9ee4a35fc654ba897e",
  "start_time": "2025-03-24 23:34:12",
  "metrics": {
    "test_hf_ray_policy::test_lm_policy_generation": {
      "avg_prob_mult_error": 1.0000039339065552,
      "mean_lps": -1.5399343967437744,
      "_elapsed": 17.323044061660767
    }
  },
  "gpu_types": [
    "NVIDIA H100 80GB HBM3"
  ],
  "coverage": 24.55897613282601
}
```

:::{tip}
Past unit test results are logged in `tests/unit/unit_results/`. These are helpful to view trends over time and commits.

Here's an example `jq` command to view trends:

```sh
jq -r '[.start_time, .git_commit, .metrics["test_hf_ray_policy::test_lm_policy_generation"].avg_prob_mult_error] | @tsv' tests/unit/unit_results/*

# Example output:
#2025-03-24 23:35:39     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:36:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:37:37     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:38:14     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
#2025-03-24 23:38:50     778d288bb5d2edfd3eec4d07bb7dffffad5ef21b        1.0000039339065552
```
:::

## CI/CD Integration for DevOps

For DevOps professionals managing automated testing pipelines:

```yaml
# .github/workflows/test-nemo-rl.yml
name: NeMo RL Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10]
        cuda-version: [11.8, 12.1]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install CUDA ${{ matrix.cuda-version }}
      uses: Jimver/cuda-toolkit@v0.2.12
      with:
        cuda: ${{ matrix.cuda-version }}
    
    - name: Run unit tests
      run: |
        uv run --group test bash tests/run_unit.sh
        python -m pytest tests/unit/ --cov=nemo_rl --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Performance Testing for MLEs

For ML Engineers validating training performance:

```python
def performance_benchmark(model_config, dataset_config):
    """Benchmark NeMo RL training performance"""
    import time
    import psutil
    
    benchmark_results = {
        'training_time': 0,
        'memory_usage': [],
        'gpu_utilization': [],
        'throughput': 0
    }
    
    start_time = time.time()
    
    # Monitor system resources during training
    def monitor_resources():
        while training_active:
            memory = psutil.virtual_memory().percent
            gpu_util = get_gpu_utilization()
            benchmark_results['memory_usage'].append(memory)
            benchmark_results['gpu_utilization'].append(gpu_util)
            time.sleep(5)
    
    # Run training with monitoring
    with ThreadPoolExecutor() as executor:
        monitor_future = executor.submit(monitor_resources)
        training_future = executor.submit(run_training, model_config, dataset_config)
        
        training_future.result()
        training_active = False
        monitor_future.result()
    
    benchmark_results['training_time'] = time.time() - start_time
    benchmark_results['throughput'] = calculate_throughput()
    
    return benchmark_results
```

## Functional Tests

:::{important}
Functional tests may require multiple GPUs to run. See each script to understand the requirements.
:::

Functional tests are located under `tests/functional/`.

```sh
# Run the functional test for sft
uv run bash tests/functional/sft.sh
```

At the end of each functional test, the metric checks will be printed as well as
whether they pass or fail. Here is an example:

```text
                              Metric Checks
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Status ┃ Check                          ┃ Value             ┃ Message ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ PASS   │ data["train/loss"]["9"] < 1500 │ 817.4517822265625 │         │
└────────┴────────────────────────────────┴───────────────────┴─────────┘
```

### Run Functional Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in Docker with this script:

```sh
CONTAINER=... bash run_functional_in_docker.sh functional/sft.sh
```


## Static Type Checking with [MyPy](https://mypy-lang.org/)
Static type checking can be run with no GPU resources:

```sh
uv run --group test mypy {program}.py
```

For example,
```sh
uv run --group test mypy examples/run_grpo_math.py
uv run --group test mypy examples/run_sft.py
```

mypy.ini controls the configuration of mypy.