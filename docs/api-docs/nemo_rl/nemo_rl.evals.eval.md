# {py:mod}`nemo_rl.evals.eval`

```{py:module} nemo_rl.evals.eval
```

```{autodoc2-docstring} nemo_rl.evals.eval
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalConfig <nemo_rl.evals.eval.EvalConfig>`
  -
* - {py:obj}`MasterConfig <nemo_rl.evals.eval.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup <nemo_rl.evals.eval.setup>`
  - ```{autodoc2-docstring} nemo_rl.evals.eval.setup
    :summary:
    ```
* - {py:obj}`eval_pass_k <nemo_rl.evals.eval.eval_pass_k>`
  - ```{autodoc2-docstring} nemo_rl.evals.eval.eval_pass_k
    :summary:
    ```
* - {py:obj}`run_env_eval <nemo_rl.evals.eval.run_env_eval>`
  - ```{autodoc2-docstring} nemo_rl.evals.eval.run_env_eval
    :summary:
    ```
````

### API

`````{py:class} EvalConfig()
:canonical: nemo_rl.evals.eval.EvalConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} metric
:canonical: nemo_rl.evals.eval.EvalConfig.metric
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.EvalConfig.metric
```

````

````{py:attribute} num_tests_per_prompt
:canonical: nemo_rl.evals.eval.EvalConfig.num_tests_per_prompt
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.EvalConfig.num_tests_per_prompt
```

````

````{py:attribute} seed
:canonical: nemo_rl.evals.eval.EvalConfig.seed
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.EvalConfig.seed
```

````

````{py:attribute} pass_k_value
:canonical: nemo_rl.evals.eval.EvalConfig.pass_k_value
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.EvalConfig.pass_k_value
```

````

`````

`````{py:class} MasterConfig()
:canonical: nemo_rl.evals.eval.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} eval
:canonical: nemo_rl.evals.eval.MasterConfig.eval
:type: nemo_rl.evals.eval.EvalConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.MasterConfig.eval
```

````

````{py:attribute} generate
:canonical: nemo_rl.evals.eval.MasterConfig.generate
:type: nemo_rl.models.generation.interfaces.GenerationConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.MasterConfig.generate
```

````

````{py:attribute} data
:canonical: nemo_rl.evals.eval.MasterConfig.data
:type: nemo_rl.data.MathDataConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.MasterConfig.data
```

````

````{py:attribute} env
:canonical: nemo_rl.evals.eval.MasterConfig.env
:type: nemo_rl.environments.math_environment.MathEnvConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.MasterConfig.env
```

````

````{py:attribute} cluster
:canonical: nemo_rl.evals.eval.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.evals.eval.MasterConfig.cluster
```

````

`````

````{py:function} setup(master_config: nemo_rl.evals.eval.MasterConfig, tokenizer: transformers.AutoTokenizer, dataset: nemo_rl.data.datasets.AllTaskProcessedDataset) -> tuple[nemo_rl.models.generation.vllm.VllmGeneration, torch.utils.data.DataLoader, nemo_rl.evals.eval.MasterConfig]
:canonical: nemo_rl.evals.eval.setup

```{autodoc2-docstring} nemo_rl.evals.eval.setup
```
````

````{py:function} eval_pass_k(rewards: torch.Tensor, num_tests_per_prompt: int, k: int) -> float
:canonical: nemo_rl.evals.eval.eval_pass_k

```{autodoc2-docstring} nemo_rl.evals.eval.eval_pass_k
```
````

````{py:function} run_env_eval(vllm_generation, dataloader, env, master_config)
:canonical: nemo_rl.evals.eval.run_env_eval

```{autodoc2-docstring} nemo_rl.evals.eval.run_env_eval
```
````
