# {py:mod}`nemo_rl.algorithms.grpo`

```{py:module} nemo_rl.algorithms.grpo
```

```{autodoc2-docstring} nemo_rl.algorithms.grpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GRPOConfig <nemo_rl.algorithms.grpo.GRPOConfig>`
  -
* - {py:obj}`GRPOSaveState <nemo_rl.algorithms.grpo.GRPOSaveState>`
  -
* - {py:obj}`GRPOLoggerConfig <nemo_rl.algorithms.grpo.GRPOLoggerConfig>`
  -
* - {py:obj}`MasterConfig <nemo_rl.algorithms.grpo.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_grpo_save_state <nemo_rl.algorithms.grpo._default_grpo_save_state>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo._default_grpo_save_state
    :summary:
    ```
* - {py:obj}`setup <nemo_rl.algorithms.grpo.setup>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo.setup
    :summary:
    ```
* - {py:obj}`_should_use_async_rollouts <nemo_rl.algorithms.grpo._should_use_async_rollouts>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo._should_use_async_rollouts
    :summary:
    ```
* - {py:obj}`refit_policy_generation <nemo_rl.algorithms.grpo.refit_policy_generation>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo.refit_policy_generation
    :summary:
    ```
* - {py:obj}`grpo_train <nemo_rl.algorithms.grpo.grpo_train>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo.grpo_train
    :summary:
    ```
* - {py:obj}`validate <nemo_rl.algorithms.grpo.validate>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo.validate
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <nemo_rl.algorithms.grpo.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.grpo.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: nemo_rl.algorithms.grpo.TokenizerType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} nemo_rl.algorithms.grpo.TokenizerType
```

````

`````{py:class} GRPOConfig()
:canonical: nemo_rl.algorithms.grpo.GRPOConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} num_prompts_per_step
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.num_prompts_per_step
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.num_prompts_per_step
```

````

````{py:attribute} num_generations_per_prompt
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.num_generations_per_prompt
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.num_generations_per_prompt
```

````

````{py:attribute} max_num_steps
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.max_num_steps
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.max_num_steps
```

````

````{py:attribute} max_rollout_turns
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.max_rollout_turns
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.max_rollout_turns
```

````

````{py:attribute} normalize_rewards
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.normalize_rewards
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.normalize_rewards
```

````

````{py:attribute} use_leave_one_out_baseline
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.use_leave_one_out_baseline
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.use_leave_one_out_baseline
```

````

````{py:attribute} val_period
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.val_period
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.val_period
```

````

````{py:attribute} val_batch_size
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.val_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.val_batch_size
```

````

````{py:attribute} val_at_start
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.val_at_start
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.val_at_start
```

````

````{py:attribute} max_val_samples
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.max_val_samples
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.max_val_samples
```

````

````{py:attribute} checkpoint_dir
:canonical: nemo_rl.algorithms.grpo.GRPOConfig.checkpoint_dir
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOConfig.checkpoint_dir
```

````

`````

`````{py:class} GRPOSaveState()
:canonical: nemo_rl.algorithms.grpo.GRPOSaveState

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} step
:canonical: nemo_rl.algorithms.grpo.GRPOSaveState.step
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOSaveState.step
```

````

````{py:attribute} val_reward
:canonical: nemo_rl.algorithms.grpo.GRPOSaveState.val_reward
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOSaveState.val_reward
```

````

````{py:attribute} consumed_samples
:canonical: nemo_rl.algorithms.grpo.GRPOSaveState.consumed_samples
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOSaveState.consumed_samples
```

````

`````

````{py:function} _default_grpo_save_state() -> nemo_rl.algorithms.grpo.GRPOSaveState
:canonical: nemo_rl.algorithms.grpo._default_grpo_save_state

```{autodoc2-docstring} nemo_rl.algorithms.grpo._default_grpo_save_state
```
````

`````{py:class} GRPOLoggerConfig()
:canonical: nemo_rl.algorithms.grpo.GRPOLoggerConfig

Bases: {py:obj}`nemo_rl.utils.logger.LoggerConfig`

````{py:attribute} num_val_samples_to_print
:canonical: nemo_rl.algorithms.grpo.GRPOLoggerConfig.num_val_samples_to_print
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.GRPOLoggerConfig.num_val_samples_to_print
```

````

`````

`````{py:class} MasterConfig()
:canonical: nemo_rl.algorithms.grpo.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} policy
:canonical: nemo_rl.algorithms.grpo.MasterConfig.policy
:type: nemo_rl.models.policy.PolicyConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.policy
```

````

````{py:attribute} loss_fn
:canonical: nemo_rl.algorithms.grpo.MasterConfig.loss_fn
:type: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.loss_fn
```

````

````{py:attribute} env
:canonical: nemo_rl.algorithms.grpo.MasterConfig.env
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.env
```

````

````{py:attribute} data
:canonical: nemo_rl.algorithms.grpo.MasterConfig.data
:type: nemo_rl.data.DataConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.data
```

````

````{py:attribute} grpo
:canonical: nemo_rl.algorithms.grpo.MasterConfig.grpo
:type: nemo_rl.algorithms.grpo.GRPOConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.grpo
```

````

````{py:attribute} logger
:canonical: nemo_rl.algorithms.grpo.MasterConfig.logger
:type: nemo_rl.algorithms.grpo.GRPOLoggerConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.logger
```

````

````{py:attribute} cluster
:canonical: nemo_rl.algorithms.grpo.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.cluster
```

````

````{py:attribute} checkpointing
:canonical: nemo_rl.algorithms.grpo.MasterConfig.checkpointing
:type: nemo_rl.utils.checkpoint.CheckpointingConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.grpo.MasterConfig.checkpointing
```

````

`````

````{py:function} setup(master_config: nemo_rl.algorithms.grpo.MasterConfig, tokenizer: nemo_rl.algorithms.grpo.TokenizerType, dataset: nemo_rl.data.datasets.AllTaskProcessedDataset, val_dataset: typing.Optional[nemo_rl.data.datasets.AllTaskProcessedDataset]) -> tuple[nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, typing.Optional[nemo_rl.models.generation.interfaces.GenerationInterface], typing.Tuple[nemo_rl.distributed.virtual_cluster.RayVirtualCluster, nemo_rl.distributed.virtual_cluster.RayVirtualCluster], torchdata.stateful_dataloader.StatefulDataLoader, typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], nemo_rl.algorithms.loss_functions.ClippedPGLossFn, nemo_rl.utils.logger.Logger, nemo_rl.utils.checkpoint.CheckpointManager, nemo_rl.algorithms.grpo.GRPOSaveState, nemo_rl.algorithms.grpo.MasterConfig]
:canonical: nemo_rl.algorithms.grpo.setup

```{autodoc2-docstring} nemo_rl.algorithms.grpo.setup
```
````

````{py:function} _should_use_async_rollouts(master_config: nemo_rl.algorithms.grpo.MasterConfig) -> bool
:canonical: nemo_rl.algorithms.grpo._should_use_async_rollouts

```{autodoc2-docstring} nemo_rl.algorithms.grpo._should_use_async_rollouts
```
````

````{py:function} refit_policy_generation(policy: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, colocated_inference: bool, _refit_buffer_size_gb: typing.Optional[int] = None) -> None
:canonical: nemo_rl.algorithms.grpo.refit_policy_generation

```{autodoc2-docstring} nemo_rl.algorithms.grpo.refit_policy_generation
```
````

````{py:function} grpo_train(policy: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, policy_generation: typing.Optional[nemo_rl.models.generation.interfaces.GenerationInterface], dataloader: torchdata.stateful_dataloader.StatefulDataLoader, val_dataloader: typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], tokenizer: nemo_rl.algorithms.grpo.TokenizerType, loss_fn: nemo_rl.algorithms.interfaces.LossFunction, task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface], val_task_to_env: typing.Optional[dict[str, nemo_rl.environments.interfaces.EnvironmentInterface]], logger: nemo_rl.utils.logger.Logger, checkpointer: nemo_rl.utils.checkpoint.CheckpointManager, grpo_save_state: nemo_rl.algorithms.grpo.GRPOSaveState, master_config: nemo_rl.algorithms.grpo.MasterConfig) -> None
:canonical: nemo_rl.algorithms.grpo.grpo_train

```{autodoc2-docstring} nemo_rl.algorithms.grpo.grpo_train
```
````

````{py:function} validate(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, val_dataloader: typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], tokenizer, val_task_to_env: typing.Optional[dict[str, nemo_rl.environments.interfaces.EnvironmentInterface]], step: int, master_config: nemo_rl.algorithms.grpo.MasterConfig) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]
:canonical: nemo_rl.algorithms.grpo.validate

```{autodoc2-docstring} nemo_rl.algorithms.grpo.validate
```
````
