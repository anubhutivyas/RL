# {py:mod}`nemo_rl.algorithms.sft`

```{py:module} nemo_rl.algorithms.sft
```

```{autodoc2-docstring} nemo_rl.algorithms.sft
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SFTSaveState <nemo_rl.algorithms.sft.SFTSaveState>`
  -
* - {py:obj}`SFTConfig <nemo_rl.algorithms.sft.SFTConfig>`
  -
* - {py:obj}`MasterConfig <nemo_rl.algorithms.sft.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_sft_save_state <nemo_rl.algorithms.sft._default_sft_save_state>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.sft._default_sft_save_state
    :summary:
    ```
* - {py:obj}`setup <nemo_rl.algorithms.sft.setup>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.sft.setup
    :summary:
    ```
* - {py:obj}`validate <nemo_rl.algorithms.sft.validate>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.sft.validate
    :summary:
    ```
* - {py:obj}`sft_train <nemo_rl.algorithms.sft.sft_train>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.sft.sft_train
    :summary:
    ```
````

### API

`````{py:class} SFTSaveState()
:canonical: nemo_rl.algorithms.sft.SFTSaveState

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} epoch
:canonical: nemo_rl.algorithms.sft.SFTSaveState.epoch
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTSaveState.epoch
```

````

````{py:attribute} step
:canonical: nemo_rl.algorithms.sft.SFTSaveState.step
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTSaveState.step
```

````

````{py:attribute} total_steps
:canonical: nemo_rl.algorithms.sft.SFTSaveState.total_steps
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTSaveState.total_steps
```

````

````{py:attribute} val_loss
:canonical: nemo_rl.algorithms.sft.SFTSaveState.val_loss
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTSaveState.val_loss
```

````

````{py:attribute} consumed_samples
:canonical: nemo_rl.algorithms.sft.SFTSaveState.consumed_samples
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTSaveState.consumed_samples
```

````

`````

````{py:function} _default_sft_save_state() -> nemo_rl.algorithms.sft.SFTSaveState
:canonical: nemo_rl.algorithms.sft._default_sft_save_state

```{autodoc2-docstring} nemo_rl.algorithms.sft._default_sft_save_state
```
````

`````{py:class} SFTConfig()
:canonical: nemo_rl.algorithms.sft.SFTConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} max_num_steps
:canonical: nemo_rl.algorithms.sft.SFTConfig.max_num_steps
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.max_num_steps
```

````

````{py:attribute} max_num_epochs
:canonical: nemo_rl.algorithms.sft.SFTConfig.max_num_epochs
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.max_num_epochs
```

````

````{py:attribute} val_period
:canonical: nemo_rl.algorithms.sft.SFTConfig.val_period
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.val_period
```

````

````{py:attribute} val_batches
:canonical: nemo_rl.algorithms.sft.SFTConfig.val_batches
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.val_batches
```

````

````{py:attribute} val_global_batch_size
:canonical: nemo_rl.algorithms.sft.SFTConfig.val_global_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.val_global_batch_size
```

````

````{py:attribute} val_micro_batch_size
:canonical: nemo_rl.algorithms.sft.SFTConfig.val_micro_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.val_micro_batch_size
```

````

````{py:attribute} val_at_start
:canonical: nemo_rl.algorithms.sft.SFTConfig.val_at_start
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.val_at_start
```

````

````{py:attribute} seed
:canonical: nemo_rl.algorithms.sft.SFTConfig.seed
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.SFTConfig.seed
```

````

`````

`````{py:class} MasterConfig()
:canonical: nemo_rl.algorithms.sft.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} policy
:canonical: nemo_rl.algorithms.sft.MasterConfig.policy
:type: nemo_rl.models.policy.PolicyConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.policy
```

````

````{py:attribute} data
:canonical: nemo_rl.algorithms.sft.MasterConfig.data
:type: nemo_rl.data.DataConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.data
```

````

````{py:attribute} sft
:canonical: nemo_rl.algorithms.sft.MasterConfig.sft
:type: nemo_rl.algorithms.sft.SFTConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.sft
```

````

````{py:attribute} logger
:canonical: nemo_rl.algorithms.sft.MasterConfig.logger
:type: nemo_rl.utils.logger.LoggerConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.logger
```

````

````{py:attribute} cluster
:canonical: nemo_rl.algorithms.sft.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.cluster
```

````

````{py:attribute} checkpointing
:canonical: nemo_rl.algorithms.sft.MasterConfig.checkpointing
:type: nemo_rl.utils.checkpoint.CheckpointingConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.sft.MasterConfig.checkpointing
```

````

`````

````{py:function} setup(master_config: nemo_rl.algorithms.sft.MasterConfig, tokenizer: transformers.AutoTokenizer, train_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset, val_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset) -> tuple[nemo_rl.models.policy.lm_policy.Policy, nemo_rl.distributed.virtual_cluster.RayVirtualCluster, torchdata.stateful_dataloader.StatefulDataLoader, torchdata.stateful_dataloader.StatefulDataLoader, nemo_rl.algorithms.loss_functions.NLLLoss, nemo_rl.algorithms.sft.MasterConfig, nemo_rl.utils.logger.Logger, nemo_rl.data.interfaces.TaskDataSpec, nemo_rl.algorithms.sft.SFTSaveState]
:canonical: nemo_rl.algorithms.sft.setup

```{autodoc2-docstring} nemo_rl.algorithms.sft.setup
```
````

````{py:function} validate(policy: nemo_rl.models.policy.interfaces.PolicyInterface, val_dataloader: torchdata.stateful_dataloader.StatefulDataLoader, tokenizer, loss_fn, step: int, master_config: nemo_rl.algorithms.sft.MasterConfig, sft_task_spec: nemo_rl.data.interfaces.TaskDataSpec, val_batches: int, val_batch_size: int, val_mbs: int)
:canonical: nemo_rl.algorithms.sft.validate

```{autodoc2-docstring} nemo_rl.algorithms.sft.validate
```
````

````{py:function} sft_train(policy, train_dataloader, val_dataloader, tokenizer, loss_fn, master_config, logger, sft_task_spec, checkpointer, sft_save_state)
:canonical: nemo_rl.algorithms.sft.sft_train

```{autodoc2-docstring} nemo_rl.algorithms.sft.sft_train
```
````
