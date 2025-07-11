# {py:mod}`nemo_rl.models.policy.lm_policy`

```{py:module} nemo_rl.models.policy.lm_policy
```

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Policy <nemo_rl.models.policy.lm_policy.Policy>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathLike <nemo_rl.models.policy.lm_policy.PathLike>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.PathLike
    :summary:
    ```
````

### API

````{py:data} PathLike
:canonical: nemo_rl.models.policy.lm_policy.PathLike
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.PathLike
```

````

`````{py:class} Policy(cluster: nemo_rl.distributed.virtual_cluster.RayVirtualCluster, config: nemo_rl.models.policy.PolicyConfig, tokenizer: transformers.PreTrainedTokenizerBase, name_prefix: str = 'lm_policy', workers_per_node: typing.Optional[typing.Union[int, list[int]]] = None, init_optimizer: bool = True, weights_path: typing.Optional[nemo_rl.models.policy.lm_policy.PathLike] = None, optimizer_path: typing.Optional[nemo_rl.models.policy.lm_policy.PathLike] = None, init_reference_model: bool = True)
:canonical: nemo_rl.models.policy.lm_policy.Policy

Bases: {py:obj}`nemo_rl.models.policy.interfaces.ColocatablePolicyInterface`, {py:obj}`nemo_rl.models.generation.interfaces.GenerationInterface`

````{py:method} init_collective(ip: str, port: int, world_size: int) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.policy.lm_policy.Policy.init_collective

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.init_collective
```

````

````{py:method} get_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec]) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.LogprobOutputSpec]
:canonical: nemo_rl.models.policy.lm_policy.Policy.get_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.get_logprobs
```

````

````{py:method} get_reference_policy_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec]
:canonical: nemo_rl.models.policy.lm_policy.Policy.get_reference_policy_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.get_reference_policy_logprobs
```

````

````{py:method} train(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], loss_fn: nemo_rl.algorithms.interfaces.LossFunction, eval_mode: bool = False, gbs: typing.Optional[int] = None, mbs: typing.Optional[int] = None) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.lm_policy.Policy.train

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.train
```

````

````{py:method} generate(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.policy.lm_policy.Policy.generate

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.generate
```

````

````{py:method} prepare_for_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.policy.lm_policy.Policy.prepare_for_generation

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.prepare_for_generation
```

````

````{py:method} prepare_for_training(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.prepare_for_training

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.prepare_for_training
```

````

````{py:method} prepare_for_lp_inference(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.prepare_for_lp_inference

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.prepare_for_lp_inference
```

````

````{py:method} finish_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.policy.lm_policy.Policy.finish_generation

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.finish_generation
```

````

````{py:method} finish_training(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.finish_training

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.finish_training
```

````

````{py:method} prepare_weights_for_ipc(_refit_buffer_size_gb: typing.Optional[int] = None) -> list[list[str]]
:canonical: nemo_rl.models.policy.lm_policy.Policy.prepare_weights_for_ipc

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.prepare_weights_for_ipc
```

````

````{py:method} get_weights_ipc_handles(keys: list[str]) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.lm_policy.Policy.get_weights_ipc_handles

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.get_weights_ipc_handles
```

````

````{py:method} prepare_info_for_collective() -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.lm_policy.Policy.prepare_info_for_collective

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.prepare_info_for_collective
```

````

````{py:method} broadcast_weights_for_collective() -> list[ray.ObjectRef]
:canonical: nemo_rl.models.policy.lm_policy.Policy.broadcast_weights_for_collective

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.broadcast_weights_for_collective
```

````

````{py:method} offload_before_refit() -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.offload_before_refit

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.offload_before_refit
```

````

````{py:method} offload_after_refit() -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.offload_after_refit

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.offload_after_refit
```

````

````{py:method} save_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None, tokenizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.save_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.save_checkpoint
```

````

````{py:method} shutdown() -> bool
:canonical: nemo_rl.models.policy.lm_policy.Policy.shutdown

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.shutdown
```

````

````{py:method} __del__() -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.__del__

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.__del__
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.lm_policy.Policy.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.lm_policy.Policy.stop_gpu_profiling
```

````

`````
