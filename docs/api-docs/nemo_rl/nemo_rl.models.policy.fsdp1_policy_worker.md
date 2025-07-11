# {py:mod}`nemo_rl.models.policy.fsdp1_policy_worker`

```{py:module} nemo_rl.models.policy.fsdp1_policy_worker
```

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FSDP1PolicyWorker <nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker
    :summary:
    ```
````

### API

`````{py:class} FSDP1PolicyWorker(config: nemo_rl.models.policy.PolicyConfig, tokenizer: transformers.PreTrainedTokenizerBase, weights_path: typing.Optional[str] = None, optimizer_path: typing.Optional[str] = None, init_optimizer: bool = True, init_reference_model: bool = True, **kwargs: typing.Any)
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.__init__
```

````{py:method} __repr__() -> str
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.__repr__

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.__repr__
```

````

````{py:method} is_alive() -> bool
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.is_alive

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.is_alive
```

````

````{py:method} reset_peak_memory_stats() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.reset_peak_memory_stats

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.reset_peak_memory_stats
```

````

````{py:method} get_gpu_info() -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_gpu_info

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_gpu_info
```

````

````{py:method} train(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], loss_fn: nemo_rl.algorithms.interfaces.LossFunction, eval_mode: bool = False, gbs: typing.Optional[int] = None, mbs: typing.Optional[int] = None) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.train

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.train
```

````

````{py:method} get_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.LogprobOutputSpec]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_logprobs
```

````

````{py:method} use_reference_model() -> typing.Generator[None, None, None]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.use_reference_model

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.use_reference_model
```

````

````{py:method} get_reference_policy_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_reference_policy_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_reference_policy_logprobs
```

````

````{py:method} generate(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.generate

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.generate
```

````

````{py:method} _add_noise_to_weights() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker._add_noise_to_weights

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker._add_noise_to_weights
```

````

````{py:method} report_device_id() -> str
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.report_device_id

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.report_device_id
```

````

````{py:method} prepare_weights_for_ipc() -> tuple[list[tuple[str, int]], float]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_weights_for_ipc

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_weights_for_ipc
```

````

````{py:method} get_weights_ipc_handles(keys: list[str]) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_weights_ipc_handles

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.get_weights_ipc_handles
```

````

````{py:method} prepare_for_lp_inference() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_for_lp_inference

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_for_lp_inference
```

````

````{py:method} prepare_for_training(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_for_training

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.prepare_for_training
```

````

````{py:method} offload_before_refit() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.offload_before_refit

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.offload_before_refit
```

````

````{py:method} offload_after_refit() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.offload_after_refit

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.offload_after_refit
```

````

````{py:method} manual_offload_to_cpu(model: torch.nn.Module) -> torch.nn.Module
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.manual_offload_to_cpu

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.manual_offload_to_cpu
```

````

````{py:method} manual_load_to_gpu(model: torch.nn.Module) -> torch.nn.Module
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.manual_load_to_gpu

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.manual_load_to_gpu
```

````

````{py:method} save_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None, tokenizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.save_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.save_checkpoint
```

````

````{py:method} load_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.load_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.load_checkpoint
```

````

````{py:method} shutdown() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.shutdown

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.shutdown
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.fsdp1_policy_worker.FSDP1PolicyWorker.stop_gpu_profiling
```

````

`````
