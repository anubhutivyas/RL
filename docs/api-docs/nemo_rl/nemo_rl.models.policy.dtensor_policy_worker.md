# {py:mod}`nemo_rl.models.policy.dtensor_policy_worker`

```{py:module} nemo_rl.models.policy.dtensor_policy_worker
```

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DTensorPolicyWorker <nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`unshard_fsdp2_model <nemo_rl.models.policy.dtensor_policy_worker.unshard_fsdp2_model>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.unshard_fsdp2_model
    :summary:
    ```
* - {py:obj}`get_cpu_state_dict <nemo_rl.models.policy.dtensor_policy_worker.get_cpu_state_dict>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.get_cpu_state_dict
    :summary:
    ```
````

### API

````{py:function} unshard_fsdp2_model(model: torch.nn.Module) -> typing.Generator[None, None, None]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.unshard_fsdp2_model

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.unshard_fsdp2_model
```
````

````{py:function} get_cpu_state_dict(state_generator: typing.Iterable[tuple[str, typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]]], pin_memory: bool = False) -> dict[str, torch.Tensor]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.get_cpu_state_dict

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.get_cpu_state_dict
```
````

`````{py:class} DTensorPolicyWorker(config: nemo_rl.models.policy.PolicyConfig, tokenizer: transformers.AutoTokenizer, weights_path: typing.Optional[str] = None, optimizer_path: typing.Optional[str] = None, init_optimizer: bool = True, init_reference_model: bool = True, **kwargs: typing.Any)
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.__init__
```

````{py:method} __repr__() -> str
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.__repr__

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.__repr__
```

````

````{py:method} create_context_parallel_ctx(cp_mesh: torch.distributed.device_mesh.DeviceMesh, cp_buffers: typing.List[torch.Tensor], cp_seq_dims: typing.List[int], cp_no_restore_buffers: typing.Set[torch.Tensor], cp_rotate_method: typing.Optional[str] = None)
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.create_context_parallel_ctx
:staticmethod:

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.create_context_parallel_ctx
```

````

````{py:method} train_context(cp_context: typing.Optional[typing.Generator[None, None, None]] = None)
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.train_context
:staticmethod:

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.train_context
```

````

````{py:method} init_collective(ip: str, port: int, world_size: int) -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.init_collective

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.init_collective
```

````

````{py:method} is_alive() -> bool
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.is_alive

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.is_alive
```

````

````{py:method} reset_peak_memory_stats() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.reset_peak_memory_stats

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.reset_peak_memory_stats
```

````

````{py:method} get_gpu_info() -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_gpu_info

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_gpu_info
```

````

````{py:method} train(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], loss_fn: nemo_rl.algorithms.interfaces.LossFunction, eval_mode: bool = False, gbs: typing.Optional[int] = None, mbs: typing.Optional[int] = None) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.train

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.train
```

````

````{py:method} get_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.LogprobOutputSpec]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_logprobs
```

````

````{py:method} use_reference_model() -> typing.Generator[None, None, None]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.use_reference_model

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.use_reference_model
```

````

````{py:method} get_reference_policy_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_reference_policy_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_reference_policy_logprobs
```

````

````{py:method} _add_noise_to_weights() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker._add_noise_to_weights

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker._add_noise_to_weights
```

````

````{py:method} return_state_dict()
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.return_state_dict

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.return_state_dict
```

````

````{py:method} report_device_id() -> str
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.report_device_id

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.report_device_id
```

````

````{py:method} prepare_weights_for_ipc() -> tuple[list[tuple[str, int]], float]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_weights_for_ipc

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_weights_for_ipc
```

````

````{py:method} get_weights_ipc_handles(keys: typing.Iterable[str]) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_weights_ipc_handles

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.get_weights_ipc_handles
```

````

````{py:method} prepare_info_for_collective() -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_info_for_collective

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_info_for_collective
```

````

````{py:method} broadcast_weights_for_collective() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.broadcast_weights_for_collective

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.broadcast_weights_for_collective
```

````

````{py:method} prepare_for_lp_inference() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_for_lp_inference

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_for_lp_inference
```

````

````{py:method} prepare_for_training(*args, **kwargs) -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_for_training

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.prepare_for_training
```

````

````{py:method} offload_before_refit() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.offload_before_refit

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.offload_before_refit
```

````

````{py:method} offload_after_refit() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.offload_after_refit

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.offload_after_refit
```

````

````{py:method} move_to_device(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_device

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_device
```

````

````{py:method} move_buffer_to_device(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_buffer_to_device

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_buffer_to_device
```

````

````{py:method} move_to_cuda(model: torch.nn.Module) -> torch.nn.Module
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_cuda

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_cuda
```

````

````{py:method} move_to_cpu(model: torch.nn.Module) -> torch.nn.Module
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_cpu

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.move_to_cpu
```

````

````{py:method} save_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None, tokenizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.save_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.save_checkpoint
```

````

````{py:method} load_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.load_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.load_checkpoint
```

````

````{py:method} shutdown() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.shutdown

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.shutdown
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker.stop_gpu_profiling
```

````

`````
