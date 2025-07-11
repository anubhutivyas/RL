# {py:mod}`nemo_rl.models.policy.megatron_policy_worker`

```{py:module} nemo_rl.models.policy.megatron_policy_worker
```

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MegatronPolicyWorker <nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup_megatron_model <nemo_rl.models.policy.megatron_policy_worker.setup_megatron_model>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.setup_megatron_model
    :summary:
    ```
* - {py:obj}`destroy_parallel_state <nemo_rl.models.policy.megatron_policy_worker.destroy_parallel_state>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.destroy_parallel_state
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <nemo_rl.models.policy.megatron_policy_worker.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: nemo_rl.models.policy.megatron_policy_worker.TokenizerType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.TokenizerType
```

````

````{py:function} setup_megatron_model(policy_cfg: nemo_rl.models.policy.PolicyConfig, cfg: nemo.tron.config.ConfigContainer, load_optimizer: bool = True, get_embedding_ranks=None, get_position_embedding_ranks=None)
:canonical: nemo_rl.models.policy.megatron_policy_worker.setup_megatron_model

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.setup_megatron_model
```
````

````{py:function} destroy_parallel_state()
:canonical: nemo_rl.models.policy.megatron_policy_worker.destroy_parallel_state

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.destroy_parallel_state
```
````

`````{py:class} MegatronPolicyWorker(config: nemo_rl.models.policy.PolicyConfig, tokenizer: nemo_rl.models.policy.megatron_policy_worker.TokenizerType, weights_path: typing.Optional[str] = None, optimizer_path: typing.Optional[str] = None, init_optimizer: bool = True, init_reference_model: bool = True, *, worker_sharding_annotations: nemo_rl.distributed.named_sharding.NamedSharding, pre_init_communication_queue: ray.util.queue.Queue, megatron_checkpoint_home: typing.Optional[str] = None, **kwargs: typing.Any)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.__init__
```

````{py:method} __repr__()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.__repr__

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.__repr__
```

````

````{py:method} configure_worker(num_gpus: int, bundle_indices: typing.Optional[tuple] = None)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.configure_worker

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.configure_worker
```

````

````{py:method} is_alive()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.is_alive

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.is_alive
```

````

````{py:method} reset_peak_memory_stats() -> None
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.reset_peak_memory_stats

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.reset_peak_memory_stats
```

````

````{py:method} get_gpu_info()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_gpu_info

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_gpu_info
```

````

````{py:method} enable_forward_pre_hook()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.enable_forward_pre_hook

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.enable_forward_pre_hook
```

````

````{py:method} disable_forward_pre_hook(param_sync=True)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.disable_forward_pre_hook

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.disable_forward_pre_hook
```

````

````{py:method} train(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict, loss_fn: nemo_rl.algorithms.interfaces.LossFunction, eval_mode: bool = False, gbs: typing.Optional[int] = None, mbs: typing.Optional[int] = None) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.train

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.train
```

````

````{py:method} get_logprobs(*, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.LogprobOutputSpec]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_logprobs
```

````

````{py:method} use_reference_model()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.use_reference_model

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.use_reference_model
```

````

````{py:method} get_reference_policy_logprobs(*, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], micro_batch_size: typing.Optional[int] = None) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_reference_policy_logprobs

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_reference_policy_logprobs
```

````

````{py:method} generate(*, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.generate

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.generate
```

````

````{py:method} zero_out_weights()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.zero_out_weights

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.zero_out_weights
```

````

````{py:method} report_device_id() -> str
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.report_device_id

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.report_device_id
```

````

````{py:method} prepare_weights_for_ipc() -> tuple[list[tuple[str, int]], float]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_weights_for_ipc

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_weights_for_ipc
```

````

````{py:method} get_weights_ipc_handles(*, keys: list[str]) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_weights_ipc_handles

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.get_weights_ipc_handles
```

````

````{py:method} prepare_for_lp_inference()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_for_lp_inference

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_for_lp_inference
```

````

````{py:method} prepare_for_training(*args, **kwargs)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_for_training

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.prepare_for_training
```

````

````{py:method} offload_before_refit()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.offload_before_refit

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.offload_before_refit
```

````

````{py:method} offload_after_refit()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.offload_after_refit

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.offload_after_refit
```

````

````{py:method} move_model(model, device: str, move_params=True, move_grads=True)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.move_model

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.move_model
```

````

````{py:method} save_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None, **kwargs)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.save_checkpoint

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.save_checkpoint
```

````

````{py:method} load_checkpoint(weights_path: str, optimizer_path: typing.Optional[str] = None)
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.load_checkpoint
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.load_checkpoint
```

````

````{py:method} shutdown()
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.shutdown

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.shutdown
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.policy.megatron_policy_worker.MegatronPolicyWorker.stop_gpu_profiling
```

````

`````
