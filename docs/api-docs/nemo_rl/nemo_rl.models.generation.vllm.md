# {py:mod}`nemo_rl.models.generation.vllm`

```{py:module} nemo_rl.models.generation.vllm
```

```{autodoc2-docstring} nemo_rl.models.generation.vllm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VllmSpecificArgs <nemo_rl.models.generation.vllm.VllmSpecificArgs>`
  -
* - {py:obj}`VllmConfig <nemo_rl.models.generation.vllm.VllmConfig>`
  -
* - {py:obj}`VllmGenerationWorker <nemo_rl.models.generation.vllm.VllmGenerationWorker>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker
    :summary:
    ```
* - {py:obj}`VllmGeneration <nemo_rl.models.generation.vllm.VllmGeneration>`
  -
````

### API

`````{py:class} VllmSpecificArgs()
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} tensor_parallel_size
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.tensor_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.tensor_parallel_size
```

````

````{py:attribute} pipeline_parallel_size
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.pipeline_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.pipeline_parallel_size
```

````

````{py:attribute} gpu_memory_utilization
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.gpu_memory_utilization
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.gpu_memory_utilization
```

````

````{py:attribute} max_model_len
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.max_model_len
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.max_model_len
```

````

````{py:attribute} skip_tokenizer_init
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.skip_tokenizer_init
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.skip_tokenizer_init
```

````

````{py:attribute} async_engine
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.async_engine
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.async_engine
```

````

````{py:attribute} load_format
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.load_format
:type: typing.NotRequired[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.load_format
```

````

````{py:attribute} precision
:canonical: nemo_rl.models.generation.vllm.VllmSpecificArgs.precision
:type: typing.NotRequired[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmSpecificArgs.precision
```

````

`````

`````{py:class} VllmConfig()
:canonical: nemo_rl.models.generation.vllm.VllmConfig

Bases: {py:obj}`nemo_rl.models.generation.interfaces.GenerationConfig`

````{py:attribute} vllm_cfg
:canonical: nemo_rl.models.generation.vllm.VllmConfig.vllm_cfg
:type: nemo_rl.models.generation.vllm.VllmSpecificArgs
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmConfig.vllm_cfg
```

````

````{py:attribute} vllm_kwargs
:canonical: nemo_rl.models.generation.vllm.VllmConfig.vllm_kwargs
:type: typing.NotRequired[dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmConfig.vllm_kwargs
```

````

`````

`````{py:class} VllmGenerationWorker(config: nemo_rl.models.generation.vllm.VllmConfig, bundle_indices: typing.Optional[list[int]] = None, fraction_of_gpus: float = 1.0, seed: typing.Optional[int] = None)
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.__init__
```

````{py:method} __repr__() -> str
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.__repr__

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.__repr__
```

````

````{py:method} configure_worker(num_gpus: int | float, bundle_indices: typing.Optional[tuple[int, list[int]]] = None) -> tuple[dict[str, typing.Any], dict[str, str], dict[str, typing.Any]]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.configure_worker
:staticmethod:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.configure_worker
```

````

````{py:method} init_collective(rank_prefix: int, ip: str, port: int, world_size: int) -> None
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.init_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.init_collective
```

````

````{py:method} init_collective_async(rank_prefix: int, ip: str, port: int, world_size: int) -> None
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.init_collective_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.init_collective_async
```

````

````{py:method} llm()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.llm

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.llm
```

````

````{py:method} is_alive()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.is_alive

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.is_alive
```

````

````{py:method} _merge_stop_strings(batch_stop_strings)
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker._merge_stop_strings

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker._merge_stop_strings
```

````

````{py:method} _build_sampling_params(*, greedy: bool, stop_strings, max_new_tokens: typing.Optional[int] = None)
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker._build_sampling_params

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker._build_sampling_params
```

````

````{py:method} generate(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.generate

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.generate
```

````

````{py:method} generate_async(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> typing.AsyncGenerator[tuple[int, nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]], None]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.generate_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.generate_async
```

````

````{py:method} generate_text(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.generate_text

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.generate_text
```

````

````{py:method} shutdown() -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.shutdown

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.shutdown
```

````

````{py:method} report_device_id() -> list[str]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.report_device_id

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.report_device_id
```

````

````{py:method} report_device_id_async() -> list[str]
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.report_device_id_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.report_device_id_async
```

````

````{py:method} update_weights_from_ipc_handles(ipc_handles: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_ipc_handles

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_ipc_handles
```

````

````{py:method} update_weights_from_ipc_handles_async(ipc_handles: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_ipc_handles_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_ipc_handles_async
```

````

````{py:method} update_weights_from_collective(info: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_collective
```

````

````{py:method} update_weights_from_collective_async(info: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_collective_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.update_weights_from_collective_async
```

````

````{py:method} reset_prefix_cache()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.reset_prefix_cache

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.reset_prefix_cache
```

````

````{py:method} reset_prefix_cache_async()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.reset_prefix_cache_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.reset_prefix_cache_async
```

````

````{py:method} sleep()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.sleep

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.sleep
```

````

````{py:method} sleep_async()
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.sleep_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.sleep_async
```

````

````{py:method} wake_up(**kwargs)
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.wake_up

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.wake_up
```

````

````{py:method} wake_up_async(**kwargs)
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.wake_up_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.wake_up_async
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.generation.vllm.VllmGenerationWorker.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGenerationWorker.stop_gpu_profiling
```

````

`````

`````{py:class} VllmGeneration(cluster: nemo_rl.distributed.virtual_cluster.RayVirtualCluster, config: nemo_rl.models.generation.vllm.VllmConfig, name_prefix: str = 'vllm_policy', workers_per_node: typing.Optional[typing.Union[int, list[int]]] = None)
:canonical: nemo_rl.models.generation.vllm.VllmGeneration

Bases: {py:obj}`nemo_rl.models.generation.interfaces.GenerationInterface`

````{py:method} _get_tied_worker_bundle_indices(cluster: nemo_rl.distributed.virtual_cluster.RayVirtualCluster) -> typing.List[typing.Tuple[int, typing.List[int]]]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration._get_tied_worker_bundle_indices

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration._get_tied_worker_bundle_indices
```

````

````{py:method} _report_device_id() -> list[list[str]]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration._report_device_id

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration._report_device_id
```

````

````{py:method} init_collective(ip: str, port: int, world_size: int) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.init_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.init_collective
```

````

````{py:method} generate(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.generate

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.generate
```

````

````{py:method} generate_text(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.generate_text

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.generate_text
```

````

````{py:method} generate_async(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool = False) -> typing.AsyncGenerator[tuple[int, nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]], None]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.generate_async
:async:

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.generate_async
```

````

````{py:method} prepare_for_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.prepare_for_generation

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.prepare_for_generation
```

````

````{py:method} finish_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.finish_generation

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.finish_generation
```

````

````{py:method} shutdown() -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.shutdown

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.shutdown
```

````

````{py:method} update_weights(ipc_handles: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.update_weights

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.update_weights
```

````

````{py:method} update_weights_from_collective(info: dict[str, typing.Any]) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.update_weights_from_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.update_weights_from_collective
```

````

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.stop_gpu_profiling
```

````

````{py:method} __del__() -> None
:canonical: nemo_rl.models.generation.vllm.VllmGeneration.__del__

```{autodoc2-docstring} nemo_rl.models.generation.vllm.VllmGeneration.__del__
```

````

`````
