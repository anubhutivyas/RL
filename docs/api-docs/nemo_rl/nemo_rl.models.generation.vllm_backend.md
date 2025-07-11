# {py:mod}`nemo_rl.models.generation.vllm_backend`

```{py:module} nemo_rl.models.generation.vllm_backend
```

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VllmInternalWorkerExtension <nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension
    :summary:
    ```
````

### API

`````{py:class} VllmInternalWorkerExtension
:canonical: nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension
```

````{py:method} init_collective(rank_prefix: int, ip: str, port: int, world_size: int) -> None
:canonical: nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.init_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.init_collective
```

````

````{py:method} report_device_id() -> str
:canonical: nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.report_device_id

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.report_device_id
```

````

````{py:method} update_weights_from_ipc_handles(ipc_handles)
:canonical: nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.update_weights_from_ipc_handles

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.update_weights_from_ipc_handles
```

````

````{py:method} update_weights_from_collective(info: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.update_weights_from_collective

```{autodoc2-docstring} nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension.update_weights_from_collective
```

````

`````
