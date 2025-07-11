# {py:mod}`nemo_rl.utils.nvml`

```{py:module} nemo_rl.utils.nvml
```

```{autodoc2-docstring} nemo_rl.utils.nvml
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`nvml_context <nemo_rl.utils.nvml.nvml_context>`
  - ```{autodoc2-docstring} nemo_rl.utils.nvml.nvml_context
    :summary:
    ```
* - {py:obj}`device_id_to_physical_device_id <nemo_rl.utils.nvml.device_id_to_physical_device_id>`
  - ```{autodoc2-docstring} nemo_rl.utils.nvml.device_id_to_physical_device_id
    :summary:
    ```
* - {py:obj}`get_device_uuid <nemo_rl.utils.nvml.get_device_uuid>`
  - ```{autodoc2-docstring} nemo_rl.utils.nvml.get_device_uuid
    :summary:
    ```
* - {py:obj}`get_free_memory_bytes <nemo_rl.utils.nvml.get_free_memory_bytes>`
  - ```{autodoc2-docstring} nemo_rl.utils.nvml.get_free_memory_bytes
    :summary:
    ```
````

### API

````{py:function} nvml_context() -> typing.Generator[None, None, None]
:canonical: nemo_rl.utils.nvml.nvml_context

```{autodoc2-docstring} nemo_rl.utils.nvml.nvml_context
```
````

````{py:function} device_id_to_physical_device_id(device_id: int) -> int
:canonical: nemo_rl.utils.nvml.device_id_to_physical_device_id

```{autodoc2-docstring} nemo_rl.utils.nvml.device_id_to_physical_device_id
```
````

````{py:function} get_device_uuid(device_idx: int) -> str
:canonical: nemo_rl.utils.nvml.get_device_uuid

```{autodoc2-docstring} nemo_rl.utils.nvml.get_device_uuid
```
````

````{py:function} get_free_memory_bytes(device_idx: int) -> float
:canonical: nemo_rl.utils.nvml.get_free_memory_bytes

```{autodoc2-docstring} nemo_rl.utils.nvml.get_free_memory_bytes
```
````
