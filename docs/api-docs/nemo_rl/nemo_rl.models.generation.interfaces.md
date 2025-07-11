# {py:mod}`nemo_rl.models.generation.interfaces`

```{py:module} nemo_rl.models.generation.interfaces
```

```{autodoc2-docstring} nemo_rl.models.generation.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ColocationConfig <nemo_rl.models.generation.interfaces.ColocationConfig>`
  -
* - {py:obj}`GenerationConfig <nemo_rl.models.generation.interfaces.GenerationConfig>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig
    :summary:
    ```
* - {py:obj}`GenerationDatumSpec <nemo_rl.models.generation.interfaces.GenerationDatumSpec>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec
    :summary:
    ```
* - {py:obj}`GenerationOutputSpec <nemo_rl.models.generation.interfaces.GenerationOutputSpec>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec
    :summary:
    ```
* - {py:obj}`GenerationInterface <nemo_rl.models.generation.interfaces.GenerationInterface>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`verify_right_padding <nemo_rl.models.generation.interfaces.verify_right_padding>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.interfaces.verify_right_padding
    :summary:
    ```
````

### API

````{py:function} verify_right_padding(data: typing.Union[nemo_rl.distributed.batched_data_dict.BatchedDataDict[GenerationDatumSpec], nemo_rl.distributed.batched_data_dict.BatchedDataDict[GenerationOutputSpec]], pad_value: int = 0, raise_error: bool = True) -> tuple[bool, typing.Union[str, None]]
:canonical: nemo_rl.models.generation.interfaces.verify_right_padding

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.verify_right_padding
```
````

``````{py:class} ColocationConfig()
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig

Bases: {py:obj}`typing.TypedDict`

`````{py:class} ResourcesConfig()
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} gpus_per_node
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig.gpus_per_node
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig.gpus_per_node
```

````

````{py:attribute} num_nodes
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig.num_nodes
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig.num_nodes
```

````

`````

````{py:attribute} enabled
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.ColocationConfig.enabled
```

````

````{py:attribute} resources
:canonical: nemo_rl.models.generation.interfaces.ColocationConfig.resources
:type: typing.NotRequired[nemo_rl.models.generation.interfaces.ColocationConfig.ResourcesConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.ColocationConfig.resources
```

````

``````

`````{py:class} GenerationConfig()
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.__init__
```

````{py:attribute} backend
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.backend
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.backend
```

````

````{py:attribute} max_new_tokens
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.max_new_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.max_new_tokens
```

````

````{py:attribute} temperature
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.temperature
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.temperature
```

````

````{py:attribute} top_p
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.top_p
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.top_p
```

````

````{py:attribute} top_k
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.top_k
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.top_k
```

````

````{py:attribute} model_name
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.model_name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.model_name
```

````

````{py:attribute} stop_token_ids
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.stop_token_ids
:type: list[int]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.stop_token_ids
```

````

````{py:attribute} stop_strings
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.stop_strings
:type: typing.NotRequired[list[str]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.stop_strings
```

````

````{py:attribute} pad_token_id
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.pad_token_id
:type: typing.NotRequired[int]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.pad_token_id
```

````

````{py:attribute} colocated
:canonical: nemo_rl.models.generation.interfaces.GenerationConfig.colocated
:type: typing.NotRequired[nemo_rl.models.generation.interfaces.ColocationConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationConfig.colocated
```

````

`````

`````{py:class} GenerationDatumSpec()
:canonical: nemo_rl.models.generation.interfaces.GenerationDatumSpec

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec.__init__
```

````{py:attribute} input_ids
:canonical: nemo_rl.models.generation.interfaces.GenerationDatumSpec.input_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec.input_ids
```

````

````{py:attribute} input_lengths
:canonical: nemo_rl.models.generation.interfaces.GenerationDatumSpec.input_lengths
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec.input_lengths
```

````

````{py:attribute} stop_strings
:canonical: nemo_rl.models.generation.interfaces.GenerationDatumSpec.stop_strings
:type: typing.Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec.stop_strings
```

````

````{py:attribute} __extra__
:canonical: nemo_rl.models.generation.interfaces.GenerationDatumSpec.__extra__
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationDatumSpec.__extra__
```

````

`````

`````{py:class} GenerationOutputSpec()
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.__init__
```

````{py:attribute} output_ids
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec.output_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.output_ids
```

````

````{py:attribute} generation_lengths
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec.generation_lengths
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.generation_lengths
```

````

````{py:attribute} unpadded_sequence_lengths
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec.unpadded_sequence_lengths
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.unpadded_sequence_lengths
```

````

````{py:attribute} logprobs
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec.logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.logprobs
```

````

````{py:attribute} __extra__
:canonical: nemo_rl.models.generation.interfaces.GenerationOutputSpec.__extra__
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationOutputSpec.__extra__
```

````

`````

`````{py:class} GenerationInterface
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface
```

````{py:method} init_collective(ip: str, port: int, world_size: int) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.init_collective
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.init_collective
```

````

````{py:method} generate(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], greedy: bool) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationOutputSpec]
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.generate
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.generate
```

````

````{py:method} prepare_for_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.prepare_for_generation
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.prepare_for_generation
```

````

````{py:method} finish_generation(*args: typing.Any, **kwargs: typing.Any) -> bool
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.finish_generation
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.finish_generation
```

````

````{py:method} update_weights(ipc_handles: dict[str, typing.Any]) -> bool
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.update_weights
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.update_weights
```

````

````{py:method} update_weights_from_collective(info: dict[str, typing.Any]) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.generation.interfaces.GenerationInterface.update_weights_from_collective
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.generation.interfaces.GenerationInterface.update_weights_from_collective
```

````

`````
