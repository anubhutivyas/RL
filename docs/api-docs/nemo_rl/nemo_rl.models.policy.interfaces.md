# {py:mod}`nemo_rl.models.policy.interfaces`

```{py:module} nemo_rl.models.policy.interfaces
```

```{autodoc2-docstring} nemo_rl.models.policy.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LogprobOutputSpec <nemo_rl.models.policy.interfaces.LogprobOutputSpec>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.interfaces.LogprobOutputSpec
    :summary:
    ```
* - {py:obj}`ReferenceLogprobOutputSpec <nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec
    :summary:
    ```
* - {py:obj}`PolicyInterface <nemo_rl.models.policy.interfaces.PolicyInterface>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface
    :summary:
    ```
* - {py:obj}`ColocatablePolicyInterface <nemo_rl.models.policy.interfaces.ColocatablePolicyInterface>`
  -
````

### API

`````{py:class} LogprobOutputSpec()
:canonical: nemo_rl.models.policy.interfaces.LogprobOutputSpec

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.LogprobOutputSpec
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.LogprobOutputSpec.__init__
```

````{py:attribute} logprobs
:canonical: nemo_rl.models.policy.interfaces.LogprobOutputSpec.logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.LogprobOutputSpec.logprobs
```

````

`````

`````{py:class} ReferenceLogprobOutputSpec()
:canonical: nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec.__init__
```

````{py:attribute} reference_logprobs
:canonical: nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec.reference_logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec.reference_logprobs
```

````

`````

`````{py:class} PolicyInterface
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface
```

````{py:method} get_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec]) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.LogprobOutputSpec]
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.get_logprobs
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.get_logprobs
```

````

````{py:method} get_reference_policy_logprobs(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec]) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.policy.interfaces.ReferenceLogprobOutputSpec]
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.get_reference_policy_logprobs
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.get_reference_policy_logprobs
```

````

````{py:method} train(data: nemo_rl.distributed.batched_data_dict.BatchedDataDict, loss_fn: nemo_rl.algorithms.interfaces.LossFunction) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.train
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.train
```

````

````{py:method} prepare_for_training(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.prepare_for_training
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.prepare_for_training
```

````

````{py:method} finish_training(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.finish_training
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.finish_training
```

````

````{py:method} save_checkpoint(*args: typing.Any, **kwargs: typing.Any) -> None
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.save_checkpoint
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.save_checkpoint
```

````

````{py:method} shutdown() -> bool
:canonical: nemo_rl.models.policy.interfaces.PolicyInterface.shutdown
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.PolicyInterface.shutdown
```

````

`````

`````{py:class} ColocatablePolicyInterface
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface

Bases: {py:obj}`nemo_rl.models.policy.interfaces.PolicyInterface`

````{py:method} init_collective(ip: str, port: int, world_size: int) -> list[ray.ObjectRef]
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.init_collective
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.init_collective
```

````

````{py:method} offload_before_refit() -> None
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.offload_before_refit
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.offload_before_refit
```

````

````{py:method} offload_after_refit() -> None
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.offload_after_refit
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.offload_after_refit
```

````

````{py:method} prepare_weights_for_ipc(*args: typing.Any, **kwargs: typing.Any) -> list[list[str]]
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.prepare_weights_for_ipc
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.prepare_weights_for_ipc
```

````

````{py:method} get_weights_ipc_handles(keys: list[str]) -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.get_weights_ipc_handles
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.get_weights_ipc_handles
```

````

````{py:method} prepare_info_for_collective() -> dict[str, typing.Any]
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.prepare_info_for_collective
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.prepare_info_for_collective
```

````

````{py:method} broadcast_weights_for_collective() -> list[ray.ObjectRef]
:canonical: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.broadcast_weights_for_collective
:abstractmethod:

```{autodoc2-docstring} nemo_rl.models.policy.interfaces.ColocatablePolicyInterface.broadcast_weights_for_collective
```

````

`````
