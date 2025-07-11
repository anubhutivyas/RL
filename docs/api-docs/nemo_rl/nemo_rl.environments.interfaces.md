# {py:mod}`nemo_rl.environments.interfaces`

```{py:module} nemo_rl.environments.interfaces
```

```{autodoc2-docstring} nemo_rl.environments.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnvironmentReturn <nemo_rl.environments.interfaces.EnvironmentReturn>`
  - ```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn
    :summary:
    ```
* - {py:obj}`EnvironmentInterface <nemo_rl.environments.interfaces.EnvironmentInterface>`
  -
````

### API

`````{py:class} EnvironmentReturn
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn
```

````{py:attribute} observations
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn.observations
:type: list[dict[str, str]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn.observations
```

````

````{py:attribute} metadata
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn.metadata
:type: list[typing.Optional[dict]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn.metadata
```

````

````{py:attribute} next_stop_strings
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn.next_stop_strings
:type: list[list[str] | None] | list[None]
:value: >
   None

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn.next_stop_strings
```

````

````{py:attribute} rewards
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn.rewards
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn.rewards
```

````

````{py:attribute} terminateds
:canonical: nemo_rl.environments.interfaces.EnvironmentReturn.terminateds
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentReturn.terminateds
```

````

`````

`````{py:class} EnvironmentInterface
:canonical: nemo_rl.environments.interfaces.EnvironmentInterface

Bases: {py:obj}`abc.ABC`

````{py:method} step(message_log_batch: list[list[dict[str, str]]], metadata: list[typing.Optional[dict]], *args, **kwargs) -> nemo_rl.environments.interfaces.EnvironmentReturn
:canonical: nemo_rl.environments.interfaces.EnvironmentInterface.step
:abstractmethod:

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentInterface.step
```

````

````{py:method} global_post_process_and_metrics(batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict, dict]
:canonical: nemo_rl.environments.interfaces.EnvironmentInterface.global_post_process_and_metrics
:abstractmethod:

```{autodoc2-docstring} nemo_rl.environments.interfaces.EnvironmentInterface.global_post_process_and_metrics
```

````

`````
