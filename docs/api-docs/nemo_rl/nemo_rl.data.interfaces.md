# {py:mod}`nemo_rl.data.interfaces`

```{py:module} nemo_rl.data.interfaces
```

```{autodoc2-docstring} nemo_rl.data.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DatumSpec <nemo_rl.data.interfaces.DatumSpec>`
  -
* - {py:obj}`DPODatumSpec <nemo_rl.data.interfaces.DPODatumSpec>`
  -
* - {py:obj}`TaskDataSpec <nemo_rl.data.interfaces.TaskDataSpec>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec
    :summary:
    ```
* - {py:obj}`TaskDataProcessFnCallable <nemo_rl.data.interfaces.TaskDataProcessFnCallable>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataProcessFnCallable
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LLMMessageLogType <nemo_rl.data.interfaces.LLMMessageLogType>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.LLMMessageLogType
    :summary:
    ```
* - {py:obj}`FlatMessagesType <nemo_rl.data.interfaces.FlatMessagesType>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.FlatMessagesType
    :summary:
    ```
* - {py:obj}`PathLike <nemo_rl.data.interfaces.PathLike>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.PathLike
    :summary:
    ```
* - {py:obj}`TokenizerType <nemo_rl.data.interfaces.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.data.interfaces.TokenizerType
    :summary:
    ```
````

### API

````{py:data} LLMMessageLogType
:canonical: nemo_rl.data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.LLMMessageLogType
```

````

````{py:data} FlatMessagesType
:canonical: nemo_rl.data.interfaces.FlatMessagesType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.FlatMessagesType
```

````

````{py:data} PathLike
:canonical: nemo_rl.data.interfaces.PathLike
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.PathLike
```

````

````{py:data} TokenizerType
:canonical: nemo_rl.data.interfaces.TokenizerType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.TokenizerType
```

````

`````{py:class} DatumSpec()
:canonical: nemo_rl.data.interfaces.DatumSpec

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} message_log
:canonical: nemo_rl.data.interfaces.DatumSpec.message_log
:type: nemo_rl.data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.message_log
```

````

````{py:attribute} length
:canonical: nemo_rl.data.interfaces.DatumSpec.length
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.length
```

````

````{py:attribute} extra_env_info
:canonical: nemo_rl.data.interfaces.DatumSpec.extra_env_info
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.extra_env_info
```

````

````{py:attribute} loss_multiplier
:canonical: nemo_rl.data.interfaces.DatumSpec.loss_multiplier
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.loss_multiplier
```

````

````{py:attribute} idx
:canonical: nemo_rl.data.interfaces.DatumSpec.idx
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.idx
```

````

````{py:attribute} task_name
:canonical: nemo_rl.data.interfaces.DatumSpec.task_name
:type: typing.NotRequired[str]
:value: >
   'default'

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.task_name
```

````

````{py:attribute} stop_strings
:canonical: nemo_rl.data.interfaces.DatumSpec.stop_strings
:type: typing.NotRequired[list[str]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.stop_strings
```

````

````{py:attribute} __extra__
:canonical: nemo_rl.data.interfaces.DatumSpec.__extra__
:type: typing.NotRequired[typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DatumSpec.__extra__
```

````

`````

`````{py:class} DPODatumSpec()
:canonical: nemo_rl.data.interfaces.DPODatumSpec

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} message_log_chosen
:canonical: nemo_rl.data.interfaces.DPODatumSpec.message_log_chosen
:type: nemo_rl.data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.message_log_chosen
```

````

````{py:attribute} message_log_rejected
:canonical: nemo_rl.data.interfaces.DPODatumSpec.message_log_rejected
:type: nemo_rl.data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.message_log_rejected
```

````

````{py:attribute} length_chosen
:canonical: nemo_rl.data.interfaces.DPODatumSpec.length_chosen
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.length_chosen
```

````

````{py:attribute} length_rejected
:canonical: nemo_rl.data.interfaces.DPODatumSpec.length_rejected
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.length_rejected
```

````

````{py:attribute} loss_multiplier
:canonical: nemo_rl.data.interfaces.DPODatumSpec.loss_multiplier
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.loss_multiplier
```

````

````{py:attribute} idx
:canonical: nemo_rl.data.interfaces.DPODatumSpec.idx
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.DPODatumSpec.idx
```

````

`````

`````{py:class} TaskDataSpec
:canonical: nemo_rl.data.interfaces.TaskDataSpec

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec
```

````{py:attribute} task_name
:canonical: nemo_rl.data.interfaces.TaskDataSpec.task_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec.task_name
```

````

````{py:attribute} prompt_file
:canonical: nemo_rl.data.interfaces.TaskDataSpec.prompt_file
:type: typing.Optional[nemo_rl.data.interfaces.PathLike]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec.prompt_file
```

````

````{py:attribute} system_prompt_file
:canonical: nemo_rl.data.interfaces.TaskDataSpec.system_prompt_file
:type: typing.Optional[nemo_rl.data.interfaces.PathLike]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec.system_prompt_file
```

````

````{py:method} __post_init__() -> None
:canonical: nemo_rl.data.interfaces.TaskDataSpec.__post_init__

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec.__post_init__
```

````

````{py:method} copy_defaults(from_spec: nemo_rl.data.interfaces.TaskDataSpec) -> None
:canonical: nemo_rl.data.interfaces.TaskDataSpec.copy_defaults

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataSpec.copy_defaults
```

````

`````

`````{py:class} TaskDataProcessFnCallable
:canonical: nemo_rl.data.interfaces.TaskDataProcessFnCallable

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataProcessFnCallable
```

````{py:method} __call__(datum_dict: dict[str, typing.Any], task_data_spec: nemo_rl.data.interfaces.TaskDataSpec, tokenizer: nemo_rl.data.interfaces.TokenizerType, max_seq_length: int, idx: int) -> nemo_rl.data.interfaces.DatumSpec
:canonical: nemo_rl.data.interfaces.TaskDataProcessFnCallable.__call__
:abstractmethod:

```{autodoc2-docstring} nemo_rl.data.interfaces.TaskDataProcessFnCallable.__call__
```

````

`````
