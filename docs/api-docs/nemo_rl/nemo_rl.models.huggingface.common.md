# {py:mod}`nemo_rl.models.huggingface.common`

```{py:module} nemo_rl.models.huggingface.common
```

```{autodoc2-docstring} nemo_rl.models.huggingface.common
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelFlag <nemo_rl.models.huggingface.common.ModelFlag>`
  - ```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_gemma_model <nemo_rl.models.huggingface.common.is_gemma_model>`
  - ```{autodoc2-docstring} nemo_rl.models.huggingface.common.is_gemma_model
    :summary:
    ```
````

### API

`````{py:class} ModelFlag(*args, **kwds)
:canonical: nemo_rl.models.huggingface.common.ModelFlag

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag.__init__
```

````{py:attribute} SKIP_DTENSOR_TIED_WEIGHTS_CHECK
:canonical: nemo_rl.models.huggingface.common.ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK
:value: >
   'auto(...)'

```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK
```

````

````{py:attribute} VLLM_LOAD_FORMAT_AUTO
:canonical: nemo_rl.models.huggingface.common.ModelFlag.VLLM_LOAD_FORMAT_AUTO
:value: >
   'auto(...)'

```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag.VLLM_LOAD_FORMAT_AUTO
```

````

````{py:method} matches(model_name: str) -> bool
:canonical: nemo_rl.models.huggingface.common.ModelFlag.matches

```{autodoc2-docstring} nemo_rl.models.huggingface.common.ModelFlag.matches
```

````

`````

````{py:function} is_gemma_model(model_name: str) -> bool
:canonical: nemo_rl.models.huggingface.common.is_gemma_model

```{autodoc2-docstring} nemo_rl.models.huggingface.common.is_gemma_model
```
````
