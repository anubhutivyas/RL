# {py:mod}`nemo_rl.models.generation`

```{py:module} nemo_rl.models.generation
```

```{autodoc2-docstring} nemo_rl.models.generation
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

nemo_rl.models.generation.interfaces
nemo_rl.models.generation.vllm
nemo_rl.models.generation.vllm_backend
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`configure_generation_config <nemo_rl.models.generation.configure_generation_config>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.configure_generation_config
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <nemo_rl.models.generation.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.models.generation.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: nemo_rl.models.generation.TokenizerType
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.generation.TokenizerType
```

````

````{py:function} configure_generation_config(config: nemo_rl.models.generation.interfaces.GenerationConfig, tokenizer: nemo_rl.models.generation.TokenizerType, is_eval=False) -> nemo_rl.models.generation.interfaces.GenerationConfig
:canonical: nemo_rl.models.generation.configure_generation_config

```{autodoc2-docstring} nemo_rl.models.generation.configure_generation_config
```
````
