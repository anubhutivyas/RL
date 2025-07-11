# {py:mod}`nemo_rl.data`

```{py:module} nemo_rl.data
```

```{autodoc2-docstring} nemo_rl.data
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

nemo_rl.data.hf_datasets
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

nemo_rl.data.datasets
nemo_rl.data.interfaces
nemo_rl.data.llm_message_utils
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataConfig <nemo_rl.data.DataConfig>`
  -
* - {py:obj}`MathDataConfig <nemo_rl.data.MathDataConfig>`
  -
````

### API

`````{py:class} DataConfig()
:canonical: nemo_rl.data.DataConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} max_input_seq_length
:canonical: nemo_rl.data.DataConfig.max_input_seq_length
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.max_input_seq_length
```

````

````{py:attribute} prompt_file
:canonical: nemo_rl.data.DataConfig.prompt_file
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.prompt_file
```

````

````{py:attribute} system_prompt_file
:canonical: nemo_rl.data.DataConfig.system_prompt_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.system_prompt_file
```

````

````{py:attribute} dataset_name
:canonical: nemo_rl.data.DataConfig.dataset_name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.dataset_name
```

````

````{py:attribute} val_dataset_name
:canonical: nemo_rl.data.DataConfig.val_dataset_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.val_dataset_name
```

````

````{py:attribute} add_bos
:canonical: nemo_rl.data.DataConfig.add_bos
:type: typing.Optional[bool]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.add_bos
```

````

````{py:attribute} add_eos
:canonical: nemo_rl.data.DataConfig.add_eos
:type: typing.Optional[bool]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.add_eos
```

````

````{py:attribute} input_key
:canonical: nemo_rl.data.DataConfig.input_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.input_key
```

````

````{py:attribute} output_key
:canonical: nemo_rl.data.DataConfig.output_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.DataConfig.output_key
```

````

`````

`````{py:class} MathDataConfig()
:canonical: nemo_rl.data.MathDataConfig

Bases: {py:obj}`nemo_rl.data.DataConfig`

````{py:attribute} problem_key
:canonical: nemo_rl.data.MathDataConfig.problem_key
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.MathDataConfig.problem_key
```

````

````{py:attribute} solution_key
:canonical: nemo_rl.data.MathDataConfig.solution_key
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.MathDataConfig.solution_key
```

````

`````
