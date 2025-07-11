# {py:mod}`nemo_rl.data.hf_datasets.deepscaler`

```{py:module} nemo_rl.data.hf_datasets.deepscaler
```

```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeepScalerDataset <nemo_rl.data.hf_datasets.deepscaler.DeepScalerDataset>`
  - ```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.DeepScalerDataset
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_math <nemo_rl.data.hf_datasets.deepscaler.format_math>`
  - ```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.format_math
    :summary:
    ```
* - {py:obj}`prepare_deepscaler_dataset <nemo_rl.data.hf_datasets.deepscaler.prepare_deepscaler_dataset>`
  - ```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.prepare_deepscaler_dataset
    :summary:
    ```
````

### API

````{py:function} format_math(data: dict[str, str | float | int]) -> dict[str, list[typing.Any] | str]
:canonical: nemo_rl.data.hf_datasets.deepscaler.format_math

```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.format_math
```
````

````{py:function} prepare_deepscaler_dataset(seed: int = 42) -> dict[str, datasets.Dataset | None]
:canonical: nemo_rl.data.hf_datasets.deepscaler.prepare_deepscaler_dataset

```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.prepare_deepscaler_dataset
```
````

````{py:class} DeepScalerDataset(seed: int = 42)
:canonical: nemo_rl.data.hf_datasets.deepscaler.DeepScalerDataset

```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.DeepScalerDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.data.hf_datasets.deepscaler.DeepScalerDataset.__init__
```

````
