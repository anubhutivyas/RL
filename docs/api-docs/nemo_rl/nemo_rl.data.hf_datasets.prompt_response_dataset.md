# {py:mod}`nemo_rl.data.hf_datasets.prompt_response_dataset`

```{py:module} nemo_rl.data.hf_datasets.prompt_response_dataset
```

```{autodoc2-docstring} nemo_rl.data.hf_datasets.prompt_response_dataset
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PromptResponseDataset <nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset>`
  - ```{autodoc2-docstring} nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset
    :summary:
    ```
````

### API

`````{py:class} PromptResponseDataset(train_ds_path: str, val_ds_path: str, input_key: str = 'input', output_key: str = 'output')
:canonical: nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset

```{autodoc2-docstring} nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset.__init__
```

````{py:method} add_messages_key(example: dict[str, typing.Any]) -> dict[str, list[dict[str, typing.Any]]]
:canonical: nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset.add_messages_key

```{autodoc2-docstring} nemo_rl.data.hf_datasets.prompt_response_dataset.PromptResponseDataset.add_messages_key
```

````

`````
