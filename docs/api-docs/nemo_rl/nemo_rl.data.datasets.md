# {py:mod}`nemo_rl.data.datasets`

```{py:module} nemo_rl.data.datasets
```

```{autodoc2-docstring} nemo_rl.data.datasets
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AllTaskProcessedDataset <nemo_rl.data.datasets.AllTaskProcessedDataset>`
  - ```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rl_collate_fn <nemo_rl.data.datasets.rl_collate_fn>`
  - ```{autodoc2-docstring} nemo_rl.data.datasets.rl_collate_fn
    :summary:
    ```
* - {py:obj}`eval_collate_fn <nemo_rl.data.datasets.eval_collate_fn>`
  - ```{autodoc2-docstring} nemo_rl.data.datasets.eval_collate_fn
    :summary:
    ```
* - {py:obj}`dpo_collate_fn <nemo_rl.data.datasets.dpo_collate_fn>`
  - ```{autodoc2-docstring} nemo_rl.data.datasets.dpo_collate_fn
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <nemo_rl.data.datasets.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.data.datasets.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: nemo_rl.data.datasets.TokenizerType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.datasets.TokenizerType
```

````

`````{py:class} AllTaskProcessedDataset(dataset: typing.Union[datasets.Dataset, typing.Any], tokenizer: nemo_rl.data.datasets.TokenizerType, default_task_data_spec: nemo_rl.data.interfaces.TaskDataSpec, task_data_processors: typing.Union[dict[str, tuple[nemo_rl.data.interfaces.TaskDataSpec, nemo_rl.data.interfaces.TaskDataProcessFnCallable]], nemo_rl.data.interfaces.TaskDataProcessFnCallable], max_seq_length: typing.Optional[int] = None)
:canonical: nemo_rl.data.datasets.AllTaskProcessedDataset

```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset.__init__
```

````{py:method} __len__() -> int
:canonical: nemo_rl.data.datasets.AllTaskProcessedDataset.__len__

```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset.__len__
```

````

````{py:method} encode_single(text: typing.Union[str, list[str]]) -> tuple[list[int] | torch.Tensor, int]
:canonical: nemo_rl.data.datasets.AllTaskProcessedDataset.encode_single

```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset.encode_single
```

````

````{py:method} __getitem__(idx: int) -> nemo_rl.data.interfaces.DatumSpec
:canonical: nemo_rl.data.datasets.AllTaskProcessedDataset.__getitem__

```{autodoc2-docstring} nemo_rl.data.datasets.AllTaskProcessedDataset.__getitem__
```

````

`````

````{py:function} rl_collate_fn(data_batch: list[nemo_rl.data.interfaces.DatumSpec]) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any]
:canonical: nemo_rl.data.datasets.rl_collate_fn

```{autodoc2-docstring} nemo_rl.data.datasets.rl_collate_fn
```
````

````{py:function} eval_collate_fn(data_batch: list[nemo_rl.data.interfaces.DatumSpec]) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any]
:canonical: nemo_rl.data.datasets.eval_collate_fn

```{autodoc2-docstring} nemo_rl.data.datasets.eval_collate_fn
```
````

````{py:function} dpo_collate_fn(data_batch: list[nemo_rl.data.interfaces.DPODatumSpec], tokenizer: nemo_rl.data.datasets.TokenizerType, make_sequence_length_divisible_by: int) -> nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any]
:canonical: nemo_rl.data.datasets.dpo_collate_fn

```{autodoc2-docstring} nemo_rl.data.datasets.dpo_collate_fn
```
````
