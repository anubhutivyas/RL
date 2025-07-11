# {py:mod}`nemo_rl.distributed.collectives`

```{py:module} nemo_rl.distributed.collectives
```

```{autodoc2-docstring} nemo_rl.distributed.collectives
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rebalance_nd_tensor <nemo_rl.distributed.collectives.rebalance_nd_tensor>`
  - ```{autodoc2-docstring} nemo_rl.distributed.collectives.rebalance_nd_tensor
    :summary:
    ```
* - {py:obj}`gather_jagged_object_lists <nemo_rl.distributed.collectives.gather_jagged_object_lists>`
  - ```{autodoc2-docstring} nemo_rl.distributed.collectives.gather_jagged_object_lists
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <nemo_rl.distributed.collectives.T>`
  - ```{autodoc2-docstring} nemo_rl.distributed.collectives.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: nemo_rl.distributed.collectives.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} nemo_rl.distributed.collectives.T
```

````

````{py:function} rebalance_nd_tensor(tensor: torch.Tensor, group: typing.Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor
:canonical: nemo_rl.distributed.collectives.rebalance_nd_tensor

```{autodoc2-docstring} nemo_rl.distributed.collectives.rebalance_nd_tensor
```
````

````{py:function} gather_jagged_object_lists(local_objects: list[nemo_rl.distributed.collectives.T], group: typing.Optional[torch.distributed.ProcessGroup] = None) -> list[nemo_rl.distributed.collectives.T]
:canonical: nemo_rl.distributed.collectives.gather_jagged_object_lists

```{autodoc2-docstring} nemo_rl.distributed.collectives.gather_jagged_object_lists
```
````
