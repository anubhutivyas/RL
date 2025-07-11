# {py:mod}`nemo_rl.distributed.named_sharding`

```{py:module} nemo_rl.distributed.named_sharding
```

```{autodoc2-docstring} nemo_rl.distributed.named_sharding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NamedSharding <nemo_rl.distributed.named_sharding.NamedSharding>`
  - ```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding
    :summary:
    ```
````

### API

`````{py:class} NamedSharding(layout: typing.Sequence[typing.Any] | numpy.ndarray, names: list[str])
:canonical: nemo_rl.distributed.named_sharding.NamedSharding

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.__init__
```

````{py:property} shape
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.shape
:type: dict[str, int]

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.shape
```

````

````{py:property} names
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.names
:type: list[str]

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.names
```

````

````{py:property} ndim
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.ndim
:type: int

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.ndim
```

````

````{py:property} size
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.size
:type: int

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.size
```

````

````{py:property} layout
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.layout
:type: numpy.ndarray[typing.Any, numpy.dtype[numpy.int_]]

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.layout
```

````

````{py:method} get_worker_coords(worker_id: int) -> dict[str, int]
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.get_worker_coords

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.get_worker_coords
```

````

````{py:method} get_ranks_by_coord(**coords: int) -> list[int]
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.get_ranks_by_coord

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.get_ranks_by_coord
```

````

````{py:method} get_ranks(**kwargs: int) -> typing.Union[nemo_rl.distributed.named_sharding.NamedSharding, int]
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.get_ranks

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.get_ranks
```

````

````{py:method} get_axis_index(name: str) -> int
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.get_axis_index

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.get_axis_index
```

````

````{py:method} get_axis_size(name: str) -> int
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.get_axis_size

```{autodoc2-docstring} nemo_rl.distributed.named_sharding.NamedSharding.get_axis_size
```

````

````{py:method} __repr__() -> str
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.__repr__

````

````{py:method} __eq__(other: object) -> bool
:canonical: nemo_rl.distributed.named_sharding.NamedSharding.__eq__

````

`````
