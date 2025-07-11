# {py:mod}`nemo_rl.algorithms.interfaces`

```{py:module} nemo_rl.algorithms.interfaces
```

```{autodoc2-docstring} nemo_rl.algorithms.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LossType <nemo_rl.algorithms.interfaces.LossType>`
  -
* - {py:obj}`LossFunction <nemo_rl.algorithms.interfaces.LossFunction>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossFunction
    :summary:
    ```
````

### API

`````{py:class} LossType(*args, **kwds)
:canonical: nemo_rl.algorithms.interfaces.LossType

Bases: {py:obj}`enum.Enum`

````{py:attribute} TOKEN_LEVEL
:canonical: nemo_rl.algorithms.interfaces.LossType.TOKEN_LEVEL
:value: >
   'token_level'

```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossType.TOKEN_LEVEL
```

````

````{py:attribute} SEQUENCE_LEVEL
:canonical: nemo_rl.algorithms.interfaces.LossType.SEQUENCE_LEVEL
:value: >
   'sequence_level'

```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossType.SEQUENCE_LEVEL
```

````

`````

`````{py:class} LossFunction
:canonical: nemo_rl.algorithms.interfaces.LossFunction

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossFunction
```

````{py:attribute} loss_type
:canonical: nemo_rl.algorithms.interfaces.LossFunction.loss_type
:type: nemo_rl.algorithms.interfaces.LossType
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossFunction.loss_type
```

````

````{py:method} __call__(next_token_logits: torch.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict, global_valid_seqs: torch.Tensor, global_valid_toks: torch.Tensor) -> tuple[torch.Tensor, dict[str, typing.Any]]
:canonical: nemo_rl.algorithms.interfaces.LossFunction.__call__

```{autodoc2-docstring} nemo_rl.algorithms.interfaces.LossFunction.__call__
```

````

`````
