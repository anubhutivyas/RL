# {py:mod}`nemo_rl.utils.native_checkpoint`

```{py:module} nemo_rl.utils.native_checkpoint
```

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelState <nemo_rl.utils.native_checkpoint.ModelState>`
  - ```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.ModelState
    :summary:
    ```
* - {py:obj}`OptimizerState <nemo_rl.utils.native_checkpoint.OptimizerState>`
  - ```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.OptimizerState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`save_checkpoint <nemo_rl.utils.native_checkpoint.save_checkpoint>`
  - ```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.save_checkpoint
    :summary:
    ```
* - {py:obj}`load_checkpoint <nemo_rl.utils.native_checkpoint.load_checkpoint>`
  - ```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.load_checkpoint
    :summary:
    ```
* - {py:obj}`convert_dcp_to_hf <nemo_rl.utils.native_checkpoint.convert_dcp_to_hf>`
  - ```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.convert_dcp_to_hf
    :summary:
    ```
````

### API

`````{py:class} ModelState(model: torch.nn.Module)
:canonical: nemo_rl.utils.native_checkpoint.ModelState

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.ModelState
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.ModelState.__init__
```

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: nemo_rl.utils.native_checkpoint.ModelState.state_dict

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.ModelState.state_dict
```

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: nemo_rl.utils.native_checkpoint.ModelState.load_state_dict

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.ModelState.load_state_dict
```

````

`````

`````{py:class} OptimizerState(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: typing.Optional[typing.Any] = None)
:canonical: nemo_rl.utils.native_checkpoint.OptimizerState

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.OptimizerState
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.OptimizerState.__init__
```

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: nemo_rl.utils.native_checkpoint.OptimizerState.state_dict

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.OptimizerState.state_dict
```

````

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: nemo_rl.utils.native_checkpoint.OptimizerState.load_state_dict

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.OptimizerState.load_state_dict
```

````

`````

````{py:function} save_checkpoint(model: torch.nn.Module, weights_path: str, optimizer: typing.Optional[torch.optim.Optimizer] = None, scheduler: typing.Optional[typing.Any] = None, optimizer_path: typing.Optional[str] = None, tokenizer: typing.Optional[typing.Any] = None, tokenizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.native_checkpoint.save_checkpoint

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.save_checkpoint
```
````

````{py:function} load_checkpoint(model: torch.nn.Module, weights_path: str, optimizer: typing.Optional[torch.optim.Optimizer] = None, scheduler: typing.Optional[typing.Any] = None, optimizer_path: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.native_checkpoint.load_checkpoint

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.load_checkpoint
```
````

````{py:function} convert_dcp_to_hf(dcp_ckpt_path: str, hf_ckpt_path: str, model_name_or_path: str, tokenizer_name_or_path: str, overwrite: bool = False) -> str
:canonical: nemo_rl.utils.native_checkpoint.convert_dcp_to_hf

```{autodoc2-docstring} nemo_rl.utils.native_checkpoint.convert_dcp_to_hf
```
````
