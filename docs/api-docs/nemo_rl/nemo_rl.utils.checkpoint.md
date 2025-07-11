# {py:mod}`nemo_rl.utils.checkpoint`

```{py:module} nemo_rl.utils.checkpoint
```

```{autodoc2-docstring} nemo_rl.utils.checkpoint
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheckpointingConfig <nemo_rl.utils.checkpoint.CheckpointingConfig>`
  - ```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig
    :summary:
    ```
* - {py:obj}`CheckpointManager <nemo_rl.utils.checkpoint.CheckpointManager>`
  - ```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_load_checkpoint_history <nemo_rl.utils.checkpoint._load_checkpoint_history>`
  - ```{autodoc2-docstring} nemo_rl.utils.checkpoint._load_checkpoint_history
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathLike <nemo_rl.utils.checkpoint.PathLike>`
  - ```{autodoc2-docstring} nemo_rl.utils.checkpoint.PathLike
    :summary:
    ```
````

### API

````{py:data} PathLike
:canonical: nemo_rl.utils.checkpoint.PathLike
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.PathLike
```

````

`````{py:class} CheckpointingConfig()
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.__init__
```

````{py:attribute} enabled
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.enabled
```

````

````{py:attribute} checkpoint_dir
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.checkpoint_dir
:type: nemo_rl.utils.checkpoint.PathLike
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.checkpoint_dir
```

````

````{py:attribute} metric_name
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.metric_name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.metric_name
```

````

````{py:attribute} higher_is_better
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.higher_is_better
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.higher_is_better
```

````

````{py:attribute} save_period
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.save_period
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.save_period
```

````

````{py:attribute} keep_top_k
:canonical: nemo_rl.utils.checkpoint.CheckpointingConfig.keep_top_k
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointingConfig.keep_top_k
```

````

`````

`````{py:class} CheckpointManager(config: nemo_rl.utils.checkpoint.CheckpointingConfig)
:canonical: nemo_rl.utils.checkpoint.CheckpointManager

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.__init__
```

````{py:method} init_tmp_checkpoint(step: int, training_info: dict[str, typing.Any], run_config: typing.Optional[dict[str, typing.Any]] = None) -> nemo_rl.utils.checkpoint.PathLike
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.init_tmp_checkpoint

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.init_tmp_checkpoint
```

````

````{py:method} finalize_checkpoint(checkpoint_path: nemo_rl.utils.checkpoint.PathLike) -> None
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.finalize_checkpoint

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.finalize_checkpoint
```

````

````{py:method} remove_old_checkpoints(exclude_latest: bool = True) -> None
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.remove_old_checkpoints

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.remove_old_checkpoints
```

````

````{py:method} get_best_checkpoint_path() -> typing.Optional[str]
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.get_best_checkpoint_path

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.get_best_checkpoint_path
```

````

````{py:method} get_latest_checkpoint_path() -> typing.Optional[str]
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.get_latest_checkpoint_path

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.get_latest_checkpoint_path
```

````

````{py:method} load_training_info(checkpoint_path: typing.Optional[nemo_rl.utils.checkpoint.PathLike] = None) -> typing.Optional[dict[str, typing.Any]]
:canonical: nemo_rl.utils.checkpoint.CheckpointManager.load_training_info

```{autodoc2-docstring} nemo_rl.utils.checkpoint.CheckpointManager.load_training_info
```

````

`````

````{py:function} _load_checkpoint_history(checkpoint_dir: pathlib.Path) -> list[tuple[int, nemo_rl.utils.checkpoint.PathLike, dict[str, typing.Any]]]
:canonical: nemo_rl.utils.checkpoint._load_checkpoint_history

```{autodoc2-docstring} nemo_rl.utils.checkpoint._load_checkpoint_history
```
````
