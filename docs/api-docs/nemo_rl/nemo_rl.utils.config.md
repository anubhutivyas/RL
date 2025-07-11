# {py:mod}`nemo_rl.utils.config`

```{py:module} nemo_rl.utils.config
```

```{autodoc2-docstring} nemo_rl.utils.config
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`resolve_path <nemo_rl.utils.config.resolve_path>`
  - ```{autodoc2-docstring} nemo_rl.utils.config.resolve_path
    :summary:
    ```
* - {py:obj}`load_config_with_inheritance <nemo_rl.utils.config.load_config_with_inheritance>`
  - ```{autodoc2-docstring} nemo_rl.utils.config.load_config_with_inheritance
    :summary:
    ```
* - {py:obj}`load_config <nemo_rl.utils.config.load_config>`
  - ```{autodoc2-docstring} nemo_rl.utils.config.load_config
    :summary:
    ```
* - {py:obj}`parse_hydra_overrides <nemo_rl.utils.config.parse_hydra_overrides>`
  - ```{autodoc2-docstring} nemo_rl.utils.config.parse_hydra_overrides
    :summary:
    ```
````

### API

````{py:function} resolve_path(base_path: pathlib.Path, path: str) -> pathlib.Path
:canonical: nemo_rl.utils.config.resolve_path

```{autodoc2-docstring} nemo_rl.utils.config.resolve_path
```
````

````{py:function} load_config_with_inheritance(config_path: typing.Union[str, pathlib.Path], base_dir: typing.Optional[typing.Union[str, pathlib.Path]] = None) -> omegaconf.DictConfig
:canonical: nemo_rl.utils.config.load_config_with_inheritance

```{autodoc2-docstring} nemo_rl.utils.config.load_config_with_inheritance
```
````

````{py:function} load_config(config_path: typing.Union[str, pathlib.Path]) -> omegaconf.DictConfig
:canonical: nemo_rl.utils.config.load_config

```{autodoc2-docstring} nemo_rl.utils.config.load_config
```
````

````{py:exception} OverridesError()
:canonical: nemo_rl.utils.config.OverridesError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} nemo_rl.utils.config.OverridesError
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.config.OverridesError.__init__
```

````

````{py:function} parse_hydra_overrides(cfg: omegaconf.DictConfig, overrides: list[str]) -> omegaconf.DictConfig
:canonical: nemo_rl.utils.config.parse_hydra_overrides

```{autodoc2-docstring} nemo_rl.utils.config.parse_hydra_overrides
```
````
