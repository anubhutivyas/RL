# {py:mod}`nemo_rl.utils.logger`

```{py:module} nemo_rl.utils.logger
```

```{autodoc2-docstring} nemo_rl.utils.logger
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WandbConfig <nemo_rl.utils.logger.WandbConfig>`
  -
* - {py:obj}`TensorboardConfig <nemo_rl.utils.logger.TensorboardConfig>`
  -
* - {py:obj}`GPUMonitoringConfig <nemo_rl.utils.logger.GPUMonitoringConfig>`
  -
* - {py:obj}`LoggerConfig <nemo_rl.utils.logger.LoggerConfig>`
  -
* - {py:obj}`LoggerInterface <nemo_rl.utils.logger.LoggerInterface>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.LoggerInterface
    :summary:
    ```
* - {py:obj}`TensorboardLogger <nemo_rl.utils.logger.TensorboardLogger>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger
    :summary:
    ```
* - {py:obj}`WandbLogger <nemo_rl.utils.logger.WandbLogger>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger
    :summary:
    ```
* - {py:obj}`GpuMetricSnapshot <nemo_rl.utils.logger.GpuMetricSnapshot>`
  -
* - {py:obj}`RayGpuMonitorLogger <nemo_rl.utils.logger.RayGpuMonitorLogger>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger
    :summary:
    ```
* - {py:obj}`Logger <nemo_rl.utils.logger.Logger>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.Logger
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`flatten_dict <nemo_rl.utils.logger.flatten_dict>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.flatten_dict
    :summary:
    ```
* - {py:obj}`configure_rich_logging <nemo_rl.utils.logger.configure_rich_logging>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.configure_rich_logging
    :summary:
    ```
* - {py:obj}`print_message_log_samples <nemo_rl.utils.logger.print_message_log_samples>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.print_message_log_samples
    :summary:
    ```
* - {py:obj}`get_next_experiment_dir <nemo_rl.utils.logger.get_next_experiment_dir>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger.get_next_experiment_dir
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_rich_logging_configured <nemo_rl.utils.logger._rich_logging_configured>`
  - ```{autodoc2-docstring} nemo_rl.utils.logger._rich_logging_configured
    :summary:
    ```
````

### API

````{py:data} _rich_logging_configured
:canonical: nemo_rl.utils.logger._rich_logging_configured
:value: >
   False

```{autodoc2-docstring} nemo_rl.utils.logger._rich_logging_configured
```

````

`````{py:class} WandbConfig()
:canonical: nemo_rl.utils.logger.WandbConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} project
:canonical: nemo_rl.utils.logger.WandbConfig.project
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.WandbConfig.project
```

````

````{py:attribute} name
:canonical: nemo_rl.utils.logger.WandbConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.WandbConfig.name
```

````

`````

`````{py:class} TensorboardConfig()
:canonical: nemo_rl.utils.logger.TensorboardConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} log_dir
:canonical: nemo_rl.utils.logger.TensorboardConfig.log_dir
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardConfig.log_dir
```

````

`````

`````{py:class} GPUMonitoringConfig()
:canonical: nemo_rl.utils.logger.GPUMonitoringConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} collection_interval
:canonical: nemo_rl.utils.logger.GPUMonitoringConfig.collection_interval
:type: int | float
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.GPUMonitoringConfig.collection_interval
```

````

````{py:attribute} flush_interval
:canonical: nemo_rl.utils.logger.GPUMonitoringConfig.flush_interval
:type: int | float
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.GPUMonitoringConfig.flush_interval
```

````

`````

`````{py:class} LoggerConfig()
:canonical: nemo_rl.utils.logger.LoggerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} log_dir
:canonical: nemo_rl.utils.logger.LoggerConfig.log_dir
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.log_dir
```

````

````{py:attribute} wandb_enabled
:canonical: nemo_rl.utils.logger.LoggerConfig.wandb_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.wandb_enabled
```

````

````{py:attribute} tensorboard_enabled
:canonical: nemo_rl.utils.logger.LoggerConfig.tensorboard_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.tensorboard_enabled
```

````

````{py:attribute} wandb
:canonical: nemo_rl.utils.logger.LoggerConfig.wandb
:type: nemo_rl.utils.logger.WandbConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.wandb
```

````

````{py:attribute} tensorboard
:canonical: nemo_rl.utils.logger.LoggerConfig.tensorboard
:type: nemo_rl.utils.logger.TensorboardConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.tensorboard
```

````

````{py:attribute} monitor_gpus
:canonical: nemo_rl.utils.logger.LoggerConfig.monitor_gpus
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.monitor_gpus
```

````

````{py:attribute} gpu_monitoring
:canonical: nemo_rl.utils.logger.LoggerConfig.gpu_monitoring
:type: nemo_rl.utils.logger.GPUMonitoringConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerConfig.gpu_monitoring
```

````

`````

`````{py:class} LoggerInterface
:canonical: nemo_rl.utils.logger.LoggerInterface

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerInterface
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.logger.LoggerInterface.log_metrics
:abstractmethod:

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerInterface.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: nemo_rl.utils.logger.LoggerInterface.log_hyperparams
:abstractmethod:

```{autodoc2-docstring} nemo_rl.utils.logger.LoggerInterface.log_hyperparams
```

````

`````

`````{py:class} TensorboardLogger(cfg: nemo_rl.utils.logger.TensorboardConfig, log_dir: typing.Optional[str] = None)
:canonical: nemo_rl.utils.logger.TensorboardLogger

Bases: {py:obj}`nemo_rl.utils.logger.LoggerInterface`

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger.__init__
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.logger.TensorboardLogger.log_metrics

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: nemo_rl.utils.logger.TensorboardLogger.log_hyperparams

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger.log_hyperparams
```

````

````{py:method} log_plot(figure: matplotlib.pyplot.Figure, step: int, name: str) -> None
:canonical: nemo_rl.utils.logger.TensorboardLogger.log_plot

```{autodoc2-docstring} nemo_rl.utils.logger.TensorboardLogger.log_plot
```

````

`````

`````{py:class} WandbLogger(cfg: nemo_rl.utils.logger.WandbConfig, log_dir: typing.Optional[str] = None)
:canonical: nemo_rl.utils.logger.WandbLogger

Bases: {py:obj}`nemo_rl.utils.logger.LoggerInterface`

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger.__init__
```

````{py:method} _log_diffs()
:canonical: nemo_rl.utils.logger.WandbLogger._log_diffs

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger._log_diffs
```

````

````{py:method} _log_code()
:canonical: nemo_rl.utils.logger.WandbLogger._log_code

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger._log_code
```

````

````{py:method} define_metric(name: str, step_metric: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.logger.WandbLogger.define_metric

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger.define_metric
```

````

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.logger.WandbLogger.log_metrics

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: nemo_rl.utils.logger.WandbLogger.log_hyperparams

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger.log_hyperparams
```

````

````{py:method} log_plot(figure: matplotlib.pyplot.Figure, step: int, name: str) -> None
:canonical: nemo_rl.utils.logger.WandbLogger.log_plot

```{autodoc2-docstring} nemo_rl.utils.logger.WandbLogger.log_plot
```

````

`````

`````{py:class} GpuMetricSnapshot()
:canonical: nemo_rl.utils.logger.GpuMetricSnapshot

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} step
:canonical: nemo_rl.utils.logger.GpuMetricSnapshot.step
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.GpuMetricSnapshot.step
```

````

````{py:attribute} metrics
:canonical: nemo_rl.utils.logger.GpuMetricSnapshot.metrics
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.utils.logger.GpuMetricSnapshot.metrics
```

````

`````

`````{py:class} RayGpuMonitorLogger(collection_interval: int | float, flush_interval: int | float, metric_prefix: str, step_metric: str, parent_logger: typing.Optional[nemo_rl.utils.logger.Logger] = None)
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger.__init__
```

````{py:method} start() -> None
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger.start

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger.start
```

````

````{py:method} stop() -> None
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger.stop

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger.stop
```

````

````{py:method} _collection_loop() -> None
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._collection_loop

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._collection_loop
```

````

````{py:method} _parse_metric(sample: prometheus_client.samples.Sample, node_idx: int) -> dict[str, typing.Any]
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._parse_metric

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._parse_metric
```

````

````{py:method} _parse_gpu_sku(sample: prometheus_client.samples.Sample, node_idx: int) -> dict[str, str]
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._parse_gpu_sku

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._parse_gpu_sku
```

````

````{py:method} _collect_gpu_sku() -> dict[str, str]
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._collect_gpu_sku

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._collect_gpu_sku
```

````

````{py:method} _collect_metrics() -> dict[str, typing.Any]
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._collect_metrics

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._collect_metrics
```

````

````{py:method} _collect(metrics: bool = False, sku: bool = False) -> dict[str, typing.Any]
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._collect

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._collect
```

````

````{py:method} _fetch_and_parse_metrics(node_idx: int, metric_address: str, parser_fn: typing.Callable)
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger._fetch_and_parse_metrics

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger._fetch_and_parse_metrics
```

````

````{py:method} flush() -> None
:canonical: nemo_rl.utils.logger.RayGpuMonitorLogger.flush

```{autodoc2-docstring} nemo_rl.utils.logger.RayGpuMonitorLogger.flush
```

````

`````

`````{py:class} Logger(cfg: nemo_rl.utils.logger.LoggerConfig)
:canonical: nemo_rl.utils.logger.Logger

Bases: {py:obj}`nemo_rl.utils.logger.LoggerInterface`

```{autodoc2-docstring} nemo_rl.utils.logger.Logger
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.__init__
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: nemo_rl.utils.logger.Logger.log_metrics

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: nemo_rl.utils.logger.Logger.log_hyperparams

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.log_hyperparams
```

````

````{py:method} log_batched_dict_as_jsonl(to_log: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any] | dict[str, typing.Any], filename: str) -> None
:canonical: nemo_rl.utils.logger.Logger.log_batched_dict_as_jsonl

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.log_batched_dict_as_jsonl
```

````

````{py:method} log_plot_token_mult_prob_error(data: dict[str, typing.Any], step: int, name: str) -> None
:canonical: nemo_rl.utils.logger.Logger.log_plot_token_mult_prob_error

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.log_plot_token_mult_prob_error
```

````

````{py:method} __del__() -> None
:canonical: nemo_rl.utils.logger.Logger.__del__

```{autodoc2-docstring} nemo_rl.utils.logger.Logger.__del__
```

````

`````

````{py:function} flatten_dict(d: typing.Mapping[str, typing.Any], sep: str = '.') -> dict[str, typing.Any]
:canonical: nemo_rl.utils.logger.flatten_dict

```{autodoc2-docstring} nemo_rl.utils.logger.flatten_dict
```
````

````{py:function} configure_rich_logging(level: str = 'INFO', show_time: bool = True, show_path: bool = True) -> None
:canonical: nemo_rl.utils.logger.configure_rich_logging

```{autodoc2-docstring} nemo_rl.utils.logger.configure_rich_logging
```
````

````{py:function} print_message_log_samples(message_logs: list[nemo_rl.data.interfaces.LLMMessageLogType], rewards: list[float], num_samples: int = 5, step: int = 0) -> None
:canonical: nemo_rl.utils.logger.print_message_log_samples

```{autodoc2-docstring} nemo_rl.utils.logger.print_message_log_samples
```
````

````{py:function} get_next_experiment_dir(base_log_dir: str) -> str
:canonical: nemo_rl.utils.logger.get_next_experiment_dir

```{autodoc2-docstring} nemo_rl.utils.logger.get_next_experiment_dir
```
````
