# {py:mod}`nemo_rl.models.policy`

```{py:module} nemo_rl.models.policy
```

```{autodoc2-docstring} nemo_rl.models.policy
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

nemo_rl.models.policy.dtensor_policy_worker
nemo_rl.models.policy.fsdp1_policy_worker
nemo_rl.models.policy.interfaces
nemo_rl.models.policy.lm_policy
nemo_rl.models.policy.megatron_policy_worker
nemo_rl.models.policy.utils
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DTensorConfig <nemo_rl.models.policy.DTensorConfig>`
  -
* - {py:obj}`SequencePackingConfig <nemo_rl.models.policy.SequencePackingConfig>`
  -
* - {py:obj}`MegatronOptimizerConfig <nemo_rl.models.policy.MegatronOptimizerConfig>`
  -
* - {py:obj}`MegatronSchedulerConfig <nemo_rl.models.policy.MegatronSchedulerConfig>`
  -
* - {py:obj}`MegatronDDPConfig <nemo_rl.models.policy.MegatronDDPConfig>`
  -
* - {py:obj}`MegatronConfig <nemo_rl.models.policy.MegatronConfig>`
  -
* - {py:obj}`TokenizerConfig <nemo_rl.models.policy.TokenizerConfig>`
  -
* - {py:obj}`PytorchOptimizerConfig <nemo_rl.models.policy.PytorchOptimizerConfig>`
  -
* - {py:obj}`SinglePytorchSchedulerConfig <nemo_rl.models.policy.SinglePytorchSchedulerConfig>`
  -
* - {py:obj}`DynamicBatchingConfig <nemo_rl.models.policy.DynamicBatchingConfig>`
  -
* - {py:obj}`PolicyConfig <nemo_rl.models.policy.PolicyConfig>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SchedulerMilestones <nemo_rl.models.policy.SchedulerMilestones>`
  - ```{autodoc2-docstring} nemo_rl.models.policy.SchedulerMilestones
    :summary:
    ```
````

### API

`````{py:class} DTensorConfig()
:canonical: nemo_rl.models.policy.DTensorConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: nemo_rl.models.policy.DTensorConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.enabled
```

````

````{py:attribute} cpu_offload
:canonical: nemo_rl.models.policy.DTensorConfig.cpu_offload
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.cpu_offload
```

````

````{py:attribute} sequence_parallel
:canonical: nemo_rl.models.policy.DTensorConfig.sequence_parallel
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.sequence_parallel
```

````

````{py:attribute} activation_checkpointing
:canonical: nemo_rl.models.policy.DTensorConfig.activation_checkpointing
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.activation_checkpointing
```

````

````{py:attribute} tensor_parallel_size
:canonical: nemo_rl.models.policy.DTensorConfig.tensor_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.tensor_parallel_size
```

````

````{py:attribute} context_parallel_size
:canonical: nemo_rl.models.policy.DTensorConfig.context_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.context_parallel_size
```

````

````{py:attribute} custom_parallel_plan
:canonical: nemo_rl.models.policy.DTensorConfig.custom_parallel_plan
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DTensorConfig.custom_parallel_plan
```

````

`````

`````{py:class} SequencePackingConfig()
:canonical: nemo_rl.models.policy.SequencePackingConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: nemo_rl.models.policy.SequencePackingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SequencePackingConfig.enabled
```

````

````{py:attribute} train_mb_tokens
:canonical: nemo_rl.models.policy.SequencePackingConfig.train_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SequencePackingConfig.train_mb_tokens
```

````

````{py:attribute} logprob_mb_tokens
:canonical: nemo_rl.models.policy.SequencePackingConfig.logprob_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SequencePackingConfig.logprob_mb_tokens
```

````

````{py:attribute} algorithm
:canonical: nemo_rl.models.policy.SequencePackingConfig.algorithm
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SequencePackingConfig.algorithm
```

````

`````

`````{py:class} MegatronOptimizerConfig()
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} optimizer
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.optimizer
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.optimizer
```

````

````{py:attribute} lr
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.lr
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.lr
```

````

````{py:attribute} min_lr
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.min_lr
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.min_lr
```

````

````{py:attribute} weight_decay
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.weight_decay
```

````

````{py:attribute} bf16
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.bf16
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.bf16
```

````

````{py:attribute} fp16
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.fp16
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.fp16
```

````

````{py:attribute} params_dtype
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.params_dtype
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.params_dtype
```

````

````{py:attribute} adam_beta1
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.adam_beta1
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.adam_beta1
```

````

````{py:attribute} adam_beta2
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.adam_beta2
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.adam_beta2
```

````

````{py:attribute} adam_eps
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.adam_eps
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.adam_eps
```

````

````{py:attribute} sgd_momentum
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.sgd_momentum
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.sgd_momentum
```

````

````{py:attribute} use_distributed_optimizer
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.use_distributed_optimizer
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.use_distributed_optimizer
```

````

````{py:attribute} use_precision_aware_optimizer
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.use_precision_aware_optimizer
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.use_precision_aware_optimizer
```

````

````{py:attribute} clip_grad
:canonical: nemo_rl.models.policy.MegatronOptimizerConfig.clip_grad
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronOptimizerConfig.clip_grad
```

````

`````

`````{py:class} MegatronSchedulerConfig()
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} start_weight_decay
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.start_weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.start_weight_decay
```

````

````{py:attribute} end_weight_decay
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.end_weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.end_weight_decay
```

````

````{py:attribute} weight_decay_incr_style
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.weight_decay_incr_style
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.weight_decay_incr_style
```

````

````{py:attribute} lr_decay_style
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.lr_decay_style
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.lr_decay_style
```

````

````{py:attribute} lr_decay_iters
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.lr_decay_iters
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.lr_decay_iters
```

````

````{py:attribute} lr_warmup_iters
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.lr_warmup_iters
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.lr_warmup_iters
```

````

````{py:attribute} lr_warmup_init
:canonical: nemo_rl.models.policy.MegatronSchedulerConfig.lr_warmup_init
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronSchedulerConfig.lr_warmup_init
```

````

`````

`````{py:class} MegatronDDPConfig()
:canonical: nemo_rl.models.policy.MegatronDDPConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} grad_reduce_in_fp32
:canonical: nemo_rl.models.policy.MegatronDDPConfig.grad_reduce_in_fp32
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.grad_reduce_in_fp32
```

````

````{py:attribute} overlap_grad_reduce
:canonical: nemo_rl.models.policy.MegatronDDPConfig.overlap_grad_reduce
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.overlap_grad_reduce
```

````

````{py:attribute} overlap_param_gather
:canonical: nemo_rl.models.policy.MegatronDDPConfig.overlap_param_gather
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.overlap_param_gather
```

````

````{py:attribute} average_in_collective
:canonical: nemo_rl.models.policy.MegatronDDPConfig.average_in_collective
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.average_in_collective
```

````

````{py:attribute} use_custom_fsdp
:canonical: nemo_rl.models.policy.MegatronDDPConfig.use_custom_fsdp
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.use_custom_fsdp
```

````

````{py:attribute} data_parallel_sharding_strategy
:canonical: nemo_rl.models.policy.MegatronDDPConfig.data_parallel_sharding_strategy
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronDDPConfig.data_parallel_sharding_strategy
```

````

`````

`````{py:class} MegatronConfig()
:canonical: nemo_rl.models.policy.MegatronConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: nemo_rl.models.policy.MegatronConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.enabled
```

````

````{py:attribute} empty_unused_memory_level
:canonical: nemo_rl.models.policy.MegatronConfig.empty_unused_memory_level
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.empty_unused_memory_level
```

````

````{py:attribute} activation_checkpointing
:canonical: nemo_rl.models.policy.MegatronConfig.activation_checkpointing
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.activation_checkpointing
```

````

````{py:attribute} converter_type
:canonical: nemo_rl.models.policy.MegatronConfig.converter_type
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.converter_type
```

````

````{py:attribute} tensor_model_parallel_size
:canonical: nemo_rl.models.policy.MegatronConfig.tensor_model_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.tensor_model_parallel_size
```

````

````{py:attribute} pipeline_model_parallel_size
:canonical: nemo_rl.models.policy.MegatronConfig.pipeline_model_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.pipeline_model_parallel_size
```

````

````{py:attribute} num_layers_in_first_pipeline_stage
:canonical: nemo_rl.models.policy.MegatronConfig.num_layers_in_first_pipeline_stage
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.num_layers_in_first_pipeline_stage
```

````

````{py:attribute} num_layers_in_last_pipeline_stage
:canonical: nemo_rl.models.policy.MegatronConfig.num_layers_in_last_pipeline_stage
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.num_layers_in_last_pipeline_stage
```

````

````{py:attribute} context_parallel_size
:canonical: nemo_rl.models.policy.MegatronConfig.context_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.context_parallel_size
```

````

````{py:attribute} pipeline_dtype
:canonical: nemo_rl.models.policy.MegatronConfig.pipeline_dtype
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.pipeline_dtype
```

````

````{py:attribute} sequence_parallel
:canonical: nemo_rl.models.policy.MegatronConfig.sequence_parallel
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.sequence_parallel
```

````

````{py:attribute} optimizer
:canonical: nemo_rl.models.policy.MegatronConfig.optimizer
:type: typing.NotRequired[nemo_rl.models.policy.MegatronOptimizerConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.optimizer
```

````

````{py:attribute} scheduler
:canonical: nemo_rl.models.policy.MegatronConfig.scheduler
:type: typing.NotRequired[nemo_rl.models.policy.MegatronSchedulerConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.scheduler
```

````

````{py:attribute} distributed_data_parallel_config
:canonical: nemo_rl.models.policy.MegatronConfig.distributed_data_parallel_config
:type: nemo_rl.models.policy.MegatronDDPConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.MegatronConfig.distributed_data_parallel_config
```

````

`````

`````{py:class} TokenizerConfig()
:canonical: nemo_rl.models.policy.TokenizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: nemo_rl.models.policy.TokenizerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.TokenizerConfig.name
```

````

````{py:attribute} chat_template
:canonical: nemo_rl.models.policy.TokenizerConfig.chat_template
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.TokenizerConfig.chat_template
```

````

`````

`````{py:class} PytorchOptimizerConfig()
:canonical: nemo_rl.models.policy.PytorchOptimizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: nemo_rl.models.policy.PytorchOptimizerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PytorchOptimizerConfig.name
```

````

````{py:attribute} kwargs
:canonical: nemo_rl.models.policy.PytorchOptimizerConfig.kwargs
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PytorchOptimizerConfig.kwargs
```

````

`````

`````{py:class} SinglePytorchSchedulerConfig()
:canonical: nemo_rl.models.policy.SinglePytorchSchedulerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: nemo_rl.models.policy.SinglePytorchSchedulerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SinglePytorchSchedulerConfig.name
```

````

````{py:attribute} kwargs
:canonical: nemo_rl.models.policy.SinglePytorchSchedulerConfig.kwargs
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SinglePytorchSchedulerConfig.kwargs
```

````

`````

````{py:data} SchedulerMilestones
:canonical: nemo_rl.models.policy.SchedulerMilestones
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.SchedulerMilestones
```

````

`````{py:class} DynamicBatchingConfig()
:canonical: nemo_rl.models.policy.DynamicBatchingConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: nemo_rl.models.policy.DynamicBatchingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DynamicBatchingConfig.enabled
```

````

````{py:attribute} train_mb_tokens
:canonical: nemo_rl.models.policy.DynamicBatchingConfig.train_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DynamicBatchingConfig.train_mb_tokens
```

````

````{py:attribute} logprob_mb_tokens
:canonical: nemo_rl.models.policy.DynamicBatchingConfig.logprob_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DynamicBatchingConfig.logprob_mb_tokens
```

````

````{py:attribute} sequence_length_round
:canonical: nemo_rl.models.policy.DynamicBatchingConfig.sequence_length_round
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.DynamicBatchingConfig.sequence_length_round
```

````

`````

`````{py:class} PolicyConfig()
:canonical: nemo_rl.models.policy.PolicyConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} model_name
:canonical: nemo_rl.models.policy.PolicyConfig.model_name
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.model_name
```

````

````{py:attribute} tokenizer
:canonical: nemo_rl.models.policy.PolicyConfig.tokenizer
:type: nemo_rl.models.policy.TokenizerConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.tokenizer
```

````

````{py:attribute} train_global_batch_size
:canonical: nemo_rl.models.policy.PolicyConfig.train_global_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.train_global_batch_size
```

````

````{py:attribute} train_micro_batch_size
:canonical: nemo_rl.models.policy.PolicyConfig.train_micro_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.train_micro_batch_size
```

````

````{py:attribute} learning_rate
:canonical: nemo_rl.models.policy.PolicyConfig.learning_rate
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.learning_rate
```

````

````{py:attribute} logprob_batch_size
:canonical: nemo_rl.models.policy.PolicyConfig.logprob_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.logprob_batch_size
```

````

````{py:attribute} generation
:canonical: nemo_rl.models.policy.PolicyConfig.generation
:type: typing.Optional[nemo_rl.models.generation.interfaces.GenerationConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.generation
```

````

````{py:attribute} generation_batch_size
:canonical: nemo_rl.models.policy.PolicyConfig.generation_batch_size
:type: typing.NotRequired[int]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.generation_batch_size
```

````

````{py:attribute} precision
:canonical: nemo_rl.models.policy.PolicyConfig.precision
:type: str
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.precision
```

````

````{py:attribute} dtensor_cfg
:canonical: nemo_rl.models.policy.PolicyConfig.dtensor_cfg
:type: nemo_rl.models.policy.DTensorConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.dtensor_cfg
```

````

````{py:attribute} megatron_cfg
:canonical: nemo_rl.models.policy.PolicyConfig.megatron_cfg
:type: nemo_rl.models.policy.MegatronConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.megatron_cfg
```

````

````{py:attribute} dynamic_batching
:canonical: nemo_rl.models.policy.PolicyConfig.dynamic_batching
:type: nemo_rl.models.policy.DynamicBatchingConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.dynamic_batching
```

````

````{py:attribute} sequence_packing
:canonical: nemo_rl.models.policy.PolicyConfig.sequence_packing
:type: nemo_rl.models.policy.SequencePackingConfig
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.sequence_packing
```

````

````{py:attribute} make_sequence_length_divisible_by
:canonical: nemo_rl.models.policy.PolicyConfig.make_sequence_length_divisible_by
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.make_sequence_length_divisible_by
```

````

````{py:attribute} max_total_sequence_length
:canonical: nemo_rl.models.policy.PolicyConfig.max_total_sequence_length
:type: int
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.max_total_sequence_length
```

````

````{py:attribute} max_grad_norm
:canonical: nemo_rl.models.policy.PolicyConfig.max_grad_norm
:type: typing.Optional[typing.Union[float, int]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.max_grad_norm
```

````

````{py:attribute} fsdp_offload_enabled
:canonical: nemo_rl.models.policy.PolicyConfig.fsdp_offload_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.fsdp_offload_enabled
```

````

````{py:attribute} activation_checkpointing_enabled
:canonical: nemo_rl.models.policy.PolicyConfig.activation_checkpointing_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.activation_checkpointing_enabled
```

````

````{py:attribute} optimizer
:canonical: nemo_rl.models.policy.PolicyConfig.optimizer
:type: typing.NotRequired[nemo_rl.models.policy.PytorchOptimizerConfig]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.optimizer
```

````

````{py:attribute} scheduler
:canonical: nemo_rl.models.policy.PolicyConfig.scheduler
:type: typing.NotRequired[list[nemo_rl.models.policy.SinglePytorchSchedulerConfig] | nemo_rl.models.policy.SchedulerMilestones]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.policy.PolicyConfig.scheduler
```

````

`````
