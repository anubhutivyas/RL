# {py:mod}`nemo_rl.algorithms.loss_functions`

```{py:module} nemo_rl.algorithms.loss_functions
```

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClippedPGLossConfig <nemo_rl.algorithms.loss_functions.ClippedPGLossConfig>`
  -
* - {py:obj}`ClippedPGLossDataDict <nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict
    :summary:
    ```
* - {py:obj}`ClippedPGLossFn <nemo_rl.algorithms.loss_functions.ClippedPGLossFn>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossFn
    :summary:
    ```
* - {py:obj}`NLLLoss <nemo_rl.algorithms.loss_functions.NLLLoss>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.NLLLoss
    :summary:
    ```
* - {py:obj}`DPOLossConfig <nemo_rl.algorithms.loss_functions.DPOLossConfig>`
  -
* - {py:obj}`DPOLossDataDict <nemo_rl.algorithms.loss_functions.DPOLossDataDict>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict
    :summary:
    ```
* - {py:obj}`DPOLossFn <nemo_rl.algorithms.loss_functions.DPOLossFn>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossFn
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tensor <nemo_rl.algorithms.loss_functions.Tensor>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.Tensor
    :summary:
    ```
````

### API

````{py:data} Tensor
:canonical: nemo_rl.algorithms.loss_functions.Tensor
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.Tensor
```

````

`````{py:class} ClippedPGLossConfig()
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} reference_policy_kl_penalty
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.reference_policy_kl_penalty
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.reference_policy_kl_penalty
```

````

````{py:attribute} ratio_clip_min
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_min
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_min
```

````

````{py:attribute} ratio_clip_max
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_max
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_max
```

````

````{py:attribute} ratio_clip_c
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_c
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.ratio_clip_c
```

````

````{py:attribute} use_on_policy_kl_approximation
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.use_on_policy_kl_approximation
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.use_on_policy_kl_approximation
```

````

````{py:attribute} use_importance_sampling_correction
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.use_importance_sampling_correction
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.use_importance_sampling_correction
```

````

````{py:attribute} token_level_loss
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.token_level_loss
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossConfig.token_level_loss
```

````

`````

`````{py:class} ClippedPGLossDataDict()
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.__init__
```

````{py:attribute} input_ids
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.input_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.input_ids
```

````

````{py:attribute} advantages
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.advantages
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.advantages
```

````

````{py:attribute} prev_logprobs
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.prev_logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.prev_logprobs
```

````

````{py:attribute} generation_logprobs
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.generation_logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.generation_logprobs
```

````

````{py:attribute} reference_policy_logprobs
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.reference_policy_logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.reference_policy_logprobs
```

````

````{py:attribute} token_mask
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.token_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.token_mask
```

````

````{py:attribute} sample_mask
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.sample_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.sample_mask
```

````

````{py:attribute} __extra__
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.__extra__
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict.__extra__
```

````

`````

`````{py:class} ClippedPGLossFn(cfg: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig)
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossFn

Bases: {py:obj}`nemo_rl.algorithms.interfaces.LossFunction`

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossFn
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossFn.__init__
```

````{py:method} __call__(next_token_logits: nemo_rl.algorithms.loss_functions.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.algorithms.loss_functions.ClippedPGLossDataDict], global_valid_seqs: torch.Tensor, global_valid_toks: torch.Tensor, vocab_parallel_rank: typing.Optional[int] = None, vocab_parallel_group: typing.Optional[torch.distributed.ProcessGroup] = None) -> tuple[torch.Tensor, dict]
:canonical: nemo_rl.algorithms.loss_functions.ClippedPGLossFn.__call__

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.ClippedPGLossFn.__call__
```

````

`````

`````{py:class} NLLLoss
:canonical: nemo_rl.algorithms.loss_functions.NLLLoss

Bases: {py:obj}`nemo_rl.algorithms.interfaces.LossFunction`

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.NLLLoss
```

````{py:attribute} loss_type
:canonical: nemo_rl.algorithms.loss_functions.NLLLoss.loss_type
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.NLLLoss.loss_type
```

````

````{py:method} __call__(next_token_logits: nemo_rl.algorithms.loss_functions.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], global_valid_seqs: nemo_rl.algorithms.loss_functions.Tensor | None, global_valid_toks: nemo_rl.algorithms.loss_functions.Tensor, vocab_parallel_rank: typing.Optional[int] = None, vocab_parallel_group: typing.Optional[torch.distributed.ProcessGroup] = None, dpo_loss: bool = False, dpo_average_log_probs: bool = False) -> tuple[torch.Tensor, dict[str, typing.Any]]
:canonical: nemo_rl.algorithms.loss_functions.NLLLoss.__call__

````

`````

`````{py:class} DPOLossConfig()
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} reference_policy_kl_penalty
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig.reference_policy_kl_penalty
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossConfig.reference_policy_kl_penalty
```

````

````{py:attribute} preference_loss_weight
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig.preference_loss_weight
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossConfig.preference_loss_weight
```

````

````{py:attribute} sft_loss_weight
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig.sft_loss_weight
:type: float
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossConfig.sft_loss_weight
```

````

````{py:attribute} preference_average_log_probs
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig.preference_average_log_probs
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossConfig.preference_average_log_probs
```

````

````{py:attribute} sft_average_log_probs
:canonical: nemo_rl.algorithms.loss_functions.DPOLossConfig.sft_average_log_probs
:type: bool
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossConfig.sft_average_log_probs
```

````

`````

`````{py:class} DPOLossDataDict()
:canonical: nemo_rl.algorithms.loss_functions.DPOLossDataDict

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict.__init__
```

````{py:attribute} input_ids
:canonical: nemo_rl.algorithms.loss_functions.DPOLossDataDict.input_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict.input_ids
```

````

````{py:attribute} reference_policy_logprobs
:canonical: nemo_rl.algorithms.loss_functions.DPOLossDataDict.reference_policy_logprobs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict.reference_policy_logprobs
```

````

````{py:attribute} token_mask
:canonical: nemo_rl.algorithms.loss_functions.DPOLossDataDict.token_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict.token_mask
```

````

````{py:attribute} sample_mask
:canonical: nemo_rl.algorithms.loss_functions.DPOLossDataDict.sample_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossDataDict.sample_mask
```

````

`````

`````{py:class} DPOLossFn(cfg: nemo_rl.algorithms.loss_functions.DPOLossConfig)
:canonical: nemo_rl.algorithms.loss_functions.DPOLossFn

Bases: {py:obj}`nemo_rl.algorithms.interfaces.LossFunction`

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossFn
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossFn.__init__
```

````{py:method} split_output_tensor(tensor: nemo_rl.algorithms.loss_functions.Tensor) -> tuple[nemo_rl.algorithms.loss_functions.Tensor, nemo_rl.algorithms.loss_functions.Tensor]
:canonical: nemo_rl.algorithms.loss_functions.DPOLossFn.split_output_tensor

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossFn.split_output_tensor
```

````

````{py:method} _preference_loss(next_token_logits: nemo_rl.algorithms.loss_functions.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.algorithms.loss_functions.DPOLossDataDict], global_valid_seqs: nemo_rl.algorithms.loss_functions.Tensor, vocab_parallel_rank: typing.Optional[int] = None, vocab_parallel_group: typing.Optional[torch.distributed.ProcessGroup] = None) -> tuple[nemo_rl.algorithms.loss_functions.Tensor, nemo_rl.algorithms.loss_functions.Tensor, nemo_rl.algorithms.loss_functions.Tensor, nemo_rl.algorithms.loss_functions.Tensor]
:canonical: nemo_rl.algorithms.loss_functions.DPOLossFn._preference_loss

```{autodoc2-docstring} nemo_rl.algorithms.loss_functions.DPOLossFn._preference_loss
```

````

````{py:method} __call__(next_token_logits: nemo_rl.algorithms.loss_functions.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.algorithms.loss_functions.DPOLossDataDict], global_valid_seqs: nemo_rl.algorithms.loss_functions.Tensor, global_valid_toks: nemo_rl.algorithms.loss_functions.Tensor | None, vocab_parallel_rank: typing.Optional[int] = None, vocab_parallel_group: typing.Optional[torch.distributed.ProcessGroup] = None) -> tuple[torch.Tensor, dict[str, typing.Any]]
:canonical: nemo_rl.algorithms.loss_functions.DPOLossFn.__call__

````

`````
