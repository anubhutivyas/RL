# {py:mod}`nemo_rl.models.dtensor.parallelize`

```{py:module} nemo_rl.models.dtensor.parallelize
```

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RotaryEmbedParallel <nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_parallelize_gemma3 <nemo_rl.models.dtensor.parallelize._parallelize_gemma3>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_gemma3
    :summary:
    ```
* - {py:obj}`_parallelize_llama <nemo_rl.models.dtensor.parallelize._parallelize_llama>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_llama
    :summary:
    ```
* - {py:obj}`_parallelize_qwen <nemo_rl.models.dtensor.parallelize._parallelize_qwen>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_qwen
    :summary:
    ```
* - {py:obj}`translate_parallel_style <nemo_rl.models.dtensor.parallelize.translate_parallel_style>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.translate_parallel_style
    :summary:
    ```
* - {py:obj}`get_hf_tp_plan <nemo_rl.models.dtensor.parallelize.get_hf_tp_plan>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_hf_tp_plan
    :summary:
    ```
* - {py:obj}`_parallelize_model <nemo_rl.models.dtensor.parallelize._parallelize_model>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_model
    :summary:
    ```
* - {py:obj}`to_local_if_dtensor <nemo_rl.models.dtensor.parallelize.to_local_if_dtensor>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.to_local_if_dtensor
    :summary:
    ```
* - {py:obj}`clip_grad_by_total_norm_ <nemo_rl.models.dtensor.parallelize.clip_grad_by_total_norm_>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.clip_grad_by_total_norm_
    :summary:
    ```
* - {py:obj}`get_grad_norm <nemo_rl.models.dtensor.parallelize.get_grad_norm>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_grad_norm
    :summary:
    ```
* - {py:obj}`get_logprobs_from_vocab_parallel_logits <nemo_rl.models.dtensor.parallelize.get_logprobs_from_vocab_parallel_logits>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_logprobs_from_vocab_parallel_logits
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PARALLIZE_FUNCTIONS <nemo_rl.models.dtensor.parallelize.PARALLIZE_FUNCTIONS>`
  - ```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.PARALLIZE_FUNCTIONS
    :summary:
    ```
````

### API

`````{py:class} RotaryEmbedParallel
:canonical: nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel

Bases: {py:obj}`torch.distributed.tensor.parallel.SequenceParallel`

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel
```

````{py:method} _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh)
:canonical: nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel._prepare_input_fn
:staticmethod:

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel._prepare_input_fn
```

````

````{py:method} _prepare_output_fn(use_local_output, mod, outputs, device_mesh)
:canonical: nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel._prepare_output_fn
:staticmethod:

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.RotaryEmbedParallel._prepare_output_fn
```

````

`````

````{py:function} _parallelize_gemma3(model: typing.Union[transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM, transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration], sequence_parallel: bool = False)
:canonical: nemo_rl.models.dtensor.parallelize._parallelize_gemma3

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_gemma3
```
````

````{py:function} _parallelize_llama(model: transformers.models.llama.modeling_llama.LlamaForCausalLM, sequence_parallel: bool = False)
:canonical: nemo_rl.models.dtensor.parallelize._parallelize_llama

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_llama
```
````

````{py:function} _parallelize_qwen(model: typing.Union[transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM, transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM], sequence_parallel: bool = False)
:canonical: nemo_rl.models.dtensor.parallelize._parallelize_qwen

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_qwen
```
````

````{py:data} PARALLIZE_FUNCTIONS
:canonical: nemo_rl.models.dtensor.parallelize.PARALLIZE_FUNCTIONS
:type: dict[type[torch.nn.Module], typing.Callable[..., dict[str, torch.distributed.tensor.parallel.ParallelStyle]]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.PARALLIZE_FUNCTIONS
```

````

````{py:function} translate_parallel_style(style: str)
:canonical: nemo_rl.models.dtensor.parallelize.translate_parallel_style

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.translate_parallel_style
```
````

````{py:function} get_hf_tp_plan(model)
:canonical: nemo_rl.models.dtensor.parallelize.get_hf_tp_plan

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_hf_tp_plan
```
````

````{py:function} _parallelize_model(model: typing.Union[transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM, transformers.models.llama.modeling_llama.LlamaForCausalLM, transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM, transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration], dp_mesh: torch.distributed.device_mesh.DeviceMesh, tp_mesh: torch.distributed.device_mesh.DeviceMesh, param_dtype: torch.dtype, sequence_parallel: bool = False, activation_checkpointing: bool = False, cpu_offload: bool = False, custom_parallel_plan: typing.Optional[typing.Union[dict, str]] = None)
:canonical: nemo_rl.models.dtensor.parallelize._parallelize_model

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize._parallelize_model
```
````

````{py:function} to_local_if_dtensor(tensor: typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]) -> torch.Tensor
:canonical: nemo_rl.models.dtensor.parallelize.to_local_if_dtensor

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.to_local_if_dtensor
```
````

````{py:function} clip_grad_by_total_norm_(parameters: typing.Union[list[typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], max_grad_norm: typing.Union[int, float], total_norm: float, dtype: torch.dtype = torch.float32)
:canonical: nemo_rl.models.dtensor.parallelize.clip_grad_by_total_norm_

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.clip_grad_by_total_norm_
```
````

````{py:function} get_grad_norm(parameters: typing.Union[list[typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], typing.Union[torch.Tensor, torch.distributed.tensor.DTensor]], dp_cp_group: torch.distributed.ProcessGroup, tp_group: torch.distributed.ProcessGroup, norm_type: typing.Union[int, float] = 2, dtype: torch.dtype = torch.float32) -> float
:canonical: nemo_rl.models.dtensor.parallelize.get_grad_norm

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_grad_norm
```
````

````{py:function} get_logprobs_from_vocab_parallel_logits(vocab_parallel_logits: torch.distributed.tensor.DTensor, input_ids: torch.Tensor)
:canonical: nemo_rl.models.dtensor.parallelize.get_logprobs_from_vocab_parallel_logits

```{autodoc2-docstring} nemo_rl.models.dtensor.parallelize.get_logprobs_from_vocab_parallel_logits
```
````
