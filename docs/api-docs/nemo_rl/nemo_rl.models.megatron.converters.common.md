# {py:mod}`nemo_rl.models.megatron.converters.common`

```{py:module} nemo_rl.models.megatron.converters.common
```

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MegatronToHFConverter <nemo_rl.models.megatron.converters.common.MegatronToHFConverter>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_local_layer_num <nemo_rl.models.megatron.converters.common.get_local_layer_num>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_local_layer_num
    :summary:
    ```
* - {py:obj}`get_global_layer_num <nemo_rl.models.megatron.converters.common.get_global_layer_num>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_global_layer_num
    :summary:
    ```
* - {py:obj}`get_global_key_from_local_key <nemo_rl.models.megatron.converters.common.get_global_key_from_local_key>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_global_key_from_local_key
    :summary:
    ```
* - {py:obj}`split_fc1_tp <nemo_rl.models.megatron.converters.common.split_fc1_tp>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_fc1_tp
    :summary:
    ```
* - {py:obj}`split_qkv_gpu <nemo_rl.models.megatron.converters.common.split_qkv_gpu>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_qkv_gpu
    :summary:
    ```
* - {py:obj}`split_qkv_bias_gpu <nemo_rl.models.megatron.converters.common.split_qkv_bias_gpu>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_qkv_bias_gpu
    :summary:
    ```
* - {py:obj}`update_transforms_for_nemorl <nemo_rl.models.megatron.converters.common.update_transforms_for_nemorl>`
  - ```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.update_transforms_for_nemorl
    :summary:
    ```
````

### API

````{py:function} get_local_layer_num(s)
:canonical: nemo_rl.models.megatron.converters.common.get_local_layer_num

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_local_layer_num
```
````

````{py:function} get_global_layer_num(s, cfg)
:canonical: nemo_rl.models.megatron.converters.common.get_global_layer_num

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_global_layer_num
```
````

````{py:function} get_global_key_from_local_key(local_key, model_cfg)
:canonical: nemo_rl.models.megatron.converters.common.get_global_key_from_local_key

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.get_global_key_from_local_key
```
````

````{py:function} split_fc1_tp(ctx: nemo.lightning.io.state.TransformCTX, linear_fc1: torch.Tensor)
:canonical: nemo_rl.models.megatron.converters.common.split_fc1_tp

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_fc1_tp
```
````

````{py:function} split_qkv_gpu(ctx: nemo.lightning.io.state.TransformCTX, linear_qkv: torch.Tensor)
:canonical: nemo_rl.models.megatron.converters.common.split_qkv_gpu

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_qkv_gpu
```
````

````{py:function} split_qkv_bias_gpu(ctx: nemo.lightning.io.state.TransformCTX, qkv_bias: torch.Tensor)
:canonical: nemo_rl.models.megatron.converters.common.split_qkv_bias_gpu

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.split_qkv_bias_gpu
```
````

````{py:function} update_transforms_for_nemorl(export_transforms)
:canonical: nemo_rl.models.megatron.converters.common.update_transforms_for_nemorl

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.update_transforms_for_nemorl
```
````

`````{py:class} MegatronToHFConverter(hf_model_name, megatron_model)
:canonical: nemo_rl.models.megatron.converters.common.MegatronToHFConverter

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter.__init__
```

````{py:method} _get_empty_state_dict(source_keys=None)
:canonical: nemo_rl.models.megatron.converters.common.MegatronToHFConverter._get_empty_state_dict

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter._get_empty_state_dict
```

````

````{py:method} _group(state_dict, key, item, main_state_dict_keys, main_items, exception_state_dict_keys_list, exception_items)
:canonical: nemo_rl.models.megatron.converters.common.MegatronToHFConverter._group

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter._group
```

````

````{py:method} _get_groups(state_dict)
:canonical: nemo_rl.models.megatron.converters.common.MegatronToHFConverter._get_groups

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter._get_groups
```

````

````{py:method} convert(state_dict, megatron_config)
:canonical: nemo_rl.models.megatron.converters.common.MegatronToHFConverter.convert

```{autodoc2-docstring} nemo_rl.models.megatron.converters.common.MegatronToHFConverter.convert
```

````

`````
