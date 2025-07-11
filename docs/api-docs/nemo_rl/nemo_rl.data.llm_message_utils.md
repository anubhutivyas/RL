# {py:mod}`nemo_rl.data.llm_message_utils`

```{py:module} nemo_rl.data.llm_message_utils
```

```{autodoc2-docstring} nemo_rl.data.llm_message_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`message_log_to_flat_messages <nemo_rl.data.llm_message_utils.message_log_to_flat_messages>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.message_log_to_flat_messages
    :summary:
    ```
* - {py:obj}`get_keys_from_message_log <nemo_rl.data.llm_message_utils.get_keys_from_message_log>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_keys_from_message_log
    :summary:
    ```
* - {py:obj}`add_loss_mask_to_message_log <nemo_rl.data.llm_message_utils.add_loss_mask_to_message_log>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.add_loss_mask_to_message_log
    :summary:
    ```
* - {py:obj}`_pad_tensor <nemo_rl.data.llm_message_utils._pad_tensor>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils._pad_tensor
    :summary:
    ```
* - {py:obj}`_validate_tensor_consistency <nemo_rl.data.llm_message_utils._validate_tensor_consistency>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils._validate_tensor_consistency
    :summary:
    ```
* - {py:obj}`batched_message_log_to_flat_message <nemo_rl.data.llm_message_utils.batched_message_log_to_flat_message>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.batched_message_log_to_flat_message
    :summary:
    ```
* - {py:obj}`message_log_shape <nemo_rl.data.llm_message_utils.message_log_shape>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.message_log_shape
    :summary:
    ```
* - {py:obj}`get_first_index_that_differs <nemo_rl.data.llm_message_utils.get_first_index_that_differs>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_first_index_that_differs
    :summary:
    ```
* - {py:obj}`get_formatted_message_log <nemo_rl.data.llm_message_utils.get_formatted_message_log>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_formatted_message_log
    :summary:
    ```
* - {py:obj}`remap_dataset_keys <nemo_rl.data.llm_message_utils.remap_dataset_keys>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.remap_dataset_keys
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tensor <nemo_rl.data.llm_message_utils.Tensor>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.Tensor
    :summary:
    ```
* - {py:obj}`TokenizerType <nemo_rl.data.llm_message_utils.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.data.llm_message_utils.TokenizerType
    :summary:
    ```
````

### API

````{py:data} Tensor
:canonical: nemo_rl.data.llm_message_utils.Tensor
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.Tensor
```

````

````{py:data} TokenizerType
:canonical: nemo_rl.data.llm_message_utils.TokenizerType
:value: >
   None

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.TokenizerType
```

````

````{py:function} message_log_to_flat_messages(message_log: nemo_rl.data.interfaces.LLMMessageLogType) -> nemo_rl.data.interfaces.FlatMessagesType
:canonical: nemo_rl.data.llm_message_utils.message_log_to_flat_messages

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.message_log_to_flat_messages
```
````

````{py:function} get_keys_from_message_log(message_log: nemo_rl.data.interfaces.LLMMessageLogType, keys: list[str]) -> nemo_rl.data.interfaces.LLMMessageLogType
:canonical: nemo_rl.data.llm_message_utils.get_keys_from_message_log

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_keys_from_message_log
```
````

````{py:function} add_loss_mask_to_message_log(batch_message_log: list[nemo_rl.data.interfaces.LLMMessageLogType], roles_to_train_on: list[str] = ['assistant'], only_unmask_final: bool = False) -> None
:canonical: nemo_rl.data.llm_message_utils.add_loss_mask_to_message_log

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.add_loss_mask_to_message_log
```
````

````{py:function} _pad_tensor(tensor: nemo_rl.data.llm_message_utils.Tensor, max_len: int, pad_side: str, pad_value: int = 0) -> nemo_rl.data.llm_message_utils.Tensor
:canonical: nemo_rl.data.llm_message_utils._pad_tensor

```{autodoc2-docstring} nemo_rl.data.llm_message_utils._pad_tensor
```
````

````{py:function} _validate_tensor_consistency(tensors: list[nemo_rl.data.llm_message_utils.Tensor]) -> None
:canonical: nemo_rl.data.llm_message_utils._validate_tensor_consistency

```{autodoc2-docstring} nemo_rl.data.llm_message_utils._validate_tensor_consistency
```
````

````{py:function} batched_message_log_to_flat_message(message_log_batch: list[nemo_rl.data.interfaces.LLMMessageLogType], pad_value_dict: typing.Optional[dict[str, int]] = None, make_sequence_length_divisible_by: int = 1) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.FlatMessagesType], nemo_rl.data.llm_message_utils.Tensor]
:canonical: nemo_rl.data.llm_message_utils.batched_message_log_to_flat_message

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.batched_message_log_to_flat_message
```
````

````{py:function} message_log_shape(message_log: nemo_rl.data.interfaces.LLMMessageLogType) -> list[dict[str, torch.Size]]
:canonical: nemo_rl.data.llm_message_utils.message_log_shape

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.message_log_shape
```
````

````{py:function} get_first_index_that_differs(str1: str, str2: str) -> int
:canonical: nemo_rl.data.llm_message_utils.get_first_index_that_differs

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_first_index_that_differs
```
````

````{py:function} get_formatted_message_log(message_log: nemo_rl.data.interfaces.LLMMessageLogType, tokenizer: nemo_rl.data.llm_message_utils.TokenizerType, task_data_spec: nemo_rl.data.interfaces.TaskDataSpec, add_bos_token: bool = True, add_eos_token: bool = True, add_generation_prompt: bool = False) -> nemo_rl.data.interfaces.LLMMessageLogType
:canonical: nemo_rl.data.llm_message_utils.get_formatted_message_log

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.get_formatted_message_log
```
````

````{py:function} remap_dataset_keys(dataset: datasets.Dataset, mapping_dict: dict[str, str]) -> datasets.Dataset
:canonical: nemo_rl.data.llm_message_utils.remap_dataset_keys

```{autodoc2-docstring} nemo_rl.data.llm_message_utils.remap_dataset_keys
```
````
