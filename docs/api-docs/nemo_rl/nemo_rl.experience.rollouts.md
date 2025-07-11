# {py:mod}`nemo_rl.experience.rollouts`

```{py:module} nemo_rl.experience.rollouts
```

```{autodoc2-docstring} nemo_rl.experience.rollouts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_responses <nemo_rl.experience.rollouts.generate_responses>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.generate_responses
    :summary:
    ```
* - {py:obj}`generate_responses_async <nemo_rl.experience.rollouts.generate_responses_async>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.generate_responses_async
    :summary:
    ```
* - {py:obj}`calculate_rewards <nemo_rl.experience.rollouts.calculate_rewards>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.calculate_rewards
    :summary:
    ```
* - {py:obj}`run_multi_turn_rollout <nemo_rl.experience.rollouts.run_multi_turn_rollout>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.run_multi_turn_rollout
    :summary:
    ```
* - {py:obj}`async_generate_response_for_sample_turn <nemo_rl.experience.rollouts.async_generate_response_for_sample_turn>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.async_generate_response_for_sample_turn
    :summary:
    ```
* - {py:obj}`run_sample_multi_turn_rollout <nemo_rl.experience.rollouts.run_sample_multi_turn_rollout>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.run_sample_multi_turn_rollout
    :summary:
    ```
* - {py:obj}`run_async_multi_turn_rollout <nemo_rl.experience.rollouts.run_async_multi_turn_rollout>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.run_async_multi_turn_rollout
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <nemo_rl.experience.rollouts.TokenizerType>`
  - ```{autodoc2-docstring} nemo_rl.experience.rollouts.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: nemo_rl.experience.rollouts.TokenizerType
:value: >
   None

```{autodoc2-docstring} nemo_rl.experience.rollouts.TokenizerType
```

````

````{py:function} generate_responses(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, generation_input_data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], tokenizer: nemo_rl.experience.rollouts.TokenizerType, input_lengths: torch.Tensor, include_logprobs: bool = True, greedy: bool = False) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], list[torch.Tensor], dict[str, float | int]]
:canonical: nemo_rl.experience.rollouts.generate_responses

```{autodoc2-docstring} nemo_rl.experience.rollouts.generate_responses
```
````

````{py:function} generate_responses_async(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, generation_input_data: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.models.generation.interfaces.GenerationDatumSpec], batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], tokenizer: nemo_rl.experience.rollouts.TokenizerType, input_lengths: torch.Tensor, include_logprobs: bool = True, greedy: bool = False) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], list[torch.Tensor], dict[str, float | int]]
:canonical: nemo_rl.experience.rollouts.generate_responses_async
:async:

```{autodoc2-docstring} nemo_rl.experience.rollouts.generate_responses_async
```
````

````{py:function} calculate_rewards(batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface]) -> nemo_rl.environments.interfaces.EnvironmentReturn
:canonical: nemo_rl.experience.rollouts.calculate_rewards

```{autodoc2-docstring} nemo_rl.experience.rollouts.calculate_rewards
```
````

````{py:function} run_multi_turn_rollout(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, input_batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], tokenizer: nemo_rl.experience.rollouts.TokenizerType, task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface], max_seq_len: int, max_rollout_turns: int = 999999, greedy: bool = False) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], dict[str, typing.Any]]
:canonical: nemo_rl.experience.rollouts.run_multi_turn_rollout

```{autodoc2-docstring} nemo_rl.experience.rollouts.run_multi_turn_rollout
```
````

````{py:function} async_generate_response_for_sample_turn(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, sample_message_log: list[dict], sample_stop_strings: list[str] | None, tokenizer: nemo_rl.experience.rollouts.TokenizerType, max_seq_len: int, greedy: bool = False) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, float]]
:canonical: nemo_rl.experience.rollouts.async_generate_response_for_sample_turn
:async:

```{autodoc2-docstring} nemo_rl.experience.rollouts.async_generate_response_for_sample_turn
```
````

````{py:function} run_sample_multi_turn_rollout(sample_idx: int, initial_sample_state: dict, policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, tokenizer: nemo_rl.experience.rollouts.TokenizerType, task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface], max_seq_len: int, max_rollout_turns: int = 999999, greedy: bool = False) -> tuple[dict, dict[str, typing.Any]]
:canonical: nemo_rl.experience.rollouts.run_sample_multi_turn_rollout
:async:

```{autodoc2-docstring} nemo_rl.experience.rollouts.run_sample_multi_turn_rollout
```
````

````{py:function} run_async_multi_turn_rollout(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, input_batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], tokenizer: nemo_rl.experience.rollouts.TokenizerType, task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface], max_seq_len: int, max_rollout_turns: int = 999999, greedy: bool = False) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[nemo_rl.data.interfaces.DatumSpec], dict[str, typing.Any]]
:canonical: nemo_rl.experience.rollouts.run_async_multi_turn_rollout

```{autodoc2-docstring} nemo_rl.experience.rollouts.run_async_multi_turn_rollout
```
````
