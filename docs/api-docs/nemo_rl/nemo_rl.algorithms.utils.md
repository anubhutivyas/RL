# {py:mod}`nemo_rl.algorithms.utils`

```{py:module} nemo_rl.algorithms.utils
```

```{autodoc2-docstring} nemo_rl.algorithms.utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`calculate_kl_penalty_joschu2020 <nemo_rl.algorithms.utils.calculate_kl_penalty_joschu2020>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.calculate_kl_penalty_joschu2020
    :summary:
    ```
* - {py:obj}`calculate_baseline_and_std_per_prompt <nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt
    :summary:
    ```
* - {py:obj}`surpress_user_warnings <nemo_rl.algorithms.utils.surpress_user_warnings>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.surpress_user_warnings
    :summary:
    ```
* - {py:obj}`masked_mean <nemo_rl.algorithms.utils.masked_mean>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.masked_mean
    :summary:
    ```
* - {py:obj}`set_seed <nemo_rl.algorithms.utils.set_seed>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.set_seed
    :summary:
    ```
* - {py:obj}`get_tokenizer <nemo_rl.algorithms.utils.get_tokenizer>`
  - ```{autodoc2-docstring} nemo_rl.algorithms.utils.get_tokenizer
    :summary:
    ```
````

### API

````{py:function} calculate_kl_penalty_joschu2020(logprobs_policy: torch.Tensor, logprobs_reference: torch.Tensor) -> torch.Tensor
:canonical: nemo_rl.algorithms.utils.calculate_kl_penalty_joschu2020

```{autodoc2-docstring} nemo_rl.algorithms.utils.calculate_kl_penalty_joschu2020
```
````

````{py:function} calculate_baseline_and_std_per_prompt(prompts: torch.Tensor, rewards: torch.Tensor, valid_mask: torch.Tensor, leave_one_out_baseline: bool = True) -> tuple[torch.Tensor, torch.Tensor]
:canonical: nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt

```{autodoc2-docstring} nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt
```
````

````{py:function} surpress_user_warnings(f)
:canonical: nemo_rl.algorithms.utils.surpress_user_warnings

```{autodoc2-docstring} nemo_rl.algorithms.utils.surpress_user_warnings
```
````

````{py:function} masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: typing.Optional[int] = None, global_normalization_factor: typing.Optional[torch.Tensor | float] = None)
:canonical: nemo_rl.algorithms.utils.masked_mean

```{autodoc2-docstring} nemo_rl.algorithms.utils.masked_mean
```
````

````{py:function} set_seed(seed: int) -> None
:canonical: nemo_rl.algorithms.utils.set_seed

```{autodoc2-docstring} nemo_rl.algorithms.utils.set_seed
```
````

````{py:function} get_tokenizer(tokenizer_config: nemo_rl.models.policy.TokenizerConfig) -> transformers.PreTrainedTokenizerBase
:canonical: nemo_rl.algorithms.utils.get_tokenizer

```{autodoc2-docstring} nemo_rl.algorithms.utils.get_tokenizer
```
````
