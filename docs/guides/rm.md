# Reward Model Training in NeMo RL

This document explains how to train reward models (RM) within NeMo RL. Currently, only Bradley-Terry reward models are supported.

## Launch a Training Job

The script, [examples/run_rm.py](../../examples/run_rm.py), is used to train a Bradley-Terry reward model. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch a training job is as follows:

```bash
uv run examples/run_rm.py --config examples/configs/rm.yaml

# Can also add overrides on CLI, like changing the model
uv run examples/run_rm.py --config examples/configs/rm.yaml policy.model_name=Qwen/Qwen2.5-1.5B
```

The YAML config must be specified. It uses the same base template as the SFT config but includes a new `reward_model_type` key that triggers Reward Model training. An example RM config file can be found at [examples/configs/rm.yaml](../../examples/configs/rm.yaml).

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

By default, NeMo RL supports the `HelpSteer3` dataset. This dataset is downloaded from Hugging Face and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

You can also use custom preference datasets by configuring as follows:
```
data:
  dataset_name: "PreferenceData:<Name>:<LocalPath>"
  validation_dataset_name: ["PreferenceData:<Name>:<LocalPath>"]
```

Each custom preference dataset should be a JSON file formatted as:
```
{
    "context": list of dicts, # The input message
    "completions": list of dicts, # The list of completions
        {
            "rank": int, # The rank of the completion (lower rank is preferred)
            "completion": list of dicts, # The completion message
        }
}
```

NeMo RL supports using multiple custom validation preference datasets during RM training.
```
data:
  dataset_name: "PreferenceData:<Name>:<LocalPath>"
  validation_dataset_name: [
    "PreferenceData:<Name>:<LocalPath>",
    "PreferenceData:<Name>:<LocalPath>",
   ]
```