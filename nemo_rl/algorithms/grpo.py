# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict, List

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, calculate_math_majority_at_k
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer

# ===============================================================================
# Configuration
# ===============================================================================


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    checkpoint_dir: str
    num_epochs: int
    max_rollout_turns: int
    max_val_samples: int
    # Aggregation stage configuration
    enable_aggregation: bool  # Whether to enable aggregation stage training
    aggregation_prompt_template: str  # Template for aggregation prompts
    first_stage_generation_max_seq_len: int  # Max new tokens for first stage generation (controls generation length, not total sequence length)


class GRPOSaveState(TypedDict):
    step: int
    optim_step: int
    val_reward: float
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "optim_step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env_configs: Dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> Tuple[
    PolicyInterface,
    GenerationInterface,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, logger, master_config, val_dataloader
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    data_config = master_config["data"]
    grpo_config = master_config["grpo"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # config validation checks
    if master_config["checkpointing"]["enabled"]:
        assert master_config["checkpointing"]["save_period"] > 0
        assert (
            master_config["checkpointing"]["save_period"]
            % master_config["grpo"]["val_period"]
            == 0
        ), (
            f"Checkpointing save period {master_config['checkpointing']['save_period']} "
            f"must be a multiple of validation period {master_config['grpo']['val_period']}"
            f", or we won't know what metric to save!"
        )

    # ==========================
    #           Data
    # ==========================
    shuffle_train = master_config["data"]["train"]["shuffle"]
    shuffle_val = master_config["data"]["val"]["shuffle"]

    train_data_generator = None
    val_data_generator = None

    if shuffle_train:
        train_data_generator = torch.Generator()
        train_data_generator.manual_seed(master_config["data"]["train"]["seed"])

    if shuffle_val:
        val_data_generator = torch.Generator()
        val_data_generator.manual_seed(master_config["data"]["val"]["seed"])

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=shuffle_train,
        generator=train_data_generator,
        collate_fn=rl_collate_fn,
        drop_last=master_config["data"]["train"]["drop_last"],
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_dataloader = None
    # If validation is enabled, load the validation dataloader
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=shuffle_val,
            collate_fn=rl_collate_fn,
            generator=val_data_generator,
            drop_last=master_config["data"]["val"]["drop_last"],
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["backend"] != "hf"
    cluster = RayVirtualCluster(
        name="grpo_policy_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2 if colocated_inference else 1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "hf":
        policy_generation = None
        print(f"  ✓ Using HF backend for generation with {policy_config['model_name']}")
    elif backend == "vllm":
        policy_generation = VllmGeneration(cluster=cluster, config=generation_config)
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    policy = HfPolicy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
    )

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


def get_reasoning_split_word(env_configs: Dict[str, Any]) -> Optional[str]:
    """Get reasoning_split_word from any enabled environment."""
    for env_name, env_config in env_configs.items():
        if env_config.get("enable", False) and "reasoning_split_word" in env_config:
            return env_config["reasoning_split_word"]
    return None


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def create_aggregation_prompts(
    original_batch: BatchedDataDict[DatumSpec],
    generation_responses: List[List[str]],
    aggregation_prompt_template: str,
    tokenizer,
) -> BatchedDataDict[DatumSpec]:
    """Create aggregation prompts by combining original prompts with generation responses.
    
    Args:
        original_batch: Original batch containing the prompts
        generation_responses: List of lists, where each inner list contains the generation responses for a prompt
        aggregation_prompt_template: Template string for formatting aggregation prompts
        tokenizer: Tokenizer to use for tokenizing the aggregation prompts
    
    Returns:
        New batch with aggregation prompts
    """
    aggregation_batch = deepcopy(original_batch)
    
    # Create new message logs for aggregation
    new_message_logs = []
    new_extra_env_info = []
    new_loss_multiplier = []
    new_task_name = []
    new_dataset = []
    valid_indices = []
    
    for i, message_log in enumerate(original_batch["message_log"]):
        # Skip if no responses for this prompt
        if len(generation_responses[i]) == 0:
            print(f"DEBUG: Skipping prompt {i} - no responses available")
            continue
            
        # Extract the original user question from metadata
        if "extra_env_info" not in original_batch or not original_batch["extra_env_info"][i]:
            raise ValueError(f"No extra_env_info found in batch for sample {i}")
        
        original_prompt = original_batch["extra_env_info"][i].get("question")
        if original_prompt is None:
            raise ValueError(f"No question found in metadata for sample {i}")
        
        # Format the generation responses
        responses_text = ""
        for j, response in enumerate(generation_responses[i]):
            responses_text += f"Solution {j+1}:\n{response}\n"
        
        # Create the aggregation prompt using the template
        aggregation_prompt = aggregation_prompt_template.format(
            original_prompt=original_prompt,
            responses=responses_text.strip()
        )
        
        # Debug: Print aggregation prompt details
        # print(f"DEBUG: Sample {i} - Using {len(generation_responses[i])} responses")
        # print(f"DEBUG: Original prompt: {original_prompt}...")
        print(f"DEBUG: Aggregation prompt: {aggregation_prompt}")
        
        # Create a proper message structure for chat template (like in run_grpo.py) 
        # Caveat: no system prompt is added here
        aggregation_message = [
            {
                "role": "user",
                "content": aggregation_prompt,
            }
        ]
        
        # Apply chat template to get properly formatted content and token_ids
        formatted_content = tokenizer.apply_chat_template(
            aggregation_message,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        token_ids = tokenizer.apply_chat_template(
            aggregation_message,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        
        # Create new message log with the aggregation prompt
        new_message_log = [
            {
                "role": "user",
                "content": formatted_content,
                "token_ids": token_ids,
            }
        ]
        new_message_logs.append(new_message_log)
    
        # Keep track of corresponding metadata for valid prompts
        new_extra_env_info.append(original_batch["extra_env_info"][i])
        new_loss_multiplier.append(original_batch["loss_multiplier"][i])
        new_task_name.append(original_batch["task_name"][i])
        # TODO: add back dataset
        # new_dataset.append(original_batch["dataset"][i])
        valid_indices.append(i)
    
    # Ensure we have at least some valid aggregation prompts
    if len(new_message_logs) == 0:
        raise ValueError("No valid aggregation prompts could be created from the generation responses")
    
    print(f"DEBUG: Created {len(new_message_logs)} aggregation prompts out of {len(original_batch['message_log'])} original prompts")
    
    # Update aggregation batch with filtered data
    aggregation_batch["message_log"] = new_message_logs
    aggregation_batch["extra_env_info"] = new_extra_env_info
    aggregation_batch["loss_multiplier"] = torch.tensor(new_loss_multiplier)
    aggregation_batch["task_name"] = new_task_name
    # TODO: add back dataset
    # aggregation_batch["dataset"] = new_dataset
    
    return aggregation_batch


def combine_training_data(
    generation_train_data: BatchedDataDict[ClippedPGLossDataDict],
    aggregation_train_data: BatchedDataDict[ClippedPGLossDataDict],
) -> BatchedDataDict[ClippedPGLossDataDict]:
    """Combine training data from both generation and aggregation stages.
    
    Note: Aggregation is always longer than generation, so we pad generation data to match.
    The combined data is shuffled to mix generation and aggregation samples.
    """
    
    # Get sequence lengths (aggregation is always longer)
    gen_seq_len = generation_train_data["input_ids"].shape[1]
    agg_seq_len = aggregation_train_data["input_ids"].shape[1]
    
    # Pad generation data to match aggregation length
    pad_size = agg_seq_len - gen_seq_len
    generation_train_data["input_ids"] = torch.nn.functional.pad(
        generation_train_data["input_ids"], (0, pad_size), value=0
    )
    generation_train_data["advantages"] = torch.nn.functional.pad(
        generation_train_data["advantages"], (0, pad_size), value=0
    )
    generation_train_data["generation_logprobs"] = torch.nn.functional.pad(
        generation_train_data["generation_logprobs"], (0, pad_size), value=0
    )
    generation_train_data["token_mask"] = torch.nn.functional.pad(
        generation_train_data["token_mask"], (0, pad_size), value=0
    )
    
    # Now concatenate the tensors (both have same sequence length)
    combined_data = BatchedDataDict[ClippedPGLossDataDict]({
        "input_ids": torch.cat([generation_train_data["input_ids"], aggregation_train_data["input_ids"]], dim=0),
        "input_lengths": torch.cat([generation_train_data["input_lengths"], aggregation_train_data["input_lengths"]], dim=0),
        "advantages": torch.cat([generation_train_data["advantages"], aggregation_train_data["advantages"]], dim=0),
        "generation_logprobs": torch.cat([generation_train_data["generation_logprobs"], aggregation_train_data["generation_logprobs"]], dim=0),
        "token_mask": torch.cat([generation_train_data["token_mask"], aggregation_train_data["token_mask"]], dim=0),
        "sample_mask": torch.cat([generation_train_data["sample_mask"], aggregation_train_data["sample_mask"]], dim=0),
    })
    
    # Shuffle the combined data to mix generation and aggregation samples
    total_samples = combined_data["input_ids"].shape[0]
    shuffle_indices = torch.randperm(total_samples)
    
    # Apply shuffle to all tensors
    combined_data["input_ids"] = combined_data["input_ids"][shuffle_indices]
    combined_data["input_lengths"] = combined_data["input_lengths"][shuffle_indices]
    combined_data["advantages"] = combined_data["advantages"][shuffle_indices]
    combined_data["generation_logprobs"] = combined_data["generation_logprobs"][shuffle_indices]
    combined_data["token_mask"] = combined_data["token_mask"][shuffle_indices]
    combined_data["sample_mask"] = combined_data["sample_mask"][shuffle_indices]
    
    return combined_data


def refit_policy_generation(
    policy: PolicyInterface,
    policy_generation: GenerationInterface,
    refit_buffer_size_gb: int,  # GB
):
    """Refit the policy generation interface with the latest policy weights."""
    policy.offload_before_refit()
    policy_generation.prepare_for_generation(tags=["weights"])
    # Streaming update weights to save memory
    state_dict_info = policy.prepare_weights_for_ipc()
    # group keys to save time
    available_bytes = refit_buffer_size_gb * (1024**3)
    split_keys, keys = [], []
    for key, size_in_bytes in state_dict_info:
        if size_in_bytes > available_bytes:
            if keys:
                split_keys.append(keys)
                keys = []
            available_bytes = refit_buffer_size_gb * (1024**3)

        keys.append(key)
        available_bytes -= size_in_bytes

    if len(keys) > 0:
        split_keys.append(keys)
    # do update
    for keys in split_keys:
        ipc_handles = policy.get_weights_ipc_handles(keys)
        if not policy_generation.update_weights(ipc_handles):
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                "This often indicates an issue with cuda-ipc or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)
    policy.offload_after_refit()
    policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def grpo_train(
    policy: PolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: Dict[str, EnvironmentInterface],
    val_task_to_env: Optional[Dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: Optional[GRPOSaveState],
    master_config: MasterConfig,
):
    """Run GRPO training algorithm."""
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (hf framework backend)
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running

    # common config/state itmes
    step = grpo_save_state["step"]
    optim_step = grpo_save_state["optim_step"]

    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    refit_buffer_size_gb = master_config["policy"]["refit_buffer_size_gb"]

    num_epochs = master_config["grpo"]["num_epochs"]
    max_num_steps = num_epochs * len(dataloader)

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, refit_buffer_size_gb)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run grpo training (single-turn)
    batch: BatchedDataDict[DatumSpec]
    # for batch in dataloader:

    iter_dataloader = iter(dataloader)

    while step < max_num_steps:
        try:
            batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            batch = next(iter_dataloader)

        print(f"\n{'=' * 25} Step {step + 1}/{max_num_steps} {'=' * 25}")
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                    # First stage uses first_stage_generation_max_seq_len as max_new_tokens
                    max_new_tokens=master_config["grpo"]["first_stage_generation_max_seq_len"],
                )
                
                # Keep generation active for potential aggregation stage
                if not master_config["grpo"].get("enable_aggregation", False):
                    policy_generation.finish_generation()

            # Initialize aggregation stage variables
            aggregation_batch = None
            aggregation_rollout_metrics = {}
            
            # Aggregation stage training logic
            if master_config["grpo"].get("enable_aggregation", False):
                print("▶ Preparing Aggregation stage training...")
                with timer.time("aggregation_preparation"):
                    # Extract generation responses grouped by original prompt
                    num_prompts = len(batch["message_log"])
                    num_generations = master_config["grpo"]["num_generations_per_prompt"]
                    
                    # Get reasoning split word from any enabled environment
                    reasoning_split_word = get_reasoning_split_word(master_config["env"])
                    
                    # Group responses by original prompt
                    generation_responses = []
                    for i in range(num_prompts):
                        prompt_responses = []
                        for j in range(num_generations):
                            idx = i * num_generations + j
                            # Extract assistant response from the message log - take the last assistant turn
                            last_assistant_response = None
                            for message in repeated_batch["message_log"][idx]:
                                if message["role"] == "assistant":
                                    last_assistant_response = message["content"]
                            if last_assistant_response is not None:
                                if reasoning_split_word and reasoning_split_word in last_assistant_response:
                                    prompt_responses.append(last_assistant_response.split(reasoning_split_word)[-1].lstrip())
                                else:
                                    prompt_responses.append("None")
                            else:
                                raise ValueError(f"No assistant response found for prompt {i} generation {j} which is {repeated_batch['message_log'][idx]}")
                        
                        # Randomly select a subset of responses instead of using all
                        # Choose a random number of responses to select (between 1 and total available)
                        num_to_select = random.randint(1, len(prompt_responses))
                        # Randomly sample that many responses
                        selected_responses = random.sample(prompt_responses, num_to_select)
                        generation_responses.append(selected_responses)
                    
                    # print(f"▶ Creating aggregation prompts for {generation_responses[0]}")   
                    
                    # Create aggregation prompts
                    aggregation_batch_template = create_aggregation_prompts(
                        batch, 
                        generation_responses, 
                        master_config["grpo"]["aggregation_prompt_template"],
                        tokenizer,
                    )
                    # print(f"▶ Aggregation batch template: {aggregation_batch_template}")
                    # Repeat aggregation batch for multiple generations
                    aggregation_repeated_batch = aggregation_batch_template.repeat_interleave(
                        master_config["grpo"]["num_generations_per_prompt"]
                    )
                    
                    # Calculate input_ids for aggregation BEFORE rollout (for proper baseline calculation)
                    # This ensures all samples with same prompt are grouped together
                    aggregation_flat_pre_rollout, aggregation_input_lengths = batched_message_log_to_flat_message(
                        aggregation_repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    aggregation_input_ids = aggregation_flat_pre_rollout["token_ids"]

                print(f"▶ Generating Aggregation responses for batch of size {aggregation_repeated_batch.size}...")
                with timer.time("aggregation_generation"):
                    aggregation_batch, aggregation_rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=aggregation_repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                        # Aggregation stage uses default max_new_tokens from policy config
                    )
                    
                    # Debug: Print first aggregation prompt and responses
                    print("\n=== DEBUG: First Aggregation Prompt - All Responses ===")
                    if len(aggregation_batch["message_log"]) > 0:
                        # Show all responses for the first aggregation prompt
                        num_generations = master_config["grpo"]["num_generations_per_prompt"]
                        num_to_show = min(num_generations, len(aggregation_batch["message_log"]))
                        
                        print(f"Showing all {num_to_show} responses for first aggregation prompt:")
                        for i in range(num_to_show):
                            message_log = aggregation_batch["message_log"][i]
                            
                            # Find the last assistant response
                            last_assistant_response = None
                            for msg in message_log:
                                if msg["role"] == "assistant":
                                    last_assistant_response = msg["content"]
                            
                            reward = aggregation_batch['total_reward'][i]
                            print(f"Response {i+1}: {last_assistant_response}")
                            print(f"Reward {i+1}: {reward}")
                            print("---")
                        
                        print(f"All aggregation rewards for first prompt: {aggregation_batch['total_reward'][:num_to_show]}")
                    print("=== END DEBUG ===\n")
                    
                policy_generation.finish_generation()

                # Update metrics with aggregation info
                rollout_metrics.update({
                    "generation_" + k: v for k, v in rollout_metrics.items()
                })
                rollout_metrics.update({
                    "aggregation_" + k: v for k, v in aggregation_rollout_metrics.items()
                })

            # Calculate rewards & advantages
            print("▶ Processing rewards...")
            with timer.time("reward_calculation"):
                # Extract rewards from original generation
                rewards = repeated_batch["total_reward"]
                
                # For aggregation training, also get aggregation rewards
                if master_config["grpo"].get("enable_aggregation", False):
                    aggregation_rewards = aggregation_batch["total_reward"]

                print("▶ Computing advantages...")
                baseline, std, more_rollout_metrics = (
                    calculate_baseline_and_std_per_prompt(
                        input_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=master_config["grpo"][
                            "use_leave_one_out_baseline"
                        ],
                    )
                )
                advantages = (rewards - baseline).unsqueeze(-1)

                if master_config["grpo"].get("enable_aggregation", False):
                    # aggregation_input_ids was already calculated before the rollout above
                    
                    aggregation_baseline, aggregation_std, aggregation_more_rollout_metrics = (
                        calculate_baseline_and_std_per_prompt(
                            aggregation_input_ids,
                            aggregation_rewards,
                            torch.ones_like(aggregation_rewards),
                            leave_one_out_baseline=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                        )
                    )
                    
                    aggregation_advantages = (aggregation_rewards - aggregation_baseline).unsqueeze(-1)
                    
                    # Combine rewards and advantages for metrics
                    all_rewards = torch.cat([rewards, aggregation_rewards])
                    all_advantages = torch.cat([advantages.flatten(), aggregation_advantages.flatten()])
                    
                    rollout_metrics.update(more_rollout_metrics)
                    rollout_metrics.update({
                        "aggregation_" + k: v for k, v in aggregation_more_rollout_metrics.items()
                    })
                else:
                    # Single-stage training
                    print("▶ Computing advantages...")
                    baseline, std, more_rollout_metrics = (
                        calculate_baseline_and_std_per_prompt(
                            input_ids,
                            rewards,
                            torch.ones_like(rewards),
                            leave_one_out_baseline=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                        )
                    )
                    advantages = (rewards - baseline).unsqueeze(-1)
                    all_rewards = rewards
                    all_advantages = advantages.flatten()
                    rollout_metrics.update(more_rollout_metrics)

                # Normalize rewards if configured
                if master_config["grpo"]["normalize_rewards"]:
                    # Generation normalization
                    zero_std_mask = std > 0
                    advantages[zero_std_mask] = (
                        advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
                    )

                    # Aggregation normalization (if applicable)
                    if master_config["grpo"].get("enable_aggregation", False):
                        zero_std_mask = aggregation_std > 0
                        aggregation_advantages[zero_std_mask] = (
                            aggregation_advantages[zero_std_mask] / aggregation_std.unsqueeze(-1)[zero_std_mask]
                        )

                # Calculate metrics
                if master_config["grpo"].get("enable_aggregation", False):
                    # Calculate separate generation metrics
                    generation_advantages_flat = advantages.flatten()
                    generation_percent_valid_advantages = (
                        generation_advantages_flat.count_nonzero() / generation_advantages_flat.numel()
                    )
                    generation_percent_zero_advantages = 1 - generation_percent_valid_advantages
                    
                    generation_advantages_min, generation_advantages_mean, generation_advantages_max = (
                        generation_advantages_flat.min(),
                        generation_advantages_flat.mean(),
                        generation_advantages_flat.max(),
                    )
                    generation_reward_min, generation_reward_mean, generation_reward_max = (
                        rewards.min(),
                        rewards.mean(),
                        rewards.max(),
                    )
                    
                    # Calculate separate aggregation metrics
                    aggregation_advantages_flat = aggregation_advantages.flatten()
                    aggregation_percent_valid_advantages = (
                        aggregation_advantages_flat.count_nonzero() / aggregation_advantages_flat.numel()
                    )
                    aggregation_percent_zero_advantages = 1 - aggregation_percent_valid_advantages
                    
                    aggregation_advantages_min, aggregation_advantages_mean, aggregation_advantages_max = (
                        aggregation_advantages_flat.min(),
                        aggregation_advantages_flat.mean(),
                        aggregation_advantages_flat.max(),
                    )
                    aggregation_reward_min, aggregation_reward_mean, aggregation_reward_max = (
                        aggregation_rewards.min(),
                        aggregation_rewards.mean(),
                        aggregation_rewards.max(),
                    )
                    
                    # Calculate combined metrics for overall tracking
                    percent_valid_advantages = (
                        all_advantages.count_nonzero() / all_advantages.numel()
                    )
                    percent_zero_advantages = 1 - percent_valid_advantages

                    advantages_min, advantages_mean, advantages_max = (
                        all_advantages.min(),
                        all_advantages.mean(),
                        all_advantages.max(),
                    )
                    reward_min, reward_mean, reward_max = (
                        all_rewards.min(),
                        all_rewards.mean(),
                        all_rewards.max(),
                    )
                    
                    # Calculate majority@k for generation stage
                    generation_majority_at_k = 0.0
                    try:
                        generation_prompts = input_ids
                        generation_majority_at_k = calculate_math_majority_at_k(
                            repeated_batch["message_log"], generation_prompts, rewards
                        )
                    except Exception as e:
                        print(f"  ⚠️ Error calculating generation majority@k: {str(e)}")
                    
                    # Calculate majority@k for aggregation stage
                    aggregation_majority_at_k = 0.0
                    if aggregation_batch is not None:
                        try:
                            aggregation_majority_at_k = calculate_math_majority_at_k(
                                aggregation_batch["message_log"], aggregation_input_ids, aggregation_rewards
                            )
                        except Exception as e:
                            print(f"  ⚠️ Error calculating aggregation majority@k: {str(e)}")
                    
                    # Update rollout metrics with separate generation and aggregation metrics
                    rollout_metrics.update({
                        # Generation metrics
                        "generation_percent_zero_advantages": generation_percent_zero_advantages,
                        "generation_advantages_min": generation_advantages_min,
                        "generation_advantages_mean": generation_advantages_mean,
                        "generation_advantages_max": generation_advantages_max,
                        "generation_reward_min": generation_reward_min,
                        "generation_reward_mean": generation_reward_mean,
                        "generation_reward_max": generation_reward_max,
                        "generation_majority_at_k": generation_majority_at_k,
                        # Aggregation metrics
                        "aggregation_percent_zero_advantages": aggregation_percent_zero_advantages,
                        "aggregation_advantages_min": aggregation_advantages_min,
                        "aggregation_advantages_mean": aggregation_advantages_mean,
                        "aggregation_advantages_max": aggregation_advantages_max,
                        "aggregation_reward_min": aggregation_reward_min,
                        "aggregation_reward_mean": aggregation_reward_mean,
                        "aggregation_reward_max": aggregation_reward_max,
                        "aggregation_majority_at_k": aggregation_majority_at_k,
                        # Combined metrics
                        "combined_percent_zero_advantages": percent_zero_advantages,
                        "combined_advantages_min": advantages_min,
                        "combined_advantages_mean": advantages_mean,
                        "combined_advantages_max": advantages_max,
                        "combined_reward_min": reward_min,
                        "combined_reward_mean": reward_mean,
                        "combined_reward_max": reward_max,
                    })
                else:
                    # Single-stage training metrics
                    percent_valid_advantages = (
                        all_advantages.count_nonzero() / all_advantages.numel()
                    )
                    percent_zero_advantages = 1 - percent_valid_advantages

                    advantages_min, advantages_mean, advantages_max = (
                        all_advantages.min(),
                        all_advantages.mean(),
                        all_advantages.max(),
                    )
                    reward_min, reward_mean, reward_max = (
                        all_rewards.min(),
                        all_rewards.mean(),
                        all_rewards.max(),
                    )
                    
                    # Calculate majority@k for single-stage training
                    majority_at_k = 0.0
                    try:
                        majority_at_k = calculate_math_majority_at_k(
                            repeated_batch["message_log"], input_ids, all_rewards
                        )
                    except Exception as e:
                        print(f"  ⚠️ Error calculating majority@k: {str(e)}")
                    
                    rollout_metrics.update({
                        "percent_zero_advantages": percent_zero_advantages,
                        "advantages_min": advantages_min,
                        "advantages_mean": advantages_mean,
                        "advantages_max": advantages_max,
                        "reward_min": reward_min,
                        "reward_mean": reward_mean,
                        "reward_max": reward_max,
                        "majority_at_k": majority_at_k,
                    })

            with timer.time("data_processing"):
                # Create training data for generation (original flow)
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = advantages[i].expand(
                            message["token_ids"].shape
                        )

                # Convert to training data (original flow)
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                train_data = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )

                # Process aggregation data if aggregation training is enabled
                if master_config["grpo"].get("enable_aggregation", False):
                    # Create training data for aggregation
                    for i, message_log in enumerate(aggregation_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )
                            message["advantages"] = aggregation_advantages[i].expand(
                                message["token_ids"].shape
                            )

                    # Convert aggregation to training data
                    aggregation_flat_messages, aggregation_input_lengths = batched_message_log_to_flat_message(
                        aggregation_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    aggregation_train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": aggregation_flat_messages["token_ids"],
                            "input_lengths": aggregation_input_lengths,
                            "advantages": aggregation_flat_messages["advantages"],
                            "generation_logprobs": aggregation_flat_messages["generation_logprobs"],
                            "token_mask": aggregation_flat_messages["token_loss_mask"],
                            "sample_mask": aggregation_batch["loss_multiplier"],
                        }
                    )

                    # Combine training data from both stages
                    train_data = combine_training_data(
                        train_data,
                        aggregation_train_data,
                    )

                train_data.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                list_of_train_metrics = policy.train(train_data, loss_fn)

            is_last_step = step + 1 == min(max_num_steps, len(dataloader))

            # Run validation if it's a validation step
            if is_last_step or (val_period > 0 and (step + 1) % val_period == 0):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy,
                        policy_generation,
                        refit_buffer_size_gb,
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                    logger=logger,
                )
                policy_generation.finish_generation()
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            if master_config["checkpointing"]["enabled"] and (
                is_last_step
                or (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                policy.prepare_for_training()

                grpo_save_state["step"] = step + 1
                grpo_save_state["val_reward"] = val_metrics["accuracy"]
                grpo_save_state["consumed_samples"] = consumed_samples
                grpo_save_state["optim_step"] = optim_step + len(list_of_train_metrics)
                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)
                policy.offload_after_refit()

        # Logging
        # Log training data (use generation data for logging since it's more interpretable)
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"][:len(rewards)].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"][:len(rewards)].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")
        table = logger.log_batched_dict_as_table(log_data, prefix="train", step=step)

        print("\n📊 Training Results:")

        rollout_metrics["table"] = table
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        print(f"  • Avg Reward: {np.mean(all_rewards.numpy()):.4f}")
        if master_config["grpo"].get("enable_aggregation", False):
            print(f"  • Generation Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(f"  • Aggregation Avg Reward: {np.mean(aggregation_rewards.numpy()):.4f}")
            print(f"  • Generation Majority@K: {rollout_metrics.get('generation_majority_at_k', 0.0):.4f}")
            print(f"  • Aggregation Majority@K: {rollout_metrics.get('aggregation_majority_at_k', 0.0):.4f}")
        else:
            print(f"  • Majority@K: {rollout_metrics.get('majority_at_k', 0.0):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
        )

        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)
        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        for i, train_step_metric in enumerate(list_of_train_metrics):
            train_step_metric["optim_step"] = optim_step + i + 1
            train_step_metric["outer_loop_step"] = step + 1
            logger.log_metrics(
                train_step_metric,
                train_step_metric["optim_step"],
                prefix="train",
            )

        logger.log_metrics(rollout_metrics, step + 1, prefix="train_rollout")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1
        optim_step += len(list_of_train_metrics)

        if step >= max_num_steps:
            break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    val_task_to_env: Dict[str, EnvironmentInterface],
    step: int,
    master_config: MasterConfig,
    logger: Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            val_batch, gen_metrics = run_multi_turn_rollout(
                policy_generation,
                val_batch,
                tokenizer,
                val_task_to_env,
                max_seq_len=master_config["grpo"].get("first_stage_generation_max_seq_len") or master_config["policy"]["max_total_sequence_length"],
                max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                greedy=False,
            )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = get_keys_from_message_log(
                val_batch["message_log"], ["role", "content"]
            )
            all_message_logs.extend(to_env)

            if batch_idx == 0:
                for interaction in val_batch["message_log"][0]:
                    if interaction["role"] == "user":
                        prompt = interaction["content"]
                    elif interaction["role"] == "assistant":
                        response = interaction["content"]
                    else:
                        environment = interaction["content"]

                reward = val_batch["total_reward"][0].item()
                table = logger.log_table_contents(
                    step, prompt, response, environment, reward, "validation"
                )

        # Calculate validation metrics
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
            "table": table,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
