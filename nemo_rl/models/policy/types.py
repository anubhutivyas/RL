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

"""Type definitions for policy configurations."""

from typing import Any, Literal, NotRequired, Optional, TypedDict, Union

from nemo_rl.models.generation.interfaces import GenerationConfig


class DTensorWorkerConfig(TypedDict):
    """Configuration for DTensor worker."""

    cpu_offload: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    tensor_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: NotRequired[int]  # Optional, defaults to 1
    custom_parallel_plan: str


class SequencePackingConfig(TypedDict):
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    algorithm: str


class MegatronOptimizerConfig(TypedDict):
    optimizer: str
    lr: float
    min_lr: float
    weight_decay: float
    bf16: bool
    fp16: bool
    params_dtype: str
    # adam
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    # sgd
    sgd_momentum: float
    # distributed optimizer
    use_distributed_optimizer: bool
    use_precision_aware_optimizer: bool
    clip_grad: float


class MegatronSchedulerConfig(TypedDict):
    start_weight_decay: float
    end_weight_decay: float
    weight_decay_incr_style: str
    lr_decay_style: str
    lr_decay_iters: int
    lr_warmup_iters: int
    lr_warmup_init: float


class MegatronDDPConfig(TypedDict):
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    average_in_collective: bool
    use_custom_fsdp: bool
    data_parallel_sharding_strategy: str


class MegatronWorkerConfig(TypedDict):
    """Configuration for Megatron worker."""

    empty_unused_memory_level: int
    activation_checkpointing: bool
    converter_type: str
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int
    num_layers_in_last_pipeline_stage: int
    context_parallel_size: int
    expert_parallel_size: NotRequired[int]  # Optional, defaults to 1
    pipeline_dtype: str
    sequence_parallel: bool

    optimizer: NotRequired[MegatronOptimizerConfig]
    scheduler: NotRequired[MegatronSchedulerConfig]
    distributed_data_parallel_config: MegatronDDPConfig


class TokenizerConfig(TypedDict):
    name: str
    chat_template: str


class PytorchOptimizerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


SchedulerMilestones = dict[str, list[int]]


class DynamicBatchingConfig(TypedDict):
    # dynamic_batching improves performance by ensuring logprob and training microbatches
    # have a sufficent number of tokens to maximize GPU utilization. Specifically, variable length
    # responses are sorted by sequence length and bucketed into microbatches with a total
    # amount of tokens is approximately close to 'train_mb_tokens' and 'logprob_mb_tokens' for the
    # training and logprob stages respectively.
    enabled: bool
    train_mb_tokens: int
    logprob_mb_tokens: int
    sequence_length_round: int


# Base worker configuration structure
class WorkerConfig(TypedDict):
    """Base worker configuration.

    The 'type' field identifies the worker implementation.
    The 'config' field contains worker-specific configuration.
    Worker types are registered dynamically via the registry system.
    """

    type: str  # Worker type (e.g., "dtensor", "megatron", "fsdp1") - extensible
    config: NotRequired[
        dict[str, Any]
    ]  # Worker-specific config (optional for simple workers like FSDP1)


# Predefined worker configurations for type checking
class DTensorWorker(TypedDict):
    """DTensor worker configuration with type safety."""

    type: Literal["dtensor"]
    config: DTensorWorkerConfig


class MegatronWorker(TypedDict):
    """Megatron worker configuration with type safety."""

    type: Literal["megatron"]
    config: MegatronWorkerConfig


class FSDP1Worker(TypedDict):
    """FSDP1 worker configuration with type safety."""

    type: Literal["fsdp1"]
    # FSDP1 doesn't need additional worker-specific config


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    learning_rate: float
    logprob_batch_size: int
    generation: Optional[GenerationConfig]
    generation_batch_size: NotRequired[
        int
    ]  # used in static batched (framework) generation
    precision: str

    # Unified worker configuration - type field determines which worker is active
    # Defaults to FSDP1 if not specified
    worker: NotRequired[WorkerConfig]

    dynamic_batching: DynamicBatchingConfig
    sequence_packing: SequencePackingConfig
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    max_grad_norm: Optional[Union[float, int]]
    fsdp_offload_enabled: bool
    activation_checkpointing_enabled: bool
    optimizer: NotRequired[PytorchOptimizerConfig] = None
    scheduler: NotRequired[list[SinglePytorchSchedulerConfig] | SchedulerMilestones] = (
        None
    )


# Default worker: FSDP1 (minimal configuration, data parallel only)
DEFAULT_WORKER: FSDP1Worker = {  # Delme insert into typedict
    "type": "fsdp1",
}
