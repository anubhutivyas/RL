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

"""Megatron worker adapter implementation."""

from typing import Any, Dict

# get_worker_config import removed - using base class helpers instead
from ..parallelism import ParallelismInfo
from ..registry import PolicyWorker, PolicyWorkerRegistry
from ..types import MegatronWorkerConfig, PolicyConfig


@PolicyWorkerRegistry.register(
    worker_type="megatron",
    worker_class=None,  # Will be set by import
    config_type=MegatronWorkerConfig,
    description="Megatron-LM worker for large-scale distributed training with full parallelism support",
)
class MegatronWorkerAdapter(PolicyWorker):
    """Adapter for Megatron policy worker."""

    @classmethod
    def prepare_worker_config(cls, config: PolicyConfig) -> Dict[str, Any]:
        """Prepare Megatron worker configuration."""
        megatron_config = cls._get_worker_config_data(config)

        # Create config with megatron_cfg field expected by MegatronPolicyWorker
        worker_config = dict(config)
        worker_config["megatron_cfg"] = megatron_config

        return worker_config

    @classmethod
    def get_parallelism_info(
        cls, config: PolicyConfig, world_size: int = None
    ) -> ParallelismInfo:
        """Extract parallelism info from Megatron config."""
        megatron_config = cls._get_worker_config_data(config)

        # Calculate model parallel size
        tensor_parallel_size = megatron_config["tensor_model_parallel_size"]
        pipeline_parallel_size = megatron_config["pipeline_model_parallel_size"]
        context_parallel_size = megatron_config["context_parallel_size"]
        expert_parallel_size = megatron_config.get("expert_parallel_size", 1)

        model_parallel_size = (
            tensor_parallel_size
            * pipeline_parallel_size
            * context_parallel_size
            * expert_parallel_size
        )

        # Calculate data parallel size if world_size is provided
        if world_size is not None:
            data_parallel_size = world_size // model_parallel_size
        else:
            data_parallel_size = 1  # Default fallback

        return ParallelismInfo(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )

    @classmethod
    def validate_config(cls, config: PolicyConfig) -> None:
        """Validate Megatron-specific configuration."""
        # Call parent validation (includes type checking)
        super().validate_config(config)

        megatron_config = cls._get_worker_config_data(config)

        # Validate parallelism sizes
        if megatron_config["tensor_model_parallel_size"] < 1:
            raise ValueError("tensor_model_parallel_size must be >= 1")
        if megatron_config["pipeline_model_parallel_size"] < 1:
            raise ValueError("pipeline_model_parallel_size must be >= 1")
        if megatron_config["context_parallel_size"] < 1:
            raise ValueError("context_parallel_size must be >= 1")
        if megatron_config.get("expert_parallel_size", 1) < 1:
            raise ValueError("expert_parallel_size must be >= 1")

        # Validate context parallel support
        if megatron_config["context_parallel_size"] > 1:
            raise ValueError(
                "Context parallel is not supported in Megatron backend yet"
            )

        # Validate dynamic batching support
        if config["dynamic_batching"]["enabled"]:
            # Megatron supports dynamic batching only with single pipeline stage
            pipeline_parallel_size = megatron_config["pipeline_model_parallel_size"]
            if pipeline_parallel_size != 1:
                raise ValueError(
                    "Dynamic batching is only supported for single pipeline parallel stage. "
                    f"Current pipeline_parallel_size: {pipeline_parallel_size}"
                )

        # Validate expert parallel constraints
        expert_parallel_size = megatron_config.get("expert_parallel_size", 1)
        if expert_parallel_size > 1:
            # Expert parallelism has certain constraints with other parallelism types
            if megatron_config["pipeline_model_parallel_size"] > 1:
                # Some Megatron versions have limitations with EP + PP
                pass  # Could add specific warning or constraint checking here

    @classmethod
    def get_worker_class(cls) -> str:
        """Get Megatron worker class path."""
        return (
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        )

    @classmethod
    def get_worker_type(cls) -> str:
        """Get worker type."""
        return "megatron"
