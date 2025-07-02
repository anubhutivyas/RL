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

"""DTensor worker adapter implementation."""

from typing import Any, Dict

# get_worker_config import removed - using base class helpers instead
from ..parallelism import ParallelismInfo
from ..registry import PolicyWorker, PolicyWorkerRegistry
from ..types import DTensorWorkerConfig, PolicyConfig


@PolicyWorkerRegistry.register(
    worker_type="dtensor",
    worker_class=None,  # Will be set by import
    config_type=DTensorWorkerConfig,
    description="PyTorch DTensor worker for distributed training with tensor and context parallelism",
)
class DTensorWorkerAdapter(PolicyWorker):
    """Adapter for DTensor policy worker."""

    @classmethod
    def prepare_worker_config(cls, config: PolicyConfig) -> Dict[str, Any]:
        """Prepare DTensor worker configuration."""
        dtensor_config = cls._get_worker_config_data(config)

        # Create config with dtensor_cfg field expected by DTensorPolicyWorker
        worker_config = dict(config)
        worker_config["dtensor_cfg"] = dtensor_config

        return worker_config

    @classmethod
    def get_parallelism_info(
        cls, config: PolicyConfig, world_size: int = None
    ) -> ParallelismInfo:
        """Extract parallelism info from DTensor config."""
        dtensor_config = cls._get_worker_config_data(config)

        # Calculate model parallel size
        tensor_parallel_size = dtensor_config["tensor_parallel_size"]
        context_parallel_size = dtensor_config["context_parallel_size"]
        expert_parallel_size = dtensor_config.get("expert_parallel_size", 1)
        pipeline_parallel_size = 1  # DTensor doesn't support pipeline parallelism

        model_parallel_size = (
            tensor_parallel_size
            * context_parallel_size
            * expert_parallel_size
            * pipeline_parallel_size
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
        """Validate DTensor-specific configuration."""
        # Call parent validation (includes type checking)
        super().validate_config(config)

        dtensor_config = cls._get_worker_config_data(config)

        # Validate parallelism sizes
        if dtensor_config["tensor_parallel_size"] < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if dtensor_config["context_parallel_size"] < 1:
            raise ValueError("context_parallel_size must be >= 1")
        if dtensor_config.get("expert_parallel_size", 1) < 1:
            raise ValueError("expert_parallel_size must be >= 1")

        # Validate dynamic batching support
        if config["dynamic_batching"]["enabled"]:
            pass
            # FIXME(ahmadki): pipeline_parallel_size validation ?
            # DTensor supports dynamic batching (pipeline_parallel_size = 1)
            # pipeline_parallel_size = 1  # DTensor doesn't support pipeline parallelism
            # if pipeline_parallel_size != 1:
            #     raise ValueError(
            #         "Dynamic batching is only supported for single pipeline parallel stage. "
            #         f"Current pipeline_parallel_size: {pipeline_parallel_size}"
            #     )

        # DTensor-specific constraints
        if dtensor_config.get("expert_parallel_size", 1) > 1:
            # DTensor may have limitations with expert parallelism
            pass  # Could add specific DTensor + EP constraints here

    @classmethod
    def get_worker_class(cls) -> str:
        """Get DTensor worker class path."""
        return "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"

    @classmethod
    def get_worker_type(cls) -> str:
        """Get worker type."""
        return "dtensor"
