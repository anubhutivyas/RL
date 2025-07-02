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

"""FSDP1 worker adapter implementation."""

from typing import Any, Dict

# get_worker_config import removed - using base class helpers instead
from ..parallelism import ParallelismInfo
from ..registry import PolicyWorker, PolicyWorkerRegistry
from ..types import PolicyConfig


@PolicyWorkerRegistry.register(
    worker_type="fsdp1",
    worker_class=None,  # Will be set by import
    config_type=dict,  # FSDP1 doesn't have additional config
    description="PyTorch FSDP1 worker for distributed training with data parallelism",
)
class FSDP1WorkerAdapter(PolicyWorker):
    """Adapter for FSDP1 policy worker."""

    @classmethod
    def prepare_worker_config(cls, config: PolicyConfig) -> Dict[str, Any]:
        """Prepare FSDP1 worker configuration."""
        cls._validate_worker_type(config)

        # FSDP1 doesn't need special config transformation
        return dict(config)

    @classmethod
    def get_parallelism_info(
        cls, config: PolicyConfig, world_size: int = None
    ) -> ParallelismInfo:
        """Extract parallelism info from FSDP1 config."""
        cls._validate_worker_type(config)

        # FSDP1 only supports data parallelism - all model parallel dimensions are 1
        # The data parallel size equals the world size
        data_parallel_size = world_size if world_size is not None else 1

        return ParallelismInfo(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            expert_parallel_size=1,
        )

    @classmethod
    def validate_config(cls, config: PolicyConfig) -> None:
        """Validate FSDP1-specific configuration."""
        # Call parent validation (includes type checking)
        super().validate_config(config)

        # Validate dynamic batching support
        if config["dynamic_batching"]["enabled"]:
            raise ValueError(
                "Dynamic batching is not supported with FSDP1 worker. "
                "Please disable dynamic_batching or use DTensor/Megatron worker."
            )

        # FSDP1 has minimal additional validation requirements

    @classmethod
    def get_worker_class(cls) -> str:
        """Get FSDP1 worker class path."""
        return "nemo_rl.models.policy.workers.fsdp1_policy_worker.FSDP1PolicyWorker"

    @classmethod
    def get_worker_type(cls) -> str:
        """Get worker type."""
        return "fsdp1"
