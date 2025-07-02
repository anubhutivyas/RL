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

"""Parallelism information and utilities for policy workers."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ParallelismInfo:
    """Information about parallelism configuration for a policy worker.

    This dataclass encapsulates all parallelism dimensions and provides
    validation and utility methods.
    """

    # Core parallelism dimensions
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    @property
    def model_parallel_size(self) -> int:
        """Total size across all model parallel dimensions.

        Model parallelism encompasses tensor, pipeline, expert, and context parallelism.
        """
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.context_parallel_size
        )

    @property
    def total_parallel_size(self) -> int:
        """Total size across all parallelism dimensions."""
        return self.data_parallel_size * self.model_parallel_size

    @property
    def supports_dynamic_batching(self) -> bool:
        """Whether this parallelism configuration supports dynamic batching."""
        # Dynamic batching only supported for single pipeline stage
        return self.pipeline_parallel_size == 1

    @property
    def supports_moe(self) -> bool:
        """Whether this configuration supports Mixture of Experts models."""
        return self.expert_parallel_size > 1

    @property
    def supports_long_sequences(self) -> bool:
        """Whether this configuration supports very long sequences via context parallelism."""
        return self.context_parallel_size > 1

    def validate_world_size(self, world_size: int) -> None:
        """Validate that parallelism configuration is consistent with world size.

        Args:
            world_size: Total number of processes/GPUs

        Raises:
            ValueError: If parallelism configuration is inconsistent
        """
        total_expected = self.total_parallel_size

        if total_expected != world_size:
            raise ValueError(
                f"Parallelism configuration is inconsistent with world size. "
                f"Expected: DP({self.data_parallel_size}) * "
                f"TP({self.tensor_parallel_size}) * "
                f"PP({self.pipeline_parallel_size}) * "
                f"EP({self.expert_parallel_size}) * "
                f"CP({self.context_parallel_size}) = {total_expected}, "
                f"but got world_size={world_size}"
            )

    def validate_parallelism_constraints(self) -> None:
        """Validate parallelism constraints and compatibility.

        Raises:
            ValueError: If parallelism configuration is invalid
        """
        # All parallelism sizes must be positive
        if any(
            size < 1
            for size in [
                self.data_parallel_size,
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                self.expert_parallel_size,
                self.context_parallel_size,
            ]
        ):
            raise ValueError("All parallelism sizes must be >= 1")

        # Expert parallelism constraints
        if self.expert_parallel_size > 1:
            # Expert parallelism typically requires certain constraints
            if self.pipeline_parallel_size > 1:
                # Some implementations don't support EP + PP together
                pass  # Could add warning or constraint here

        # Context parallelism constraints
        if self.context_parallel_size > 1:
            # Context parallelism has limitations with certain model types
            pass  # Could add specific model compatibility checks

    def get_parallelism_summary(self) -> Dict[str, Any]:
        """Get a summary of the parallelism configuration.

        Returns:
            Dictionary with parallelism information and derived properties
        """
        return {
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "model_parallel_size": self.model_parallel_size,
            "total_parallel_size": self.total_parallel_size,
            "supports_dynamic_batching": self.supports_dynamic_batching,
            "supports_moe": self.supports_moe,
            "supports_long_sequences": self.supports_long_sequences,
        }
