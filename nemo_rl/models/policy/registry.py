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

"""Registry system for policy workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .parallelism import ParallelismInfo
    from .types import PolicyConfig


class PolicyWorker(ABC):  # TODO(ahmadki): is PolicyWorkerAdapter a better name ?
    """Abstract base class for policy workers.

    This class defines the interface that all policy workers must implement
    to work with the registry system.
    """

    @classmethod
    @abstractmethod
    def prepare_worker_config(cls, config: "PolicyConfig") -> Dict[str, Any]:
        """Prepare configuration for the actual worker class.

        This method transforms the unified PolicyConfig into the format
        expected by the underlying worker implementation.

        Args:
            config: Unified policy configuration

        Returns:
            Configuration dictionary for the worker
        """
        pass

    @classmethod
    @abstractmethod
    def get_parallelism_info(
        cls, config: "PolicyConfig", world_size: int = None
    ) -> "ParallelismInfo":
        """Extract parallelism information from configuration.

        Args:
            config: Policy configuration
            world_size: Total number of processes/GPUs (optional, used for data parallel size calculation)

        Returns:
            ParallelismInfo instance with all parallelism dimensions
        """
        pass

    @classmethod
    def validate_config(cls, config: "PolicyConfig") -> None:
        """Validate configuration for this worker type.

        Args:
            config: Policy configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Default implementation - subclasses can override for specific validation
        cls._validate_worker_type(config)

    @classmethod
    def _validate_worker_type(cls, config: "PolicyConfig") -> None:
        """Validate that the worker type matches this adapter.

        Args:
            config: Policy configuration to validate

        Raises:
            ValueError: If worker type doesn't match
        """
        from .helpers import get_worker_config

        worker_spec = get_worker_config(config)
        expected_type = cls.get_worker_type()
        actual_type = worker_spec["type"]

        if actual_type != expected_type:
            raise ValueError(f"Expected {expected_type} worker, got {actual_type}")

    @classmethod
    def _get_worker_config_data(cls, config: "PolicyConfig") -> dict:
        """Get worker-specific configuration data with type validation.

        Args:
            config: Policy configuration

        Returns:
            Worker-specific configuration data

        Raises:
            ValueError: If worker type doesn't match
        """
        from .helpers import get_worker_config

        cls._validate_worker_type(config)
        worker_spec = get_worker_config(config)
        return worker_spec.get("config", {})

    @classmethod
    @abstractmethod
    def get_worker_class(cls) -> str:
        """Get the fully qualified class name of the actual worker.

        Returns:
            String path to the worker class (e.g., "module.path.WorkerClass")
        """
        pass

    @classmethod
    @abstractmethod
    def get_worker_type(cls) -> str:
        """Get the worker type identifier.

        Returns:
            Worker type string (e.g., "dtensor", "megatron", "fsdp1")
        """
        pass


class PolicyWorkerRegistry:
    """Registry for policy workers with type-safe creation and validation."""

    _workers: Dict[str, Type[PolicyWorker]] = {}
    _config_types: Dict[str, Type[Dict[str, Any]]] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        worker_type: str,
        worker_class: Type[Any],  # The actual worker class (e.g., DTensorPolicyWorker)
        config_type: Type[Dict[str, Any]],
        description: str = "",
    ):
        """Decorator to register a policy worker adapter.

        Args:
            worker_type: Type identifier (e.g., "dtensor")
            worker_class: The actual worker class to instantiate
            config_type: TypedDict class for configuration validation
            description: Human-readable description

        Returns:
            Decorator function
        """

        def decorator(adapter_class: Type[PolicyWorker]) -> Type[PolicyWorker]:
            cls._workers[worker_type] = adapter_class
            cls._config_types[worker_type] = config_type
            cls._descriptions[worker_type] = description

            # Store references to the actual worker class for direct access
            adapter_class._worker_class = worker_class
            adapter_class._config_type = config_type

            return adapter_class

        return decorator

    @classmethod
    def create(
        cls,
        config: "PolicyConfig",
        world_size: int,
    ) -> tuple[str, "ParallelismInfo", Dict[str, Any]]:
        """Create a worker configuration from policy config.

        Args:
            config: Policy configuration
            world_size: Total number of processes/GPUs

        Returns:
            Tuple of (worker_class_path, parallelism_info, worker_config)

        Raises:
            ValueError: If backend type is not registered or configuration is invalid
        """
        from .helpers import get_worker_config

        worker_spec = get_worker_config(config)
        worker_type = worker_spec["type"]

        if worker_type not in cls._workers:
            raise ValueError(
                f"Unknown worker type: {worker_type}. "
                f"Available types: {list(cls._workers.keys())}"
            )

        adapter_class = cls._workers[worker_type]

        # Validate configuration
        adapter_class.validate_config(config)

        # Get parallelism info and validate against world size
        parallelism_info = adapter_class.get_parallelism_info(config, world_size)
        parallelism_info.validate_parallelism_constraints()
        parallelism_info.validate_world_size(world_size)

        # Prepare worker-specific configuration
        worker_config = adapter_class.prepare_worker_config(config)

        # Get worker class path
        worker_class_path = adapter_class.get_worker_class()

        return worker_class_path, parallelism_info, worker_config

    # FIXME(ahmadki): add usage to the examples file
    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of registered worker types."""
        return list(cls._workers.keys())

    @classmethod
    def get_description(cls, worker_type: str) -> str:
        """Get description for a worker type."""
        return cls._descriptions.get(worker_type, "")
