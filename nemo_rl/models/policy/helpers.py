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

"""Helper functions for worker configuration management."""

from typing import Literal

from .types import (
    DEFAULT_WORKER,
    PolicyConfig,
    WorkerConfig,
)


def get_worker_config(
    config: PolicyConfig,
) -> (
    WorkerConfig
):  # FIXME(ahmadki): temporary solution, should eventually be deleted/replaced
    """Get worker configuration with FSDP1 as default."""
    return config.get("worker", DEFAULT_WORKER)


def get_worker_type(
    config: PolicyConfig,
) -> Literal[
    "dtensor", "megatron", "fsdp1"
]:  # FIXME(ahmadki): temporary solution, should eventually be deleted/replaced
    """Get the active worker type."""
    return get_worker_config(config)["type"]
