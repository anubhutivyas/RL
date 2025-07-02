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

"""Policy workers package - core worker implementations."""

# TODO(ahmadki): do we want to expose the workers as modules or completely rely on the registry pattern ?

from .dtensor_policy_worker import DTensorPolicyWorker
from .fsdp1_policy_worker import FSDP1PolicyWorker

# from .megatron_policy_worker import MegatronPolicyWorker # TODO(ahmadki): currently commented out due to nemo dependency.

__all__ = [
    "DTensorPolicyWorker",
    "FSDP1PolicyWorker",
    # "MegatronPolicyWorker",
]
