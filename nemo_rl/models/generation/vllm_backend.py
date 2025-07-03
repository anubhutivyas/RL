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
from typing import Any

import torch

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
    )


class VllmInternalWorkerExtension:
    def init_shared_buffer(self, ipc_handle):
        """Initialize a shared buffer from an IPC handle."""
        # The ipc_handle is a tuple returned by `reduce_tensor`.
        # The first element is the function to rebuild the tensor,
        # and the second is a tuple of arguments for that function.
        device_uuid = self.report_device_id()
        func, args = ipc_handle[device_uuid]
        self.shared_buffer = func(*args)

    def init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(pg, device=self.device)

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def update_weights_from_shared_buffer(self, metadata: dict[str, Any]) -> bool:
        """Update weights using data from the shared buffer."""
        try:
            # Get metadata for this device
            device_uuid = self.report_device_id()
            if device_uuid not in metadata:
                return True
            metadata = metadata[device_uuid]

            # Wait for the producer to finish copying data.

            key = metadata["key"]
            shape = metadata["shape"]
            dtype = metadata["dtype"]

            # Create a view into the shared buffer.
            tensor_numel = torch.prod(torch.tensor(shape)).item()
            tensor_nbytes = tensor_numel * torch.tensor([], dtype=dtype).element_size()
            buffer_view = self.shared_buffer.narrow(0, 0, tensor_nbytes).view(dtype)
            tensor = buffer_view[:tensor_numel].view(shape)

            # Load the weight from the view.
            # self.model_runner.model.load_weights(weights=[(key, tensor.clone())])
            self.model_runner.model.load_weights(weights=[(key, tensor)])

            event = torch.cuda.Event.from_ipc_handle(self.device.index, metadata["event_handle"])
            event.record()

            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_shared_buffer: {e}"
            )
            return False

    def update_weights_from_collective(self, info: dict[str, Any]) -> bool:
        """Update the model weights from collective communication."""
        try:
            for name, (shape, dtype) in info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)
                self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True
