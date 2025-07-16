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

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_alpaca(data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            },
            {
                "role": "user",
                "content": data["instruction"]
                + (("\n" + data["input"]) if data["input"] else "")
                + "\n",
            },
            {
                "role": "assistant",
                "content": data["output"],
            },
        ]
    }


class AlpacaDataset:
    def __init__(self) -> None:
        original_ds = load_dataset("tatsu-lab/alpaca")
        ds_mapped = original_ds.map(format_alpaca)
        print(f"ds_mapped: {ds_mapped}")
        ds_split = ds_mapped["train"].train_test_split(test_size=0.1)
        self.formatted_ds = {
            "train": ds_split["train"],
            "validation": ds_split["test"],
        }
        self.task_spec = TaskDataSpec(
            task_name="Alpaca",
        )
