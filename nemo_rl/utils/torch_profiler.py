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
import atexit
import os
from typing import Protocol, Optional
import torch
import fnmatch
import logging
import rich

NRL_TORCH_PROFILE_WORKER_PATTERNS = os.environ.get("NRL_TORCH_PROFILE_WORKER_PATTERNS", "")
NRL_TORCH_PROFILE_STEP_RANGE = os.environ.get("NRL_TORCH_PROFILE_STEP_RANGE", "")
NRL_TORCH_PROFILE_DIR = os.environ.get("NRL_TORCH_PROFILE_DIR", "torch_profiler_trace")

assert not (bool(NRL_TORCH_PROFILE_WORKER_PATTERNS) ^ bool(NRL_TORCH_PROFILE_STEP_RANGE)), (
    "Either both NRL_TORCH_PROFILE_WORKER_PATTERNS and NRL_TORCH_PROFILE_STEP_RANGE must be set, "
    "or neither. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/torch-profiler.md for more details."
)
NRL_TORCH_PROFILE_ENABLED = bool(NRL_TORCH_PROFILE_STEP_RANGE)

TORCH_PROFILE_START_STEP = None
TORCH_PROFILE_STOP_STEP = None

class _ProfilablePolicy(Protocol):
    def start_torch_profiling(self) -> None: ...

    def stop_torch_profiling(self) -> None: ...

def maybe_torch_profile_step(main_profiler: torch.profiler, step: int, policy: _ProfilablePolicy, policy_generation: Optional[_ProfilablePolicy]):
    """ Maybe turn on or off torch profiling based on the step. """

    if not NRL_TORCH_PROFILE_ENABLED:
        return
    
    global TORCH_PROFILE_START_STEP, TORCH_PROFILE_STOP_STEP
    if TORCH_PROFILE_START_STEP is None:

        # parse the step range
        TORCH_START_STEP, TORCH_STOP_STEP = NRL_TORCH_PROFILE_STEP_RANGE.split(":")
        try:
            TORCH_START_STEP = int(TORCH_START_STEP)
            TORCH_STOP_STEP = int(TORCH_STOP_STEP)
        except ValueError as e:
            raise ValueError(
                f"Invalid NRL_TORCH_PROFILE_STEP_RANGE: {NRL_TORCH_PROFILE_STEP_RANGE}. "
                "Please ensure the format is 'start:stop' where both values are integers. "
                "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/torch-profiler.md for more details."
            ) from e

        assert TORCH_START_STEP < TORCH_STOP_STEP, (
            f"{NRL_TORCH_PROFILE_STEP_RANGE=} must be a non-empty range"
        )
        assert TORCH_START_STEP >= 1, (
            f"The start step in {NRL_TORCH_PROFILE_STEP_RANGE=} must be >= 1"
        )
    
    # use slice syntax of left inclusive and right exclusive
    if TORCH_START_STEP <= step < TORCH_STOP_STEP:
        if not getattr(main_profiler, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Starting Torch profiling for step {step}[/bold red]"
            )
            main_profiler.start()
            main_profiler.__NRL_PROFILE_STARTED = True
            policy.start_torch_profiling()
            policy.__NRL_PROFILE_STARTED = True
            if policy_generation is not None:
                policy_generation.start_torch_profiling()
                policy_generation.__NRL_PROFILE_STARTED = True
            
            def stop_profiler_on_exit():
                rich.print(
                    f"[bold red]Stopping Torch profiling on exit for step {step}[/bold red]"
                )
                main_profiler.stop()
                policy.stop_torch_profiling()
                policy.__NRL_PROFILE_STARTED = False
                if policy_generation is not None:
                    policy_generation.stop_torch_profiling()
                    policy_generation.__NRL_PROFILE_STARTED = False

            atexit.register(stop_profiler_on_exit)
    else:
        if getattr(main_profiler, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Stopping Torch profiling for step {step}[/bold red]"
            )
            main_profiler.stop()
            main_profiler.__NRL_PROFILE_STARTED = False
            policy.stop_torch_profiling()
            policy.__NRL_PROFILE_STARTED = False
            if policy_generation is not None:
                policy_generation.stop_torch_profiling()
                policy_generation.__NRL_PROFILE_STARTED = False

def pattern_match(worker_name: str) -> bool:
    """ Match the worker name against the pattern. """
    if not NRL_TORCH_PROFILE_WORKER_PATTERNS:
        return False
    
    patterns = [pattern.strip() for pattern in NRL_TORCH_PROFILE_WORKER_PATTERNS.split(",") if pattern.strip()]
    for pattern in patterns:
        if fnmatch.fnmatch(worker_name, pattern):
            logging.info(
                f"Torch profiling enabled for worker '{worker_name}' (matched pattern '{pattern}')"
            )
            return True
    return False