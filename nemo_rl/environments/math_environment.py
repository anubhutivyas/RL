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
import contextlib
import io
import logging
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env
    reasoning_split_word: Optional[str] = None  # Word to split reasoning from final answer


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class HFVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self):
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self, pred_responses: List[str], ground_truths: List[str]
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            ground_truths: List[str]. The ground truth responses.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(
                            [ground_truth_parsable], [response]
                        )
                    except Exception:
                        ret_score = 0.0

                results.append(float(ret_score))
            except Exception:
                results.append(0.0)
        return results


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str
    question: Optional[str]  # Added to store the question in metadata
    format_checking: Optional[bool]  # Whether to apply format checking (default: False)


@ray.remote
class MathEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def _check_aggregation_format(self, response: str) -> float:
        """Check if response has required XML tags for aggregation.
        
        Args:
            response: The assistant's response to check
            
        Returns:
            float: 1.0 if <analysis> tags are present in correct order and answer content after analysis is longer, 0.0 otherwise
        """
        response_lower = response.lower()
        
        # Check analysis tag order
        analysis_start = response_lower.find("<analysis>")
        analysis_end = response_lower.find("</analysis>")
        has_proper_analysis = analysis_start != -1 and analysis_end != -1 and analysis_start < analysis_end
        
        if has_proper_analysis:
            # Extract content between analysis tags
            analysis_content = response[analysis_start + len("<analysis>"):analysis_end].strip()
            
            # Extract answer content as everything after </analysis> tag
            answer_content = response[analysis_end + len("</analysis>"):].strip()
            
            # Answer section should be longer than analysis section
            answer_longer = len(answer_content) > len(analysis_content)
            
            return 1.0 if answer_longer else 0.0
        else:
            return 0.0

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[MathEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        math_results = ray.get(futures)

        # flatten the math results
        math_results = [item for sublist in math_results for item in sublist]
        
        # Apply format checking individually to each response based on its metadata
        final_results = []
        format_checked_count = 0
        format_perfect_count = 0
        
        for i, (math_score, response, meta) in enumerate(zip(math_results, assistant_response_batch, metadata)):
            if meta.get("format_checking", False):
                # Apply format checking to this specific response
                format_score = self._check_aggregation_format(response)
                final_reward = math_score if format_score == 1.0 else 0.0
                final_results.append(final_reward)
                
                format_checked_count += 1
                if format_score == 1.0:
                    format_perfect_count += 1
            else:
                # No format checking for this response - use math result only
                final_results.append(math_score)
        
        # Print format checking summary if any responses were format checked
        if format_checked_count > 0:
            math_mean = sum(math_results) / len(math_results)
            final_mean = sum(final_results) / len(final_results)
            print(f"Math Environment: Applied format checking to {format_checked_count}/{len(math_results)} responses")
            print(f"  Math rewards - Mean: {math_mean:.3f}")
            print(f"  Perfect format responses: {format_perfect_count}/{format_checked_count}")
            print(f"  Final rewards - Mean: {final_mean:.3f}")

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in final_results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(final_results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
