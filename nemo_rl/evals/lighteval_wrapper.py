import collections
from dataclasses import dataclass
from typing import Any, Dict, List

from lighteval.models.model_output import GenerativeResponse
from lighteval.tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from lighteval.tasks.registry import Registry, taskinfo_selector
from lighteval.tasks.requests import Request, SampleUid


@dataclass
class EvaluationInput:
    """Input structure for inference engines."""

    id: str
    prompt: str


@dataclass
class EvaluationOutput:
    """Output structure from inference engines."""

    id: str
    generated_text: str


class FakeLogger:
    """Fake task config logger for testing."""

    def log(self, *args, **kwargs):
        pass

    def log_num_docs(self, *args, **kwargs):
        pass


class FakeEvaluationTracker:
    """Fake evaluation tracker for testing."""

    def __init__(self):
        self.details_logger = FakeLogger()
        self.metrics_logger = FakeLogger()
        self.versions_logger = FakeLogger()
        self.general_config_logger = FakeLogger()
        self.task_config_logger = FakeLogger()


class LightEvalWrapper:
    """Wrapper around LightEval for use with custom inference engines."""

    def __init__(self, task_name: str, num_fewshot: int = 0) -> None:
        """Initialize with a specific dataset/task.

        Taken and modified from: https://github.com/huggingface/lighteval/blob/v0.10.0/src/lighteval/pipeline.py#L210
        """
        self.task_name = task_name
        self.evaluation_tracker = FakeEvaluationTracker()

        # parse task specification: suite|task|num_fewshot|truncation
        if "|" in task_name:
            # full specification
            parts = task_name.split("|")
            if len(parts) >= 3:
                num_fewshot = int(parts[2])
            full_task_name = task_name
        else:
            # simple name
            full_task_name = f"lighteval|{task_name}|{num_fewshot}|0"

        # load task
        registry = Registry(custom_tasks=None)
        task_names_list, fewshots_dict = taskinfo_selector(full_task_name, registry)
        task_dict = registry.get_task_dict(task_names_list)
        LightevalTask.load_datasets(list(task_dict.values()))

        # prepare requests
        requests, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict=fewshots_dict,
            num_fewshot_seeds=1,
            lm=None,
            max_samples=None,
            evaluation_tracker=self.evaluation_tracker,
            use_chat_template=False,
            system_prompt=None,
            cot_prompt=None,
        )

        self.task_names_list = task_names_list
        self.task_dict = task_dict
        self.fewshot_dict = fewshots_dict
        self.requests = requests
        self.docs = docs

        print(f"Initialized task '{self.task_name}' with {len(self.docs)=} samples")

    def _get_request_id(self, request: Request) -> str:
        """Get a unique ID for each request."""
        return f"{request.task_name}_{request.sample_index}"

    def get_input(self) -> List[EvaluationInput]:
        """Get inputs formatted for inference engine."""
        inputs = []
        for request_type, request_list in self.requests.items():
            for request in request_list:
                # create a unique ID for each request
                request_id = self._get_request_id(request)

                # extract prompt text from request
                if hasattr(request, "prompt"):
                    prompt_text = request.prompt
                elif hasattr(request, "query"):
                    prompt_text = request.query
                else:
                    prompt_text = getattr(request, "context", "") + getattr(
                        request, "continuation", ""
                    )

                inputs.append(EvaluationInput(id=request_id, prompt=prompt_text))

        return inputs

    def score(self, outputs: List[EvaluationOutput]) -> Dict[str, Any]:
        """Score the results using LightEval metrics."""
        output_map = {output.id: output for output in outputs}

        # convert outputs to a map from (sample_id, metric_category) to output
        # https://github.com/huggingface/lighteval/blob/v0.10.0/src/lighteval/pipeline.py#L380
        sample_id_to_responses = collections.defaultdict(list)
        for request_type, request_list in self.requests.items():
            for request in request_list:
                # get the corresponding output
                request_id = self._get_request_id(request)
                if request_id not in output_map:
                    raise ValueError(f"No output found for request {request_id}")
                output = output_map[request_id]

                # create mock response based on request type
                if request_type.name in ["GREEDY_UNTIL", "GREEDY_UNTIL_MULTI_TURN"]:
                    # generative tasks
                    response = GenerativeResponse(
                        result=output.generated_text,
                        input_tokens=[],
                        generated_tokens=[],
                        truncated_tokens_count=0,
                        padded_tokens_count=0,
                        logits=None,
                    )
                else:
                    # other request types
                    response = GenerativeResponse(
                        result=output.generated_text,
                        input_tokens=[],
                        generated_tokens=[],
                        truncated_tokens_count=0,
                        padded_tokens_count=0,
                        logits=None,
                    )

                # add response for each metric category
                for metric_category in request.metric_categories:
                    sample_id = SampleUid(request.task_name, request.sample_index)
                    sample_id_to_responses[(sample_id, metric_category)].append(
                        response
                    )

        # compute metrics
        # https://github.com/huggingface/lighteval/blob/v0.10.0/src/lighteval/pipeline.py#L499
        task_metric_category_groups = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )
        for (
            sample_id,
            metric_category,
        ), sample_responses in sample_id_to_responses.items():
            task_metric_category_groups[sample_id.task_name][metric_category][
                "ids"
            ].append(sample_id.doc_id_seed)
            task_metric_category_groups[sample_id.task_name][metric_category][
                "responses"
            ].append(sample_responses)
            task_metric_category_groups[sample_id.task_name][metric_category][
                "docs"
            ].append(self.docs[sample_id])

        # calculate metrics
        final_scores = {}
        for task_name, samples_per_metric in task_metric_category_groups.items():
            # https://github.com/huggingface/lighteval/blob/v0.10.0/src/lighteval/pipeline.py#L495
            short_task_name = task_name.rsplit("|", 1)[0]
            task = self.task_dict[short_task_name]

            for metric_category, samples in samples_per_metric.items():
                sample_ids = samples["ids"]
                responses = samples["responses"]
                docs = samples["docs"]

                # get the metric function for this category
                metric_function = task.get_metric_method_from_category(
                    metric_category=metric_category
                )
                metric_category_metrics = [
                    metric
                    for metric in task.metrics
                    if metric.category == metric_category
                ]

                # compute metrics
                try:
                    outputs = metric_function(
                        sample_ids=sample_ids,
                        responses=responses,
                        formatted_docs=docs,
                        metrics=metric_category_metrics,
                    )

                    # aggregate results
                    for output in outputs:
                        for metric_name, metric_value in output.items():
                            if metric_name not in final_scores:
                                final_scores[metric_name] = []
                            final_scores[metric_name].append(metric_value)

                except Exception as e:
                    print(f"Error computing metrics for {metric_category}: {e}")
                    continue

        # calculate average scores
        aggregated_scores = {}
        for metric_name, values in final_scores.items():
            if values:
                aggregated_scores[metric_name] = sum(values) / len(values)
            else:
                aggregated_scores[metric_name] = 0.0

        return aggregated_scores


if __name__ == "__main__":
    lighteval_wrapper = LightEvalWrapper("aime24")

    inputs = lighteval_wrapper.get_input()

    outputs = inputs
    answers = ["\\boxed{321}"] * 30
    for i, answer in enumerate(answers):
        outputs[i].generated_text = answer

    scores = lighteval_wrapper.score(outputs)
    print(f"{scores=}")
