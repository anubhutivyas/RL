from typing import Any
import json
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.generation.interfaces import GenerationInterface

def _report_memory_usage(label, policy: ColocatablePolicyInterface, policy_generation: GenerationInterface) -> dict[str, Any]:
    """Report the memory usage of the policy and policy generation."""

    policy_memory_usage = policy.get_memory_usage()
    policy_generation_memory_usage = policy_generation.get_memory_usage()
    combined_memory_usage = {}
    for k in set(policy_memory_usage.keys()) | set(policy_generation_memory_usage.keys()):
        policy_memory_allocated = policy_memory_usage[k]["current_memory_allocated"] if k in policy_memory_usage else 0
        policy_memory_reserved = policy_memory_usage[k]["current_memory_reserved"] if k in policy_memory_usage else 0
        policy_memory_max_allocated = policy_memory_usage[k]["max_memory_allocated"] if k in policy_memory_usage else 0
        policy_memory_max_reserved = policy_memory_usage[k]["max_memory_reserved"] if k in policy_memory_usage else 0
        policy_generation_memory_allocated = policy_generation_memory_usage[k]["current_memory_allocated"] if k in policy_generation_memory_usage else 0
        policy_generation_memory_reserved = policy_generation_memory_usage[k]["current_memory_reserved"] if k in policy_generation_memory_usage else 0
        policy_generation_max_memory_allocated = policy_generation_memory_usage[k]["max_memory_allocated"] if k in policy_generation_memory_usage else 0
        policy_generation_max_memory_reserved = policy_generation_memory_usage[k]["max_memory_reserved"] if k in policy_generation_memory_usage else 0
        combined_memory_usage[k] = {
            "policy_memory_allocated_gb": policy_memory_allocated/(1024**3),
            "policy_memory_reserved_gb": policy_memory_reserved/(1024**3),
            "policy_generation_memory_allocated_gb": policy_generation_memory_allocated/(1024**3),
            "policy_generation_memory_reserved_gb": policy_generation_memory_reserved/(1024**3),
            "policy_memory_max_allocated_gb": policy_memory_max_allocated/(1024**3),
            "policy_memory_max_reserved_gb": policy_memory_max_reserved/(1024**3),
            "policy_generation_max_allocated_gb": policy_generation_max_memory_allocated/(1024**3),
            "policy_generation_max_reserved_gb": policy_generation_max_memory_reserved/(1024**3),
            "total_memory_allocated_gb": (policy_memory_allocated + policy_generation_memory_allocated)/(1024**3),
            "total_memory_reserved_gb": (policy_memory_reserved + policy_generation_memory_reserved)/(1024**3),
        }
    return combined_memory_usage

class MemoryLogStore:
    def __init__(self):
        self.memory_log_store = {}

    def add_memory_log(self, label, policy: ColocatablePolicyInterface, policy_generation: GenerationInterface):
        self.memory_log_store[label] = _report_memory_usage(label, policy, policy_generation)
    
    def dump_memory_log(self, output_file: str):
        # dump self.memory_log_store to a json file
        with open(output_file, "w") as f:
            json.dump(self.memory_log_store, f)
