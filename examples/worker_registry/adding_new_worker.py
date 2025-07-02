#!/usr/bin/env python3
"""Example: Adding a new worker to the registry system.

This example shows how to add a new distributed training worker
without modifying any existing code in the core system.
"""

from typing import Any, Dict, NotRequired, TypedDict

from nemo_rl.models.policy.parallelism import ParallelismInfo
from nemo_rl.models.policy.registry import PolicyWorker, PolicyWorkerRegistry
from nemo_rl.models.policy.types import PolicyConfig


# Step 1: Define your worker-specific configuration
class FakeDDPWorkerConfig(TypedDict):
    """Configuration for FakeDDP worker (example)."""

    reduce_bucket_size: int
    gradient_compression: bool
    overlap_communication: bool
    tensor_parallel_size: NotRequired[int]  # Optional, defaults to 1


# Step 2: Create the adapter class
@PolicyWorkerRegistry.register(
    worker_type="fake_ddp",
    worker_class=None,  # Could point to actual worker class
    config_type=FakeDDPWorkerConfig,
    description="Example FakeDDP worker for demonstration",
)
class FakeDDPWorkerAdapter(PolicyWorker):
    """Adapter for FakeDDP worker (example implementation)."""

    @classmethod
    def prepare_worker_config(cls, config: PolicyConfig) -> Dict[str, Any]:
        """Prepare FakeDDP worker configuration."""
        fake_ddp_config = cls._get_worker_config_data(config)

        # Transform config for the actual worker
        worker_config = dict(config)
        worker_config["fake_ddp_cfg"] = (
            fake_ddp_config if fake_ddp_config else cls.get_default_config()
        )

        return worker_config

    @classmethod
    def get_parallelism_info(
        cls, config: PolicyConfig, world_size: int = None
    ) -> ParallelismInfo:
        """Extract parallelism info from FakeDDP config."""
        fake_ddp_config = cls._get_worker_config_data(config)

        # Calculate model parallel size (just tensor parallel for FakeDDP)
        tensor_parallel_size = fake_ddp_config.get("tensor_parallel_size", 1)
        pipeline_parallel_size = 1  # FakeDDP doesn't support pipeline parallelism
        context_parallel_size = 1  # FakeDDP doesn't support context parallelism
        expert_parallel_size = 1  # FakeDDP doesn't support expert parallelism

        model_parallel_size = tensor_parallel_size  # Only TP is supported

        # Calculate data parallel size if world_size is provided
        if world_size is not None:
            data_parallel_size = world_size // model_parallel_size
        else:
            data_parallel_size = 1  # Default fallback

        return ParallelismInfo(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )

    @classmethod
    def validate_config(cls, config: PolicyConfig) -> None:
        """Validate FakeDDP-specific configuration."""
        # Call parent validation (includes type checking)
        super().validate_config(config)

        fake_ddp_config = cls._get_worker_config_data(config)

        # Validate configuration
        if fake_ddp_config.get("reduce_bucket_size", 0) <= 0:
            raise ValueError("reduce_bucket_size must be > 0")

        # Validate dynamic batching support
        if config["dynamic_batching"]["enabled"]:
            raise ValueError(
                "Dynamic batching is not supported with FakeDDP worker. "
                "Please disable dynamic_batching or use a different worker."
            )

    @classmethod
    def get_worker_class(cls) -> str:
        """Get FakeDDP worker class path."""
        return "examples.fake_ddp_worker.FakeDDPWorker"  # Hypothetical

    @classmethod
    def get_worker_type(cls) -> str:
        """Get worker type."""
        return "fake_ddp"

    @classmethod
    def get_default_config(cls) -> FakeDDPWorkerConfig:
        """Get default configuration for FakeDDP worker."""
        return {
            "reduce_bucket_size": 25 * 1024 * 1024,  # 25MB
            "gradient_compression": False,
            "overlap_communication": True,
            "tensor_parallel_size": 1,
        }


def test_new_worker():
    """Test that our new worker is properly registered and functional."""
    from nemo_rl.models.policy import PolicyWorkerRegistry
    from nemo_rl.models.policy.types import PolicyConfig

    # Configuration using our new worker
    # TODO(ahmadki): configure through function and not a dict
    config: PolicyConfig = {
        "model_name": "test_model",
        "worker": {
            "type": "fake_ddp",
            "config": {
                "reduce_bucket_size": 50 * 1024 * 1024,
                "gradient_compression": True,
                "overlap_communication": True,
                "tensor_parallel_size": 1,
            },
        },
        "precision": "bfloat16",
        "train_global_batch_size": 32,
        "train_micro_batch_size": 4,
        "logprob_batch_size": 16,
        "generation_batch_size": 8,
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 4096,
            "logprob_mb_tokens": 4096,
            "sequence_length_round": 64,
        },
        "sequence_packing": {
            "enabled": False,
            "train_mb_tokens": 0,
            "logprob_mb_tokens": 0,
            "algorithm": "",
        },
        "make_sequence_length_divisible_by": 1,
        "max_total_sequence_length": 1024,
        "max_grad_norm": 1.0,
        "fsdp_offload_enabled": False,
        "activation_checkpointing_enabled": False,
        "tokenizer": {"name": "test_tokenizer", "chat_template": ""},
        "optimizer": {"name": "torch.optim.AdamW", "kwargs": {"lr": 1e-4}},
        "generation": None,
    }

    try:
        # Test that our worker is registered
        registered_types = PolicyWorkerRegistry.get_registered_types()
        assert "fake_ddp" in registered_types, f"fake_ddp not in {registered_types}"
        print("✓ FakeDDP worker is registered")

        # Test worker creation
        worker_class, parallelism_info, worker_config = PolicyWorkerRegistry.create(
            config, world_size=2
        )
        print(f"✓ Worker created: {worker_class}")
        print(f"✓ Parallelism: {parallelism_info}")

        # Test validation
        FakeDDPWorkerAdapter.validate_config(config)
        print("✓ Configuration validation passed")

        print("\n🎉 New worker integration successful!")
        return True

    except Exception as e:
        print(f"✗ New worker test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Example: Adding New Worker to Registry System")
    print("=" * 50)
    print("New worker type 'fake_ddp' will be automatically registered")
    print("when this module is imported (via @register decorator)")
    print()

    success = test_new_worker()

    if success:
        print("\n✅ Key Benefits of Extensible Design:")
        print("• No modifications to core code required")
        print("• Plugin-style architecture")
        print("• Type safety maintained through adapters")
        print("• Automatic validation and configuration transformation")
        print("• Seamless integration with existing Policy class")
    else:
        print("\n❌ Integration failed - check implementation")
