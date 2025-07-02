import os
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from nemo.collections.llm.gpt.model.qwen3 import Qwen3Model, Qwen3MoEConfig
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_rl.models.megatron.converters.common import MegatronToHFConverter


### TODO: add dense test
def dummy_qwen3_megatron_config():
    return Qwen3MoEConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        num_moe_experts=2,
    )


def create_dummy_hf_config():
    """Create a dummy HF config and save it to a temporary directory."""
    # Create a minimal HF config that matches the megatron config
    hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)

    # Update config to match our dummy megatron config
    hf_config.num_hidden_layers = 2
    hf_config.hidden_size = 64
    hf_config.num_attention_heads = 4
    hf_config.num_key_value_heads = 2
    hf_config.intermediate_size = 128
    hf_config.moe_intermediate_size = 32
    hf_config.num_experts = 2

    return hf_config


def test_conversion_to_hf():
    # Set up environment variables for distributed training
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6000"

    # Initialize torch.distributed first
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )

    # Initialize megatron parallel
    import megatron.core.parallel_state as parallel_state
    import megatron.core.tensor_parallel.random as tensor_parallel_random

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    # Set up CUDA RNG states for model parallel
    tensor_parallel_random.model_parallel_cuda_manual_seed(42)

    try:
        # Create megatron model
        mcore_config = dummy_qwen3_megatron_config()
        model = Qwen3Model(mcore_config)
        model.configure_model()

        # Create dummy HF config and save to temporary directory
        with TemporaryDirectory() as tmp_dir:
            hf_dir = os.path.join(tmp_dir, "Qwen3-tiny-test")
            hf_config = create_dummy_hf_config()
            hf_config.save_pretrained(hf_dir)

            # Create a dummy HF model to get the model class
            dummy_model = AutoModelForCausalLM.from_config(
                hf_config, trust_remote_code=True
            )
            dummy_model.save_pretrained(hf_dir)

            original_state_dict = model.module.state_dict()

            converter = MegatronToHFConverter(
                hf_model_name=hf_dir,
                megatron_model=model.module,
            )

            converted_state_dict = converter.convert(original_state_dict, model.config)

            ## filter out _extra_state keys
            original_state_dict = {
                k: v for k, v in original_state_dict.items() if "_extra_state" not in k
            }

            ## check that the number of keys in the original state dict is equal to the number of keys in the converted state dict minus the number of extra state keys
            ## taking into account the qkv merging and the merging of the up and gate projections
            assert len(original_state_dict) == len(converted_state_dict) - (
                2 * hf_config.num_hidden_layers
                + (hf_config.num_hidden_layers * hf_config.num_experts)
            )

            ## TODO: do not hardcode these values
            q_chunk_size = 256
            kv_chunk_size = 256

            ## check a few of the tensors to make sure they match
            torch.testing.assert_close(
                original_state_dict[
                    "decoder.layers.0.self_attention.q_layernorm.weight"
                ],
                converted_state_dict["model.layers.0.self_attn.q_norm.weight"],
            )
            torch.testing.assert_close(
                original_state_dict[
                    "decoder.layers.0.self_attention.linear_qkv.weight"
                ][:q_chunk_size],
                converted_state_dict["model.layers.0.self_attn.q_proj.weight"][
                    :q_chunk_size
                ],
            )
            torch.testing.assert_close(
                original_state_dict[
                    "decoder.layers.1.self_attention.linear_qkv.weight"
                ][(q_chunk_size + kv_chunk_size) : (2 * q_chunk_size + kv_chunk_size)],
                converted_state_dict["model.layers.1.self_attn.q_proj.weight"][
                    q_chunk_size:
                ],
            )
            torch.testing.assert_close(
                original_state_dict["decoder.layers.1.mlp.experts.linear_fc1.weight0"][
                    mcore_config.moe_ffn_hidden_size :
                ],
                converted_state_dict["model.layers.1.mlp.experts.0.up_proj.weight"],
            )
            torch.testing.assert_close(
                original_state_dict["decoder.layers.1.mlp.experts.linear_fc1.weight0"][
                    : mcore_config.moe_ffn_hidden_size
                ],
                converted_state_dict["model.layers.1.mlp.experts.0.gate_proj.weight"],
            )
            torch.testing.assert_close(
                original_state_dict["decoder.layers.0.mlp.experts.linear_fc2.weight1"],
                converted_state_dict["model.layers.0.mlp.experts.1.down_proj.weight"],
            )

    finally:
        # Clean up megatron parallel
        parallel_state.destroy_model_parallel()
        # Clean up distributed
        if dist.is_initialized():
            dist.destroy_process_group()
