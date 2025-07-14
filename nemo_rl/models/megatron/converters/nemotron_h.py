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

import torch
from nemo.lightning import io
from nemo.lightning.io.state import TransformFns
from megatron.core import parallel_state



def concat_tp_dim_0(ctx: io.TransformCTX, param: torch.Tensor):
    """Concat tensor"""
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(tp_group) 
    return torch.cat(param.chunk(tp_size), dim=0)


@io.state_transform(
    source_key="decoder.layers.*.mixer.conv1d.weight",
    target_key="backbone.layers.*.mixer.conv1d.weight",
)
def correct_xBC_tp_order(ctx: io.TransformCTX, param: torch.Tensor):
    """Correct tensor parallel order for xBC parameters (conv1d weight).
    
    This function reorders the tensor parallel chunks to match the expected format
    where x, B, and C components are interleaved across tensor parallel ranks.
    """
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(tp_group)

    # Get configuration from context
    megatron_config = ctx.source.config
    
    # Match exact Megatron-LM calculations
    # From MambaMixer.__init__: self.d_inner = int(self.expand * self.d_model)
    # But if mamba_num_heads is provided: self.d_inner = self.nheads * self.headdim
    if hasattr(megatron_config, 'mamba_num_heads') and megatron_config.mamba_num_heads is not None:
        d_inner = megatron_config.mamba_num_heads * megatron_config.mamba_head_dim
    else:
        # Default expand is 2 in Megatron-LM
        expand = 2
        d_inner = int(expand * megatron_config.hidden_size)
    
    # From MambaMixer.__init__: self.d_inner_local = self.d_inner // self.tensor_model_parallel_size
    d_inner_local = d_inner // tp_size
    # From MambaMixer.__init__: self.ngroups_local = self.ngroups // self.tensor_model_parallel_size
    ngroups_local = megatron_config.mamba_num_groups // tp_size
    
    # From MambaMixer forward: x, B, C = torch.split(xBC, [self.d_inner_local, self.ngroups_local * self.d_state, self.ngroups_local * self.d_state], dim=-1)
    x_local_size = d_inner_local
    B_local_size = ngroups_local * megatron_config.mamba_state_dim
    C_local_size = ngroups_local * megatron_config.mamba_state_dim

    # Split parameter into tensor parallel chunks
    param_chunks = param.chunk(tp_size)

    # Reorder chunks to interleave x, B, C components
    correct_chunks = []
    # Add x components from all ranks
    correct_chunks.extend(param_chunks[tp][:x_local_size] for tp in range(tp_size))
    # Add B components from all ranks
    correct_chunks.extend(param_chunks[tp][x_local_size:x_local_size+B_local_size] for tp in range(tp_size))
    # Add C components from all ranks
    correct_chunks.extend(param_chunks[tp][x_local_size+B_local_size:x_local_size+B_local_size+C_local_size] for tp in range(tp_size))
    
    # Concatenate all chunks
    correct_param = torch.cat(correct_chunks, dim=0)
    return correct_param

@io.state_transform(
    source_key="decoder.layers.*.mixer.in_proj.weight",
    target_key="backbone.layers.*.mixer.in_proj.weight",
)
def correct_zxBCdt_tp_order(ctx: io.TransformCTX, param: torch.Tensor):
    """Correct tensor parallel order for zxBCdt parameters (in_proj weight).
    
    This function reorders the tensor parallel chunks to match the expected format
    where z, x, B, C, and dt components are interleaved across tensor parallel ranks.
    """
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(tp_group)

    # Get configuration from context
    megatron_config = ctx.source.config
    
    # Match exact Megatron-LM calculations
    # From MambaMixer.__init__: self.d_inner = int(self.expand * self.d_model)
    # But if mamba_num_heads is provided: self.d_inner = self.nheads * self.headdim
    if hasattr(megatron_config, 'mamba_num_heads') and megatron_config.mamba_num_heads is not None:
        d_inner = megatron_config.mamba_num_heads * megatron_config.mamba_head_dim
    else:
        # Default expand is 2 in Megatron-LM
        expand = 2
        d_inner = int(expand * megatron_config.hidden_size)
    
    # From MambaMixer.__init__: self.d_inner_local = self.d_inner // self.tensor_model_parallel_size
    d_inner_local = d_inner // tp_size
    # From MambaMixer.__init__: self.ngroups_local = self.ngroups // self.tensor_model_parallel_size
    ngroups_local = megatron_config.mamba_num_groups // tp_size
    # From MambaMixer.__init__: self.nheads_local = self.nheads // self.tensor_model_parallel_size
    nheads_local = megatron_config.mamba_num_heads // tp_size
    
    # From MambaMixer forward: z, xBC, dt = torch.split(xz, [self.d_inner_local, self.d_inner_local + 2 * self.ngroups_local * self.d_state, self.nheads_local], dim=-1)
    z_local_size = d_inner_local
    x_local_size = d_inner_local
    B_local_size = ngroups_local * megatron_config.mamba_state_dim
    C_local_size = ngroups_local * megatron_config.mamba_state_dim
    dt_local_size = nheads_local

    # Split parameter into tensor parallel chunks
    param_chunks = param.chunk(tp_size)

    # Reorder chunks to interleave z, x, B, C, dt components
    correct_chunks = []
    # Add z components from all ranks
    correct_chunks.extend(param_chunks[tp][:z_local_size] for tp in range(tp_size))
    # Add x components from all ranks
    correct_chunks.extend(param_chunks[tp][z_local_size:z_local_size+x_local_size] for tp in range(tp_size))
    # Add B components from all ranks
    correct_chunks.extend(param_chunks[tp][z_local_size+x_local_size:z_local_size+x_local_size+B_local_size] for tp in range(tp_size))
    # Add C components from all ranks
    correct_chunks.extend(param_chunks[tp][z_local_size+x_local_size+B_local_size:z_local_size+x_local_size+B_local_size+C_local_size] for tp in range(tp_size))
    # Add dt components from all ranks
    correct_chunks.extend(param_chunks[tp][z_local_size+x_local_size+B_local_size+C_local_size:z_local_size+x_local_size+B_local_size+C_local_size+dt_local_size] for tp in range(tp_size))
    
    # Concatenate all chunks
    correct_param = torch.cat(correct_chunks, dim=0)
    return correct_param

@io.state_transform(
    source_key="decoder.layers.*.mixer.conv1d.bias",
    target_key="backbone.layers.*.mixer.conv1d.bias",
)
def correct_xBC_tp_order_bias(ctx: io.TransformCTX, param: torch.Tensor):
    """Correct tensor parallel order for xBC parameters (conv1d bias).
    
    This function reorders the tensor parallel chunks to match the expected format
    where x, B, and C components are interleaved across tensor parallel ranks.
    """
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(tp_group)

    # Get configuration from context
    megatron_config = ctx.source.config
    
    # Match exact Megatron-LM calculations
    # From MambaMixer.__init__: self.d_inner = int(self.expand * self.d_model)
    # But if mamba_num_heads is provided: self.d_inner = self.nheads * self.headdim
    if hasattr(megatron_config, 'mamba_num_heads') and megatron_config.mamba_num_heads is not None:
        d_inner = megatron_config.mamba_num_heads * megatron_config.mamba_head_dim
    else:
        # Default expand is 2 in Megatron-LM
        expand = 2
        d_inner = int(expand * megatron_config.hidden_size)
    
    # From MambaMixer.__init__: self.d_inner_local = self.d_inner // self.tensor_model_parallel_size
    d_inner_local = d_inner // tp_size
    # From MambaMixer.__init__: self.ngroups_local = self.ngroups // self.tensor_model_parallel_size
    ngroups_local = megatron_config.mamba_num_groups // tp_size
    
    # From MambaMixer forward: x, B, C = torch.split(xBC, [self.d_inner_local, self.ngroups_local * self.d_state, self.ngroups_local * self.d_state], dim=-1)
    x_local_size = d_inner_local
    B_local_size = ngroups_local * megatron_config.mamba_state_dim
    C_local_size = ngroups_local * megatron_config.mamba_state_dim

    # Split parameter into tensor parallel chunks
    param_chunks = param.chunk(tp_size)

    # Reorder chunks to interleave x, B, C components
    correct_chunks = []
    # Add x components from all ranks
    correct_chunks.extend(param_chunks[tp][:x_local_size] for tp in range(tp_size))
    # Add B components from all ranks
    correct_chunks.extend(param_chunks[tp][x_local_size:x_local_size+B_local_size] for tp in range(tp_size))
    # Add C components from all ranks
    correct_chunks.extend(param_chunks[tp][x_local_size+B_local_size:x_local_size+B_local_size+C_local_size] for tp in range(tp_size))
    
    # Concatenate all chunks
    correct_param = torch.cat(correct_chunks, dim=0)
    return correct_param

def get_export_mapping(source):
    mapping = {
        # TODO: name='backbone.layers.28.mixer.A_log' refit=torch.Size([16]) gt=torch.Size([128])
        'decoder.layers.*.mixer.A_log': 'backbone.layers.*.mixer.A_log',
        # TODO: TP on first dim  setattr(self.D, 'tensor_model_parallel', True)
        'decoder.layers.*.mixer.D': 'backbone.layers.*.mixer.D',
        # TODO: name='backbone.layers.46.mixer.conv1d.weight' refit=torch.Size([1280, 1, 4]) gt=torch.Size([10240, 1, 4])
        #'decoder.layers.*.mixer.conv1d.weight': 'backbone.layers.*.mixer.conv1d.weight',
        # TODO: name='backbone.layers.4.mixer.conv1d.bias' refit=torch.Size([1280]) gt=torch.Size([10240])
        #'decoder.layers.*.mixer.conv1d.bias': 'backbone.layers.*.mixer.conv1d.bias',
        #'decoder.layers.*.mixer.in_proj.weight': 'backbone.layers.*.mixer.in_proj.weight',
        # TODO: TP on first dim  setattr(self.D, 'tensor_model_parallel', True)
        'decoder.layers.*.mixer.dt_bias': 'backbone.layers.*.mixer.dt_bias',
        'decoder.layers.*.mixer.out_proj.weight': 'backbone.layers.*.mixer.out_proj.weight',
        # TODO name='backbone.layers.48.mixer.norm.weight' refit=torch.Size([1024]) gt=torch.Size([8192])
        'decoder.layers.*.mixer.norm.weight': 'backbone.layers.*.mixer.norm.weight',
        'decoder.layers.*.mlp.linear_fc1.weight': 'backbone.layers.*.mixer.up_proj.weight',
        'decoder.layers.*.mlp.linear_fc2.weight': 'backbone.layers.*.mixer.down_proj.weight',
        'decoder.layers.*.self_attention.linear_proj.weight': 'backbone.layers.*.mixer.o_proj.weight',
        'decoder.final_norm.weight': 'backbone.norm_f.weight',
    }

    for i, layer_type in enumerate(source.config.hybrid_override_pattern):
        if layer_type == "M":
            mapping[f'decoder.layers.{i}.mixer.in_proj.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
        elif layer_type == "-":
            mapping[f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
        elif layer_type == "*":
            mapping[f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'] = (
                f'backbone.layers.{i}.norm.weight'
            )
        else:
            raise AttributeError(f"layer type {layer_type} not found.")
    return mapping


def get_export_transforms(hf_config):
    transforms = [
        # _export_qkv from nemo.collections.llm.gpt.model.ssm
        io.state_transform(
            source_key="decoder.layers.*.self_attention.linear_qkv.weight",
            target_key=(
                "backbone.layers.*.mixer.q_proj.weight",
                "backbone.layers.*.mixer.k_proj.weight",
                "backbone.layers.*.mixer.v_proj.weight",
            ),
            fn=TransformFns.split_qkv,
        ),
        # _export_embedding from nemo.collections.llm.gpt.model.ssm
        io.state_transform(
            source_key="embedding.word_embeddings.weight",
            target_key="backbone.embeddings.weight",
            fn=TransformFns.prune_padding,
        ),
        # Tensor parallel correction transforms for Nemotron-H
        correct_xBC_tp_order,
        correct_xBC_tp_order_bias,
        correct_zxBCdt_tp_order,
    ]

    if not hf_config.tie_word_embeddings:
        # _export_head from nemo.collections.llm.gpt.model.ssm
        transforms.append(
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        )

    return transforms
