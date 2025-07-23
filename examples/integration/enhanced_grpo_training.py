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

"""Example integration of enhanced checkpointing with safetensors support in GRPO training.

This example demonstrates how to use the EnhancedCheckpointManager to:
1. Save checkpoints in safetensors format for HuggingFace compatibility
2. Generate consolidated checkpoints for inference tools (vLLM, SGLang)
3. Maintain backward compatibility with existing nemo-rl checkpoints
"""

import os
import warnings
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

# Import enhanced checkpoint manager
from nemo_rl.utils.enhanced_checkpoint import (
    EnhancedCheckpointManager,
    EnhancedCheckpointingConfig,
)
from nemo_rl.algorithms.interfaces import ColocatablePolicyInterface, GenerationInterface
from nemo_rl.data.interfaces import StatefulDataLoader  
from nemo_rl.utils.timer import Timer

def enhanced_grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: AutoTokenizer,
    loss_fn: Any,  # LossFunction
    task_to_env: Dict[str, Any],  # dict[str, EnvironmentInterface]
    val_task_to_env: Optional[Dict[str, Any]],  # Optional[dict[str, EnvironmentInterface]]
    logger: Any,  # Logger
    checkpointer: EnhancedCheckpointManager,
    grpo_save_state: Dict[str, Any],  # GRPOSaveState
    master_config: Dict[str, Any],  # MasterConfig
) -> None:
    """Enhanced GRPO training loop with safetensors checkpoint support.
    
    This function demonstrates how to integrate the EnhancedCheckpointManager
    with existing GRPO training code to enable safetensors checkpointing
    while maintaining backward compatibility.
    
    Args:
        policy: The RL policy interface
        policy_generation: Optional generation interface for evaluation
        dataloader: Training dataloader with state management
        val_dataloader: Optional validation dataloader
        tokenizer: HuggingFace tokenizer
        loss_fn: Loss function for training
        task_to_env: Mapping of task names to environments
        val_task_to_env: Optional validation environments
        logger: Training logger
        checkpointer: Enhanced checkpoint manager with safetensors support
        grpo_save_state: Current training state
        master_config: Complete training configuration
    """
    timer = Timer()
    
    # Training loop (simplified for demonstration)
    max_steps = master_config.get("max_steps", 1000)
    save_period = master_config["checkpointing"]["save_period"]
    
    for step in range(max_steps):
        # === Training Step ===
        with timer.time("training"):
            # Get batch from dataloader
            batch = next(iter(dataloader))
            
            # Prepare policy for training
            policy.prepare_for_training()
            
            # Forward pass and loss computation
            train_results = policy.train(batch, loss_fn)
            
            # Update training state
            grpo_save_state["step"] = step + 1
            grpo_save_state["loss"] = train_results["loss"].item()
            
            # Finish training step
            policy.finish_training()
        
        # === Validation (Optional) ===
        val_metrics = None
        if val_dataloader is not None and (step + 1) % 100 == 0:
            with timer.time("validation"):
                val_metrics = run_validation(policy, val_dataloader, val_task_to_env)
                if val_metrics:
                    grpo_save_state["val_reward"] = val_metrics["accuracy"]
        
        # === Enhanced Checkpointing ===
        is_last_step = (step + 1) == max_steps
        should_checkpoint = (
            master_config["checkpointing"]["enabled"] and 
            (is_last_step or (step + 1) % save_period == 0)
        )
        
        if should_checkpoint:
            save_enhanced_checkpoint(
                step=step + 1,
                policy=policy,
                dataloader=dataloader,
                tokenizer=tokenizer,
                checkpointer=checkpointer,
                grpo_save_state=grpo_save_state,
                master_config=master_config,
                timer=timer,
            )
        
        # === Logging ===
        if (step + 1) % 10 == 0:
            metrics = {
                "step": step + 1,
                "loss": train_results["loss"].item(),
                "grad_norm": train_results.get("grad_norm", 0.0),
            }
            if val_metrics:
                metrics.update(val_metrics)
                
            logger.log_metrics(metrics, step=step + 1)
            
            # Log timing information
            timing_metrics = timer.get_timing_metrics()
            print(f"Step {step + 1}: Loss = {metrics['loss']:.4f}, "
                  f"Time = {timing_metrics.get('training', 0):.2f}s")


def save_enhanced_checkpoint(
    step: int,
    policy: ColocatablePolicyInterface,
    dataloader: StatefulDataLoader,
    tokenizer: AutoTokenizer,
    checkpointer: EnhancedCheckpointManager,
    grpo_save_state: Dict[str, Any],
    master_config: Dict[str, Any],
    timer: Timer,
) -> None:
    """Save checkpoint using enhanced manager with safetensors support.
    
    This function demonstrates the new checkpoint saving workflow that:
    1. Uses the enhanced checkpoint manager
    2. Saves in safetensors format when configured
    3. Creates consolidated HF-compatible checkpoints
    4. Maintains backward compatibility
    
    Args:
        step: Current training step
        policy: RL policy to checkpoint
        dataloader: Dataloader with state to save
        tokenizer: Tokenizer to include in checkpoint
        checkpointer: Enhanced checkpoint manager
        grpo_save_state: Current training state
        master_config: Training configuration
        timer: Timer for performance tracking
    """
    with timer.time("checkpointing"):
        print(f"Saving enhanced checkpoint for step {step}...")
        
        # Validate metric for top-k management
        metric_name = master_config["checkpointing"]["metric_name"]
        if metric_name is not None and metric_name not in grpo_save_state:
            warnings.warn(
                f"Metric '{metric_name}' not found in training state. "
                "Using step-based checkpoint management instead."
            )
            master_config["checkpointing"]["metric_name"] = None
        
        # Initialize temporary checkpoint directory
        checkpoint_path = checkpointer.init_tmp_checkpoint(
            step=step,
            training_info=grpo_save_state,
            run_config=master_config,
        )
        
        try:
            # === Save Model and Training State ===
            # Note: We need to access the actual model from the policy
            # This is simplified - in practice you'd extract the model appropriately
            model = extract_model_from_policy(policy)
            optimizer = extract_optimizer_from_policy(policy)  
            scheduler = extract_scheduler_from_policy(policy)
            
            # Save all checkpoint data using enhanced manager
            checkpointer.save_checkpoint_data(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                dataloader_state=dataloader.state_dict(),
            )
            
            # Finalize checkpoint (atomic rename + cleanup)
            checkpointer.finalize_checkpoint(checkpoint_path)
            
            # === Post-checkpoint Actions ===
            # Check if consolidated checkpoint was created
            final_checkpoint_path = str(checkpoint_path).replace("tmp_step_", "step_")
            consolidated_path = checkpointer.get_consolidated_checkpoint_path(final_checkpoint_path)
            
            if consolidated_path:
                print(f"✅ Consolidated HF checkpoint available at: {consolidated_path}")
                print("   This checkpoint can be used directly with:")
                print("   - Hugging Face Transformers")
                print("   - vLLM for inference")  
                print("   - SGLang for serving")
                print("   - Any HF-compatible tool")
            
            # Log checkpoint format info
            format_info = (
                f"Checkpoint saved in {checkpointer.model_save_format} format"
                + (f" with consolidation" if checkpointer.save_consolidated else "")
            )
            print(f"📁 {format_info}")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Clean up temporary directory on failure
            if os.path.exists(checkpoint_path):
                import shutil
                shutil.rmtree(checkpoint_path)
            raise
        
        finally:
            # Ensure policy is ready for next training step
            policy.offload_after_refit()


def setup_enhanced_checkpointing(config: Dict[str, Any]) -> EnhancedCheckpointManager:
    """Set up enhanced checkpoint manager from configuration.
    
    This function shows how to configure the enhanced checkpoint manager
    based on your training configuration, with sensible defaults and
    automatic format detection.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Configured enhanced checkpoint manager
    """
    ckpt_config = config["checkpointing"]
    
    # Create enhanced checkpointing configuration
    enhanced_config = EnhancedCheckpointingConfig(
        enabled=ckpt_config["enabled"],
        checkpoint_dir=ckpt_config["checkpoint_dir"],
        metric_name=ckpt_config["metric_name"],
        higher_is_better=ckpt_config["higher_is_better"],
        save_period=ckpt_config["save_period"],
        keep_top_k=ckpt_config.get("keep_top_k"),
        
        # Enhanced features
        model_save_format=ckpt_config.get("model_save_format", "torch_save"),
        save_consolidated=ckpt_config.get("save_consolidated", False),
        model_cache_dir=ckpt_config.get("model_cache_dir"),
        model_repo_id=config.get("policy", {}).get("model_name"),
    )
    
    # Create and return enhanced checkpoint manager
    checkpointer = EnhancedCheckpointManager(enhanced_config)
    
    # Print configuration info
    print("🔧 Enhanced Checkpoint Manager Configuration:")
    print(f"   Format: {checkpointer.model_save_format}")
    print(f"   Consolidated: {checkpointer.save_consolidated}")
    print(f"   Directory: {checkpointer.checkpoint_dir}")
    if checkpointer.keep_top_k:
        print(f"   Keep top-{checkpointer.keep_top_k} based on: {checkpointer.metric_name}")
    
    return checkpointer


def load_enhanced_checkpoint(
    checkpointer: EnhancedCheckpointManager,
    policy: ColocatablePolicyInterface,
    dataloader: StatefulDataLoader,
    prefer_consolidated: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint using enhanced manager with format auto-detection.
    
    This function demonstrates how to load checkpoints with the enhanced
    manager, including automatic format detection and consolidated checkpoint
    preference for inference scenarios.
    
    Args:
        checkpointer: Enhanced checkpoint manager
        policy: RL policy to load checkpoint into
        dataloader: Dataloader to restore state
        prefer_consolidated: Whether to prefer consolidated checkpoints for loading
        
    Returns:
        Loaded training info, or None if no checkpoint found
    """
    # Find latest checkpoint
    latest_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    if latest_checkpoint_path is None:
        print("No checkpoint found for resuming")
        return None
    
    print(f"Loading checkpoint from: {latest_checkpoint_path}")
    
    # Check for consolidated checkpoint if preferred
    if prefer_consolidated:
        consolidated_path = checkpointer.get_consolidated_checkpoint_path(latest_checkpoint_path)
        if consolidated_path:
            print(f"Using consolidated checkpoint: {consolidated_path}")
            # For inference, you might load directly with HF transformers:
            # model = AutoModelForCausalLM.from_pretrained(consolidated_path)
            # return load_training_info(checkpointer, latest_checkpoint_path)
    
    # Load training info
    training_info = checkpointer.load_training_info(latest_checkpoint_path)
    
    # TODO: Implement actual checkpoint loading into policy and dataloader
    # This would depend on the specific policy implementation
    print("✅ Checkpoint loading placeholder - implement based on your policy type")
    
    return training_info


# Helper functions (implementation depends on your policy architecture)
def extract_model_from_policy(policy: ColocatablePolicyInterface) -> torch.nn.Module:
    """Extract the underlying model from a policy interface."""
    # This is policy-specific - you'd implement based on your architecture
    # For example, for DTensor policies:
    # return policy.model
    # For Megatron policies:
    # return policy.worker_group.workers[0].model.remote()
    raise NotImplementedError("Implement based on your policy type")

def extract_optimizer_from_policy(policy: ColocatablePolicyInterface) -> Optional[torch.optim.Optimizer]:
    """Extract optimizer from policy if available."""
    # Similar to extract_model_from_policy, this depends on your policy implementation
    return None

def extract_scheduler_from_policy(policy: ColocatablePolicyInterface) -> Optional[Any]:
    """Extract scheduler from policy if available.""" 
    return None

def run_validation(policy, val_dataloader, val_task_to_env) -> Optional[Dict[str, Any]]:
    """Run validation step and return metrics."""
    # Placeholder for validation logic
    return {"accuracy": 0.85}


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration showing safetensors integration
    example_config = {
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "./enhanced_checkpoints",
            "metric_name": "val_reward", 
            "higher_is_better": True,
            "save_period": 100,
            "keep_top_k": 3,
            
            # Enhanced features
            "model_save_format": "safetensors",  # Enable safetensors
            "save_consolidated": True,           # Create HF-compatible checkpoints
            "model_cache_dir": "./cache",
        },
        "policy": {
            "model_name": "microsoft/DialoGPT-medium",
        },
        "max_steps": 1000,
    }
    
    print("🚀 Enhanced Checkpoint Manager Example")
    print("\nThis example shows how to integrate Automodel's safetensors")
    print("capabilities with nemo-rl checkpointing for:")
    print("✅ HuggingFace ecosystem compatibility")
    print("✅ Direct vLLM/SGLang loading") 
    print("✅ Consolidated and sharded formats")
    print("✅ Backward compatibility")
    
    # Set up enhanced checkpointing
    checkpointer = setup_enhanced_checkpointing(example_config)
    
    # The enhanced_grpo_train function would be called with this checkpointer
    print(f"\n📝 Use this checkpointer in your training loop:")
    print(f"   enhanced_grpo_train(..., checkpointer=checkpointer, ...)") 