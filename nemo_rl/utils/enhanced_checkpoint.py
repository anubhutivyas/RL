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

"""Enhanced checkpoint management utilities integrating Automodel's safetensors capabilities.

This module provides a unified checkpoint manager that supports both the original nemo-rl
checkpoint management and Automodel's advanced safetensors capabilities for better
HuggingFace ecosystem integration.
"""

import glob
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, NotRequired, Optional, TypedDict, Union

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)

try:
    # Try to import Automodel components if available
    from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
    from nemo_automodel.components.checkpoint._backports.hf_storage import (
        _HuggingFaceStorageWriter,
        get_fqn_to_file_index_mapping,
    )
    from nemo_automodel.components.checkpoint.stateful_wrappers import (
        ModelState as AutomodelModelState,
        OptimizerState as AutomodelOptimizerState,
    )
    AUTOMODEL_AVAILABLE = True
except ImportError:
    AUTOMODEL_AVAILABLE = False
    # Define fallback enums if Automodel is not available
    class SerializationFormat:
        SAFETENSORS = "safetensors"
        TORCH_SAVE = "torch_save"

PathLike = Union[str, "os.PathLike[Any]"]


class EnhancedCheckpointingConfig(TypedDict):
    """Enhanced configuration for checkpoint management with safetensors support.

    Attributes:
        enabled (bool): Whether checkpointing is enabled.
        checkpoint_dir (PathLike): Directory where checkpoints will be saved.
        metric_name (str): Name of the metric to use for determining best checkpoints.
        higher_is_better (bool): Whether higher values of the metric indicate better performance.
        save_period (int): How often to save checkpoints (in steps).
        keep_top_k (Optional[int]): Number of best checkpoints to keep. If None, all checkpoints are kept.
        model_save_format (str): Format for saving model weights ('safetensors' or 'torch_save').
        save_consolidated (bool): Whether to save consolidated HF-compatible checkpoints.
        model_cache_dir (Optional[str]): Cache directory for model artifacts.
        model_repo_id (Optional[str]): Repository ID for the base model.
    """
    enabled: bool
    checkpoint_dir: PathLike
    metric_name: str
    higher_is_better: bool
    save_period: int
    keep_top_k: NotRequired[int]
    model_save_format: NotRequired[str]  # 'safetensors' or 'torch_save'
    save_consolidated: NotRequired[bool]
    model_cache_dir: NotRequired[str]
    model_repo_id: NotRequired[str]


class EnhancedCheckpointManager:
    """Enhanced checkpoint manager with safetensors support.
    
    This manager extends the original nemo-rl checkpoint functionality with:
    - Safetensors support for HuggingFace compatibility
    - Consolidated checkpoint generation
    - Backward compatibility with existing nemo-rl checkpoints
    
    The checkpointing structure supports both formats:
    
    Legacy format (torch_save):
    ```
    checkpoint_dir/
        step_0/
            training_info.json
            config.yaml
            policy/
                weights/     (DCP format)
                optimizer/   (DCP format)
                tokenizer/   (HF format)
            train_dataloader.pt
    ```
    
    Enhanced format (safetensors):
    ```
    checkpoint_dir/
        step_0/
            training_info.json
            config.yaml
            model/
                consolidated/        (HF-compatible safetensors)
                shard-XX-model-...   (sharded safetensors)
            optim/                   (DCP format)
            policy/
                tokenizer/           (HF format)
            train_dataloader.pt
    ```
    """

    def __init__(self, config: EnhancedCheckpointingConfig):
        """Initialize the enhanced checkpoint manager.

        Args:
            config: Enhanced checkpointing configuration
        """
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.metric_name = config["metric_name"]
        self.higher_is_better = config["higher_is_better"]
        self.keep_top_k = config.get("keep_top_k")
        
        # Enhanced features
        self.model_save_format = config.get("model_save_format", "torch_save")
        self.save_consolidated = config.get("save_consolidated", False)
        self.model_cache_dir = config.get("model_cache_dir")
        self.model_repo_id = config.get("model_repo_id")
        
        # Validate safetensors availability
        if self.model_save_format == "safetensors" and not AUTOMODEL_AVAILABLE:
            warnings.warn(
                "Safetensors format requested but Automodel components not available. "
                "Falling back to torch_save format."
            )
            self.model_save_format = "torch_save"

    def init_tmp_checkpoint(
        self,
        step: int,
        training_info: Dict[str, Any],
        run_config: Optional[Dict[str, Any]] = None,
    ) -> PathLike:
        """Initialize a temporary checkpoint directory.

        Creates a temporary directory for a new checkpoint and saves training info
        and configuration. The directory is named 'tmp_step_{step}' and will be renamed
        to 'step_{step}' when the checkpoint is completed.

        Args:
            step: The training step number.
            training_info: Dictionary containing training metrics and info.
            run_config: Optional configuration for the training run.

        Returns:
            Path to the temporary checkpoint directory.
        """
        save_dir = self.checkpoint_dir / f"tmp_step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save training info
        with open(save_dir / "training_info.json", "w") as f:
            # Make any numpy items serializable
            for k, v in training_info.items():
                if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                    training_info[k] = v.item()
            json.dump(training_info, f)

        # Save config
        if run_config is not None:
            with open(save_dir / "config.yaml", "w") as f:
                yaml.safe_dump(run_config, f)

        return Path(os.path.abspath(save_dir))

    def save_model_safetensors(
        self,
        model: torch.nn.Module,
        checkpoint_path: PathLike,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Save model in safetensors format with optional consolidation.
        
        Args:
            model: The PyTorch model to save
            checkpoint_path: Path to the checkpoint directory
            tokenizer: Optional tokenizer to save
        """
        if not AUTOMODEL_AVAILABLE:
            raise RuntimeError("Automodel components required for safetensors format")
            
        checkpoint_path = Path(checkpoint_path)
        model_path = checkpoint_path / "model"
        model_path.mkdir(exist_ok=True)
        
        # Create model state using Automodel's wrapper
        model_state = AutomodelModelState(model, is_peft=False)
        model_state_dict = {"model": model_state}
        
        if self.save_consolidated:
            # Save both sharded and consolidated formats
            consolidated_model_path = model_path / "consolidated"
            consolidated_model_path.mkdir(exist_ok=True)
            
            # Get FQN to file mapping for sharding
            fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                model_state_dict,
                world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            )
            
            storage_writer = _HuggingFaceStorageWriter(
                path=str(model_path),
                save_sharded=True,
                consolidated_output_path=str(consolidated_model_path),
                fqn_to_index_mapping=fqn_to_file_index_mapping,
            )
            
            dcp.save(
                model_state_dict,
                checkpoint_id=str(model_path),
                storage_writer=storage_writer,
            )
            
            # Save tokenizer in consolidated directory for HF compatibility
            if tokenizer is not None:
                tokenizer.save_pretrained(consolidated_model_path)
        else:
            # Save only sharded safetensors
            storage_writer = _HuggingFaceStorageWriter(
                path=str(model_path),
                save_sharded=True,
                consolidated_output_path=None,
                fqn_to_index_mapping=None,
            )
            
            dcp.save(
                model_state_dict,
                checkpoint_id=str(model_path),
                storage_writer=storage_writer,
            )

    def save_model_legacy(
        self,
        model: torch.nn.Module,
        checkpoint_path: PathLike,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Save model in legacy DCP format for backward compatibility.
        
        Args:
            model: The PyTorch model to save
            checkpoint_path: Path to the checkpoint directory
            tokenizer: Optional tokenizer to save
        """
        checkpoint_path = Path(checkpoint_path)
        weights_path = checkpoint_path / "policy" / "weights"
        tokenizer_path = checkpoint_path / "policy" / "tokenizer"
        
        # Use the existing native checkpoint functionality
        from nemo_rl.utils.native_checkpoint import save_checkpoint as native_save
        
        native_save(
            model=model,
            weights_path=str(weights_path),
            tokenizer=tokenizer,
            tokenizer_path=str(tokenizer_path) if tokenizer else None,
        )

    def save_checkpoint_data(
        self,
        checkpoint_path: PathLike,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        dataloader_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model, optimizer, and training state to checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            model: The PyTorch model to save
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save  
            tokenizer: Optional tokenizer to save
            dataloader_state: Optional dataloader state to save
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Save model based on format
        if self.model_save_format == "safetensors":
            self.save_model_safetensors(model, checkpoint_path, tokenizer)
        else:
            self.save_model_legacy(model, checkpoint_path, tokenizer)
        
        # Save optimizer (always use DCP format for consistency)
        if optimizer is not None:
            optim_path = checkpoint_path / "optim"
            optim_path.mkdir(exist_ok=True)
            
            if AUTOMODEL_AVAILABLE:
                # Use Automodel's optimizer state wrapper
                optimizer_state = {"optim": AutomodelOptimizerState(model, optimizer, scheduler)}
                dcp.save(optimizer_state, checkpoint_id=str(optim_path))
            else:
                # Fallback to native checkpoint
                from nemo_rl.utils.native_checkpoint import save_checkpoint as native_save
                optimizer_path = checkpoint_path / "policy" / "optimizer" 
                native_save(
                    model=model,
                    weights_path="",  # Not saving weights here
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_path=str(optimizer_path),
                )
        
        # Save dataloader state
        if dataloader_state is not None:
            torch.save(dataloader_state, checkpoint_path / "train_dataloader.pt")

    def finalize_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Complete a checkpoint by moving it from temporary to permanent location.

        If a checkpoint at the target location already exists (i.e when resuming training),
        we override the old one. Also triggers cleanup of old checkpoints based on the 
        keep_top_k setting.

        Args:
            checkpoint_path: Path to the temporary checkpoint directory.
        """
        checkpoint_path = Path(checkpoint_path)
        to_checkpoint_path = (
            checkpoint_path.parent / f"step_{checkpoint_path.name.split('_')[2]}"
        )
        
        if to_checkpoint_path.exists():
            # Pseudo-atomic checkpoint save
            old_checkpoint_path = (
                checkpoint_path.parent
                / f"old_step_{checkpoint_path.name.split('_')[2]}"
            )
            os.rename(to_checkpoint_path, old_checkpoint_path)
            os.rename(checkpoint_path, to_checkpoint_path)
            # Delete old checkpoint
            if old_checkpoint_path.exists():
                shutil.rmtree(old_checkpoint_path)
        else:
            os.rename(checkpoint_path, to_checkpoint_path)
            
        self.remove_old_checkpoints()

    def remove_old_checkpoints(self, exclude_latest: bool = True) -> None:
        """Remove checkpoints that are not in the top-k based on metrics.

        Args:
            exclude_latest: Whether to exclude the latest checkpoint from deletion.
        """
        if self.keep_top_k is None:
            return
            
        checkpoint_history = self._load_checkpoint_history()
        latest_step = (
            max([step for step, _, _ in checkpoint_history])
            if checkpoint_history
            else None
        )

        if self.metric_name is None:
            checkpoint_history.sort(key=lambda x: x[0], reverse=True)
        else:
            try:
                if self.higher_is_better:
                    checkpoint_history.sort(
                        key=lambda x: (x[2][self.metric_name], x[0]), reverse=True
                    )
                else:
                    checkpoint_history.sort(
                        key=lambda x: (x[2][self.metric_name], -x[0])
                    )
            except KeyError:
                warnings.warn(
                    f"Metric {self.metric_name} not found in checkpoint history. "
                    "Keeping most recent k checkpoints."
                )
                checkpoint_history.sort(key=lambda x: x[0], reverse=True)
                self.metric_name = None

        # Remove checkpoints outside top-k
        for checkpoint in checkpoint_history[self.keep_top_k :]:
            if exclude_latest and checkpoint[0] == latest_step:
                continue
            print(f"Removing checkpoint {checkpoint[1]} (outside top-{self.keep_top_k})")
            shutil.rmtree(checkpoint[1])

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get the path to the best checkpoint based on the metric.

        Returns:
            Path to the best checkpoint, or None if no valid checkpoints exist.
        """
        checkpoint_history = self._load_checkpoint_history()
        if len(checkpoint_history) == 0:
            return None
            
        if self.metric_name not in checkpoint_history[0][2]:
            warnings.warn(
                f"Metric {self.metric_name} not found in checkpoint history. Returning latest"
            )
            return self.get_latest_checkpoint_path()

        checkpoint_history.sort(
            key=lambda x: x[2][self.metric_name], reverse=self.higher_is_better
        )
        return str(checkpoint_history[0][1])

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        step_dirs = glob.glob(str(self.checkpoint_dir / "step_*"))
        step_dirs.sort(key=lambda x: int(Path(x).name.split("_")[1]))
        if len(step_dirs) == 0:
            return None
        return str(step_dirs[-1])

    def get_consolidated_checkpoint_path(self, checkpoint_path: str) -> Optional[str]:
        """Get the path to the consolidated HF-compatible checkpoint if it exists.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            
        Returns:
            Path to consolidated checkpoint, or None if not available
        """
        consolidated_path = Path(checkpoint_path) / "model" / "consolidated"
        if consolidated_path.exists():
            return str(consolidated_path)
        return None

    def load_training_info(
        self, checkpoint_path: Optional[PathLike] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the training info from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint. If None, returns None.

        Returns:
            Dictionary containing the training info, or None if checkpoint_path is None.
        """
        if checkpoint_path is None:
            return None
        with open(Path(checkpoint_path) / "training_info.json", "r") as f:
            return json.load(f)

    def _load_checkpoint_history(self) -> list[tuple[int, PathLike, Dict[str, Any]]]:
        """Load the history of checkpoints and their metrics.

        Returns:
            List of tuples containing (step_number, checkpoint_path, info) for each checkpoint.
        """
        checkpoint_history = []
        step_dirs = glob.glob(str(self.checkpoint_dir / "step_*"))

        for step_dir in step_dirs:
            info_file = Path(step_dir) / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                    step = int(Path(step_dir).name.split("_")[1])
                    checkpoint_history.append((step, step_dir, info))

        return checkpoint_history


# Backward compatibility alias
CheckpointManager = EnhancedCheckpointManager
CheckpointingConfig = EnhancedCheckpointingConfig 