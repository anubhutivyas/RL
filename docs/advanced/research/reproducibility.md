---
description: "Best practices for ensuring reproducible research including seed management and environment setup."
tags: ["reproducibility", "research", "best practices", "seeds"]
categories: ["research"]
---

# Reproducibility

This document provides comprehensive guidelines for ensuring reproducible research with NeMo RL, including seed management, environment setup, and result validation.

## Overview

Reproducibility is fundamental to scientific research. This guide covers deterministic training, environment management, and comprehensive validation protocols to ensure reproducible experiments.

## Key Components

### Seed Management

#### Deterministic Random Number Generation
Ensure consistent random number generation across all components:

```python
import torch
import numpy as np
import random
import os

def set_deterministic_seeds(seed=42, rank=0):
    """
    Set all random seeds for reproducible training
    
    Args:
        seed: Base seed value
        rank: Process rank for distributed training
    """
    # Set base seeds
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"Set deterministic seeds: base={seed}, rank={rank}")

def verify_determinism(model, dataloader, num_steps=10):
    """
    Verify that training is deterministic
    
    Args:
        model: Model to test
        dataloader: Data loader for testing
        num_steps: Number of training steps to test
    """
    # Run training twice with same seed
    set_deterministic_seeds(42)
    outputs1 = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        with torch.no_grad():
            output = model(batch)
            outputs1.append(output.cpu().numpy())
    
    # Reset and run again
    set_deterministic_seeds(42)
    outputs2 = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        with torch.no_grad():
            output = model(batch)
            outputs2.append(output.cpu().numpy())
    
    # Compare outputs
    for i, (out1, out2) in enumerate(zip(outputs1, outputs2)):
        if not np.allclose(out1, out2, rtol=1e-5, atol=1e-5):
            print(f"Non-deterministic output at step {i}")
            return False
    
    print("Training is deterministic")
    return True
```

#### Consistent Initialization
Ensure model initialization is deterministic:

```python
def deterministic_model_init(model, seed=42):
    """
    Initialize model weights deterministically
    
    Args:
        model: Model to initialize
        seed: Seed for initialization
    """
    set_deterministic_seeds(seed)
    
    # Initialize weights deterministically
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.xavier_uniform_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    return model

def save_model_state(model, optimizer, epoch, filename):
    """
    Save complete model state including random states
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        filename: Output filename
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    
    torch.save(state, filename)

def load_model_state(model, optimizer, filename):
    """
    Load complete model state including random states
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filename: State file to load
    """
    state = torch.load(filename, map_location='cpu')
    
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    # Restore random states
    torch.set_rng_state(state['torch_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    random.setstate(state['python_rng_state'])
    
    if state['cuda_rng_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    
    return state['epoch']
```

### Environment Setup

#### Containerized Environments
Use Docker for consistent environments:

```dockerfile
# Dockerfile for NeMo RL environment
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy NeMo RL code
COPY . /workspace/

# Set environment variables for reproducibility
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDA_LAUNCH_BLOCKING=1

# Default command
CMD ["python3"]
```

#### Dependency Management
Pin exact versions for reproducibility:

```python
# requirements.txt with exact versions
torch==2.0.0+cu118
transformers==4.30.0
numpy==1.24.0
scipy==1.10.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
pandas==2.0.0
tqdm==4.65.0
wandb==0.15.0
tensorboard==2.13.0
ray==2.6.0
optuna==3.2.0
```

#### Environment Validation
Verify environment consistency:

```python
def validate_environment():
    """
    Validate that environment is consistent with requirements
    """
    import sys
    import subprocess
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor != 9:
        raise ValueError(f"Python 3.9 required, got {python_version}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available")
    else:
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
    
    # Check package versions
    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.24.0'
    }
    
    for package, required_version in required_packages.items():
        try:
            module = __import__(package)
            actual_version = getattr(module, '__version__', 'unknown')
            if actual_version != required_version:
                print(f"Warning: {package} version mismatch. "
                      f"Required: {required_version}, Got: {actual_version}")
        except ImportError:
            print(f"Warning: {package} not installed")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")
    
    print("Environment validation complete")

def create_environment_snapshot():
    """
    Create a snapshot of the current environment
    """
    import subprocess
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save pip freeze
    with open(f"environment_{timestamp}.txt", "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)
    
    # Save system info
    with open(f"system_info_{timestamp}.txt", "w") as f:
        subprocess.run(["nvidia-smi"], stdout=f)
        subprocess.run(["python", "-c", "import torch; print(torch.__version__)"], stdout=f)
    
    print(f"Environment snapshot saved: environment_{timestamp}.txt")
```

### Data Versioning

#### Dataset Version Control
Implement dataset versioning:

```python
import hashlib
import json
from pathlib import Path

class DatasetVersioning:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.version_file = self.dataset_path / "version_info.json"
    
    def calculate_dataset_hash(self):
        """
        Calculate SHA256 hash of dataset files
        """
        hasher = hashlib.sha256()
        
        for file_path in sorted(self.dataset_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def save_version_info(self, metadata):
        """
        Save dataset version information
        
        Args:
            metadata: Additional metadata about the dataset
        """
        version_info = {
            "dataset_hash": self.calculate_dataset_hash(),
            "creation_date": datetime.datetime.now().isoformat(),
            "metadata": metadata
        }
        
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
    
    def verify_dataset_integrity(self):
        """
        Verify dataset hasn't changed since version was saved
        """
        if not self.version_file.exists():
            print("No version info found")
            return False
        
        with open(self.version_file, 'r') as f:
            version_info = json.load(f)
        
        current_hash = self.calculate_dataset_hash()
        saved_hash = version_info["dataset_hash"]
        
        if current_hash != saved_hash:
            print(f"Dataset hash mismatch: {current_hash} != {saved_hash}")
            return False
        
        print("Dataset integrity verified")
        return True
```

#### Preprocessing Pipeline Documentation
Document all data preprocessing steps:

```python
class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.preprocessing_steps = []
    
    def add_step(self, step_name, step_function, parameters):
        """
        Add a preprocessing step
        
        Args:
            step_name: Name of the step
            step_function: Function to apply
            parameters: Parameters for the function
        """
        self.preprocessing_steps.append({
            "name": step_name,
            "function": step_function,
            "parameters": parameters
        })
    
    def apply_pipeline(self, data):
        """
        Apply all preprocessing steps
        
        Args:
            data: Input data
        """
        processed_data = data
        
        for step in self.preprocessing_steps:
            print(f"Applying step: {step['name']}")
            processed_data = step['function'](processed_data, **step['parameters'])
        
        return processed_data
    
    def save_pipeline_config(self, filename):
        """
        Save pipeline configuration for reproducibility
        
        Args:
            filename: Output filename
        """
        config = {
            "steps": [
                {
                    "name": step["name"],
                    "function": step["function"].__name__,
                    "parameters": step["parameters"]
                }
                for step in self.preprocessing_steps
            ],
            "config": self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
```

## Best Practices

### Code Reproducibility

#### Version Control Integration
Integrate with Git for code versioning:

```python
import git
import os

def get_git_info():
    """
    Get current Git repository information
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "commit_hash": repo.head.object.hexsha,
            "branch": repo.active_branch.name,
            "is_dirty": repo.is_dirty(),
            "remote_url": repo.remotes.origin.url if repo.remotes else None
        }
    except git.InvalidGitRepositoryError:
        return {"error": "Not a Git repository"}

def save_experiment_info(experiment_name, config, results):
    """
    Save complete experiment information
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        results: Experiment results
    """
    experiment_info = {
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_info": get_git_info(),
        "config": config,
        "results": results,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    filename = f"experiments/{experiment_name}_info.json"
    os.makedirs("experiments", exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"Experiment info saved: {filename}")
```

#### Configuration Management
Use structured configuration management:

```python
import yaml
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    """Structured experiment configuration"""
    # Model parameters
    model_name: str
    model_size: str
    
    # Training parameters
    learning_rate: float
    batch_size: int
    num_epochs: int
    
    # Algorithm parameters
    algorithm: str
    epsilon: float = 0.2  # for GRPO
    beta: float = 0.2     # for DPO
    
    # Reproducibility parameters
    seed: int = 42
    deterministic: bool = True
    
    def save_config(self, filename):
        """Save configuration to file"""
        with open(filename, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, filename):
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

### Result Validation

#### Multiple Independent Runs
Implement multiple run validation:

```python
class MultiRunValidation:
    def __init__(self, n_runs=5):
        self.n_runs = n_runs
        self.results = []
    
    def run_experiment(self, experiment_function, config):
        """
        Run experiment multiple times
        
        Args:
            experiment_function: Function to run
            config: Experiment configuration
        """
        for run in range(self.n_runs):
            print(f"Run {run + 1}/{self.n_runs}")
            
            # Set different seed for each run
            run_config = config.copy()
            run_config["seed"] = config["seed"] + run
            
            # Run experiment
            result = experiment_function(run_config)
            self.results.append(result)
    
    def calculate_statistics(self):
        """
        Calculate statistical measures across runs
        """
        import numpy as np
        from scipy import stats
        
        # Extract primary metric from results
        primary_metrics = [result["primary_metric"] for result in self.results]
        
        statistics = {
            "mean": np.mean(primary_metrics),
            "std": np.std(primary_metrics),
            "min": np.min(primary_metrics),
            "max": np.max(primary_metrics),
            "median": np.median(primary_metrics),
            "confidence_interval": stats.t.interval(
                0.95, len(primary_metrics)-1, 
                loc=np.mean(primary_metrics), 
                scale=stats.sem(primary_metrics)
            )
        }
        
        return statistics
    
    def check_reproducibility(self, tolerance=0.01):
        """
        Check if results are reproducible within tolerance
        
        Args:
            tolerance: Acceptable standard deviation as fraction of mean
        """
        primary_metrics = [result["primary_metric"] for result in self.results]
        mean_metric = np.mean(primary_metrics)
        std_metric = np.std(primary_metrics)
        
        reproducibility_score = std_metric / mean_metric
        
        is_reproducible = reproducibility_score < tolerance
        
        return {
            "is_reproducible": is_reproducible,
            "reproducibility_score": reproducibility_score,
            "tolerance": tolerance,
            "mean": mean_metric,
            "std": std_metric
        }
```

#### Cross-Validation Strategies
Implement robust validation:

```python
class CrossValidation:
    def __init__(self, n_folds=5, n_repeats=3):
        self.n_folds = n_folds
        self.n_repeats = n_repeats
    
    def stratified_kfold(self, data, labels):
        """
        Perform stratified k-fold cross-validation
        
        Args:
            data: Input data
            labels: Target labels
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            train_data, val_data = data[train_idx], data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Train model on this fold
            model = self.train_model(train_data, train_labels)
            
            # Evaluate on validation set
            result = self.evaluate_model(model, val_data, val_labels)
            fold_results.append(result)
        
        return fold_results
    
    def repeated_cross_validation(self, data, labels):
        """
        Perform repeated cross-validation
        
        Args:
            data: Input data
            labels: Target labels
        """
        all_results = []
        
        for repeat in range(self.n_repeats):
            print(f"Repeat {repeat + 1}/{self.n_repeats}")
            
            # Set different seed for each repeat
            np.random.seed(42 + repeat)
            
            fold_results = self.stratified_kfold(data, labels)
            all_results.extend(fold_results)
        
        return all_results
```

## Implementation Guidelines

### Environment Management

#### Automated Environment Setup
Create automated setup scripts:

```python
#!/usr/bin/env python3
"""
Automated environment setup script for NeMo RL
"""

import subprocess
import sys
import os

def setup_environment():
    """Setup reproducible environment"""
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Verify CUDA installation
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.version.cuda}")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed")
    
    # Create necessary directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("Environment setup complete")

if __name__ == "__main__":
    setup_environment()
```

#### Experiment Tracking
Implement comprehensive experiment tracking:

```python
import wandb
import tensorboard
from datetime import datetime

class ExperimentTracker:
    def __init__(self, project_name, experiment_name, config):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=["nemo-rl", "reproducibility"]
        )
        
        # Initialize tensorboard
        self.tensorboard_dir = f"logs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.tensorboard_dir, exist_ok=True)
    
    def log_metrics(self, metrics, step):
        """
        Log metrics to both wandb and tensorboard
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
        """
        # Log to wandb
        wandb.log(metrics, step=step)
        
        # Log to tensorboard
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(self.tensorboard_dir) as writer:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, step)
    
    def log_model(self, model, step):
        """
        Log model checkpoint
        
        Args:
            model: Model to log
            step: Current step
        """
        checkpoint_path = f"checkpoints/{self.experiment_name}_step_{step}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Log to wandb
        wandb.save(checkpoint_path)
    
    def finish(self):
        """Finish experiment tracking"""
        wandb.finish()
```

### Validation Protocols

#### Independent Verification
Implement independent verification procedures:

```python
class IndependentVerification:
    def __init__(self, reference_results):
        self.reference_results = reference_results
    
    def verify_results(self, new_results, tolerance=0.01):
        """
        Verify new results against reference results
        
        Args:
            new_results: New experiment results
            tolerance: Acceptable difference tolerance
        """
        verification_report = {}
        
        for metric_name in self.reference_results.keys():
            if metric_name in new_results:
                ref_value = self.reference_results[metric_name]
                new_value = new_results[metric_name]
                
                relative_diff = abs(new_value - ref_value) / abs(ref_value)
                
                verification_report[metric_name] = {
                    "reference": ref_value,
                    "new": new_value,
                    "relative_difference": relative_diff,
                    "within_tolerance": relative_diff < tolerance
                }
        
        return verification_report
    
    def generate_verification_report(self, verification_results):
        """
        Generate human-readable verification report
        
        Args:
            verification_results: Results from verify_results
        """
        report = []
        report.append("Independent Verification Report")
        report.append("=" * 40)
        
        all_within_tolerance = True
        
        for metric_name, result in verification_results.items():
            status = "✓" if result["within_tolerance"] else "✗"
            report.append(f"{status} {metric_name}:")
            report.append(f"  Reference: {result['reference']:.6f}")
            report.append(f"  New: {result['new']:.6f}")
            report.append(f"  Relative difference: {result['relative_difference']:.4f}")
            report.append("")
            
            if not result["within_tolerance"]:
                all_within_tolerance = False
        
        report.append(f"Overall status: {'PASS' if all_within_tolerance else 'FAIL'}")
        
        return "\n".join(report)
```

## Next Steps

After implementing reproducibility practices:

1. **Set Up Environment**: Use containerized environments with pinned dependencies
2. **Implement Seed Management**: Ensure deterministic training across all components
3. **Version Control Data**: Implement dataset versioning and integrity checks
4. **Track Experiments**: Use comprehensive logging and experiment tracking
5. **Validate Results**: Perform multiple runs and independent verification

## References

- Peng, R.D. "Reproducible Research in Computational Science." Science (2011).
- Stodden, V., et al. "Enhancing reproducibility for computational research." Science (2016).
- Sandve, G.K., et al. "Ten Simple Rules for Reproducible Computational Research." PLoS Comput Biol (2013).
- Wilson, G., et al. "Good Enough Practices in Scientific Computing." PLoS Comput Biol (2017). 