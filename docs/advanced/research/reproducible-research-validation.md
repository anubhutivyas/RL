---
description: "Comprehensive guide for ensuring reproducible research with NeMo RL including seed management, environment setup, experiment tracking, and result validation"
tags: ["reproducibility", "research", "versioning", "environment", "validation", "seeds", "experiment-tracking"]
categories: ["research-validation"]
---

# Reproducible Research

This comprehensive guide covers how to ensure reproducible results and scientific rigor in NeMo RL research through proper seed management, environment setup, experiment tracking, and result validation.

## Overview

Reproducible research is fundamental to scientific progress. This guide provides frameworks and best practices for ensuring that NeMo RL research can be reliably reproduced by other researchers through deterministic training, comprehensive environment management, and rigorous validation protocols.

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

#### Consistent Model Initialization
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

### Environment Management

#### Comprehensive Environment Capture
Implement comprehensive environment management:

```python
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import subprocess

class EnvironmentManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.environment_file = self.project_root / "environment.yml"
        self.requirements_file = self.project_root / "requirements.txt"
        self.docker_file = self.project_root / "Dockerfile"
        
    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture complete environment information
        """
        environment_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'architecture': sys.maxsize > 2**32 and "64 bit" or "32 bit",
            'installed_packages': self.get_installed_packages(),
            'environment_variables': self.get_environment_variables(),
            'hardware_info': self.get_hardware_info(),
            'gpu_info': self.get_gpu_info(),
            'timestamp': time.time()
        }
        
        return environment_info
    
    def get_installed_packages(self) -> Dict[str, str]:
        """
        Get installed package versions
        """
        try:
            import pkg_resources
            installed_packages = {}
            for dist in pkg_resources.working_set:
                installed_packages[dist.project_name] = dist.version
            return installed_packages
        except Exception as e:
            print(f"Error getting installed packages: {e}")
            return {}
    
    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get relevant environment variables
        """
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES',
            'PYTHONPATH',
            'LD_LIBRARY_PATH',
            'PATH',
            'HOME',
            'USER'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information
        """
        import psutil
        
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').total
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information
        """
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    'cuda_version': torch.version.cuda,
                    'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
                }
            else:
                return {'gpu_count': 0}
        except Exception as e:
            return {'error': str(e)}
    
    def create_conda_environment(self, env_name: str, python_version: str = "3.9"):
        """
        Create conda environment file
        """
        environment_content = {
            'name': env_name,
            'channels': ['conda-forge', 'defaults'],
            'dependencies': [
                f'python={python_version}',
                'pip',
                'pytorch',
                'torchvision',
                'cudatoolkit',
                'numpy',
                'scipy',
                'matplotlib',
                'jupyter',
                'pip'
            ]
        }
        
        with open(self.environment_file, 'w') as f:
            yaml.dump(environment_content, f, default_flow_style=False)
        
        print(f"Conda environment file created: {self.environment_file}")
    
    def create_requirements_file(self, packages: List[str]):
        """
        Create requirements.txt file with exact versions
        """
        requirements_content = [
            "torch==2.0.0+cu118",
            "transformers==4.30.0",
            "numpy==1.24.0",
            "scipy==1.10.0",
            "scikit-learn==1.3.0",
            "matplotlib==3.7.0",
            "seaborn==0.12.0",
            "pandas==2.0.0",
            "tqdm==4.65.0",
            "wandb==0.15.0",
            "tensorboard==2.13.0",
            "ray==2.6.0",
            "optuna==3.2.0"
        ]
        
        with open(self.requirements_file, 'w') as f:
            f.write('\n'.join(requirements_content))
        
        print(f"Requirements file created: {self.requirements_file}")
    
    def create_dockerfile(self, base_image: str = "pytorch/pytorch:latest"):
        """
        Create Dockerfile for containerization
        """
        dockerfile_content = f"""FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDA_LAUNCH_BLOCKING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m -u 1000 researcher
RUN chown -R researcher:researcher /app
USER researcher

# Default command
CMD ["python", "main.py"]
"""
        
        with open(self.docker_file, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"Dockerfile created: {self.docker_file}")
    
    def save_environment_snapshot(self, filename: str = "environment_snapshot.json"):
        """
        Save environment snapshot
        """
        environment_info = self.capture_environment()
        
        snapshot_file = self.project_root / filename
        with open(snapshot_file, 'w') as f:
            json.dump(environment_info, f, indent=2)
        
        print(f"Environment snapshot saved: {snapshot_file}")
        return snapshot_file
    
    def verify_environment_compatibility(self, snapshot_file: str = "environment_snapshot.json") -> Dict[str, Any]:
        """
        Verify current environment matches snapshot
        """
        snapshot = self.load_environment_snapshot(snapshot_file)
        current_env = self.capture_environment()
        
        compatibility_report = {
            'python_version_match': snapshot['python_version'] == current_env['python_version'],
            'platform_match': snapshot['platform'] == current_env['platform'],
            'package_matches': {},
            'gpu_compatibility': snapshot['gpu_info'] == current_env['gpu_info'],
            'overall_compatible': True
        }
        
        # Check package versions
        for package, version in snapshot['installed_packages'].items():
            current_version = current_env['installed_packages'].get(package)
            compatibility_report['package_matches'][package] = version == current_version
            if version != current_version:
                compatibility_report['overall_compatible'] = False
        
        return compatibility_report
```

### Experiment Tracking

#### Comprehensive Experiment Management
Implement comprehensive experiment tracking:

```python
import git
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentTracker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.experiments_dir = self.project_root / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
    def create_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Create new experiment
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Capture git information
        git_info = self.capture_git_info()
        git_file = experiment_dir / "git_info.json"
        with open(git_file, 'w') as f:
            json.dump(git_info, f, indent=2)
        
        # Create experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'created_at': timestamp,
            'status': 'created',
            'config_file': str(config_file),
            'git_info': git_info
        }
        
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Experiment created: {experiment_id}")
        return experiment_id
    
    def capture_git_info(self) -> Dict[str, Any]:
        """
        Capture git repository information
        """
        try:
            repo = git.Repo(self.project_root)
            return {
                'commit_hash': repo.head.commit.hexsha,
                'branch': repo.active_branch.name,
                'is_dirty': repo.is_dirty(),
                'remote_url': repo.remotes.origin.url if repo.remotes else None,
                'commit_message': repo.head.commit.message,
                'commit_date': repo.head.commit.committed_datetime.isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def log_experiment_result(self, experiment_id: str, result: Dict[str, Any]):
        """
        Log experiment result
        """
        experiment_dir = self.experiments_dir / experiment_id
        
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Save result
        result_file = experiment_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update metadata
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = 'completed'
        metadata['completed_at'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata['result_file'] = str(result_file)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Result logged for experiment: {experiment_id}")
    
    def get_experiment_info(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment information
        """
        experiment_dir = self.experiments_dir / experiment_id
        
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load metadata
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_file = experiment_dir / "config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load result if exists
        result_file = experiment_dir / "result.json"
        result = None
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
        
        return {
            'metadata': metadata,
            'config': config,
            'result': result
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments
        """
        experiments = []
        
        for experiment_dir in self.experiments_dir.iterdir():
            if experiment_dir.is_dir():
                metadata_file = experiment_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
    
    def reproduce_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Reproduce experiment
        """
        experiment_info = self.get_experiment_info(experiment_id)
        
        # Create new experiment for reproduction
        reproduction_id = self.create_experiment(
            f"reproduction_{experiment_id}",
            experiment_info['config']
        )
        
        # Run experiment with same configuration
        result = self.run_experiment(experiment_info['config'])
        
        # Log result
        self.log_experiment_result(reproduction_id, result)
        
        # Compare results
        comparison = self.compare_experiments(experiment_id, reproduction_id)
        
        return {
            'original_experiment': experiment_id,
            'reproduction_experiment': reproduction_id,
            'result': result,
            'comparison': comparison
        }
    
    def compare_experiments(self, experiment1_id: str, experiment2_id: str) -> Dict[str, Any]:
        """
        Compare two experiments
        """
        exp1_info = self.get_experiment_info(experiment1_id)
        exp2_info = self.get_experiment_info(experiment2_id)
        
        comparison = {
            'config_match': exp1_info['config'] == exp2_info['config'],
            'git_info_match': exp1_info['metadata']['git_info'] == exp2_info['metadata']['git_info'],
            'result_comparison': {}
        }
        
        # Compare results if both exist
        if exp1_info['result'] and exp2_info['result']:
            for key in exp1_info['result'].keys():
                if key in exp2_info['result']:
                    val1 = exp1_info['result'][key]
                    val2 = exp2_info['result'][key]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        comparison['result_comparison'][key] = {
                            'difference': abs(val1 - val2),
                            'relative_difference': abs(val1 - val2) / max(abs(val1), abs(val2)) if max(abs(val1), abs(val2)) > 0 else 0
                        }
                    else:
                        comparison['result_comparison'][key] = {
                            'match': val1 == val2
                        }
        
        return comparison
```

### Data Versioning and Management

#### Dataset Version Control
Implement data versioning and management:

```python
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

class DataVersionManager:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.version_file = self.data_root / "versions.json"
        self.load_versions()
    
    def load_versions(self):
        """
        Load version information
        """
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def save_versions(self):
        """
        Save version information
        """
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def version_dataset(self, dataset_path: str, description: str = "") -> str:
        """
        Version a dataset
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Calculate dataset hash
        if dataset_path.is_file():
            dataset_hash = self.calculate_file_hash(dataset_path)
        else:
            # For directories, hash all files
            hashes = []
            for file_path in dataset_path.rglob("*"):
                if file_path.is_file():
                    hashes.append(self.calculate_file_hash(file_path))
            dataset_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
        
        # Create version entry
        version_id = f"v{len(self.versions) + 1}"
        version_info = {
            'version_id': version_id,
            'dataset_path': str(dataset_path),
            'dataset_hash': dataset_hash,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'file_size': self.get_directory_size(dataset_path) if dataset_path.is_dir() else dataset_path.stat().st_size
        }
        
        self.versions[version_id] = version_info
        self.save_versions()
        
        print(f"Dataset versioned: {version_id}")
        return version_id
    
    def get_directory_size(self, directory: Path) -> int:
        """
        Calculate total size of directory
        """
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def get_dataset_version(self, version_id: str) -> Dict[str, Any]:
        """
        Get dataset version information
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        return self.versions[version_id]
    
    def list_dataset_versions(self) -> List[Dict[str, Any]]:
        """
        List all dataset versions
        """
        return list(self.versions.values())
    
    def verify_dataset_integrity(self, version_id: str) -> bool:
        """
        Verify dataset integrity
        """
        version_info = self.get_dataset_version(version_id)
        dataset_path = Path(version_info['dataset_path'])
        
        if not dataset_path.exists():
            return False
        
        # Recalculate hash
        if dataset_path.is_file():
            current_hash = self.calculate_file_hash(dataset_path)
        else:
            hashes = []
            for file_path in dataset_path.rglob("*"):
                if file_path.is_file():
                    hashes.append(self.calculate_file_hash(file_path))
            current_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
        
        return current_hash == version_info['dataset_hash']
```

### Result Validation and Verification

#### Multiple Run Validation
Implement result validation and verification:

```python
class ResultValidator:
    def __init__(self, tolerance_threshold: float = 0.01):
        self.tolerance_threshold = tolerance_threshold
    
    def validate_numerical_results(self, expected: Dict[str, float], 
                                 actual: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate numerical results
        """
        validation_results = {}
        
        for key in expected.keys():
            if key in actual:
                expected_val = expected[key]
                actual_val = actual[key]
                
                if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                    relative_error = abs(expected_val - actual_val) / max(abs(expected_val), 1e-8)
                    validation_results[key] = {
                        'expected': expected_val,
                        'actual': actual_val,
                        'relative_error': relative_error,
                        'within_tolerance': relative_error <= self.tolerance_threshold
                    }
                else:
                    validation_results[key] = {
                        'expected': expected_val,
                        'actual': actual_val,
                        'match': expected_val == actual_val
                    }
        
        return validation_results
    
    def perform_cross_validation(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform cross-validation of results
        """
        if len(experiment_results) < 2:
            return {'error': 'Need at least 2 experiments for cross-validation'}
        
        # Extract common metrics
        common_metrics = set(experiment_results[0].keys())
        for result in experiment_results[1:]:
            common_metrics = common_metrics.intersection(set(result.keys()))
        
        cross_validation_results = {}
        
        for metric in common_metrics:
            values = [result[metric] for result in experiment_results if isinstance(result[metric], (int, float))]
            
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                
                cross_validation_results[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'coefficient_of_variation': cv,
                    'stable': cv <= 0.1  # Consider stable if CV <= 10%
                }
        
        return cross_validation_results

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

## Configuration

### Reproducibility Configuration

```yaml
# configs/reproducibility.yaml
reproducibility:
  enabled: true
  
  # Environment management
  environment:
    capture_snapshot: true
    verify_compatibility: true
    create_conda_env: true
    create_dockerfile: true
  
  # Experiment tracking
  experiment_tracking:
    enabled: true
    experiments_dir: "experiments"
    auto_save_results: true
    git_integration: true
  
  # Data versioning
  data_versioning:
    enabled: true
    data_root: "data"
    auto_version_datasets: true
    verify_integrity: true
  
  # Random seed management
  random_seeds:
    python: 42
    numpy: 42
    torch: 42
    cuda: 42
  
  # Logging
  logging:
    level: "INFO"
    format: "json"
    include_timestamps: true
    log_experiment_steps: true
```

## Best Practices

### 1. Complete Reproducibility Pipeline

Implement a complete reproducibility pipeline:

```python
class ReproducibilityPipeline:
    def __init__(self, project_root: str, config: Dict[str, Any]):
        self.project_root = Path(project_root)
        self.config = config
        
        # Initialize components
        self.env_manager = EnvironmentManager(project_root)
        self.exp_tracker = ExperimentTracker(project_root)
        self.data_manager = DataVersionManager(project_root / "data")
        self.result_validator = ResultValidator(
            tolerance_threshold=config.get('tolerance_threshold', 0.01)
        )
    
    def setup_reproducible_environment(self):
        """
        Setup reproducible environment
        """
        # Capture environment snapshot
        snapshot_file = self.env_manager.save_environment_snapshot()
        
        # Create conda environment
        if self.config.get('create_conda_env', True):
            self.env_manager.create_conda_environment("nemo-rl", "3.9")
        
        # Create Dockerfile
        if self.config.get('create_dockerfile', True):
            self.env_manager.create_dockerfile()
        
        # Create requirements file
        self.env_manager.create_requirements_file([
            'torch', 'transformers', 'numpy', 'scipy', 'matplotlib', 'jupyter'
        ])
        
        print("Reproducible environment setup completed")
    
    def run_reproducible_experiment(self, experiment_name: str, 
                                  config: Dict[str, Any],
                                  dataset_path: str = None) -> str:
        """
        Run reproducible experiment
        """
        # Version dataset if provided
        dataset_version = None
        if dataset_path:
            dataset_version = self.data_manager.version_dataset(
                dataset_path, f"Dataset for {experiment_name}"
            )
        
        # Create experiment
        experiment_id = self.exp_tracker.create_experiment(experiment_name, config)
        
        # Set random seeds
        random_seeds = self.config.get('random_seeds', {})
        if 'python' in random_seeds:
            random.seed(random_seeds['python'])
        if 'numpy' in random_seeds:
            np.random.seed(random_seeds['numpy'])
        if 'torch' in random_seeds:
            torch.manual_seed(random_seeds['torch'])
        
        # Run experiment
        try:
            result = self.run_experiment(config)
            
            # Add metadata
            result['experiment_id'] = experiment_id
            result['dataset_version'] = dataset_version
            result['random_seeds'] = random_seeds
            result['environment_snapshot'] = self.env_manager.capture_environment()
            
            # Log result
            self.exp_tracker.log_experiment_result(experiment_id, result)
            
            print(f"Experiment completed: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            raise
```

### 2. Automated Reproducibility Testing

Implement automated reproducibility testing:

```python
class AutomatedReproducibilityTester:
    def __init__(self, pipeline: ReproducibilityPipeline):
        self.pipeline = pipeline
        self.test_results = []
    
    def run_reproducibility_tests(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        Run automated reproducibility tests
        """
        test_config = {
            'model_type': 'test_model',
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 2
        }
        
        # Run multiple experiments
        experiment_ids = []
        for i in range(num_runs):
            experiment_id = self.pipeline.run_reproducible_experiment(
                f"reproducibility_test_{i+1}", test_config
            )
            experiment_ids.append(experiment_id)
        
        # Compare results
        comparison_results = {}
        for i in range(len(experiment_ids)):
            for j in range(i+1, len(experiment_ids)):
                comparison = self.pipeline.exp_tracker.compare_experiments(
                    experiment_ids[i], experiment_ids[j]
                )
                comparison_results[f"run_{i+1}_vs_run_{j+1}"] = comparison
        
        # Validate results
        validation_results = self.pipeline.result_validator.perform_cross_validation([
            self.pipeline.exp_tracker.get_experiment_info(exp_id)['result']
            for exp_id in experiment_ids
        ])
        
        return {
            'experiment_ids': experiment_ids,
            'comparison_results': comparison_results,
            'validation_results': validation_results,
            'overall_reproducible': self.assess_reproducibility(comparison_results, validation_results)
        }
    
    def assess_reproducibility(self, comparison_results: Dict[str, Any],
                             validation_results: Dict[str, Any]) -> bool:
        """
        Assess overall reproducibility
        """
        # Check if all comparisons show good agreement
        all_comparable = True
        for comparison in comparison_results.values():
            if not comparison.get('config_match', False):
                all_comparable = False
                break
        
        # Check if results are stable
        all_stable = True
        for metric_result in validation_results.values():
            if not metric_result.get('stable', False):
                all_stable = False
                break
        
        return all_comparable and all_stable
```

## Troubleshooting

### Common Reproducibility Issues

1. **Environment Differences**: Use containerization and environment snapshots
2. **Random Seed Issues**: Ensure all random seeds are properly set
3. **Data Versioning**: Implement proper data versioning and integrity checks

### Debugging Tips

```python
# Add debugging to reproducibility
def debug_reproducibility_issues(self):
    """
    Debug reproducibility issues
    """
    print("=== Reproducibility Debug ===")
    
    # Check environment
    env_info = self.env_manager.capture_environment()
    print(f"Python version: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']}")
    print(f"GPU info: {env_info['gpu_info']}")
    
    # Check git status
    git_info = self.exp_tracker.capture_git_info()
    print(f"Git commit: {git_info['commit_hash']}")
    print(f"Git dirty: {git_info['is_dirty']}")
    
    # Check data integrity
    data_versions = self.data_manager.list_dataset_versions()
    print(f"Number of dataset versions: {len(data_versions)}")
    
    # Check experiments
    experiments = self.exp_tracker.list_experiments()
    print(f"Number of experiments: {len(experiments)}")
    
    print("============================")
```

## Next Steps

- Learn about [Experimental Design](experimental-design-validation) for rigorous research
- Review [Model Evaluation](model-evaluation-validation) for comprehensive assessment
- Explore [Algorithm Development](../algorithm-development/index) for advanced training 