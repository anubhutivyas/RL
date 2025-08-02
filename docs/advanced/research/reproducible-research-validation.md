---
description: "Ensure reproducible results and scientific rigor with proper versioning, environment management, and result validation"
tags: ["reproducibility", "research", "versioning", "environment", "validation"]
categories: ["research-validation"]
---

# Reproducible Research

This guide covers how to ensure reproducible results and scientific rigor in NeMo RL research through proper versioning, environment management, and result validation.

## Overview

Reproducible research is fundamental to scientific progress. This guide provides frameworks and best practices for ensuring that NeMo RL research can be reliably reproduced by other researchers.

## Key Components

### Environment Management

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
        Create requirements.txt file
        """
        requirements_content = []
        
        for package in packages:
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package).version
                requirements_content.append(f"{package}=={version}")
            except Exception:
                requirements_content.append(package)
        
        with open(self.requirements_file, 'w') as f:
            f.write('\n'.join(requirements_content))
        
        print(f"Requirements file created: {self.requirements_file}")
    
    def create_dockerfile(self, base_image: str = "pytorch/pytorch:latest"):
        """
        Create Dockerfile for containerization
        """
        dockerfile_content = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

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
    
    def load_environment_snapshot(self, filename: str = "environment_snapshot.json") -> Dict[str, Any]:
        """
        Load environment snapshot
        """
        snapshot_file = self.project_root / filename
        
        if snapshot_file.exists():
            with open(snapshot_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Environment snapshot not found: {snapshot_file}")
    
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

### Version Control and Experiment Tracking

Implement comprehensive version control and experiment tracking:

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
    
    def export_dataset_version(self, version_id: str, export_path: str):
        """
        Export dataset version
        """
        version_info = self.get_dataset_version(version_id)
        source_path = Path(version_info['dataset_path'])
        export_path = Path(export_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset not found: {source_path}")
        
        # Copy dataset
        if source_path.is_file():
            shutil.copy2(source_path, export_path)
        else:
            shutil.copytree(source_path, export_path, dirs_exist_ok=True)
        
        # Save version info
        version_info_file = export_path / "version_info.json"
        with open(version_info_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"Dataset version {version_id} exported to: {export_path}")
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

### Advanced Reproducibility Configuration

```yaml
# configs/advanced_reproducibility.yaml
reproducibility:
  # Containerization
  containerization:
    enabled: true
    base_image: "pytorch/pytorch:latest"
    include_gpu_support: true
    mount_data_volumes: true
  
  # Dependency management
  dependency_management:
    pin_versions: true
    create_requirements_file: true
    create_environment_file: true
    verify_dependencies: true
  
  # Result validation
  result_validation:
    enabled: true
    tolerance_threshold: 0.01
    statistical_tests: true
    cross_validation: true
  
  # Documentation
  documentation:
    auto_generate_readme: true
    include_setup_instructions: true
    include_usage_examples: true
    include_troubleshooting: true
```

## Advanced Reproducibility Techniques

### Containerization for Reproducibility

Implement containerization for maximum reproducibility:

```python
class ContainerManager:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dockerfile_path = self.project_root / "Dockerfile"
        self.docker_compose_path = self.project_root / "docker-compose.yml"
    
    def create_dockerfile(self, base_image: str = "pytorch/pytorch:latest", 
                         python_version: str = "3.9"):
        """
        Create comprehensive Dockerfile
        """
        dockerfile_content = f"""FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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
        
        with open(self.dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"Dockerfile created: {self.dockerfile_path}")
    
    def create_docker_compose(self, gpu_support: bool = True):
        """
        Create docker-compose.yml for easy deployment
        """
        compose_content = {
            'version': '3.8',
            'services': {
                'nemo-rl': {
                    'build': '.',
                    'volumes': [
                        './data:/app/data',
                        './experiments:/app/experiments',
                        './results:/app/results'
                    ],
                    'environment': [
                        'CUDA_VISIBLE_DEVICES=0'
                    ] if gpu_support else [],
                    'deploy': {
                        'resources': {
                            'reservations': {
                                'devices': [
                                    {
                                        'driver': 'nvidia',
                                        'count': 1,
                                        'capabilities': ['gpu']
                                    }
                                ]
                            }
                        }
                    } if gpu_support else {}
                }
            }
        }
        
        with open(self.docker_compose_path, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
        
        print(f"Docker Compose file created: {self.docker_compose_path}")
    
    def build_container(self, tag: str = "nemo-rl:latest"):
        """
        Build Docker container
        """
        try:
            subprocess.run([
                'docker', 'build', '-t', tag, '.'
            ], check=True, cwd=self.project_root)
            print(f"Container built successfully: {tag}")
        except subprocess.CalledProcessError as e:
            print(f"Error building container: {e}")
    
    def run_container(self, tag: str = "nemo-rl:latest", 
                     command: str = "python main.py",
                     volumes: List[str] = None):
        """
        Run Docker container
        """
        if volumes is None:
            volumes = [
                f"{self.project_root}/data:/app/data",
                f"{self.project_root}/experiments:/app/experiments",
                f"{self.project_root}/results:/app/results"
            ]
        
        docker_cmd = ['docker', 'run', '--rm']
        
        # Add volume mounts
        for volume in volumes:
            docker_cmd.extend(['-v', volume])
        
        # Add GPU support if available
        try:
            subprocess.run(['nvidia-smi'], check=True, capture_output=True)
            docker_cmd.extend(['--gpus', 'all'])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("GPU not available, running without GPU support")
        
        docker_cmd.extend([tag, 'bash', '-c', command])
        
        try:
            subprocess.run(docker_cmd, check=True)
            print("Container run completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running container: {e}")
```

### Result Validation and Verification

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
    
    def validate_statistical_results(self, expected_stats: Dict[str, Any],
                                  actual_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate statistical results
        """
        validation_results = {}
        
        # Validate means
        if 'mean' in expected_stats and 'mean' in actual_stats:
            mean_error = abs(expected_stats['mean'] - actual_stats['mean'])
            validation_results['mean'] = {
                'expected': expected_stats['mean'],
                'actual': actual_stats['mean'],
                'error': mean_error,
                'within_tolerance': mean_error <= self.tolerance_threshold
            }
        
        # Validate standard deviations
        if 'std' in expected_stats and 'std' in actual_stats:
            std_error = abs(expected_stats['std'] - actual_stats['std'])
            validation_results['std'] = {
                'expected': expected_stats['std'],
                'actual': actual_stats['std'],
                'error': std_error,
                'within_tolerance': std_error <= self.tolerance_threshold
            }
        
        # Validate p-values
        if 'p_value' in expected_stats and 'p_value' in actual_stats:
            p_value_error = abs(expected_stats['p_value'] - actual_stats['p_value'])
            validation_results['p_value'] = {
                'expected': expected_stats['p_value'],
                'actual': actual_stats['p_value'],
                'error': p_value_error,
                'within_tolerance': p_value_error <= self.tolerance_threshold
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
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate validation report
        """
        report = "# Result Validation Report\n\n"
        
        # Summary
        total_checks = 0
        passed_checks = 0
        
        for category, results in validation_results.items():
            if isinstance(results, dict):
                for check_name, check_result in results.items():
                    total_checks += 1
                    if check_result.get('within_tolerance', check_result.get('match', False)):
                        passed_checks += 1
        
        report += f"## Summary\n\n"
        report += f"- Total checks: {total_checks}\n"
        report += f"- Passed checks: {passed_checks}\n"
        report += f"- Failed checks: {total_checks - passed_checks}\n"
        report += f"- Success rate: {passed_checks/total_checks*100:.1f}%\n\n"
        
        # Detailed results
        for category, results in validation_results.items():
            report += f"## {category.replace('_', ' ').title()}\n\n"
            
            if isinstance(results, dict):
                for check_name, check_result in results.items():
                    status = "✅ PASS" if check_result.get('within_tolerance', check_result.get('match', False)) else "❌ FAIL"
                    report += f"### {check_name}\n"
                    report += f"Status: {status}\n"
                    
                    for key, value in check_result.items():
                        if key not in ['within_tolerance', 'match']:
                            report += f"- {key}: {value}\n"
                    
                    report += "\n"
        
        return report
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
        self.container_manager = ContainerManager(project_root)
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
            self.container_manager.create_dockerfile()
            self.container_manager.create_docker_compose()
        
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
    
    def reproduce_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Reproduce experiment
        """
        # Get original experiment info
        original_info = self.exp_tracker.get_experiment_info(experiment_id)
        
        # Verify environment compatibility
        compatibility = self.env_manager.verify_environment_compatibility()
        if not compatibility['overall_compatible']:
            print("Warning: Environment may not be fully compatible")
        
        # Reproduce experiment
        reproduction_result = self.exp_tracker.reproduce_experiment(experiment_id)
        
        # Validate results
        if original_info['result'] and reproduction_result['result']:
            validation_results = self.result_validator.validate_numerical_results(
                original_info['result'], reproduction_result['result']
            )
            reproduction_result['validation'] = validation_results
        
        return reproduction_result
    
    def generate_reproducibility_report(self, experiment_id: str) -> str:
        """
        Generate comprehensive reproducibility report
        """
        experiment_info = self.exp_tracker.get_experiment_info(experiment_id)
        
        report = f"# Reproducibility Report for {experiment_id}\n\n"
        
        # Experiment information
        report += "## Experiment Information\n\n"
        report += f"- Experiment ID: {experiment_info['metadata']['experiment_id']}\n"
        report += f"- Created: {experiment_info['metadata']['created_at']}\n"
        report += f"- Status: {experiment_info['metadata']['status']}\n\n"
        
        # Git information
        git_info = experiment_info['metadata']['git_info']
        report += "## Git Information\n\n"
        report += f"- Commit: {git_info['commit_hash']}\n"
        report += f"- Branch: {git_info['branch']}\n"
        report += f"- Message: {git_info['commit_message']}\n"
        report += f"- Date: {git_info['commit_date']}\n\n"
        
        # Configuration
        report += "## Configuration\n\n"
        report += "```json\n"
        report += json.dumps(experiment_info['config'], indent=2)
        report += "\n```\n\n"
        
        # Results
        if experiment_info['result']:
            report += "## Results\n\n"
            report += "```json\n"
            report += json.dumps(experiment_info['result'], indent=2)
            report += "\n```\n\n"
        
        # Environment information
        if 'environment_snapshot' in experiment_info['result']:
            env_info = experiment_info['result']['environment_snapshot']
            report += "## Environment\n\n"
            report += f"- Python: {env_info['python_version']}\n"
            report += f"- Platform: {env_info['platform']}\n"
            report += f"- GPU: {env_info['gpu_info']}\n\n"
        
        return report
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

- Learn about [Experimental Design](experimental-design) for rigorous research
- Review [Model Evaluation](model-evaluation) for comprehensive assessment
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 