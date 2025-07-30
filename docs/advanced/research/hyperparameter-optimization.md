# Hyperparameter Optimization

This guide covers hyperparameter optimization techniques for NeMo RL, including automated tuning strategies, search algorithms, and best practices.

> **Note**: This guide focuses on **research methodology and experimental design**. For practical training optimization techniques, see [Training Hyperparameter Optimization](../training/hyperparameter-optimization.md).

## Overview

Hyperparameter optimization is crucial for achieving optimal performance in NeMo RL training. This guide covers automated tuning methods, search strategies, and practical implementation.

## Key Concepts

### Hyperparameters

NeMo RL has several key hyperparameters that significantly impact training performance:

```python
# Learning rate and optimization
learning_rate = 1e-4
weight_decay = 0.01
warmup_steps = 1000

# Training configuration
batch_size = 16
gradient_accumulation_steps = 4
max_grad_norm = 1.0

# Model-specific parameters
dropout_rate = 0.1
attention_dropout = 0.1
hidden_dropout = 0.1
```

### Optimization Objectives

Define clear optimization objectives:

```python
class OptimizationObjective:
    def __init__(self, primary_metric, constraints=None):
        self.primary_metric = primary_metric  # e.g., 'validation_loss'
        self.constraints = constraints or {}
    
    def evaluate(self, trial_results):
        """Evaluate trial results against objectives."""
        primary_value = trial_results[self.primary_metric]
        
        # Check constraints
        for constraint_metric, (min_val, max_val) in self.constraints.items():
            constraint_value = trial_results[constraint_metric]
            if not (min_val <= constraint_value <= max_val):
                return float('inf')  # Invalid trial
        
        return primary_value
```

## Search Algorithms

### Grid Search

Exhaustive search over parameter combinations:

```python
def grid_search_hyperparameters():
    """Perform grid search over hyperparameters."""
    param_grid = {
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
        'batch_size': [8, 16, 32],
        'weight_decay': [0.0, 0.01, 0.1],
        'warmup_steps': [500, 1000, 2000]
    }
    
    best_config = None
    best_score = float('inf')
    
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for wd in param_grid['weight_decay']:
                for ws in param_grid['warmup_steps']:
                    config = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'weight_decay': wd,
                        'warmup_steps': ws
                    }
                    
                    score = evaluate_config(config)
                    
                    if score < best_score:
                        best_score = score
                        best_config = config
    
    return best_config, best_score
```

### Random Search

Random sampling of parameter space:

```python
import random
import numpy as np

def random_search_hyperparameters(n_trials=100):
    """Perform random search over hyperparameters."""
    param_ranges = {
        'learning_rate': (1e-5, 1e-3),
        'batch_size': (4, 64),
        'weight_decay': (0.0, 0.1),
        'warmup_steps': (100, 5000)
    }
    
    best_config = None
    best_score = float('inf')
    
    for trial in range(n_trials):
        config = {}
        for param, (min_val, max_val) in param_ranges.items():
            if param == 'batch_size':
                config[param] = int(2**random.uniform(np.log2(min_val), np.log2(max_val)))
            elif param == 'warmup_steps':
                config[param] = int(2**random.uniform(np.log2(min_val), np.log2(max_val)))
            else:
                config[param] = 10**random.uniform(np.log10(min_val), np.log10(max_val))
        
        score = evaluate_config(config)
        
        if score < best_score:
            best_score = score
            best_config = config
    
    return best_config, best_score
```

### Bayesian Optimization

Efficient search using probabilistic models:

```python
from optuna import create_study, Trial
import optuna

def bayesian_optimization_hyperparameters(n_trials=100):
    """Perform Bayesian optimization of hyperparameters."""
    
    def objective(trial: Trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        warmup_steps = trial.suggest_int('warmup_steps', 100, 5000, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
        
        config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'dropout_rate': dropout_rate
        }
        
        return evaluate_config(config)
    
    # Create study
    study = create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value
```

## Automated Tuning

### Optuna Integration

Use Optuna for advanced hyperparameter optimization:

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def advanced_hyperparameter_optimization():
    """Advanced hyperparameter optimization with Optuna."""
    
    def objective(trial):
        # Model architecture parameters
        hidden_size = trial.suggest_categorical('hidden_size', [512, 768, 1024, 1536])
        num_layers = trial.suggest_int('num_layers', 6, 24)
        num_attention_heads = trial.suggest_categorical('num_attention_heads', [8, 12, 16, 24])
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 8)
        
        # Regularization parameters
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
        attention_dropout = trial.suggest_float('attention_dropout', 0.0, 0.3)
        
        # Schedule parameters
        warmup_steps = trial.suggest_int('warmup_steps', 100, 5000, log=True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 2.0)
        
        config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'attention_dropout': attention_dropout,
            'warmup_steps': warmup_steps,
            'max_grad_norm': max_grad_norm
        }
        
        return evaluate_config(config)
    
    # Create study with advanced settings
    study = create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    )
    
    # Optimize
    study.optimize(objective, n_trials=200, timeout=3600)  # 1 hour timeout
    
    return study.best_params, study.best_value
```

### Ray Tune Integration

Use Ray Tune for distributed hyperparameter optimization:

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def ray_tune_hyperparameter_optimization():
    """Distributed hyperparameter optimization with Ray Tune."""
    
    def training_function(config):
        # Training function for Ray Tune
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        
        for epoch in range(config['max_epochs']):
            train_epoch(model, optimizer, config)
            validation_loss = validate_model(model)
            
            # Report metrics to Ray Tune
            tune.report(validation_loss=validation_loss, epoch=epoch)
    
    # Define search space
    config = {
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'batch_size': tune.choice([4, 8, 16, 32]),
        'weight_decay': tune.uniform(0.0, 0.1),
        'warmup_steps': tune.lograndint(100, 5000),
        'dropout_rate': tune.uniform(0.0, 0.3),
        'max_epochs': 10
    }
    
    # Create scheduler
    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='validation_loss',
        mode='min',
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    
    # Create search algorithm
    search_alg = OptunaSearch(metric='validation_loss', mode='min')
    
    # Run optimization
    analysis = tune.run(
        training_function,
        config=config,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=100,
        resources_per_trial={'gpu': 1, 'cpu': 4}
    )
    
    return analysis.best_config, analysis.best_result
```

## Evaluation Strategies

### Cross-Validation

Implement cross-validation for robust evaluation:

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_hyperparameters(config, n_splits=5):
    """Cross-validate hyperparameters for robust evaluation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # Split data
        train_data = dataset[train_idx]
        val_data = dataset[val_idx]
        
        # Train model with current config
        model = train_model(train_data, config)
        
        # Evaluate on validation set
        val_score = evaluate_model(model, val_data)
        cv_scores.append(val_score)
        
        print(f"Fold {fold + 1}: {val_score:.4f}")
    
    return {
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'scores': cv_scores
    }
```

### Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```python
def multi_objective_optimization():
    """Multi-objective hyperparameter optimization."""
    
    def objective(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3)
        }
        
        # Train and evaluate
        model = train_model(config)
        validation_loss = evaluate_model(model, val_data)
        training_time = measure_training_time(model, config)
        memory_usage = measure_memory_usage(model, config)
        
        return validation_loss, training_time, memory_usage
    
    # Create multi-objective study
    study = optuna.create_study(
        directions=['minimize', 'minimize', 'minimize'],
        sampler=optuna.samplers.NSGAIISampler()
    )
    
    study.optimize(objective, n_trials=100)
    
    return study.best_trials
```

## Practical Implementation

### Configuration Management

Organize hyperparameter configurations:

```python
import yaml
from pathlib import Path

class HyperparameterManager:
    def __init__(self, config_dir):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config, name):
        """Save hyperparameter configuration."""
        config_file = self.config_dir / f"{name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def load_config(self, name):
        """Load hyperparameter configuration."""
        config_file = self.config_dir / f"{name}.yaml"
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def list_configs(self):
        """List available configurations."""
        return [f.stem for f in self.config_dir.glob("*.yaml")]
```

### Automated Pipeline

Create automated hyperparameter optimization pipeline:

```python
class AutomatedHyperparameterOptimization:
    def __init__(self, search_algorithm='bayesian', n_trials=100):
        self.search_algorithm = search_algorithm
        self.n_trials = n_trials
        self.results = []
    
    def optimize(self, objective_function, search_space):
        """Run automated hyperparameter optimization."""
        if self.search_algorithm == 'bayesian':
            return self._bayesian_optimization(objective_function, search_space)
        elif self.search_algorithm == 'random':
            return self._random_search(objective_function, search_space)
        elif self.search_algorithm == 'grid':
            return self._grid_search(objective_function, search_space)
        else:
            raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")
    
    def _bayesian_optimization(self, objective_function, search_space):
        """Bayesian optimization implementation."""
        study = optuna.create_study(direction='minimize')
        
        def objective(trial):
            config = {}
            for param, (param_type, *args) in search_space.items():
                if param_type == 'float':
                    config[param] = trial.suggest_float(param, *args)
                elif param_type == 'int':
                    config[param] = trial.suggest_int(param, *args)
                elif param_type == 'categorical':
                    config[param] = trial.suggest_categorical(param, args[0])
            
            return objective_function(config)
        
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value
    
    def generate_report(self):
        """Generate optimization report."""
        return {
            'best_config': self.results[-1]['config'] if self.results else None,
            'best_score': self.results[-1]['score'] if self.results else None,
            'total_trials': len(self.results),
            'optimization_history': self.results
        }
```

## Best Practices

### Search Space Design

1. **Parameter Ranges**
   - Use log-uniform for learning rates
   - Use categorical for discrete choices
   - Consider parameter interactions

2. **Trial Budget**
   - Balance exploration vs. exploitation
   - Use early stopping for expensive trials
   - Implement parallel evaluation

3. **Evaluation Strategy**
   - Use cross-validation for robustness
   - Consider multiple metrics
   - Account for training time

### Optimization Guidelines

1. **Start Conservative**
   - Begin with broad parameter ranges
   - Use fewer trials initially
   - Focus on most important parameters

2. **Iterative Refinement**
   - Narrow search space based on results
   - Increase trial budget for promising regions
   - Validate best configurations

3. **Practical Considerations**
   - Balance optimization time with training time
   - Consider computational resources
   - Plan for reproducibility

## Configuration Examples

### Basic Optimization

```yaml
hyperparameter_optimization:
  algorithm: "bayesian"
  n_trials: 100
  timeout_hours: 24
  
  search_space:
    learning_rate:
      type: "float"
      min: 1e-5
      max: 1e-3
      log: true
    batch_size:
      type: "categorical"
      choices: [4, 8, 16, 32]
    weight_decay:
      type: "float"
      min: 0.0
      max: 0.1
    warmup_steps:
      type: "int"
      min: 100
      max: 5000
      log: true
  
  evaluation:
    metric: "validation_loss"
    direction: "minimize"
    cross_validation: true
    n_folds: 5
```

### Advanced Optimization

```yaml
hyperparameter_optimization:
  algorithm: "bayesian"
  n_trials: 500
  timeout_hours: 72
  
  search_space:
    # Model architecture
    hidden_size:
      type: "categorical"
      choices: [512, 768, 1024, 1536]
    num_layers:
      type: "int"
      min: 6
      max: 24
    num_attention_heads:
      type: "categorical"
      choices: [8, 12, 16, 24]
    
    # Training parameters
    learning_rate:
      type: "float"
      min: 1e-5
      max: 1e-3
      log: true
    batch_size:
      type: "categorical"
      choices: [4, 8, 16, 32, 64]
    gradient_accumulation_steps:
      type: "int"
      min: 1
      max: 8
    
    # Regularization
    weight_decay:
      type: "float"
      min: 0.0
      max: 0.1
    dropout_rate:
      type: "float"
      min: 0.0
      max: 0.3
    attention_dropout:
      type: "float"
      min: 0.0
      max: 0.3
    
    # Schedule parameters
    warmup_steps:
      type: "int"
      min: 100
      max: 5000
      log: true
    max_grad_norm:
      type: "float"
      min: 0.1
      max: 2.0
  
  evaluation:
    primary_metric: "validation_loss"
    secondary_metrics: ["training_time", "memory_usage"]
    direction: "minimize"
    cross_validation: true
    n_folds: 5
    early_stopping: true
    patience: 10
```

## Next Steps

After implementing hyperparameter optimization:

1. **Establish Baselines**: Create baseline configurations for comparison
2. **Optimize Iteratively**: Start with broad search, then refine
3. **Validate Results**: Cross-validate best configurations
4. **Scale Optimization**: Extend to larger models and datasets

For more advanced topics, see:
- [Custom Algorithms](custom-algorithms.md) - Implementing custom optimization algorithms
- [Performance Analysis](performance-analysis.md) - Analyzing optimization results
- [Research Methodologies](index.md) - Research best practices 