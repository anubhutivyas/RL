# Hyperparameter Optimization

This comprehensive guide covers hyperparameter optimization techniques for NeMo RL, including both practical training optimization and research methodology approaches. Learn systematic approaches to finding optimal hyperparameters using Bayesian optimization, multi-objective search, and automated tuning strategies.

## Overview

Hyperparameter optimization is crucial for achieving the best performance from NeMo RL models. This guide covers systematic approaches to finding optimal hyperparameters, including learning rates, batch sizes, model architectures, and training strategies, with both practical implementation and research methodology perspectives.

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

## Hyperparameter Space Definition

### 1. Continuous Parameters

Define continuous hyperparameters with appropriate ranges:

```python
import optuna
from optuna.samplers import TPESampler

class HyperparameterOptimizer:
    def __init__(self):
        self.study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )
        
    def define_continuous_params(self, trial):
        """Define continuous hyperparameters."""
        params = {
            # Learning rate (log scale)
            'learning_rate': trial.suggest_float(
                'learning_rate', 1e-6, 1e-3, log=True
            ),
            
            # Batch size
            'batch_size': trial.suggest_int('batch_size', 1, 16),
            
            # Weight decay
            'weight_decay': trial.suggest_float(
                'weight_decay', 1e-5, 1e-2, log=True
            ),
            
            # Dropout rate
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            
            # Warmup steps
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 5000),
            
            # Gradient clipping
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 5.0),
        }
        
        return params
```

### 2. Categorical Parameters

Define categorical hyperparameters:

```python
def define_categorical_params(self, trial):
    """Define categorical hyperparameters."""
    params = {
        # Optimizer choice
        'optimizer': trial.suggest_categorical(
            'optimizer', ['adam', 'adamw', 'sgd']
        ),
        
        # Learning rate scheduler
        'scheduler': trial.suggest_categorical(
            'scheduler', ['cosine', 'linear', 'step', 'plateau']
        ),
        
        # Model architecture
        'model_type': trial.suggest_categorical(
            'model_type', ['llama', 'gpt', 'bert']
        ),
        
        # Loss function
        'loss_function': trial.suggest_categorical(
            'loss_function', ['cross_entropy', 'focal', 'label_smoothing']
        ),
        
        # Mixed precision
        'mixed_precision': trial.suggest_categorical(
            'mixed_precision', ['fp16', 'bf16', 'fp32']
        ),
    }
    
    return params
```

### 3. Conditional Parameters

Define conditional hyperparameters based on other choices:

```python
def define_conditional_params(self, trial, base_params):
    """Define conditional hyperparameters."""
    params = base_params.copy()
    
    # Conditional on optimizer choice
    if base_params['optimizer'] == 'sgd':
        params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        params['nesterov'] = trial.suggest_categorical('nesterov', [True, False])
    
    # Conditional on scheduler choice
    if base_params['scheduler'] == 'step':
        params['step_size'] = trial.suggest_int('step_size', 1000, 10000)
        params['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
    
    # Conditional on loss function
    if base_params['loss_function'] == 'focal':
        params['focal_alpha'] = trial.suggest_float('focal_alpha', 0.1, 1.0)
        params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 5.0)
    
    return params
```

## Search Algorithms

### 1. Grid Search

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

### 2. Random Search

Random sampling of parameter space:

```python
def random_search_hyperparameters(n_trials=100):
    """Perform random search over hyperparameters."""
    best_config = None
    best_score = float('inf')
    
    for trial in range(n_trials):
        config = {
            'learning_rate': np.random.uniform(1e-6, 1e-3),
            'batch_size': np.random.choice([8, 16, 32]),
            'weight_decay': np.random.uniform(0.0, 0.1),
            'warmup_steps': np.random.randint(100, 5000)
        }
        
        score = evaluate_config(config)
        
        if score < best_score:
            best_score = score
            best_config = config
    
    return best_config, best_score
```

## Optimization Strategies

### 1. Bayesian Optimization

Use Bayesian optimization for efficient hyperparameter search:

```python
class BayesianOptimizer:
    def __init__(self, n_trials=100):
        self.study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        self.n_trials = n_trials
        
    def objective(self, trial):
        """Objective function for optimization."""
        # Define hyperparameters
        params = self.define_hyperparameters(trial)
        
        # Train model with these parameters
        model = self.train_model(params)
        
        # Evaluate model
        score = self.evaluate_model(model)
        
        return score
        
    def optimize(self):
        """Run hyperparameter optimization."""
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        print(f"Best score: {best_score}")
        print(f"Best parameters: {best_params}")
        
        return best_params, best_score
```

### 2. Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```python
class MultiObjectiveOptimizer:
    def __init__(self):
        self.study = optuna.create_study(
            directions=["minimize", "minimize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )
        
    def objective(self, trial):
        """Multi-objective optimization."""
        params = self.define_hyperparameters(trial)
        
        # Train and evaluate model
        model = self.train_model(params)
        
        # Multiple objectives
        training_loss = self.get_training_loss(model)
        validation_loss = self.get_validation_loss(model)
        accuracy = self.get_accuracy(model)
        
        return training_loss, validation_loss, accuracy
        
    def optimize(self):
        """Run multi-objective optimization."""
        self.study.optimize(self.objective, n_trials=100)
        
        # Get Pareto front
        pareto_front = self.study.best_trials
        
        print(f"Pareto front size: {len(pareto_front)}")
        for trial in pareto_front:
            print(f"Trial {trial.number}: {trial.values}")
            
        return pareto_front
```

### 3. Population-based Training

Use population-based training for dynamic hyperparameter optimization:

```python
class PopulationBasedTrainer:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        self.generations = []
        
    def initialize_population(self):
        """Initialize population with random hyperparameters."""
        for i in range(self.population_size):
            params = self.generate_random_params()
            self.population.append({
                'params': params,
                'fitness': None,
                'generation': 0
            })
            
    def evolve_population(self):
        """Evolve population using genetic algorithm."""
        # Evaluate current population
        for individual in self.population:
            if individual['fitness'] is None:
                individual['fitness'] = self.evaluate_params(individual['params'])
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'])
        
        # Selection and crossover
        new_population = []
        for i in range(self.population_size):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child_params = self.crossover(parent1['params'], parent2['params'])
            child_params = self.mutate(child_params)
            
            new_population.append({
                'params': child_params,
                'fitness': None,
                'generation': parent1['generation'] + 1
            })
            
        self.population = new_population
        self.generations.append(self.population.copy())
        
    def tournament_selection(self):
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda x: x['fitness'])
        
    def crossover(self, params1, params2):
        """Crossover operation for genetic algorithm."""
        child_params = {}
        for key in params1:
            if random.random() < 0.5:
                child_params[key] = params1[key]
            else:
                child_params[key] = params2[key]
        return child_params
        
    def mutate(self, params):
        """Mutation operation for genetic algorithm."""
        mutation_rate = 0.1
        for key in params:
            if random.random() < mutation_rate:
                if isinstance(params[key], float):
                    params[key] *= random.uniform(0.8, 1.2)
                elif isinstance(params[key], int):
                    params[key] += random.randint(-1, 1)
        return params
```

## Advanced Optimization Techniques

### 1. Early Pruning

Implement early pruning to stop unpromising trials:

```python
class EarlyPruningOptimizer:
    def __init__(self):
        self.study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
    def objective(self, trial):
        """Objective function with early pruning."""
        params = self.define_hyperparameters(trial)
        
        # Train model with intermediate evaluations
        for epoch in range(100):
            loss = self.train_epoch(params, epoch)
            
            # Report intermediate value for pruning
            trial.report(loss, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return loss
```

### 2. Transfer Learning for Hyperparameter Optimization

Use knowledge from previous optimizations:

```python
class TransferLearningOptimizer:
    def __init__(self, previous_studies=None):
        self.previous_studies = previous_studies or []
        
        # Create study with transfer learning
        sampler = optuna.samplers.TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=1.0,
            n_startup_trials=10
        )
        
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )
        
    def transfer_knowledge(self):
        """Transfer knowledge from previous studies."""
        for study in self.previous_studies:
            # Add best trials from previous studies
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    self.study.add_trial(trial)
```

### 3. Hyperparameter Importance Analysis

Analyze the importance of different hyperparameters:

```python
class HyperparameterAnalyzer:
    def __init__(self, study):
        self.study = study
        
    def analyze_importance(self):
        """Analyze hyperparameter importance."""
        importance = optuna.importance.get_param_importances(self.study)
        
        # Sort by importance
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("Hyperparameter Importance:")
        for param, imp in sorted_importance:
            print(f"  {param}: {imp:.4f}")
            
        return importance
        
    def plot_importance(self):
        """Plot hyperparameter importance."""
        optuna.visualization.plot_param_importances(self.study)
        
    def plot_optimization_history(self):
        """Plot optimization history."""
        optuna.visualization.plot_optimization_history(self.study)
        
    def plot_parallel_coordinate(self):
        """Plot parallel coordinate plot."""
        optuna.visualization.plot_parallel_coordinate(self.study)
```

## Research Methodology

### 1. Experimental Design for Hyperparameter Optimization

Design robust experiments for hyperparameter optimization:

```python
class HyperparameterExperimentDesigner:
    def __init__(self):
        self.experiment_configs = []
        
    def design_comparison_experiment(self, algorithms, datasets):
        """Design experiment to compare hyperparameter optimization methods."""
        experiment_config = {
            'algorithms': algorithms,  # ['bayesian', 'random', 'grid']
            'datasets': datasets,      # ['preference', 'instruction', 'code']
            'metrics': ['validation_loss', 'training_time', 'convergence_rate'],
            'n_trials': 100,
            'repetitions': 5,  # For statistical significance
            'random_seeds': [42, 123, 456, 789, 101112]
        }
        
        return experiment_config
    
    def design_ablation_study(self, base_config, components_to_ablate):
        """Design ablation study for hyperparameter optimization components."""
        ablation_configs = []
        
        for component in components_to_ablate:
            # Create config without this component
            ablated_config = base_config.copy()
            ablated_config[component] = None  # or default value
            
            ablation_configs.append({
                'component_removed': component,
                'config': ablated_config
            })
        
        return ablation_configs
```

### 2. Statistical Analysis

Implement statistical analysis for hyperparameter optimization results:

```python
class HyperparameterStatisticalAnalyzer:
    def __init__(self):
        self.results = []
        
    def analyze_optimization_performance(self, results):
        """Analyze performance of different optimization methods."""
        import scipy.stats as stats
        
        # Compare methods using statistical tests
        methods = list(results.keys())
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    results[method1]['scores'],
                    results[method2]['scores']
                )
                
                print(f"{method1} vs {method2}: p-value = {p_value:.4f}")
                
                # Calculate effect size
                effect_size = self.calculate_effect_size(
                    results[method1]['scores'],
                    results[method2]['scores']
                )
                
                print(f"Effect size: {effect_size:.4f}")
    
    def calculate_effect_size(self, group1, group2):
        """Calculate Cohen's d effect size."""
        import numpy as np
        
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
```

## Integration with NeMo RL

### 1. Configuration-based Optimization

```yaml
# hyperparameter_optimization.yaml
optimization:
  method: "bayesian"
  n_trials: 100
  timeout: 3600  # 1 hour
  
  hyperparameters:
    learning_rate:
      type: "float"
      min: 1e-6
      max: 1e-3
      log: true
      
    batch_size:
      type: "int"
      min: 1
      max: 16
      
    optimizer:
      type: "categorical"
      choices: ["adam", "adamw", "sgd"]
      
    scheduler:
      type: "categorical"
      choices: ["cosine", "linear", "step"]
      
  objectives:
    - name: "validation_loss"
      direction: "minimize"
      weight: 1.0
    - name: "training_time"
      direction: "minimize"
      weight: 0.5
```

### 2. Automated Optimization Pipeline

```python
class AutomatedOptimizer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.optimizer = self.create_optimizer()
        
    def load_config(self, config_path):
        """Load optimization configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def create_optimizer(self):
        """Create optimizer based on configuration."""
        method = self.config['optimization']['method']
        
        if method == 'bayesian':
            return BayesianOptimizer(
                n_trials=self.config['optimization']['n_trials']
            )
        elif method == 'multi_objective':
            return MultiObjectiveOptimizer()
        elif method == 'population_based':
            return PopulationBasedTrainer(
                population_size=self.config['optimization'].get('population_size', 10)
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
    def run_optimization(self):
        """Run automated hyperparameter optimization."""
        print("Starting hyperparameter optimization...")
        
        # Run optimization
        best_params, best_score = self.optimizer.optimize()
        
        # Save results
        self.save_results(best_params, best_score)
        
        # Generate report
        self.generate_report()
        
        return best_params, best_score
        
    def save_results(self, best_params, best_score):
        """Save optimization results."""
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    def generate_report(self):
        """Generate optimization report."""
        # Create visualization plots
        self.optimizer.plot_importance()
        self.optimizer.plot_optimization_history()
        self.optimizer.plot_parallel_coordinate()
```

### 3. Distributed Optimization

```python
class DistributedOptimizer:
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
    def run_distributed_optimization(self):
        """Run distributed hyperparameter optimization."""
        import multiprocessing as mp
        
        # Create worker processes
        with mp.Pool(self.n_workers) as pool:
            # Distribute trials across workers
            results = pool.map(self.worker_function, range(self.n_workers))
            
        # Combine results
        best_params = min(results, key=lambda x: x[1])[0]
        return best_params
        
    def worker_function(self, worker_id):
        """Worker function for distributed optimization."""
        # Each worker runs a subset of trials
        n_trials_per_worker = 25
        
        for i in range(n_trials_per_worker):
            trial = self.study.ask()
            params = self.define_hyperparameters(trial)
            score = self.evaluate_params(params)
            self.study.tell(trial, score)
            
        return self.study.best_params, self.study.best_value
```

## Best Practices

### 1. Search Space Design

1. **Define appropriate ranges** for each hyperparameter
2. **Use log scales** for learning rates and other multiplicative parameters
3. **Consider parameter interactions** when defining the search space
4. **Start with broad ranges** and narrow down based on results

### 2. Optimization Strategy

1. **Choose the right optimization method** for your problem
2. **Use early pruning** to save computational resources
3. **Implement multi-objective optimization** when multiple metrics matter
4. **Consider transfer learning** from previous optimizations

### 3. Evaluation Strategy

1. **Use cross-validation** for robust evaluation
2. **Implement proper train/validation/test splits**
3. **Consider computational cost** in evaluation strategy
4. **Use appropriate metrics** for your specific task

### 4. Monitoring and Analysis

1. **Monitor optimization progress** continuously
2. **Analyze hyperparameter importance** to understand your model
3. **Visualize optimization results** for insights
4. **Document optimization process** for reproducibility

### 5. Research Rigor

1. **Use statistical tests** to compare optimization methods
2. **Report effect sizes** for meaningful comparisons
3. **Ensure reproducibility** with proper seeding
4. **Document experimental design** thoroughly

## Troubleshooting

### Common Issues

1. **Optimization Not Converging**
   ```python
   # Increase number of trials
   n_trials = 200  # Increase from 100
   
   # Adjust search space
   learning_rate_range = (1e-7, 1e-2)  # Broader range
   ```

2. **Computational Budget Exceeded**
   ```python
   # Use early pruning
   pruner = optuna.pruners.MedianPruner(
       n_startup_trials=3,
       n_warmup_steps=5
   )
   
   # Reduce evaluation frequency
   eval_steps = 100  # Evaluate less frequently
   ```

3. **Poor Hyperparameter Importance**
   ```python
   # Increase number of trials for better analysis
   n_trials = 500  # More trials for reliable importance
   
   # Use more diverse search space
   broader_ranges = True
   ```

4. **Overfitting to Validation Set**
   ```python
   # Use nested cross-validation
   cv_folds = 5
   
   # Use separate test set
   holdout_test_set = True
   ```

## Next Steps

- [Multi-Objective Training](multi-objective-training) - Learn multi-objective optimization
- [Loss Functions](loss-functions) - Design custom objectives
- [Curriculum Learning](curriculum-learning) - Progressive training strategies
- [Advanced Performance](../performance/index) - Performance optimization 