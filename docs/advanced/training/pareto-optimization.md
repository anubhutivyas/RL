# Pareto Optimization

This guide covers Pareto optimization techniques for multi-objective training in NeMo RL.

## Overview

Pareto optimization is a technique for handling multiple competing objectives during training. Instead of optimizing a single objective, Pareto optimization finds solutions that represent the best trade-offs between multiple objectives.

## Multi-Objective Training

### Problem Formulation

In multi-objective training, we have multiple objectives that may conflict:

```python
# Example objectives
objectives = {
    "accuracy": maximize_accuracy,
    "efficiency": minimize_computation,
    "robustness": maximize_generalization,
    "fairness": minimize_bias
}
```

### Pareto Frontier

The Pareto frontier represents the set of solutions where improving one objective requires degrading another:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_pareto_frontier(objectives_1, objectives_2):
    """Plot Pareto frontier for two objectives."""
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives_1, objectives_2, alpha=0.6)
    
    # Find Pareto optimal points
    pareto_points = find_pareto_optimal(objectives_1, objectives_2)
    plt.scatter(objectives_1[pareto_points], objectives_2[pareto_points], 
                color='red', s=100, label='Pareto Optimal')
    
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Frontier')
    plt.legend()
    plt.show()

def find_pareto_optimal(obj1, obj2):
    """Find Pareto optimal points."""
    n = len(obj1)
    pareto_optimal = []
    
    for i in range(n):
        is_pareto = True
        for j in range(n):
            if i != j:
                # Check if point j dominates point i
                if (obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and 
                    (obj1[j] > obj1[i] or obj2[j] > obj2[i])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_optimal.append(i)
    
    return pareto_optimal
```

## Implementation Strategies

### Weighted Sum Method

Combine multiple objectives using weighted sums:

```python
class WeightedSumOptimizer:
    def __init__(self, weights):
        self.weights = weights
    
    def combined_objective(self, objectives):
        """Combine multiple objectives using weighted sum."""
        return sum(w * obj for w, obj in zip(self.weights, objectives))
    
    def optimize(self, model, data, num_iterations=1000):
        """Optimize using weighted sum method."""
        for i in range(num_iterations):
            # Compute individual objectives
            accuracy = compute_accuracy(model, data)
            efficiency = compute_efficiency(model, data)
            robustness = compute_robustness(model, data)
            
            # Combine objectives
            combined_obj = self.combined_objective([accuracy, -efficiency, robustness])
            
            # Update model
            update_model(model, combined_obj)
```

### Evolutionary Algorithms

Use evolutionary algorithms to find Pareto optimal solutions:

```python
import random
from typing import List, Tuple

class ParetoEvolutionaryOptimizer:
    def __init__(self, population_size=100, generations=50):
        self.population_size = population_size
        self.generations = generations
    
    def initialize_population(self, model_configs):
        """Initialize population of model configurations."""
        population = []
        for _ in range(self.population_size):
            config = self.mutate_config(random.choice(model_configs))
            population.append(config)
        return population
    
    def evaluate_fitness(self, config, data):
        """Evaluate fitness across multiple objectives."""
        model = create_model_from_config(config)
        train_model(model, data)
        
        return {
            'accuracy': evaluate_accuracy(model, data),
            'efficiency': evaluate_efficiency(model, data),
            'robustness': evaluate_robustness(model, data)
        }
    
    def non_dominated_sort(self, population, fitness_scores):
        """Sort population by Pareto dominance."""
        fronts = []
        domination_count = {i: 0 for i in range(len(population))}
        dominated_solutions = {i: [] for i in range(len(population))}
        
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self.dominates(fitness_scores[i], fitness_scores[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(fitness_scores[j], fitness_scores[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)
        
        current_front = 0
        while current_front < len(fronts):
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def dominates(self, fitness1, fitness2):
        """Check if fitness1 dominates fitness2."""
        at_least_one_better = False
        for obj in fitness1:
            if fitness1[obj] < fitness2[obj]:
                return False
            elif fitness1[obj] > fitness2[obj]:
                at_least_one_better = True
        
        return at_least_one_better
    
    def optimize(self, initial_configs, data):
        """Run Pareto optimization."""
        population = self.initialize_population(initial_configs)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for config in population:
                fitness = self.evaluate_fitness(config, data)
                fitness_scores.append(fitness)
            
            # Non-dominated sorting
            fronts = self.non_dominated_sort(population, fitness_scores)
            
            # Selection and reproduction
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Use crowding distance for selection
                    remaining = self.population_size - len(new_population)
                    selected = self.crowding_distance_selection(front, fitness_scores, remaining)
                    new_population.extend(selected)
                    break
            
            # Mutation and crossover
            population = self.evolutionary_operators(new_population)
        
        return population, fitness_scores
```

### Multi-Objective Loss Functions

Implement multi-objective loss functions:

```python
import torch
import torch.nn as nn

class MultiObjectiveLoss(nn.Module):
    def __init__(self, objectives, weights=None):
        super().__init__()
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
    
    def forward(self, predictions, targets, model):
        """Compute multi-objective loss."""
        losses = []
        
        for objective, weight in zip(self.objectives, self.weights):
            loss = objective(predictions, targets, model)
            losses.append(weight * loss)
        
        return sum(losses)

class AccuracyEfficiencyLoss(MultiObjectiveLoss):
    def __init__(self, accuracy_weight=1.0, efficiency_weight=0.1):
        objectives = [
            self.accuracy_loss,
            self.efficiency_loss
        ]
        weights = [accuracy_weight, efficiency_weight]
        super().__init__(objectives, weights)
    
    def accuracy_loss(self, predictions, targets, model):
        """Accuracy-based loss."""
        return nn.CrossEntropyLoss()(predictions, targets)
    
    def efficiency_loss(self, predictions, targets, model):
        """Efficiency-based loss (e.g., model complexity)."""
        # Count parameters or compute FLOPs
        num_params = sum(p.numel() for p in model.parameters())
        return torch.tensor(num_params, dtype=torch.float32)
```

## Configuration

### Multi-Objective Configuration

```yaml
# multi_objective_config.yaml
training:
  multi_objective:
    enabled: true
    method: "weighted_sum"  # or "evolutionary", "pareto"
    
    objectives:
      - name: "accuracy"
        weight: 1.0
        target: "maximize"
        
      - name: "efficiency"
        weight: 0.1
        target: "minimize"
        
      - name: "robustness"
        weight: 0.5
        target: "maximize"
    
    evolutionary:
      population_size: 100
      generations: 50
      mutation_rate: 0.1
      crossover_rate: 0.8
    
    pareto:
      epsilon: 0.01
      max_iterations: 1000
```

### Dynamic Weight Adjustment

```python
class DynamicWeightAdjuster:
    def __init__(self, initial_weights, learning_rate=0.01):
        self.weights = initial_weights
        self.learning_rate = learning_rate
        self.history = []
    
    def adjust_weights(self, current_objectives, target_objectives):
        """Dynamically adjust weights based on performance."""
        for i, (current, target) in enumerate(zip(current_objectives, target_objectives)):
            error = target - current
            self.weights[i] += self.learning_rate * error
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.history.append(self.weights.copy())
    
    def get_weights(self):
        """Get current weights."""
        return self.weights
```

## Evaluation and Analysis

### Pareto Frontier Analysis

```python
def analyze_pareto_frontier(population, fitness_scores):
    """Analyze Pareto frontier characteristics."""
    pareto_front = find_pareto_optimal(fitness_scores)
    
    analysis = {
        'num_pareto_solutions': len(pareto_front),
        'diversity': compute_diversity(pareto_front, fitness_scores),
        'spread': compute_spread(pareto_front, fitness_scores),
        'hypervolume': compute_hypervolume(pareto_front, fitness_scores)
    }
    
    return analysis

def compute_diversity(pareto_front, fitness_scores):
    """Compute diversity of Pareto solutions."""
    if len(pareto_front) < 2:
        return 0.0
    
    distances = []
    for i in range(len(pareto_front)):
        for j in range(i + 1, len(pareto_front)):
            dist = euclidean_distance(
                fitness_scores[pareto_front[i]],
                fitness_scores[pareto_front[j]]
            )
            distances.append(dist)
    
    return np.mean(distances)

def compute_spread(pareto_front, fitness_scores):
    """Compute spread of Pareto solutions."""
    if len(pareto_front) < 2:
        return 0.0
    
    # Compute spread along each objective
    spreads = []
    for obj_idx in range(len(fitness_scores[0])):
        values = [fitness_scores[i][obj_idx] for i in pareto_front]
        spread = max(values) - min(values)
        spreads.append(spread)
    
    return np.mean(spreads)
```

### Visualization

```python
def visualize_pareto_results(population, fitness_scores, pareto_front):
    """Visualize Pareto optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 2D Pareto frontier
    obj1 = [fitness_scores[i]['accuracy'] for i in range(len(population))]
    obj2 = [fitness_scores[i]['efficiency'] for i in range(len(population))]
    
    axes[0, 0].scatter(obj1, obj2, alpha=0.6, label='All Solutions')
    pareto_obj1 = [fitness_scores[i]['accuracy'] for i in pareto_front]
    pareto_obj2 = [fitness_scores[i]['efficiency'] for i in pareto_front]
    axes[0, 0].scatter(pareto_obj1, pareto_obj2, color='red', s=100, label='Pareto Optimal')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Efficiency')
    axes[0, 0].set_title('Pareto Frontier')
    axes[0, 0].legend()
    
    # Plot objective distributions
    objectives = list(fitness_scores[0].keys())
    for i, obj in enumerate(objectives):
        values = [fitness_scores[j][obj] for j in range(len(population))]
        axes[0, 1].hist(values, alpha=0.7, label=obj)
    axes[0, 1].set_xlabel('Objective Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Objective Distributions')
    axes[0, 1].legend()
    
    # Plot convergence
    if hasattr(self, 'convergence_history'):
        axes[1, 0].plot(self.convergence_history)
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Hypervolume')
        axes[1, 0].set_title('Convergence')
    
    # Plot weight evolution
    if hasattr(self, 'weight_history'):
        weight_history = np.array(self.weight_history)
        for i, obj in enumerate(objectives):
            axes[1, 1].plot(weight_history[:, i], label=obj)
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].set_title('Weight Evolution')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

### 1. Objective Selection

- Choose objectives that are truly independent
- Ensure objectives are measurable and well-defined
- Consider the trade-offs between objectives

### 2. Weight Tuning

- Start with equal weights and adjust based on domain knowledge
- Use validation data to tune weights
- Consider dynamic weight adjustment

### 3. Solution Selection

- Use multiple criteria for selecting final solution
- Consider application-specific requirements
- Validate selected solution on test data

### 4. Computational Efficiency

- Use efficient algorithms for large populations
- Implement early stopping criteria
- Parallelize objective evaluation

## Troubleshooting

### Common Issues

1. **Convergence Problems**
   - Adjust population size and mutation rates
   - Use different selection strategies
   - Implement diversity preservation

2. **Objective Conflicts**
   - Analyze objective correlations
   - Consider objective normalization
   - Use preference-based methods

3. **Computational Cost**
   - Reduce population size
   - Use surrogate models
   - Implement efficient evaluation

For more advanced multi-objective optimization techniques, see [Multi-Objective Tuning](multi-objective-tuning.md) and [Hyperparameter Optimization](hyperparameter-optimization.md). 