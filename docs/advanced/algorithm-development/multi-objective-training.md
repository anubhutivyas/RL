# Multi-Objective Training and Optimization

Learn to combine multiple loss functions and objectives in a single training pipeline. Balance competing objectives like accuracy, efficiency, and safety while maintaining training stability. This guide covers both training approaches and optimization techniques.

## Overview

Multi-objective training involves optimizing multiple competing objectives simultaneously. In RL contexts, this often means balancing objectives like task performance, safety constraints, efficiency, and alignment. Unlike single-objective training, multi-objective approaches require sophisticated techniques to handle objective conflicts and ensure stable training.

## Key Challenges

### Objective Conflicts
- **Competing objectives**: Objectives that naturally conflict (e.g., performance vs. safety)
- **Trade-offs**: Finding optimal balance between multiple objectives
- **Dynamic relationships**: Objectives that change importance during training

### Training Stability
- **Gradient conflicts**: Gradients from different objectives may oppose each other
- **Convergence issues**: Multiple objectives can lead to unstable training
- **Weight balancing**: Determining optimal weights for different objectives

### Evaluation Metrics
- **Pareto efficiency**: Finding solutions that can't be improved without degrading others
- **Multi-dimensional evaluation**: Assessing performance across all objectives
- **Objective importance**: Determining relative importance of different objectives

## Multi-Objective Problem Formulation

### 1. Objective Definition

Define multiple objectives for your training problem:

```python
class MultiObjectiveProblem:
    def __init__(self):
        self.objectives = {
            'accuracy': {'direction': 'maximize', 'weight': 1.0},
            'efficiency': {'direction': 'maximize', 'weight': 0.5},
            'memory_usage': {'direction': 'minimize', 'weight': 0.3},
            'training_time': {'direction': 'minimize', 'weight': 0.2}
        }
        
    def evaluate_objectives(self, model, training_config):
        """Evaluate all objectives for a given model and configuration."""
        results = {}
        
        # Accuracy objective
        results['accuracy'] = self.evaluate_accuracy(model)
        
        # Efficiency objective (tokens per second)
        results['efficiency'] = self.evaluate_efficiency(model)
        
        # Memory usage objective
        results['memory_usage'] = self.evaluate_memory_usage(model)
        
        # Training time objective
        results['training_time'] = self.evaluate_training_time(model)
        
        return results
        
    def normalize_objectives(self, results):
        """Normalize objectives to [0, 1] range."""
        normalized = {}
        
        for obj_name, value in results.items():
            if self.objectives[obj_name]['direction'] == 'maximize':
                normalized[obj_name] = value / self.get_max_value(obj_name)
            else:
                normalized[obj_name] = 1 - (value / self.get_max_value(obj_name))
                
        return normalized
```

### 2. Pareto Front Analysis

Implement Pareto front analysis to find non-dominated solutions:

```python
class ParetoOptimizer:
    def __init__(self, objectives):
        self.objectives = objectives
        self.solutions = []
        
    def is_dominated(self, solution1, solution2):
        """Check if solution1 is dominated by solution2."""
        at_least_as_good = True
        strictly_better = False
        
        for obj_name in self.objectives:
            val1 = solution1[obj_name]
            val2 = solution2[obj_name]
            direction = self.objectives[obj_name]['direction']
            
            if direction == 'maximize':
                if val1 < val2:
                    at_least_as_good = False
                elif val1 > val2:
                    strictly_better = True
            else:  # minimize
                if val1 > val2:
                    at_least_as_good = False
                elif val1 < val2:
                    strictly_better = True
                    
        return at_least_as_good and strictly_better
        
    def find_pareto_front(self, solutions):
        """Find Pareto optimal solutions."""
        pareto_front = []
        
        for solution in solutions:
            is_pareto_optimal = True
            
            for other_solution in solutions:
                if solution != other_solution and self.is_dominated(solution, other_solution):
                    is_pareto_optimal = False
                    break
                    
            if is_pareto_optimal:
                pareto_front.append(solution)
                
        return pareto_front
```

## Architecture Patterns

### Multi-Objective Loss Function

```python
class MultiObjectiveLoss:
    def __init__(self, objectives, weights=None):
        self.objectives = objectives
        self.weights = weights or {obj: 1.0 for obj in objectives}
        self.dynamic_balancer = DynamicWeightBalancer()
    
    def compute_loss(self, predictions, targets, current_step):
        """Compute combined loss from multiple objectives"""
        losses = {}
        total_loss = 0.0
        
        # Compute individual objective losses
        for obj_name, obj_func in self.objectives.items():
            losses[obj_name] = obj_func(predictions, targets)
        
        # Apply dynamic weight balancing
        adjusted_weights = self.dynamic_balancer.adjust_weights(
            losses, self.weights, current_step
        )
        
        # Combine losses with adjusted weights
        for obj_name, loss in losses.items():
            total_loss += adjusted_weights[obj_name] * loss
        
        return total_loss, losses, adjusted_weights
```

### Dynamic Weight Balancing

```python
class DynamicWeightBalancer:
    def __init__(self, balancing_strategy='uncertainty'):
        self.strategy = balancing_strategy
        self.loss_history = {}
        self.weight_history = {}
    
    def adjust_weights(self, current_losses, base_weights, step):
        """Dynamically adjust weights based on training progress"""
        if self.strategy == 'uncertainty':
            return self._uncertainty_based_balancing(current_losses, base_weights)
        elif self.strategy == 'gradient_norm':
            return self._gradient_norm_balancing(current_losses, base_weights)
        elif self.strategy == 'adaptive':
            return self._adaptive_balancing(current_losses, base_weights, step)
        else:
            return base_weights
    
    def _uncertainty_based_balancing(self, losses, base_weights):
        """Balance weights based on loss uncertainty"""
        adjusted_weights = {}
        total_uncertainty = sum(1.0 / (loss + 1e-8) for loss in losses.values())
        
        for obj_name, loss in losses.items():
            # Weight inversely proportional to loss uncertainty
            uncertainty = 1.0 / (loss + 1e-8)
            adjusted_weights[obj_name] = (uncertainty / total_uncertainty) * base_weights[obj_name]
        
        return adjusted_weights
    
    def _gradient_norm_balancing(self, losses, base_weights):
        """Balance weights based on gradient norms"""
        adjusted_weights = {}
        total_grad_norm = sum(loss for loss in losses.values())
        
        for obj_name, loss in losses.items():
            # Weight inversely proportional to gradient norm
            grad_norm = loss
            adjusted_weights[obj_name] = (1.0 / (grad_norm + 1e-8)) * base_weights[obj_name]
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        for obj_name in adjusted_weights:
            adjusted_weights[obj_name] /= total_weight
        
        return adjusted_weights
```

## Implementation Strategies

### Objective Combination Methods

```python
class ObjectiveCombiner:
    def __init__(self, method='weighted_sum'):
        self.method = method
    
    def combine_objectives(self, objectives, weights=None):
        """Combine multiple objectives using different strategies"""
        if self.method == 'weighted_sum':
            return self._weighted_sum(objectives, weights)
        elif self.method == 'pareto_optimization':
            return self._pareto_optimization(objectives)
        elif self.method == 'constraint_optimization':
            return self._constraint_optimization(objectives)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _weighted_sum(self, objectives, weights):
        """Simple weighted sum of objectives"""
        if weights is None:
            weights = {obj: 1.0 for obj in objectives}
        
        combined_loss = 0.0
        for obj_name, obj_value in objectives.items():
            combined_loss += weights[obj_name] * obj_value
        
        return combined_loss
    
    def _pareto_optimization(self, objectives):
        """Find Pareto optimal solution"""
        # Implementation for Pareto optimization
        # This would involve more sophisticated optimization
        pass
    
    def _constraint_optimization(self, objectives):
        """Treat some objectives as constraints"""
        # Implementation for constraint-based optimization
        pass
```

### Training Pipeline Integration

```python
class MultiObjectiveTrainer:
    def __init__(self, model, objectives, combiner):
        self.model = model
        self.objectives = objectives
        self.combiner = combiner
        self.monitor = MultiObjectiveMonitor()
    
    def train_step(self, batch, step):
        """Single training step with multiple objectives"""
        # Forward pass
        predictions = self.model(batch['input'])
        
        # Compute individual objective losses
        objective_losses = {}
        for obj_name, obj_func in self.objectives.items():
            objective_losses[obj_name] = obj_func(predictions, batch)
        
        # Combine objectives
        total_loss = self.combiner.combine_objectives(objective_losses)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Monitor objectives
        self.monitor.log_objectives(objective_losses, step)
        
        return total_loss, objective_losses
```

## Advanced Optimization Techniques

### Multi-Objective Bayesian Optimization

```python
class MultiObjectiveBayesianOptimizer:
    def __init__(self, objectives, acquisition_function='ehvi'):
        self.objectives = objectives
        self.acquisition_function = acquisition_function
        self.gp_models = {obj: GaussianProcessRegressor() for obj in objectives}
        
    def suggest_next_point(self, observed_points, observed_values):
        """Suggest next point to evaluate using Bayesian optimization"""
        # Update GP models
        for obj_name in self.objectives:
            self.gp_models[obj_name].fit(observed_points, observed_values[obj_name])
        
        # Optimize acquisition function
        if self.acquisition_function == 'ehvi':
            return self._optimize_ehvi(observed_points, observed_values)
        elif self.acquisition_function == 'parego':
            return self._optimize_parego(observed_points, observed_values)
        else:
            return self._optimize_random()
    
    def _optimize_ehvi(self, observed_points, observed_values):
        """Optimize Expected Hypervolume Improvement"""
        # Implementation for EHVI optimization
        pass
    
    def _optimize_parego(self, observed_points, observed_values):
        """Optimize using ParEGO scalarization"""
        # Implementation for ParEGO optimization
        pass
```

### Constraint Handling

```python
class ConstraintHandler:
    def __init__(self, constraints):
        self.constraints = constraints
        
    def check_feasibility(self, solution):
        """Check if solution satisfies all constraints"""
        for constraint_name, constraint_func in self.constraints.items():
            if not constraint_func(solution):
                return False
        return True
    
    def penalize_infeasible(self, solution, objectives):
        """Add penalty for infeasible solutions"""
        if self.check_feasibility(solution):
            return objectives
        
        # Add penalty to all objectives
        penalized_objectives = {}
        penalty = self._compute_penalty(solution)
        
        for obj_name, obj_value in objectives.items():
            penalized_objectives[obj_name] = obj_value + penalty
            
        return penalized_objectives
    
    def _compute_penalty(self, solution):
        """Compute penalty for constraint violations"""
        total_violation = 0.0
        
        for constraint_name, constraint_func in self.constraints.items():
            if not constraint_func(solution):
                total_violation += 1.0
                
        return total_violation * 1000.0  # Large penalty
```

## Real-World Examples

### Safety-Constrained RL Training

```python
# Example: Training with performance and safety objectives
class SafetyConstrainedTrainer:
    def __init__(self):
        self.objectives = {
            'performance': PerformanceLoss(),
            'safety': SafetyLoss(),
            'efficiency': EfficiencyLoss()
        }
        
        self.constraints = {
            'safety_threshold': 0.95,  # Minimum safety score
            'efficiency_threshold': 0.8  # Minimum efficiency score
        }
    
    def train_with_constraints(self, model, data):
        """Train model with safety and efficiency constraints"""
        for step in range(max_steps):
            # Compute all objective losses
            losses = {}
            for obj_name, obj_func in self.objectives.items():
                losses[obj_name] = obj_func(model, data)
            
            # Check constraint violations
            constraint_violations = self._check_constraints(losses)
            
            if constraint_violations:
                # Adjust training strategy for constraint violations
                self._handle_constraint_violations(constraint_violations)
            
            # Combine losses with constraint-aware weighting
            total_loss = self._combine_with_constraints(losses)
            
            # Training step
            total_loss.backward()
            optimizer.step()
```

### Multi-Domain Training

```python
# Example: Training across multiple domains with different objectives
class MultiDomainTrainer:
    def __init__(self, domains):
        self.domains = domains
        self.domain_objectives = {
            'code_generation': {
                'correctness': CodeCorrectnessLoss(),
                'efficiency': CodeEfficiencyLoss(),
                'readability': CodeReadabilityLoss()
            },
            'mathematical_reasoning': {
                'accuracy': MathAccuracyLoss(),
                'reasoning': ReasoningQualityLoss(),
                'completeness': CompletenessLoss()
            },
            'dialogue': {
                'relevance': RelevanceLoss(),
                'helpfulness': HelpfulnessLoss(),
                'safety': SafetyLoss()
            }
        }
    
    def train_multi_domain(self, model, domain_data):
        """Train model across multiple domains with domain-specific objectives"""
        for domain_name, data in domain_data.items():
            objectives = self.domain_objectives[domain_name]
            
            # Compute domain-specific losses
            domain_losses = {}
            for obj_name, obj_func in objectives.items():
                domain_losses[obj_name] = obj_func(model, data)
            
            # Combine domain objectives
            domain_loss = self._combine_domain_objectives(domain_losses)
            
            # Update model
            domain_loss.backward()
            optimizer.step()
```

## Best Practices

### Objective Design
- **Clear objectives**: Define objectives that are measurable and well-defined
- **Balanced importance**: Ensure objectives have appropriate relative importance
- **Conflict awareness**: Understand when objectives naturally conflict
- **Dynamic adjustment**: Allow objective weights to change during training

### Training Stability
- **Gradient clipping**: Use gradient clipping to prevent gradient explosion
- **Learning rate scheduling**: Implement adaptive learning rates for multi-objective training
- **Monitoring**: Track individual objective progress during training
- **Early stopping**: Use multi-objective early stopping criteria

### Evaluation
- **Pareto analysis**: Evaluate solutions on the Pareto front
- **Trade-off analysis**: Understand the trade-offs between objectives
- **Robustness testing**: Test performance across different objective weightings
- **Human evaluation**: Include human evaluation for subjective objectives

### Production Considerations
- **Objective monitoring**: Monitor all objectives in production
- **Weight tuning**: Allow dynamic adjustment of objective weights
- **Fallback strategies**: Implement fallbacks when objectives conflict severely
- **Performance tracking**: Track performance across all objectives over time

## Monitoring and Observability

```python
class MultiObjectiveMonitor:
    def __init__(self, objectives):
        self.objectives = objectives
        self.history = {obj: [] for obj in objectives}
        self.weight_history = {obj: [] for obj in objectives}
    
    def log_objectives(self, objective_losses, weights, step):
        """Log objective losses and weights for monitoring"""
        for obj_name, loss in objective_losses.items():
            self.history[obj_name].append({
                'step': step,
                'loss': loss.item(),
                'weight': weights.get(obj_name, 1.0)
            })
    
    def plot_objective_trajectories(self):
        """Plot objective trajectories over training"""
        fig, axes = plt.subplots(len(self.objectives), 1, figsize=(12, 8))
        
        for i, (obj_name, history) in enumerate(self.history.items()):
            steps = [h['step'] for h in history]
            losses = [h['loss'] for h in history]
            weights = [h['weight'] for h in history]
            
            axes[i].plot(steps, losses, label=f'{obj_name} Loss')
            axes[i].twinx().plot(steps, weights, label=f'{obj_name} Weight', alpha=0.5)
            axes[i].set_title(f'{obj_name} Objective')
            axes[i].legend()
        
        plt.tight_layout()
        return fig
```

This comprehensive guide provides the foundation for implementing multi-objective training and optimization in NeMo RL, covering everything from basic objective combination to advanced Pareto optimization techniques. 