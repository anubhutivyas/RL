# Multi-Objective Training

Learn to combine multiple loss functions and objectives in a single training pipeline. Balance competing objectives like accuracy, efficiency, and safety while maintaining training stability.

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

## Advanced Techniques

### Pareto Front Optimization

```python
class ParetoOptimizer:
    def __init__(self, objectives, population_size=100):
        self.objectives = objectives
        self.population_size = population_size
        self.population = []
    
    def find_pareto_front(self, model_parameters):
        """Find Pareto optimal solutions"""
        pareto_front = []
        
        for _ in range(self.population_size):
            # Generate candidate solution
            candidate = self._generate_candidate(model_parameters)
            
            # Evaluate objectives
            objective_values = self._evaluate_objectives(candidate)
            
            # Check if Pareto optimal
            if self._is_pareto_optimal(objective_values, pareto_front):
                pareto_front.append(objective_values)
        
        return pareto_front
    
    def _is_pareto_optimal(self, candidate, pareto_front):
        """Check if candidate is Pareto optimal"""
        for solution in pareto_front:
            # Check if existing solution dominates candidate
            if self._dominates(solution, candidate):
                return False
        return True
    
    def _dominates(self, solution_a, solution_b):
        """Check if solution_a dominates solution_b"""
        at_least_as_good = True
        strictly_better = False
        
        for obj_name in self.objectives:
            if solution_a[obj_name] < solution_b[obj_name]:
                at_least_as_good = False
            elif solution_a[obj_name] > solution_b[obj_name]:
                strictly_better = True
        
        return at_least_as_good and strictly_better
```

### Multi-Task Learning Integration

```python
class MultiTaskMultiObjective:
    def __init__(self, tasks, objectives_per_task):
        self.tasks = tasks
        self.objectives_per_task = objectives_per_task
        self.task_weights = {task: 1.0 for task in tasks}
    
    def compute_multi_task_loss(self, predictions, targets):
        """Compute loss across multiple tasks with multiple objectives"""
        total_loss = 0.0
        task_losses = {}
        
        for task_name in self.tasks:
            task_loss = 0.0
            task_objectives = self.objectives_per_task[task_name]
            
            # Compute objectives for this task
            for obj_name, obj_func in task_objectives.items():
                obj_loss = obj_func(predictions[task_name], targets[task_name])
                task_loss += obj_loss
            
            # Apply task weight
            weighted_task_loss = self.task_weights[task_name] * task_loss
            total_loss += weighted_task_loss
            task_losses[task_name] = task_loss
        
        return total_loss, task_losses
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

This comprehensive guide provides the foundation for implementing multi-objective training in NeMo RL, covering everything from basic objective combination to advanced Pareto optimization techniques. 