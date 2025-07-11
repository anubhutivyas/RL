# Multi-Objective Tuning

This guide covers multi-objective tuning techniques for NeMo RL, including Pareto optimization, weighted sum methods, and constraint handling to balance multiple competing objectives during training.

## Overview

Multi-objective tuning addresses the challenge of optimizing multiple competing objectives simultaneously. In NeMo RL, this often involves balancing objectives like model performance, training efficiency, memory usage, and safety constraints. This guide provides comprehensive strategies for effective multi-objective optimization.

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

## Optimization Strategies

### 1. Weighted Sum Method

Use weighted sum to combine multiple objectives:

```python
class WeightedSumOptimizer:
    def __init__(self, objectives):
        self.objectives = objectives
        
    def compute_weighted_sum(self, results):
        """Compute weighted sum of objectives."""
        weighted_sum = 0.0
        
        for obj_name, value in results.items():
            weight = self.objectives[obj_name]['weight']
            direction = self.objectives[obj_name]['direction']
            
            # Normalize value to [0, 1]
            normalized_value = self.normalize_value(obj_name, value)
            
            # Apply direction
            if direction == 'minimize':
                normalized_value = 1 - normalized_value
                
            weighted_sum += weight * normalized_value
            
        return weighted_sum
        
    def optimize_weighted_sum(self, parameter_space, n_trials=100):
        """Optimize using weighted sum method."""
        best_params = None
        best_score = float('-inf')
        
        for trial in range(n_trials):
            # Sample parameters
            params = self.sample_parameters(parameter_space)
            
            # Train and evaluate model
            model = self.train_model(params)
            results = self.evaluate_objectives(model)
            
            # Compute weighted sum
            score = self.compute_weighted_sum(results)
            
            # Update best solution
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params, best_score
```

### 2. Epsilon-Constraint Method

Use epsilon-constraint method to handle constraints:

```python
class EpsilonConstraintOptimizer:
    def __init__(self, primary_objective, constraints):
        self.primary_objective = primary_objective
        self.constraints = constraints
        
    def optimize_with_constraints(self, parameter_space, epsilon_values):
        """Optimize primary objective subject to constraints."""
        solutions = []
        
        for epsilon in epsilon_values:
            # Set constraint bounds
            constraint_bounds = self.set_constraint_bounds(epsilon)
            
            # Optimize primary objective with constraints
            solution = self.optimize_constrained(
                parameter_space, 
                constraint_bounds
            )
            
            if solution is not None:
                solutions.append(solution)
                
        return solutions
        
    def optimize_constrained(self, parameter_space, constraint_bounds):
        """Optimize with constraint handling."""
        def objective_function(params):
            # Check constraints
            for constraint_name, bound in constraint_bounds.items():
                constraint_value = self.evaluate_constraint(params, constraint_name)
                if constraint_value > bound:
                    return float('inf')  # Penalty for constraint violation
                    
            # Evaluate primary objective
            model = self.train_model(params)
            return self.evaluate_objective(model, self.primary_objective)
            
        # Run optimization
        return self.run_optimization(objective_function, parameter_space)
```

### 3. Multi-Objective Evolutionary Algorithm

Implement NSGA-II for multi-objective optimization:

```python
class NSGAIIOptimizer:
    def __init__(self, objectives, population_size=100):
        self.objectives = objectives
        self.population_size = population_size
        self.population = []
        
    def initialize_population(self, parameter_space):
        """Initialize population with random solutions."""
        for i in range(self.population_size):
            params = self.sample_parameters(parameter_space)
            self.population.append({
                'params': params,
                'objectives': None,
                'rank': None,
                'crowding_distance': None
            })
            
    def evaluate_population(self):
        """Evaluate all solutions in population."""
        for individual in self.population:
            if individual['objectives'] is None:
                model = self.train_model(individual['params'])
                individual['objectives'] = self.evaluate_objectives(model)
                
    def fast_non_dominated_sort(self):
        """Perform fast non-dominated sorting."""
        fronts = [[]]
        
        for individual in self.population:
            individual['domination_count'] = 0
            individual['dominated_solutions'] = []
            
            for other in self.population:
                if self.dominates(individual, other):
                    individual['dominated_solutions'].append(other)
                elif self.dominates(other, individual):
                    individual['domination_count'] += 1
                    
            if individual['domination_count'] == 0:
                individual['rank'] = 0
                fronts[0].append(individual)
                
        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual['dominated_solutions']:
                    dominated['domination_count'] -= 1
                    if dominated['domination_count'] == 0:
                        dominated['rank'] = i + 1
                        next_front.append(dominated)
            i += 1
            if next_front:
                fronts.append(next_front)
                
        return fronts
        
    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for a front."""
        if len(front) <= 2:
            for individual in front:
                individual['crowding_distance'] = float('inf')
            return
            
        for individual in front:
            individual['crowding_distance'] = 0
            
        for obj_name in self.objectives:
            # Sort by objective
            front.sort(key=lambda x: x['objectives'][obj_name])
            
            # Set boundary points
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')
            
            # Calculate crowding distance
            obj_range = front[-1]['objectives'][obj_name] - front[0]['objectives'][obj_name]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    front[i]['crowding_distance'] += (
                        front[i+1]['objectives'][obj_name] - 
                        front[i-1]['objectives'][obj_name]
                    ) / obj_range
                    
    def tournament_selection(self, tournament_size=2):
        """Tournament selection based on rank and crowding distance."""
        tournament = random.sample(self.population, tournament_size)
        
        # Sort by rank, then by crowding distance
        tournament.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
        
        return tournament[0]
        
    def crossover(self, parent1, parent2):
        """Crossover operation."""
        child_params = {}
        
        for param_name in parent1['params']:
            if random.random() < 0.5:
                child_params[param_name] = parent1['params'][param_name]
            else:
                child_params[param_name] = parent2['params'][param_name]
                
        return {'params': child_params, 'objectives': None}
        
    def mutate(self, individual, mutation_rate=0.1):
        """Mutation operation."""
        for param_name, value in individual['params'].items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    individual['params'][param_name] *= random.uniform(0.8, 1.2)
                elif isinstance(value, int):
                    individual['params'][param_name] += random.randint(-1, 1)
                    
    def evolve(self, generations=50):
        """Evolve population for specified number of generations."""
        for generation in range(generations):
            # Evaluate current population
            self.evaluate_population()
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort()
            
            # Calculate crowding distance
            for front in fronts:
                self.calculate_crowding_distance(front)
                
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                offspring.append(child)
                
            # Combine parent and offspring populations
            combined = self.population + offspring
            self.evaluate_population()
            
            # Select next generation
            fronts = self.fast_non_dominated_sort()
            new_population = []
            
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Sort front by crowding distance
                    front.sort(key=lambda x: x['crowding_distance'], reverse=True)
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
                    
            self.population = new_population
            
        return self.get_pareto_front()
```

## Constraint Handling

### 1. Penalty Method

Use penalty method for constraint handling:

```python
class PenaltyOptimizer:
    def __init__(self, objectives, constraints):
        self.objectives = objectives
        self.constraints = constraints
        self.penalty_factor = 1000.0
        
    def compute_penalty(self, results):
        """Compute penalty for constraint violations."""
        penalty = 0.0
        
        for constraint_name, constraint_config in self.constraints.items():
            constraint_value = results.get(constraint_name, 0)
            constraint_bound = constraint_config['bound']
            constraint_type = constraint_config['type']  # 'upper' or 'lower'
            
            if constraint_type == 'upper' and constraint_value > constraint_bound:
                penalty += self.penalty_factor * (constraint_value - constraint_bound)
            elif constraint_type == 'lower' and constraint_value < constraint_bound:
                penalty += self.penalty_factor * (constraint_bound - constraint_value)
                
        return penalty
        
    def optimize_with_penalty(self, parameter_space, n_trials=100):
        """Optimize with penalty method."""
        best_params = None
        best_score = float('-inf')
        
        for trial in range(n_trials):
            params = self.sample_parameters(parameter_space)
            model = self.train_model(params)
            results = self.evaluate_objectives(model)
            
            # Compute weighted sum
            score = self.compute_weighted_sum(results)
            
            # Apply penalty
            penalty = self.compute_penalty(results)
            final_score = score - penalty
            
            if final_score > best_score:
                best_score = final_score
                best_params = params
                
        return best_params, best_score
```

### 2. Feasible Region Optimization

Optimize within feasible region:

```python
class FeasibleRegionOptimizer:
    def __init__(self, objectives, constraints):
        self.objectives = objectives
        self.constraints = constraints
        
    def is_feasible(self, results):
        """Check if solution is feasible."""
        for constraint_name, constraint_config in self.constraints.items():
            constraint_value = results.get(constraint_name, 0)
            constraint_bound = constraint_config['bound']
            constraint_type = constraint_config['type']
            
            if constraint_type == 'upper' and constraint_value > constraint_bound:
                return False
            elif constraint_type == 'lower' and constraint_value < constraint_bound:
                return False
                
        return True
        
    def optimize_feasible(self, parameter_space, n_trials=100):
        """Optimize within feasible region."""
        feasible_solutions = []
        
        for trial in range(n_trials):
            params = self.sample_parameters(parameter_space)
            model = self.train_model(params)
            results = self.evaluate_objectives(model)
            
            if self.is_feasible(results):
                feasible_solutions.append({
                    'params': params,
                    'results': results,
                    'score': self.compute_weighted_sum(results)
                })
                
        if feasible_solutions:
            # Sort by score and return best
            feasible_solutions.sort(key=lambda x: x['score'], reverse=True)
            best_solution = feasible_solutions[0]
            return best_solution['params'], best_solution['score']
        else:
            raise ValueError("No feasible solutions found")
```

## Advanced Techniques

### 1. Preference-Based Optimization

Use preference-based methods for multi-objective optimization:

```python
class PreferenceBasedOptimizer:
    def __init__(self, objectives, preferences):
        self.objectives = objectives
        self.preferences = preferences
        
    def compute_preference_score(self, results):
        """Compute score based on preferences."""
        score = 0.0
        
        for obj_name, value in results.items():
            preference = self.preferences.get(obj_name, {})
            target = preference.get('target', 0.5)
            weight = preference.get('weight', 1.0)
            
            # Normalize value
            normalized_value = self.normalize_value(obj_name, value)
            
            # Compute distance from target
            distance = abs(normalized_value - target)
            score += weight * (1 - distance)
            
        return score
        
    def optimize_with_preferences(self, parameter_space, n_trials=100):
        """Optimize based on preferences."""
        best_params = None
        best_score = float('-inf')
        
        for trial in range(n_trials):
            params = self.sample_parameters(parameter_space)
            model = self.train_model(params)
            results = self.evaluate_objectives(model)
            
            score = self.compute_preference_score(results)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params, best_score
```

### 2. Interactive Multi-Objective Optimization

Implement interactive optimization with human feedback:

```python
class InteractiveOptimizer:
    def __init__(self, objectives):
        self.objectives = objectives
        self.solutions = []
        self.user_feedback = []
        
    def get_user_preference(self, solution1, solution2):
        """Get user preference between two solutions."""
        print("Compare the following solutions:")
        print(f"Solution A: {solution1['results']}")
        print(f"Solution B: {solution2['results']}")
        
        preference = input("Which do you prefer? (A/B): ").upper()
        return preference
        
    def update_preference_model(self, solution1, solution2, preference):
        """Update preference model based on user feedback."""
        self.user_feedback.append({
            'solution1': solution1,
            'solution2': solution2,
            'preference': preference
        })
        
    def optimize_interactive(self, parameter_space, n_iterations=10):
        """Interactive multi-objective optimization."""
        # Generate initial population
        self.generate_initial_population(parameter_space)
        
        for iteration in range(n_iterations):
            # Present solutions to user
            pareto_front = self.get_pareto_front()
            
            if len(pareto_front) >= 2:
                # Get user preference
                solution1, solution2 = random.sample(pareto_front, 2)
                preference = self.get_user_preference(solution1, solution2)
                
                # Update preference model
                self.update_preference_model(solution1, solution2, preference)
                
                # Generate new solutions based on preferences
                self.generate_new_solutions(parameter_space)
                
        return self.get_best_solution()
```

## Integration with NeMo RL

### 1. Configuration-based Multi-Objective Optimization

```yaml
# multi_objective_config.yaml
multi_objective:
  method: "nsga_ii"
  population_size: 100
  generations: 50
  
  objectives:
    accuracy:
      direction: "maximize"
      weight: 1.0
      target: 0.9
      
    efficiency:
      direction: "maximize"
      weight: 0.5
      target: 1000  # tokens per second
      
    memory_usage:
      direction: "minimize"
      weight: 0.3
      target: 8  # GB
      
    training_time:
      direction: "minimize"
      weight: 0.2
      target: 3600  # seconds
      
  constraints:
    memory_usage:
      type: "upper"
      bound: 16  # GB
      
    training_time:
      type: "upper"
      bound: 7200  # 2 hours
```

### 2. Automated Multi-Objective Pipeline

```python
class AutomatedMultiObjectiveOptimizer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.optimizer = self.create_optimizer()
        
    def create_optimizer(self):
        """Create optimizer based on configuration."""
        method = self.config['multi_objective']['method']
        
        if method == 'nsga_ii':
            return NSGAIIOptimizer(
                objectives=self.config['multi_objective']['objectives'],
                population_size=self.config['multi_objective']['population_size']
            )
        elif method == 'weighted_sum':
            return WeightedSumOptimizer(
                objectives=self.config['multi_objective']['objectives']
            )
        elif method == 'epsilon_constraint':
            return EpsilonConstraintOptimizer(
                primary_objective=self.config['multi_objective']['primary_objective'],
                constraints=self.config['multi_objective']['constraints']
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
    def run_optimization(self):
        """Run multi-objective optimization."""
        print("Starting multi-objective optimization...")
        
        # Run optimization
        if isinstance(self.optimizer, NSGAIIOptimizer):
            pareto_front = self.optimizer.evolve(
                generations=self.config['multi_objective']['generations']
            )
        else:
            best_params, best_score = self.optimizer.optimize()
            pareto_front = [{'params': best_params, 'score': best_score}]
            
        # Save results
        self.save_results(pareto_front)
        
        # Generate report
        self.generate_report(pareto_front)
        
        return pareto_front
```

## Best Practices

### 1. Objective Selection

1. **Choose relevant objectives** for your specific problem
2. **Ensure objectives are measurable** and well-defined
3. **Consider computational cost** of objective evaluation
4. **Balance objective importance** based on requirements

### 2. Constraint Handling

1. **Define realistic constraints** based on available resources
2. **Use appropriate constraint handling** methods
3. **Monitor constraint violations** during optimization
4. **Adjust constraint bounds** if necessary

### 3. Optimization Strategy

1. **Choose appropriate optimization method** for your problem
2. **Use population-based methods** for complex multi-objective problems
3. **Implement proper termination criteria**
4. **Monitor optimization progress** continuously

### 4. Solution Analysis

1. **Analyze Pareto front** to understand trade-offs
2. **Visualize multi-objective results** for insights
3. **Consider user preferences** in solution selection
4. **Document optimization process** for reproducibility

## Troubleshooting

### Common Issues

1. **No Feasible Solutions**
   ```python
   # Relax constraints
   constraint_bounds = {
       'memory_usage': 32,  # Increase from 16
       'training_time': 14400  # Increase from 7200
   }
   
   # Adjust objective weights
   weights = {
       'accuracy': 0.8,  # Reduce from 1.0
       'efficiency': 0.6  # Increase from 0.5
   }
   ```

2. **Poor Convergence**
   ```python
   # Increase population size
   population_size = 200  # Increase from 100
   
   # Increase generations
   generations = 100  # Increase from 50
   
   # Adjust mutation rate
   mutation_rate = 0.2  # Increase from 0.1
   ```

3. **Computational Budget Exceeded**
   ```python
   # Use early termination
   early_termination = True
   patience = 10
   
   # Reduce evaluation frequency
   eval_interval = 5  # Evaluate every 5 generations
   ```

4. **Unbalanced Objectives**
   ```python
   # Normalize objectives properly
   normalization_method = "min_max"
   
   # Adjust objective scales
   objective_scales = {
       'accuracy': 1.0,
       'efficiency': 0.001,  # Scale down
       'memory_usage': 0.0625  # Scale down
   }
   ```

## Next Steps

- [Hyperparameter Optimization](hyperparameter-optimization) - Learn hyperparameter tuning
- [Training Stability](training-stability) - Ensure stable training
- [Custom Loss Functions](custom-loss-functions) - Design custom objectives
- [Advanced Performance](../performance/index) - Performance optimization 