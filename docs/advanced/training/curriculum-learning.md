# Curriculum Learning

Implement progressive difficulty scheduling to improve training efficiency and model performance. Design adaptive curricula that respond to model progress and learning dynamics.

## Overview

Curriculum learning is a training strategy that progressively increases task difficulty based on model performance. Instead of training on all data simultaneously, curriculum learning starts with simpler examples and gradually introduces more complex ones. This approach can significantly improve training efficiency, convergence speed, and final model performance.

## Key Concepts

### Progressive Difficulty
- **Task complexity**: Gradually increase the complexity of training tasks
- **Data difficulty**: Start with easier examples and progress to harder ones
- **Concept integration**: Introduce new concepts progressively

### Adaptive Scheduling
- **Performance-based**: Adjust curriculum based on model performance
- **Learning dynamics**: Respond to model learning patterns
- **Automatic progression**: Automatically advance curriculum when ready

### Curriculum Design
- **Difficulty metrics**: Define metrics for task difficulty
- **Progression strategies**: Design strategies for curriculum progression
- **Evaluation criteria**: Determine when to advance to next level

## Architecture Patterns

### Basic Curriculum Learning

```python
class CurriculumLearner:
    def __init__(self, curriculum_stages, progression_criteria):
        self.curriculum_stages = curriculum_stages
        self.progression_criteria = progression_criteria
        self.current_stage = 0
        self.stage_performance = {}
    
    def get_training_data(self, current_step):
        """Get training data for current curriculum stage"""
        current_stage_data = self.curriculum_stages[self.current_stage]
        
        # Check if ready to progress
        if self._should_progress(current_step):
            self._advance_curriculum()
            current_stage_data = self.curriculum_stages[self.current_stage]
        
        return current_stage_data
    
    def _should_progress(self, current_step):
        """Check if model is ready to progress to next stage"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False
        
        criteria = self.progression_criteria[self.current_stage]
        performance = self.stage_performance.get(self.current_stage, {})
        
        # Check performance criteria
        for metric, threshold in criteria.items():
            if performance.get(metric, 0) < threshold:
                return False
        
        return True
    
    def _advance_curriculum(self):
        """Advance to next curriculum stage"""
        self.current_stage += 1
        print(f"Advancing to curriculum stage {self.current_stage}")
```

### Adaptive Curriculum Scheduling

```python
class AdaptiveCurriculum:
    def __init__(self, difficulty_estimator, performance_tracker):
        self.difficulty_estimator = difficulty_estimator
        self.performance_tracker = performance_tracker
        self.curriculum_history = []
        self.adaptation_strategy = 'performance_based'
    
    def adapt_curriculum(self, current_performance, current_difficulty):
        """Adapt curriculum based on current performance and difficulty"""
        if self.adaptation_strategy == 'performance_based':
            return self._performance_based_adaptation(current_performance, current_difficulty)
        elif self.adaptation_strategy == 'learning_rate_based':
            return self._learning_rate_based_adaptation(current_performance, current_difficulty)
        else:
            return self._fixed_curriculum_adaptation(current_performance, current_difficulty)
    
    def _performance_based_adaptation(self, performance, difficulty):
        """Adapt curriculum based on performance metrics"""
        # Calculate learning progress
        progress_rate = self.performance_tracker.calculate_progress_rate(performance)
        
        # Determine next difficulty level
        if progress_rate > 0.8:  # High progress
            next_difficulty = min(difficulty * 1.2, self.max_difficulty)
        elif progress_rate > 0.5:  # Moderate progress
            next_difficulty = difficulty
        else:  # Low progress
            next_difficulty = max(difficulty * 0.8, self.min_difficulty)
        
        return next_difficulty
    
    def _learning_rate_based_adaptation(self, performance, difficulty):
        """Adapt curriculum based on learning rate"""
        learning_rate = self.performance_tracker.calculate_learning_rate(performance)
        
        # Adjust difficulty based on learning rate
        if learning_rate > self.target_learning_rate:
            next_difficulty = min(difficulty * 1.1, self.max_difficulty)
        elif learning_rate < self.target_learning_rate * 0.5:
            next_difficulty = max(difficulty * 0.9, self.min_difficulty)
        else:
            next_difficulty = difficulty
        
        return next_difficulty
```

## Implementation Strategies

### Difficulty Estimation

```python
class DifficultyEstimator:
    def __init__(self, difficulty_metrics):
        self.difficulty_metrics = difficulty_metrics
    
    def estimate_difficulty(self, task_data):
        """Estimate difficulty of a task or dataset"""
        difficulty_scores = {}
        
        for metric_name, metric_func in self.difficulty_metrics.items():
            difficulty_scores[metric_name] = metric_func(task_data)
        
        # Combine difficulty scores
        overall_difficulty = self._combine_difficulty_scores(difficulty_scores)
        
        return overall_difficulty, difficulty_scores
    
    def _combine_difficulty_scores(self, scores):
        """Combine multiple difficulty metrics into overall score"""
        # Weighted average of difficulty scores
        weights = {
            'complexity': 0.4,
            'length': 0.2,
            'ambiguity': 0.2,
            'domain_specificity': 0.2
        }
        
        overall_score = 0.0
        for metric, score in scores.items():
            overall_score += weights.get(metric, 0.1) * score
        
        return overall_score
```

### Performance Tracking

```python
class PerformanceTracker:
    def __init__(self, metrics, window_size=100):
        self.metrics = metrics
        self.window_size = window_size
        self.performance_history = {metric: [] for metric in metrics}
    
    def update_performance(self, current_metrics):
        """Update performance tracking with current metrics"""
        for metric_name, value in current_metrics.items():
            if metric_name in self.performance_history:
                self.performance_history[metric_name].append(value)
                
                # Keep only recent history
                if len(self.performance_history[metric_name]) > self.window_size:
                    self.performance_history[metric_name].pop(0)
    
    def calculate_progress_rate(self, current_metrics):
        """Calculate progress rate based on recent performance"""
        progress_rates = {}
        
        for metric_name in self.metrics:
            if metric_name in self.performance_history:
                recent_performance = self.performance_history[metric_name][-10:]
                if len(recent_performance) >= 2:
                    # Calculate improvement rate
                    improvement = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
                    progress_rates[metric_name] = improvement
        
        # Average progress rate across metrics
        if progress_rates:
            return sum(progress_rates.values()) / len(progress_rates)
        else:
            return 0.0
    
    def calculate_learning_rate(self, current_metrics):
        """Calculate learning rate based on performance trends"""
        learning_rates = {}
        
        for metric_name in self.metrics:
            if metric_name in self.performance_history:
                recent_performance = self.performance_history[metric_name][-20:]
                if len(recent_performance) >= 5:
                    # Fit linear trend to recent performance
                    x = np.arange(len(recent_performance))
                    y = np.array(recent_performance)
                    slope, _ = np.polyfit(x, y, 1)
                    learning_rates[metric_name] = slope
        
        # Average learning rate across metrics
        if learning_rates:
            return sum(learning_rates.values()) / len(learning_rates)
        else:
            return 0.0
```

## Advanced Curriculum Techniques

### Multi-Dimensional Curriculum

```python
class MultiDimensionalCurriculum:
    def __init__(self, dimensions):
        self.dimensions = dimensions  # e.g., ['complexity', 'length', 'domain']
        self.current_levels = {dim: 0 for dim in dimensions}
        self.max_levels = {dim: 10 for dim in dimensions}
    
    def get_current_tasks(self):
        """Get tasks for current curriculum levels across all dimensions"""
        current_tasks = []
        
        for task in self.all_tasks:
            task_levels = self._get_task_levels(task)
            
            # Check if task matches current curriculum levels
            if self._matches_current_levels(task_levels):
                current_tasks.append(task)
        
        return current_tasks
    
    def _get_task_levels(self, task):
        """Get difficulty levels for a task across all dimensions"""
        levels = {}
        for dimension in self.dimensions:
            levels[dimension] = self._estimate_dimension_level(task, dimension)
        return levels
    
    def _matches_current_levels(self, task_levels):
        """Check if task matches current curriculum levels"""
        for dimension in self.dimensions:
            if task_levels[dimension] > self.current_levels[dimension]:
                return False
        return True
    
    def advance_dimension(self, dimension):
        """Advance curriculum in a specific dimension"""
        if self.current_levels[dimension] < self.max_levels[dimension]:
            self.current_levels[dimension] += 1
            print(f"Advanced {dimension} to level {self.current_levels[dimension]}")
```

### Dynamic Task Generation

```python
class DynamicTaskGenerator:
    def __init__(self, base_tasks, difficulty_ranges):
        self.base_tasks = base_tasks
        self.difficulty_ranges = difficulty_ranges
        self.task_generator = TaskGenerator()
    
    def generate_tasks_for_level(self, current_level):
        """Generate tasks appropriate for current curriculum level"""
        generated_tasks = []
        
        for base_task in self.base_tasks:
            # Generate variations of base task
            variations = self.task_generator.generate_variations(
                base_task, 
                difficulty_level=current_level,
                num_variations=5
            )
            
            generated_tasks.extend(variations)
        
        return generated_tasks
    
    def adapt_task_difficulty(self, task, target_difficulty):
        """Adapt task difficulty to target level"""
        adapted_task = self.task_generator.adapt_difficulty(
            task, 
            current_difficulty=self._estimate_difficulty(task),
            target_difficulty=target_difficulty
        )
        
        return adapted_task
```

## Real-World Examples

### Mathematical Reasoning Curriculum

```python
# Example: Progressive curriculum for mathematical reasoning
class MathCurriculum:
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'Basic Arithmetic',
                'difficulty': 1,
                'topics': ['addition', 'subtraction', 'multiplication', 'division'],
                'problem_types': ['single_step', 'word_problems'],
                'max_complexity': 2
            },
            {
                'name': 'Algebra Fundamentals',
                'difficulty': 2,
                'topics': ['linear_equations', 'inequalities', 'factoring'],
                'problem_types': ['multi_step', 'variable_manipulation'],
                'max_complexity': 4
            },
            {
                'name': 'Advanced Algebra',
                'difficulty': 3,
                'topics': ['quadratic_equations', 'systems', 'functions'],
                'problem_types': ['proofs', 'complex_manipulation'],
                'max_complexity': 6
            },
            {
                'name': 'Calculus Introduction',
                'difficulty': 4,
                'topics': ['limits', 'derivatives', 'integrals'],
                'problem_types': ['theoretical', 'applications'],
                'max_complexity': 8
            }
        ]
        
        self.progression_criteria = {
            0: {'accuracy': 0.85, 'completion_rate': 0.9},
            1: {'accuracy': 0.80, 'completion_rate': 0.85},
            2: {'accuracy': 0.75, 'completion_rate': 0.80},
            3: {'accuracy': 0.70, 'completion_rate': 0.75}
        }
    
    def get_training_data(self, current_stage):
        """Get training data for current curriculum stage"""
        stage_config = self.curriculum_stages[current_stage]
        
        # Filter dataset based on stage criteria
        filtered_data = self._filter_by_difficulty(
            self.math_dataset,
            max_complexity=stage_config['max_complexity'],
            topics=stage_config['topics'],
            problem_types=stage_config['problem_types']
        )
        
        return filtered_data
```

### Code Generation Curriculum

```python
# Example: Progressive curriculum for code generation
class CodeGenerationCurriculum:
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'Simple Functions',
                'difficulty': 1,
                'features': ['basic_syntax', 'simple_loops', 'basic_conditionals'],
                'max_lines': 10,
                'languages': ['python']
            },
            {
                'name': 'Data Structures',
                'difficulty': 2,
                'features': ['lists', 'dictionaries', 'classes'],
                'max_lines': 25,
                'languages': ['python', 'javascript']
            },
            {
                'name': 'Algorithms',
                'difficulty': 3,
                'features': ['recursion', 'sorting', 'searching'],
                'max_lines': 50,
                'languages': ['python', 'javascript', 'java']
            },
            {
                'name': 'System Design',
                'difficulty': 4,
                'features': ['design_patterns', 'optimization', 'testing'],
                'max_lines': 100,
                'languages': ['python', 'javascript', 'java', 'cpp']
            }
        ]
    
    def adapt_code_task(self, task, target_difficulty):
        """Adapt code generation task to target difficulty"""
        adapted_task = task.copy()
        
        # Adjust task complexity based on target difficulty
        if target_difficulty == 1:
            adapted_task['constraints'] = {
                'max_lines': 10,
                'allowed_features': ['basic_syntax', 'simple_loops'],
                'required_tests': False
            }
        elif target_difficulty == 2:
            adapted_task['constraints'] = {
                'max_lines': 25,
                'allowed_features': ['lists', 'dictionaries'],
                'required_tests': True
            }
        # ... continue for other difficulty levels
        
        return adapted_task
```

## Best Practices

### Curriculum Design
- **Clear progression**: Define clear progression criteria between stages
- **Balanced difficulty**: Ensure smooth difficulty transitions
- **Domain-specific**: Design curricula specific to your domain
- **Flexible adaptation**: Allow curriculum to adapt to model performance

### Implementation
- **Performance monitoring**: Track performance across curriculum stages
- **Automatic progression**: Implement automatic curriculum progression
- **Fallback strategies**: Have fallback strategies for difficult stages
- **Evaluation**: Evaluate curriculum effectiveness regularly

### Training Stability
- **Gradual transitions**: Ensure smooth transitions between curriculum stages
- **Performance validation**: Validate performance before advancing
- **Rollback capability**: Allow rolling back to previous stages if needed
- **Monitoring**: Monitor training stability during curriculum transitions

### Production Considerations
- **Curriculum versioning**: Version curricula for reproducibility
- **A/B testing**: Test different curricula in production
- **Performance tracking**: Track performance across curriculum stages
- **Adaptive deployment**: Deploy curricula that adapt to production data

## Monitoring and Observability

```python
class CurriculumMonitor:
    def __init__(self, curriculum_stages):
        self.curriculum_stages = curriculum_stages
        self.stage_performance = {stage: [] for stage in range(len(curriculum_stages))}
        self.progression_history = []
    
    def log_stage_performance(self, stage, performance_metrics):
        """Log performance for current curriculum stage"""
        self.stage_performance[stage].append({
            'step': len(self.stage_performance[stage]),
            'metrics': performance_metrics,
            'timestamp': time.time()
        })
    
    def log_progression(self, from_stage, to_stage, reason):
        """Log curriculum progression"""
        self.progression_history.append({
            'from_stage': from_stage,
            'to_stage': to_stage,
            'reason': reason,
            'timestamp': time.time()
        })
    
    def plot_curriculum_progress(self):
        """Plot curriculum progression and performance"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot performance by stage
        for stage, performance in self.stage_performance.items():
            if performance:
                steps = [p['step'] for p in performance]
                accuracies = [p['metrics']['accuracy'] for p in performance]
                ax1.plot(steps, accuracies, label=f'Stage {stage}')
        
        ax1.set_title('Performance by Curriculum Stage')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot progression timeline
        progression_times = [p['timestamp'] for p in self.progression_history]
        progression_stages = [p['to_stage'] for p in self.progression_history]
        ax2.plot(progression_times, progression_stages, 'o-')
        ax2.set_title('Curriculum Progression Timeline')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Curriculum Stage')
        
        plt.tight_layout()
        return fig
```

This comprehensive guide provides the foundation for implementing curriculum learning in NeMo RL, covering everything from basic progressive difficulty to advanced adaptive scheduling techniques. 