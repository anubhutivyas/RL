# Curriculum Learning

Implement progressive difficulty scheduling to improve training efficiency and model performance. Design adaptive curricula that respond to model progress and learning dynamics. This guide covers both basic curriculum learning and advanced adaptive techniques.

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

## Core Concepts

### Curriculum Difficulty

The difficulty of training examples can be measured along multiple dimensions:

```python
from nemo_rl.advanced import CurriculumDifficulty

# Define difficulty metrics
difficulty_metrics = {
    "sequence_length": lambda x: len(x.split()),
    "complexity": lambda x: calculate_complexity_score(x),
    "domain_specificity": lambda x: calculate_domain_score(x),
    "reward_sparsity": lambda x: calculate_reward_sparsity(x)
}

# Create difficulty calculator
difficulty_calculator = CurriculumDifficulty(metrics=difficulty_metrics)
```

### Performance Tracking

Track model performance to inform curriculum adjustments:

```python
from nemo_rl.advanced import PerformanceTracker

# Create performance tracker
performance_tracker = PerformanceTracker(
    metrics=["accuracy", "reward", "loss"],
    window_size=100,  # Rolling average window
    thresholds={
        "improvement": 0.01,
        "plateau": 0.001,
        "degradation": -0.01
    }
)

# Update performance
performance_tracker.update(
    epoch=epoch,
    metrics={"accuracy": 0.85, "reward": 0.72, "loss": 0.15}
)
```

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
```

## Adaptive Curriculum Algorithms

### Difficulty-Based Adaptation

Adjust difficulty based on current performance:

```python
from nemo_rl.advanced import AdaptiveCurriculum

# Create adaptive curriculum
curriculum = AdaptiveCurriculum(
    initial_difficulty=0.3,
    adaptation_rate=0.1,
    performance_threshold=0.8,
    difficulty_range=(0.1, 1.0)
)

# Update curriculum based on performance
def update_curriculum(performance):
    if performance > curriculum.performance_threshold:
        # Increase difficulty
        new_difficulty = min(
            curriculum.current_difficulty + curriculum.adaptation_rate,
            curriculum.difficulty_range[1]
        )
    else:
        # Decrease difficulty
        new_difficulty = max(
            curriculum.current_difficulty - curriculum.adaptation_rate,
            curriculum.difficulty_range[0]
        )
    
    curriculum.update_difficulty(new_difficulty)
    return curriculum.get_filtered_dataset(dataset)
```

### Multi-Objective Curriculum

Balance multiple objectives in curriculum design:

```python
from nemo_rl.advanced import MultiObjectiveCurriculum

# Define curriculum objectives
curriculum_objectives = {
    "performance": {
        "weight": 0.6,
        "target": 0.85,
        "direction": "maximize"
    },
    "efficiency": {
        "weight": 0.3,
        "target": 0.9,
        "direction": "maximize"
    },
    "stability": {
        "weight": 0.1,
        "target": 0.95,
        "direction": "maximize"
    }
}

# Create multi-objective curriculum
multi_obj_curriculum = MultiObjectiveCurriculum(
    objectives=curriculum_objectives,
    adaptation_strategy="pareto_optimal"
)
```

## Implementation Strategies

### Curriculum Design Patterns

```python
class CurriculumDesigner:
    def __init__(self, difficulty_metrics, progression_strategy):
        self.difficulty_metrics = difficulty_metrics
        self.progression_strategy = progression_strategy
        self.curriculum_stages = []
    
    def design_curriculum(self, dataset, num_stages=5):
        """Design curriculum stages based on dataset characteristics"""
        # Analyze dataset difficulty distribution
        difficulty_scores = self._analyze_dataset_difficulty(dataset)
        
        # Create curriculum stages
        self.curriculum_stages = self._create_stages(difficulty_scores, num_stages)
        
        return self.curriculum_stages
    
    def _analyze_dataset_difficulty(self, dataset):
        """Analyze difficulty of dataset examples"""
        difficulty_scores = []
        
        for example in dataset:
            score = 0.0
            for metric_name, metric_func in self.difficulty_metrics.items():
                score += metric_func(example)
            difficulty_scores.append(score)
        
        return difficulty_scores
    
    def _create_stages(self, difficulty_scores, num_stages):
        """Create curriculum stages based on difficulty distribution"""
        stages = []
        
        # Sort examples by difficulty
        sorted_indices = np.argsort(difficulty_scores)
        
        # Divide into stages
        stage_size = len(sorted_indices) // num_stages
        
        for i in range(num_stages):
            start_idx = i * stage_size
            end_idx = (i + 1) * stage_size if i < num_stages - 1 else len(sorted_indices)
            
            stage_indices = sorted_indices[start_idx:end_idx]
            stages.append({
                'stage_id': i,
                'indices': stage_indices,
                'difficulty_range': (
                    difficulty_scores[stage_indices[0]],
                    difficulty_scores[stage_indices[-1]]
                )
            })
        
        return stages
```

### Progression Strategies

```python
class ProgressionStrategy:
    def __init__(self, strategy_type='performance_threshold'):
        self.strategy_type = strategy_type
        self.performance_history = []
    
    def should_progress(self, current_performance, stage_criteria):
        """Determine if curriculum should progress to next stage"""
        if self.strategy_type == 'performance_threshold':
            return self._performance_threshold_check(current_performance, stage_criteria)
        elif self.strategy_type == 'plateau_detection':
            return self._plateau_detection_check(current_performance, stage_criteria)
        elif self.strategy_type == 'confidence_based':
            return self._confidence_based_check(current_performance, stage_criteria)
        else:
            return False
    
    def _performance_threshold_check(self, performance, criteria):
        """Check if performance meets threshold criteria"""
        for metric, threshold in criteria.items():
            if performance.get(metric, 0) < threshold:
                return False
        return True
    
    def _plateau_detection_check(self, performance, criteria):
        """Check if performance has plateaued"""
        if len(self.performance_history) < 10:
            return False
        
        # Calculate improvement rate
        recent_performance = self.performance_history[-10:]
        improvement_rate = self._calculate_improvement_rate(recent_performance)
        
        # Progress if improvement rate is low (plateaued)
        return improvement_rate < criteria.get('plateau_threshold', 0.001)
    
    def _confidence_based_check(self, performance, criteria):
        """Check if model confidence is high enough to progress"""
        confidence = performance.get('confidence', 0)
        return confidence > criteria.get('confidence_threshold', 0.8)
```

## Advanced Techniques

### Dynamic Difficulty Adjustment

```python
class DynamicDifficultyAdjuster:
    def __init__(self, base_difficulty, adjustment_rate=0.1):
        self.base_difficulty = base_difficulty
        self.adjustment_rate = adjustment_rate
        self.current_difficulty = base_difficulty
        self.performance_window = []
    
    def adjust_difficulty(self, current_performance, target_performance):
        """Dynamically adjust difficulty based on performance"""
        # Update performance window
        self.performance_window.append(current_performance)
        if len(self.performance_window) > 10:
            self.performance_window.pop(0)
        
        # Calculate performance trend
        if len(self.performance_window) >= 5:
            trend = self._calculate_performance_trend()
            
            # Adjust difficulty based on trend
            if trend > 0.1:  # Improving
                self.current_difficulty = min(
                    self.current_difficulty + self.adjustment_rate,
                    1.0
                )
            elif trend < -0.1:  # Degrading
                self.current_difficulty = max(
                    self.current_difficulty - self.adjustment_rate,
                    0.1
                )
        
        return self.current_difficulty
    
    def _calculate_performance_trend(self):
        """Calculate performance trend over recent window"""
        if len(self.performance_window) < 5:
            return 0.0
        
        # Use linear regression to calculate trend
        x = np.arange(len(self.performance_window))
        y = np.array(self.performance_window)
        
        slope, _ = np.polyfit(x, y, 1)
        return slope
```

### Multi-Modal Curriculum

```python
class MultiModalCurriculum:
    def __init__(self, modalities):
        self.modalities = modalities
        self.modality_curricula = {}
        
        # Initialize curriculum for each modality
        for modality in modalities:
            self.modality_curricula[modality] = AdaptiveCurriculum(
                initial_difficulty=0.3,
                adaptation_rate=0.1
            )
    
    def get_multi_modal_data(self, current_performance):
        """Get training data for all modalities with appropriate difficulty"""
        multi_modal_data = {}
        
        for modality in self.modalities:
            # Get difficulty for this modality
            difficulty = self.modality_curricula[modality].get_current_difficulty()
            
            # Filter data for this modality and difficulty
            modality_data = self._filter_modality_data(modality, difficulty)
            multi_modal_data[modality] = modality_data
        
        return multi_modal_data
    
    def update_modality_curricula(self, modality_performance):
        """Update curriculum for each modality based on performance"""
        for modality, performance in modality_performance.items():
            if modality in self.modality_curricula:
                self.modality_curricula[modality].adapt_curriculum(performance)
```

## Real-World Examples

### Language Model Curriculum

```python
# Example: Curriculum learning for language models
class LanguageModelCurriculum:
    def __init__(self):
        self.difficulty_metrics = {
            'sequence_length': self._calculate_sequence_length,
            'vocabulary_complexity': self._calculate_vocab_complexity,
            'syntactic_complexity': self._calculate_syntactic_complexity,
            'semantic_complexity': self._calculate_semantic_complexity
        }
        
        self.curriculum_stages = [
            {'name': 'basic_vocabulary', 'difficulty': 0.2},
            {'name': 'simple_sentences', 'difficulty': 0.4},
            {'name': 'complex_sentences', 'difficulty': 0.6},
            {'name': 'paragraphs', 'difficulty': 0.8},
            {'name': 'full_documents', 'difficulty': 1.0}
        ]
    
    def _calculate_sequence_length(self, text):
        """Calculate difficulty based on sequence length"""
        tokens = text.split()
        return min(len(tokens) / 1000.0, 1.0)  # Normalize to [0, 1]
    
    def _calculate_vocab_complexity(self, text):
        """Calculate difficulty based on vocabulary complexity"""
        # Implementation for vocabulary complexity calculation
        return 0.5  # Placeholder
    
    def _calculate_syntactic_complexity(self, text):
        """Calculate difficulty based on syntactic complexity"""
        # Implementation for syntactic complexity calculation
        return 0.5  # Placeholder
    
    def _calculate_semantic_complexity(self, text):
        """Calculate difficulty based on semantic complexity"""
        # Implementation for semantic complexity calculation
        return 0.5  # Placeholder
```

### Reinforcement Learning Curriculum

```python
# Example: Curriculum learning for reinforcement learning
class RLCurriculum:
    def __init__(self):
        self.difficulty_metrics = {
            'environment_complexity': self._calculate_env_complexity,
            'task_difficulty': self._calculate_task_difficulty,
            'reward_sparsity': self._calculate_reward_sparsity,
            'action_space_size': self._calculate_action_space_size
        }
        
        self.curriculum_stages = [
            {'name': 'simple_environment', 'difficulty': 0.2},
            {'name': 'basic_tasks', 'difficulty': 0.4},
            {'name': 'complex_tasks', 'difficulty': 0.6},
            {'name': 'multi_objective', 'difficulty': 0.8},
            {'name': 'full_environment', 'difficulty': 1.0}
        ]
    
    def _calculate_env_complexity(self, environment):
        """Calculate environment complexity"""
        # Implementation for environment complexity calculation
        return 0.5  # Placeholder
    
    def _calculate_task_difficulty(self, task):
        """Calculate task difficulty"""
        # Implementation for task difficulty calculation
        return 0.5  # Placeholder
    
    def _calculate_reward_sparsity(self, environment):
        """Calculate reward sparsity"""
        # Implementation for reward sparsity calculation
        return 0.5  # Placeholder
    
    def _calculate_action_space_size(self, environment):
        """Calculate action space size"""
        # Implementation for action space size calculation
        return 0.5  # Placeholder
```

## Best Practices

### Curriculum Design
- **Gradual progression**: Ensure smooth transitions between difficulty levels
- **Performance monitoring**: Track performance at each curriculum stage
- **Adaptive adjustment**: Allow curriculum to adapt based on model performance
- **Multi-modal consideration**: Design curriculum for different data modalities

### Implementation
- **Efficient filtering**: Use efficient algorithms for data filtering
- **Memory management**: Handle large datasets efficiently
- **Distributed training**: Support curriculum learning in distributed settings
- **Reproducibility**: Ensure curriculum progression is reproducible

### Evaluation
- **Stage-wise evaluation**: Evaluate performance at each curriculum stage
- **Progression analysis**: Analyze curriculum progression patterns
- **Ablation studies**: Compare with non-curriculum training
- **Robustness testing**: Test curriculum robustness across different datasets

## Monitoring and Observability

```python
class CurriculumMonitor:
    def __init__(self, curriculum):
        self.curriculum = curriculum
        self.stage_history = []
        self.performance_history = []
        self.difficulty_history = []
    
    def log_stage_transition(self, stage_id, performance, difficulty):
        """Log curriculum stage transitions"""
        self.stage_history.append({
            'stage_id': stage_id,
            'performance': performance,
            'difficulty': difficulty,
            'timestamp': time.time()
        })
    
    def plot_curriculum_progression(self):
        """Plot curriculum progression over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot stage transitions
        stages = [h['stage_id'] for h in self.stage_history]
        timestamps = [h['timestamp'] for h in self.stage_history]
        ax1.plot(timestamps, stages, 'b-o')
        ax1.set_title('Curriculum Stage Progression')
        ax1.set_ylabel('Stage ID')
        
        # Plot performance vs difficulty
        performances = [h['performance'] for h in self.stage_history]
        difficulties = [h['difficulty'] for h in self.stage_history]
        ax2.scatter(difficulties, performances, c=stages, cmap='viridis')
        ax2.set_xlabel('Difficulty')
        ax2.set_ylabel('Performance')
        ax2.set_title('Performance vs Difficulty')
        
        plt.tight_layout()
        return fig
```

This comprehensive guide provides the foundation for implementing curriculum learning in NeMo RL, covering everything from basic curriculum design to advanced adaptive techniques. 