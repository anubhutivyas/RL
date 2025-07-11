# Adaptive Curriculum Learning

This guide covers adaptive curriculum learning techniques in NeMo RL, which dynamically adjust training difficulty based on model performance to optimize learning efficiency.

## Overview

Adaptive curriculum learning automatically adjusts the difficulty of training examples based on the model's current performance, enabling more efficient training and better convergence. This is particularly valuable for reinforcement learning with language models where task complexity varies significantly.

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
        "metric": "accuracy"
    },
    "diversity": {
        "weight": 0.3,
        "target": 0.7,
        "metric": "example_diversity"
    },
    "efficiency": {
        "weight": 0.1,
        "target": 0.9,
        "metric": "training_efficiency"
    }
}

# Create multi-objective curriculum
multi_curriculum = MultiObjectiveCurriculum(
    objectives=curriculum_objectives,
    adaptation_strategy="pareto_optimal"
)
```

### Dynamic Difficulty Adjustment

Implement dynamic difficulty adjustment based on real-time performance:

```python
from nemo_rl.advanced import DynamicDifficultyAdjuster

# Create dynamic adjuster
adjuster = DynamicDifficultyAdjuster(
    base_difficulty=0.5,
    adjustment_factor=0.1,
    performance_window=50,
    stability_threshold=0.02
)

# Adjust difficulty during training
def adjust_difficulty_during_training(performance_history):
    current_performance = performance_history[-1]
    recent_performance = performance_history[-adjuster.performance_window:]
    
    # Calculate performance trend
    trend = calculate_trend(recent_performance)
    
    if trend > adjuster.stability_threshold:
        # Performance improving, increase difficulty
        adjuster.increase_difficulty()
    elif trend < -adjuster.stability_threshold:
        # Performance degrading, decrease difficulty
        adjuster.decrease_difficulty()
    else:
        # Performance stable, maintain difficulty
        adjuster.maintain_difficulty()
    
    return adjuster.get_current_difficulty()
```

## Implementation Strategies

### Configuration Setup

```yaml
# adaptive_curriculum.yaml
training:
  curriculum:
    type: "adaptive"
    initial_difficulty: 0.3
    adaptation_rate: 0.1
    performance_threshold: 0.8
    difficulty_range: [0.1, 1.0]
    
    metrics:
      - name: "sequence_length"
        weight: 0.4
        normalization: "log"
      - name: "complexity"
        weight: 0.3
        normalization: "z_score"
      - name: "domain_specificity"
        weight: 0.3
        normalization: "min_max"
    
    adaptation_strategy:
      type: "performance_based"
      window_size: 100
      stability_threshold: 0.02
      max_adjustment_rate: 0.2
```

### Training Loop Integration

```python
from nemo_rl.advanced import AdaptiveCurriculumTrainer

# Create adaptive curriculum trainer
trainer = AdaptiveCurriculumTrainer(
    model=model,
    optimizer=optimizer,
    curriculum_config=config["training"]["curriculum"]
)

# Training loop with adaptive curriculum
for epoch in range(num_epochs):
    # Get current curriculum difficulty
    current_difficulty = trainer.get_current_difficulty()
    
    # Filter dataset based on current difficulty
    filtered_dataset = trainer.filter_dataset(
        dataset=train_dataset,
        difficulty=current_difficulty
    )
    
    # Train on filtered dataset
    performance = trainer.train_epoch(filtered_dataset)
    
    # Update curriculum based on performance
    trainer.update_curriculum(performance)
    
    # Log curriculum state
    trainer.log_curriculum_state(epoch)
```

### Difficulty Filtering

Implement difficulty-based data filtering:

```python
class DifficultyFilter:
    def __init__(self, difficulty_metrics, target_difficulty):
        self.difficulty_metrics = difficulty_metrics
        self.target_difficulty = target_difficulty
    
    def filter_examples(self, dataset):
        """Filter examples based on current difficulty level."""
        filtered_examples = []
        
        for example in dataset:
            example_difficulty = self.calculate_example_difficulty(example)
            
            if self.is_appropriate_difficulty(example_difficulty):
                filtered_examples.append(example)
        
        return filtered_examples
    
    def calculate_example_difficulty(self, example):
        """Calculate difficulty score for a single example."""
        difficulty_scores = {}
        
        for metric_name, metric_func in self.difficulty_metrics.items():
            difficulty_scores[metric_name] = metric_func(example)
        
        # Combine scores using weighted average
        total_difficulty = sum(
            score * self.difficulty_metrics[metric_name].get("weight", 1.0)
            for metric_name, score in difficulty_scores.items()
        )
        
        return total_difficulty
    
    def is_appropriate_difficulty(self, example_difficulty):
        """Check if example difficulty is appropriate for current level."""
        tolerance = 0.1
        return abs(example_difficulty - self.target_difficulty) <= tolerance
```

## Advanced Techniques

### Curriculum Scheduling

Implement sophisticated curriculum scheduling:

```python
from nemo_rl.advanced import CurriculumScheduler

# Create curriculum scheduler
scheduler = CurriculumScheduler(
    schedule_type="exponential",
    initial_difficulty=0.2,
    final_difficulty=0.9,
    total_epochs=100,
    warmup_epochs=10
)

# Get difficulty for current epoch
current_difficulty = scheduler.get_difficulty(epoch)

# Custom scheduling function
def custom_schedule(epoch, performance_history):
    """Custom curriculum schedule based on performance."""
    if epoch < 10:
        return 0.2  # Start easy
    elif epoch < 50:
        # Gradual increase based on performance
        avg_performance = np.mean(performance_history[-10:])
        return min(0.2 + (avg_performance - 0.5) * 0.4, 0.6)
    else:
        # Final phase with high difficulty
        return 0.8 + (epoch - 50) * 0.004
```

### Multi-Domain Curriculum

Handle multiple domains with different difficulty levels:

```python
from nemo_rl.advanced import MultiDomainCurriculum

# Define domains and their difficulty characteristics
domains = {
    "mathematics": {
        "difficulty_range": (0.3, 0.9),
        "adaptation_rate": 0.15,
        "performance_threshold": 0.75
    },
    "code_generation": {
        "difficulty_range": (0.4, 0.95),
        "adaptation_rate": 0.12,
        "performance_threshold": 0.8
    },
    "reasoning": {
        "difficulty_range": (0.2, 0.85),
        "adaptation_rate": 0.1,
        "performance_threshold": 0.7
    }
}

# Create multi-domain curriculum
multi_domain_curriculum = MultiDomainCurriculum(
    domains=domains,
    domain_weights={"mathematics": 0.4, "code_generation": 0.4, "reasoning": 0.2}
)
```

### Performance-Based Adaptation

Implement sophisticated performance-based adaptation:

```python
from nemo_rl.advanced import PerformanceBasedAdaptation

# Create performance-based adapter
adapter = PerformanceBasedAdaptation(
    adaptation_strategy="adaptive_rate",
    performance_window=50,
    stability_threshold=0.02,
    max_adaptation_rate=0.2
)

# Adaptive rate calculation
def calculate_adaptive_rate(performance_history):
    """Calculate adaptation rate based on performance stability."""
    recent_performance = performance_history[-adapter.performance_window:]
    performance_variance = np.var(recent_performance)
    
    if performance_variance < adapter.stability_threshold:
        # Stable performance, increase adaptation rate
        return min(adapter.current_rate * 1.1, adapter.max_adaptation_rate)
    else:
        # Unstable performance, decrease adaptation rate
        return max(adapter.current_rate * 0.9, 0.01)
```

## Monitoring and Analysis

### Curriculum Progress Tracking

```python
from nemo_rl.advanced import CurriculumTracker

# Create curriculum tracker
tracker = CurriculumTracker(
    metrics=["difficulty", "performance", "adaptation_rate"],
    save_path="curriculum_logs/"
)

# Track curriculum progress
def track_curriculum_progress(epoch, difficulty, performance, adaptation_rate):
    tracker.log(
        epoch=epoch,
        difficulty=difficulty,
        performance=performance,
        adaptation_rate=adaptation_rate
    )
    
    # Generate progress report
    if epoch % 10 == 0:
        report = tracker.generate_report()
        print(f"Curriculum Progress Report:\n{report}")
```

### Visualization

```python
import matplotlib.pyplot as plt
from nemo_rl.advanced import CurriculumVisualizer

# Create curriculum visualizer
visualizer = CurriculumVisualizer()

# Plot difficulty progression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Difficulty over time
visualizer.plot_difficulty_progression(
    curriculum_history,
    ax=ax1,
    title="Difficulty Progression"
)

# Performance vs difficulty
visualizer.plot_performance_vs_difficulty(
    performance_history,
    difficulty_history,
    ax=ax2,
    title="Performance vs Difficulty"
)

plt.tight_layout()
plt.show()
```

## Practical Applications

### DPO with Adaptive Curriculum

```python
# DPO training with adaptive curriculum
dpo_curriculum_config = {
    "algorithm": "dpo",
    "curriculum": {
        "type": "adaptive",
        "difficulty_metrics": {
            "prompt_complexity": lambda x: calculate_prompt_complexity(x),
            "response_length": lambda x: len(x["chosen"].split()),
            "domain_specificity": lambda x: calculate_domain_specificity(x)
        },
        "adaptation_strategy": "performance_based"
    }
}

# Create adaptive DPO trainer
adaptive_dpo_trainer = AdaptiveDPOTrainer(config=dpo_curriculum_config)
curriculum_history = adaptive_dpo_trainer.train(dataset)
```

### GRPO with Dynamic Difficulty

```python
# GRPO training with dynamic difficulty
grpo_curriculum_config = {
    "algorithm": "grpo",
    "curriculum": {
        "type": "dynamic",
        "difficulty_metrics": {
            "environment_complexity": lambda x: calculate_env_complexity(x),
            "reward_sparsity": lambda x: calculate_reward_sparsity(x),
            "action_space_size": lambda x: calculate_action_space_size(x)
        },
        "adaptation_rate": 0.1
    }
}

# Create dynamic GRPO trainer
dynamic_grpo_trainer = DynamicGRPOTrainer(config=grpo_curriculum_config)
training_history = dynamic_grpo_trainer.train(environment)
```

## Best Practices

### Curriculum Design

1. **Start with easy examples** and gradually increase difficulty
2. **Monitor performance closely** to avoid overwhelming the model
3. **Use multiple difficulty metrics** for comprehensive assessment
4. **Implement smooth transitions** between difficulty levels
5. **Consider domain-specific characteristics** in difficulty calculation

### Adaptation Strategy

1. **Use performance-based adaptation** for automatic difficulty adjustment
2. **Implement stability checks** to avoid rapid difficulty changes
3. **Set appropriate thresholds** for adaptation triggers
4. **Monitor adaptation rate** to ensure smooth progression
5. **Use rolling averages** for stable performance assessment

### Monitoring and Debugging

1. **Track curriculum progression** with detailed logging
2. **Visualize difficulty-performance relationships**
3. **Monitor adaptation frequency** and rate
4. **Validate curriculum effectiveness** with ablation studies
5. **Implement fallback strategies** for poor performance

## Troubleshooting

### Common Issues

1. **Difficulty Oscillation**
   ```python
   # Increase stability threshold
   curriculum.stability_threshold = 0.05
   
   # Use exponential moving average
   curriculum.use_ema = True
   curriculum.ema_alpha = 0.9
   ```

2. **Poor Performance at High Difficulty**
   ```python
   # Implement gradual difficulty increase
   curriculum.max_difficulty_increase = 0.05
   
   # Add difficulty validation
   curriculum.validate_difficulty_increase = True
   ```

3. **Curriculum Stagnation**
   ```python
   # Implement minimum difficulty increase
   curriculum.min_difficulty_increase = 0.01
   
   # Add performance plateau detection
   curriculum.plateau_detection = True
   curriculum.plateau_threshold = 0.001
   ```

4. **Domain Imbalance**
   ```python
   # Implement domain-specific curricula
   curriculum.domain_specific = True
   
   # Balance domain representation
   curriculum.domain_balancing = True
   ```

## Next Steps

- [Curriculum Learning](curriculum-learning) - Learn basic curriculum learning
- [Multi-Objective Training](multi-objective-training) - Balance multiple objectives
- [Training Stability](training-stability) - Ensure stable training
- [Advanced Performance](../performance/index) - Optimize performance 