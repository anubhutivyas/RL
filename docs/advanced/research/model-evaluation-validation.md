---
description: "Build comprehensive evaluation frameworks and implement robust model assessment and comparison strategies"
tags: ["evaluation", "metrics", "assessment", "comparison", "validation"]
categories: ["research-validation"]
---

# Model Evaluation

This guide covers how to build comprehensive evaluation frameworks and implement robust model assessment and comparison strategies for NeMo RL research.

## Overview

Comprehensive model evaluation is essential for understanding model performance, identifying strengths and weaknesses, and making informed decisions about model deployment. This guide provides frameworks for evaluating RL models across multiple dimensions.

## Key Components

### Evaluation Framework

Implement a comprehensive evaluation framework:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import torch

class EvaluationMetric(Enum):
    ACCURACY = "accuracy"
    PREFERENCE_ALIGNMENT = "preference_alignment"
    RESPONSE_QUALITY = "response_quality"
    SAFETY_SCORE = "safety_score"
    EFFICIENCY = "efficiency"
    FAIRNESS = "fairness"

@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    metric_name: str
    value: float
    confidence_interval: Optional[tuple] = None
    metadata: Dict[str, Any] = None

class ModelEvaluator:
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.evaluation_results = {}
        self.evaluation_datasets = {}
        
    def evaluate_model(self, evaluation_tasks: List[str]) -> Dict[str, EvaluationResult]:
        """
        Comprehensive model evaluation
        """
        results = {}
        
        for task in evaluation_tasks:
            if task == "preference_alignment":
                results[task] = self.evaluate_preference_alignment()
            elif task == "response_quality":
                results[task] = self.evaluate_response_quality()
            elif task == "safety":
                results[task] = self.evaluate_safety()
            elif task == "efficiency":
                results[task] = self.evaluate_efficiency()
            elif task == "fairness":
                results[task] = self.evaluate_fairness()
            else:
                raise ValueError(f"Unknown evaluation task: {task}")
        
        self.evaluation_results = results
        return results
    
    def evaluate_preference_alignment(self) -> EvaluationResult:
        """
        Evaluate preference alignment
        """
        # Load preference dataset
        preference_data = self.load_preference_dataset()
        
        alignment_scores = []
        for sample in preference_data:
            # Generate responses for chosen and rejected prompts
            chosen_response = self.generate_response(sample['chosen_prompt'])
            rejected_response = self.generate_response(sample['rejected_prompt'])
            
            # Calculate preference alignment score
            alignment_score = self.calculate_preference_alignment(
                chosen_response, rejected_response, sample['human_preference']
            )
            alignment_scores.append(alignment_score)
        
        # Calculate statistics
        mean_score = np.mean(alignment_scores)
        std_score = np.std(alignment_scores)
        confidence_interval = self.calculate_confidence_interval(alignment_scores)
        
        return EvaluationResult(
            metric_name="preference_alignment",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={
                'std': std_score,
                'num_samples': len(alignment_scores),
                'scores': alignment_scores
            }
        )
    
    def evaluate_response_quality(self) -> EvaluationResult:
        """
        Evaluate response quality
        """
        # Load quality evaluation dataset
        quality_data = self.load_quality_dataset()
        
        quality_scores = []
        for sample in quality_data:
            # Generate response
            response = self.generate_response(sample['prompt'])
            
            # Evaluate quality using multiple criteria
            coherence_score = self.evaluate_coherence(response, sample['context'])
            relevance_score = self.evaluate_relevance(response, sample['prompt'])
            fluency_score = self.evaluate_fluency(response)
            
            # Combine scores
            overall_quality = (coherence_score + relevance_score + fluency_score) / 3
            quality_scores.append(overall_quality)
        
        mean_score = np.mean(quality_scores)
        confidence_interval = self.calculate_confidence_interval(quality_scores)
        
        return EvaluationResult(
            metric_name="response_quality",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={
                'coherence_scores': [self.evaluate_coherence(r, s['context']) for r, s in zip(quality_data)],
                'relevance_scores': [self.evaluate_relevance(r, s['prompt']) for r, s in zip(quality_data)],
                'fluency_scores': [self.evaluate_fluency(r) for r in quality_data]
            }
        )
    
    def evaluate_safety(self) -> EvaluationResult:
        """
        Evaluate model safety
        """
        # Load safety evaluation dataset
        safety_data = self.load_safety_dataset()
        
        safety_scores = []
        for sample in safety_data:
            # Generate response to potentially harmful prompt
            response = self.generate_response(sample['prompt'])
            
            # Evaluate safety
            safety_score = self.evaluate_safety_score(response, sample['harm_category'])
            safety_scores.append(safety_score)
        
        mean_score = np.mean(safety_scores)
        confidence_interval = self.calculate_confidence_interval(safety_scores)
        
        return EvaluationResult(
            metric_name="safety_score",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={
                'harm_categories': [s['harm_category'] for s in safety_data],
                'safety_scores': safety_scores
            }
        )
    
    def evaluate_efficiency(self) -> EvaluationResult:
        """
        Evaluate model efficiency
        """
        # Measure inference time
        inference_times = []
        memory_usage = []
        
        test_prompts = self.load_efficiency_test_data()
        
        for prompt in test_prompts:
            # Measure inference time
            start_time = time.time()
            response = self.generate_response(prompt)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
        
        # Calculate efficiency metrics
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
        
        # Normalize efficiency score (lower is better)
        efficiency_score = 1.0 / (1.0 + avg_inference_time + avg_memory_usage)
        
        return EvaluationResult(
            metric_name="efficiency",
            value=efficiency_score,
            confidence_interval=self.calculate_confidence_interval(inference_times),
            metadata={
                'avg_inference_time': avg_inference_time,
                'avg_memory_usage': avg_memory_usage,
                'inference_times': inference_times
            }
        )
    
    def evaluate_fairness(self) -> EvaluationResult:
        """
        Evaluate model fairness
        """
        # Load fairness evaluation dataset
        fairness_data = self.load_fairness_dataset()
        
        fairness_scores = []
        demographic_performance = {}
        
        for sample in fairness_data:
            # Generate response
            response = self.generate_response(sample['prompt'])
            
            # Evaluate performance
            performance_score = self.evaluate_performance(response, sample['expected_output'])
            
            # Group by demographic
            demographic = sample['demographic']
            if demographic not in demographic_performance:
                demographic_performance[demographic] = []
            demographic_performance[demographic].append(performance_score)
        
        # Calculate fairness metrics
        demographic_means = {demo: np.mean(scores) for demo, scores in demographic_performance.items()}
        
        # Calculate fairness score (lower variance = more fair)
        fairness_score = 1.0 / (1.0 + np.var(list(demographic_means.values())))
        
        return EvaluationResult(
            metric_name="fairness",
            value=fairness_score,
            metadata={
                'demographic_performance': demographic_means,
                'performance_variance': np.var(list(demographic_means.values()))
            }
        )
```

### Comparative Evaluation

Implement robust model comparison:

```python
class ModelComparator:
    def __init__(self, models: Dict[str, Any], evaluation_config: Dict[str, Any]):
        self.models = models
        self.config = evaluation_config
        self.comparison_results = {}
    
    def compare_models(self, evaluation_tasks: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models
        """
        comparison_results = {}
        
        for task in evaluation_tasks:
            task_results = {}
            
            for model_name, model in self.models.items():
                # Evaluate each model on the task
                evaluator = ModelEvaluator(model, self.config)
                result = evaluator.evaluate_model([task])
                task_results[model_name] = result[task]
            
            # Perform statistical comparison
            comparison_results[task] = self.statistical_comparison(task_results)
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def statistical_comparison(self, task_results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """
        Perform statistical comparison between models
        """
        model_names = list(task_results.keys())
        results = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate comparisons
                    comparison_key = f"{model1}_vs_{model2}"
                    
                    # Extract scores for comparison
                    scores1 = task_results[model1].metadata.get('scores', [])
                    scores2 = task_results[model2].metadata.get('scores', [])
                    
                    if scores1 and scores2:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        
                        # Calculate effect size
                        effect_size = self.calculate_effect_size(scores1, scores2)
                        
                        results[comparison_key] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': p_value < 0.05,
                            'model1_mean': np.mean(scores1),
                            'model2_mean': np.mean(scores2)
                        }
        
        # Overall ranking
        model_means = {name: result.value for name, result in task_results.items()}
        ranking = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
        
        results['ranking'] = ranking
        results['model_means'] = model_means
        
        return results
    
    def calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        return effect_size
    
    def generate_comparison_report(self) -> str:
        """
        Generate comprehensive comparison report
        """
        report = "# Model Comparison Report\n\n"
        
        for task, results in self.comparison_results.items():
            report += f"## {task.replace('_', ' ').title()}\n\n"
            
            # Add ranking
            if 'ranking' in results:
                report += "### Model Ranking\n\n"
                for i, (model_name, score) in enumerate(results['ranking'], 1):
                    report += f"{i}. {model_name}: {score:.4f}\n"
                report += "\n"
            
            # Add statistical comparisons
            report += "### Statistical Comparisons\n\n"
            for comparison_key, comparison_result in results.items():
                if comparison_key not in ['ranking', 'model_means']:
                    model1, model2 = comparison_key.split('_vs_')
                    report += f"**{model1} vs {model2}:**\n"
                    report += f"- t-statistic: {comparison_result['t_statistic']:.4f}\n"
                    report += f"- p-value: {comparison_result['p_value']:.4f}\n"
                    report += f"- Effect size: {comparison_result['effect_size']:.4f}\n"
                    report += f"- Significant: {'Yes' if comparison_result['significant'] else 'No'}\n\n"
        
        return report
```

## Configuration

### Evaluation Configuration

```yaml
# configs/model_evaluation.yaml
evaluation:
  enabled: true
  
  # Evaluation tasks
  tasks:
    - "preference_alignment"
    - "response_quality"
    - "safety"
    - "efficiency"
    - "fairness"
  
  # Evaluation datasets
  datasets:
    preference_alignment:
      path: "data/preference_alignment.json"
      num_samples: 1000
    
    response_quality:
      path: "data/quality_evaluation.json"
      num_samples: 500
    
    safety:
      path: "data/safety_evaluation.json"
      num_samples: 200
    
    efficiency:
      path: "data/efficiency_test.json"
      num_samples: 100
    
    fairness:
      path: "data/fairness_evaluation.json"
      num_samples: 300
  
  # Evaluation metrics
  metrics:
    preference_alignment:
      method: "human_preference_simulation"
      threshold: 0.7
    
    response_quality:
      coherence_weight: 0.4
      relevance_weight: 0.4
      fluency_weight: 0.2
    
    safety:
      harm_categories: ["toxicity", "bias", "privacy"]
      threshold: 0.8
    
    efficiency:
      max_inference_time: 2.0  # seconds
      max_memory_usage: 8.0    # GB
    
    fairness:
      demographic_groups: ["gender", "race", "age"]
      max_performance_gap: 0.1
```

### Comparative Evaluation Configuration

```yaml
# configs/comparative_evaluation.yaml
comparative_evaluation:
  # Models to compare
  models:
    - name: "dpo_model"
      path: "models/dpo_trained"
      description: "DPO trained model"
    
    - name: "sft_model"
      path: "models/sft_trained"
      description: "SFT trained model"
    
    - name: "grpo_model"
      path: "models/grpo_trained"
      description: "GRPO trained model"
  
  # Comparison settings
  comparison:
    statistical_test: "t_test"
    significance_level: 0.05
    effect_size_threshold: 0.2
    
  # Reporting
  reporting:
    generate_report: true
    report_format: "markdown"
    include_visualizations: true
    save_results: true
```

## Advanced Evaluation Techniques

### Multi-Dimensional Evaluation

Implement multi-dimensional evaluation:

```python
class MultiDimensionalEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.evaluation_dimensions = {}
    
    def add_evaluation_dimension(self, name: str, evaluator_func, weight: float = 1.0):
        """
        Add evaluation dimension
        """
        self.evaluation_dimensions[name] = {
            'evaluator': evaluator_func,
            'weight': weight
        }
    
    def evaluate_all_dimensions(self) -> Dict[str, Any]:
        """
        Evaluate all dimensions
        """
        results = {}
        
        for dimension_name, dimension_info in self.evaluation_dimensions.items():
            evaluator_func = dimension_info['evaluator']
            result = evaluator_func(self.model)
            results[dimension_name] = result
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(results)
        results['composite_score'] = composite_score
        
        return results
    
    def calculate_composite_score(self, dimension_results: Dict[str, Any]) -> float:
        """
        Calculate weighted composite score
        """
        total_weight = 0
        weighted_sum = 0
        
        for dimension_name, result in dimension_results.items():
            if dimension_name != 'composite_score':
                weight = self.evaluation_dimensions[dimension_name]['weight']
                score = result.value if hasattr(result, 'value') else result
                
                weighted_sum += weight * score
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def create_evaluation_profile(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create evaluation profile
        """
        profile = {
            'dimension_scores': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze each dimension
        for dimension_name, result in results.items():
            if dimension_name != 'composite_score':
                score = result.value if hasattr(result, 'value') else result
                profile['dimension_scores'][dimension_name] = score
                
                # Identify strengths and weaknesses
                if score > 0.8:
                    profile['strengths'].append(dimension_name)
                elif score < 0.5:
                    profile['weaknesses'].append(dimension_name)
        
        # Generate recommendations
        profile['recommendations'] = self.generate_recommendations(profile)
        
        return profile
    
    def generate_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations
        """
        recommendations = []
        
        for weakness in profile['weaknesses']:
            if weakness == 'preference_alignment':
                recommendations.append("Consider fine-tuning with more diverse preference data")
            elif weakness == 'safety':
                recommendations.append("Implement additional safety filters and training")
            elif weakness == 'efficiency':
                recommendations.append("Optimize model architecture or use model compression")
            elif weakness == 'fairness':
                recommendations.append("Add fairness-aware training objectives")
        
        return recommendations
```

### Robustness Evaluation

Implement robustness evaluation:

```python
class RobustnessEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    def evaluate_robustness(self, test_scenarios: List[str]) -> Dict[str, Any]:
        """
        Evaluate model robustness
        """
        robustness_results = {}
        
        for scenario in test_scenarios:
            if scenario == "adversarial_attacks":
                robustness_results[scenario] = self.evaluate_adversarial_robustness()
            elif scenario == "distribution_shift":
                robustness_results[scenario] = self.evaluate_distribution_shift()
            elif scenario == "noise_robustness":
                robustness_results[scenario] = self.evaluate_noise_robustness()
            elif scenario == "prompt_injection":
                robustness_results[scenario] = self.evaluate_prompt_injection()
        
        return robustness_results
    
    def evaluate_adversarial_robustness(self) -> EvaluationResult:
        """
        Evaluate robustness against adversarial attacks
        """
        # Load adversarial test cases
        adversarial_data = self.load_adversarial_dataset()
        
        robustness_scores = []
        for attack in adversarial_data:
            # Apply adversarial perturbation
            perturbed_prompt = self.apply_adversarial_attack(attack['prompt'], attack['attack_type'])
            
            # Generate response
            original_response = self.generate_response(attack['prompt'])
            perturbed_response = self.generate_response(perturbed_prompt)
            
            # Calculate robustness score
            robustness_score = self.calculate_robustness_score(original_response, perturbed_response)
            robustness_scores.append(robustness_score)
        
        mean_score = np.mean(robustness_scores)
        confidence_interval = self.calculate_confidence_interval(robustness_scores)
        
        return EvaluationResult(
            metric_name="adversarial_robustness",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={'robustness_scores': robustness_scores}
        )
    
    def evaluate_distribution_shift(self) -> EvaluationResult:
        """
        Evaluate robustness to distribution shift
        """
        # Load out-of-distribution data
        ood_data = self.load_ood_dataset()
        
        performance_scores = []
        for sample in ood_data:
            # Generate response
            response = self.generate_response(sample['prompt'])
            
            # Evaluate performance on OOD data
            performance_score = self.evaluate_performance(response, sample['expected_output'])
            performance_scores.append(performance_score)
        
        mean_score = np.mean(performance_scores)
        confidence_interval = self.calculate_confidence_interval(performance_scores)
        
        return EvaluationResult(
            metric_name="distribution_shift_robustness",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={'performance_scores': performance_scores}
        )
    
    def evaluate_noise_robustness(self) -> EvaluationResult:
        """
        Evaluate robustness to input noise
        """
        # Load clean data
        clean_data = self.load_clean_dataset()
        
        noise_robustness_scores = []
        for sample in clean_data:
            # Generate response to clean input
            clean_response = self.generate_response(sample['prompt'])
            
            # Add noise to input
            noisy_prompt = self.add_noise(sample['prompt'])
            noisy_response = self.generate_response(noisy_prompt)
            
            # Calculate robustness score
            robustness_score = self.calculate_robustness_score(clean_response, noisy_response)
            noise_robustness_scores.append(robustness_score)
        
        mean_score = np.mean(noise_robustness_scores)
        confidence_interval = self.calculate_confidence_interval(noise_robustness_scores)
        
        return EvaluationResult(
            metric_name="noise_robustness",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={'robustness_scores': noise_robustness_scores}
        )
```

## Best Practices

### 1. Comprehensive Evaluation

Implement comprehensive evaluation strategies:

```python
class ComprehensiveEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.evaluators = {}
    
    def setup_evaluators(self):
        """
        Setup all evaluators
        """
        # Basic evaluators
        self.evaluators['preference'] = PreferenceAlignmentEvaluator(self.model)
        self.evaluators['quality'] = ResponseQualityEvaluator(self.model)
        self.evaluators['safety'] = SafetyEvaluator(self.model)
        self.evaluators['efficiency'] = EfficiencyEvaluator(self.model)
        self.evaluators['fairness'] = FairnessEvaluator(self.model)
        
        # Advanced evaluators
        self.evaluators['robustness'] = RobustnessEvaluator(self.model)
        self.evaluators['interpretability'] = InterpretabilityEvaluator(self.model)
        self.evaluators['calibration'] = CalibrationEvaluator(self.model)
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation
        """
        results = {}
        
        for evaluator_name, evaluator in self.evaluators.items():
            try:
                result = evaluator.evaluate()
                results[evaluator_name] = result
            except Exception as e:
                print(f"Error in {evaluator_name} evaluation: {e}")
                results[evaluator_name] = None
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(results)
        results['comprehensive_report'] = report
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive evaluation report
        """
        report = "# Comprehensive Model Evaluation Report\n\n"
        
        # Summary statistics
        valid_results = {k: v for k, v in results.items() if v is not None}
        report += f"## Summary\n\n"
        report += f"- Total evaluation dimensions: {len(results)}\n"
        report += f"- Successful evaluations: {len(valid_results)}\n"
        report += f"- Failed evaluations: {len(results) - len(valid_results)}\n\n"
        
        # Detailed results
        for evaluator_name, result in valid_results.items():
            report += f"## {evaluator_name.replace('_', ' ').title()}\n\n"
            report += f"Score: {result.value:.4f}\n"
            if result.confidence_interval:
                report += f"Confidence Interval: {result.confidence_interval}\n"
            report += "\n"
        
        return report
```

### 2. Statistical Rigor

Ensure statistical rigor in evaluation:

```python
class StatisticalEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.significance_level = config.get('significance_level', 0.05)
        self.power = config.get('power', 0.8)
    
    def calculate_sample_size(self, effect_size: float, alpha: float = None, power: float = None) -> int:
        """
        Calculate required sample size
        """
        if alpha is None:
            alpha = self.significance_level
        if power is None:
            power = self.power
        
        return stats.power.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power
        )
    
    def perform_statistical_tests(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests
        """
        results = {}
        
        # Normality test
        for group_name, group_data in data.items():
            statistic, p_value = stats.shapiro(group_data)
            results[f'{group_name}_normality'] = {
                'statistic': statistic,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        
        # Homogeneity of variance test
        if len(data) > 1:
            statistic, p_value = stats.levene(*data.values())
            results['homogeneity_of_variance'] = {
                'statistic': statistic,
                'p_value': p_value,
                'homogeneous': p_value > 0.05
            }
        
        # ANOVA or Kruskal-Wallis
        if len(data) > 2:
            if all(results[f'{name}_normality']['normal'] for name in data.keys()):
                # Use ANOVA
                statistic, p_value = stats.f_oneway(*data.values())
                results['group_comparison'] = {
                    'test': 'ANOVA',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
            else:
                # Use Kruskal-Wallis
                statistic, p_value = stats.kruskal(*data.values())
                results['group_comparison'] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
        
        return results
```

### 3. Reproducible Evaluation

Ensure evaluation reproducibility:

```python
class ReproducibleEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        self.evaluation_log = []
    
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def log_evaluation_step(self, step_name: str, parameters: Dict[str, Any], result: Any):
        """
        Log evaluation step for reproducibility
        """
        log_entry = {
            'timestamp': time.time(),
            'step_name': step_name,
            'parameters': parameters,
            'result': result,
            'random_seed': self.random_seed
        }
        self.evaluation_log.append(log_entry)
    
    def save_evaluation_log(self, filename: str):
        """
        Save evaluation log
        """
        with open(filename, 'w') as f:
            json.dump(self.evaluation_log, f, indent=2)
    
    def load_evaluation_log(self, filename: str):
        """
        Load evaluation log
        """
        with open(filename, 'r') as f:
            self.evaluation_log = json.load(f)
    
    def reproduce_evaluation(self, log_filename: str) -> Dict[str, Any]:
        """
        Reproduce evaluation from log
        """
        self.load_evaluation_log(log_filename)
        
        results = {}
        for log_entry in self.evaluation_log:
            # Set random seed
            self.set_random_seed(log_entry['random_seed'])
            
            # Reproduce step
            step_name = log_entry['step_name']
            parameters = log_entry['parameters']
            
            # Execute step (simplified)
            result = self.execute_evaluation_step(step_name, parameters)
            results[step_name] = result
        
        return results
```

## Troubleshooting

### Common Evaluation Issues

1. **Insufficient Sample Size**: Use power analysis to determine required sample size
2. **Biased Evaluation**: Use proper randomization and blinding
3. **Metric Selection**: Choose metrics that align with evaluation goals

### Debugging Tips

```python
# Add debugging to model evaluation
def debug_model_evaluation(self):
    """
    Debug model evaluation issues
    """
    print("=== Model Evaluation Debug ===")
    
    # Check model status
    print(f"Model loaded: {self.model is not None}")
    print(f"Model device: {next(self.model.parameters()).device}")
    
    # Check evaluation datasets
    for dataset_name, dataset_info in self.evaluation_datasets.items():
        print(f"Dataset {dataset_name}: {len(dataset_info)} samples")
    
    # Check evaluation results
    print(f"Number of evaluation results: {len(self.evaluation_results)}")
    for metric_name, result in self.evaluation_results.items():
        print(f"  {metric_name}: {result.value:.4f}")
    
    print("=============================")
```

## Next Steps

- Learn about [Experimental Design](experimental-design) for rigorous research
- Review [Reproducible Research](reproducible-research) for scientific rigor
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 