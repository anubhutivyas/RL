---
description: "Build comprehensive evaluation frameworks and implement robust model assessment and comparison strategies"
tags: ["evaluation", "metrics", "assessment", "comparison", "validation"]
categories: ["research-validation"]
---

# Model Evaluation

This guide covers how to build comprehensive evaluation frameworks and implement robust model assessment and comparison strategies for NeMo RL research.

## Overview

Comprehensive model evaluation is essential for understanding model performance, identifying strengths and weaknesses, and making informed decisions about model deployment. This guide provides frameworks for evaluating RL models across multiple dimensions.

**Note**: This guide provides **research methodology and theoretical frameworks** for model evaluation. The examples show how to integrate these frameworks with actual NeMo RL code.

## Key Components

### Evaluation Framework

Implement a comprehensive evaluation framework:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import torch

# Real NeMo RL imports for evaluation
from nemo_rl.evals.eval import eval_pass_k, run_env_eval
from nemo_rl.algorithms.dpo import validate as dpo_validate
from nemo_rl.algorithms.grpo import validate as grpo_validate
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer

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
        
        # Real NeMo RL components
        self.logger = Logger(config.get('logger', {}))
        self.timer = Timer()
        
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
        Evaluate preference alignment using NeMo RL patterns
        """
        # Real NeMo RL evaluation setup
        with self.timer.time("preference_alignment_evaluation"):
            # Load preference dataset using NeMo RL patterns
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
            
            # Log results using NeMo RL logger
            self.logger.log({
                'preference_alignment_mean': mean_score,
                'preference_alignment_std': std_score,
                'preference_alignment_confidence_interval': confidence_interval
            })
        
        return EvaluationResult(
            metric_name="preference_alignment",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={
                'std': std_score,
                'scores': alignment_scores,
                'evaluation_time': self.timer.get_timing_metrics().get('preference_alignment_evaluation', 0)
            }
        )
    
    def evaluate_response_quality(self) -> EvaluationResult:
        """
        Evaluate response quality using NeMo RL patterns
        """
        with self.timer.time("response_quality_evaluation"):
            # Load quality evaluation dataset
            quality_data = self.load_quality_dataset()
            
            quality_scores = []
            for sample in quality_data:
                # Generate response
                response = self.generate_response(sample['prompt'])
                
                # Evaluate quality using NeMo RL patterns
                quality_score = self.calculate_quality_score(response, sample['reference'])
                quality_scores.append(quality_score)
            
            mean_score = np.mean(quality_scores)
            confidence_interval = self.calculate_confidence_interval(quality_scores)
            
            # Log using NeMo RL logger
            self.logger.log({
                'response_quality_mean': mean_score,
                'response_quality_std': np.std(quality_scores)
            })
        
        return EvaluationResult(
            metric_name="response_quality",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={'scores': quality_scores}
        )
    
    def evaluate_safety(self) -> EvaluationResult:
        """
        Evaluate safety using NeMo RL patterns
        """
        with self.timer.time("safety_evaluation"):
            # Load safety evaluation dataset
            safety_data = self.load_safety_dataset()
            
            safety_scores = []
            for sample in safety_data:
                # Generate response
                response = self.generate_response(sample['prompt'])
                
                # Evaluate safety
                safety_score = self.calculate_safety_score(response, sample['safety_criteria'])
                safety_scores.append(safety_score)
            
            mean_score = np.mean(safety_scores)
            confidence_interval = self.calculate_confidence_interval(safety_scores)
            
            # Log using NeMo RL logger
            self.logger.log({
                'safety_mean': mean_score,
                'safety_std': np.std(safety_scores)
            })
        
        return EvaluationResult(
            metric_name="safety",
            value=mean_score,
            confidence_interval=confidence_interval,
            metadata={'scores': safety_scores}
        )
    
    def evaluate_efficiency(self) -> EvaluationResult:
        """
        Evaluate efficiency using NeMo RL patterns
        """
        with self.timer.time("efficiency_evaluation"):
            # Measure inference time and memory usage
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
                
                # Measure memory usage (simplified)
                memory_usage.append(self.get_memory_usage())
            
            avg_inference_time = np.mean(inference_times)
            avg_memory_usage = np.mean(memory_usage)
            
            # Normalize efficiency score (lower is better)
            efficiency_score = 1.0 / (1.0 + avg_inference_time + avg_memory_usage)
            
            # Log using NeMo RL logger
            self.logger.log({
                'avg_inference_time': avg_inference_time,
                'avg_memory_usage': avg_memory_usage,
                'efficiency_score': efficiency_score
            })
        
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
        Evaluate model fairness using NeMo RL patterns
        """
        with self.timer.time("fairness_evaluation"):
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
            
            # Log using NeMo RL logger
            self.logger.log({
                'fairness_score': fairness_score,
                'demographic_performance': demographic_means
            })
        
        return EvaluationResult(
            metric_name="fairness",
            value=fairness_score,
            metadata={
                'demographic_performance': demographic_means,
                'performance_variance': np.var(list(demographic_means.values()))
            }
        )
```

### Real NeMo RL Integration Examples

#### Using NeMo RL's Built-in Evaluation Functions

```python
# Real NeMo RL evaluation integration
from nemo_rl.evals.eval import eval_pass_k, run_env_eval
from nemo_rl.algorithms.dpo import validate as dpo_validate
from nemo_rl.algorithms.grpo import validate as grpo_validate
from nemo_rl.utils.config import load_config

class NeMoRLEvaluator:
    """Real NeMo RL evaluation integration"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
    
    def evaluate_dpo_model(self, policy, val_dataloader, tokenizer, loss_fn, step: int):
        """Evaluate DPO model using real NeMo RL validation"""
        with self.timer.time("dpo_validation"):
            # Use real NeMo RL DPO validation
            val_metrics, timing_metrics = dpo_validate(
                policy=policy,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                step=step,
                master_config=self.config,
                val_batches=self.config['dpo']['val_batches'],
                val_batch_size=self.config['dpo']['val_global_batch_size'],
                val_mbs=self.config['dpo']['val_micro_batch_size']
            )
            
            # Log results
            self.logger.log({
                'dpo_validation_loss': val_metrics.get('loss', 0),
                'dpo_validation_accuracy': val_metrics.get('accuracy', 0),
                'dpo_validation_time': timing_metrics.get('total_validation_time', 0)
            })
            
            return val_metrics, timing_metrics
    
    def evaluate_grpo_model(self, policy_generation, val_dataloader, tokenizer, val_task_to_env, step: int):
        """Evaluate GRPO model using real NeMo RL validation"""
        with self.timer.time("grpo_validation"):
            # Use real NeMo RL GRPO validation
            val_metrics, timing_metrics = grpo_validate(
                policy_generation=policy_generation,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                val_task_to_env=val_task_to_env,
                step=step,
                master_config=self.config
            )
            
            # Log results
            self.logger.log({
                'grpo_validation_accuracy': val_metrics.get('accuracy', 0),
                'grpo_avg_length': val_metrics.get('avg_length', 0),
                'grpo_validation_time': timing_metrics.get('total_validation_time', 0)
            })
            
            return val_metrics, timing_metrics
    
    def evaluate_pass_k(self, rewards: torch.Tensor, num_tests_per_prompt: int, k: int):
        """Evaluate pass@k using real NeMo RL function"""
        # Use real NeMo RL pass@k evaluation
        pass_k_score = eval_pass_k(rewards, num_tests_per_prompt, k)
        
        self.logger.log({
            'pass_k_score': pass_k_score,
            'k_value': k,
            'num_tests_per_prompt': num_tests_per_prompt
        })
        
        return pass_k_score
    
    def run_environment_evaluation(self, vllm_generation, dataloader, env):
        """Run environment evaluation using real NeMo RL function"""
        # Use real NeMo RL environment evaluation
        run_env_eval(vllm_generation, dataloader, env, self.config)
```

### Comparative Evaluation

Implement robust model comparison:

```python
class ModelComparator:
    def __init__(self, models: Dict[str, Any], evaluation_config: Dict[str, Any]):
        self.models = models
        self.config = evaluation_config
        self.comparison_results = {}
        
        # Real NeMo RL components
        self.logger = Logger(evaluation_config.get('logger', {}))
        self.timer = Timer()
    
    def compare_models(self, evaluation_tasks: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models using NeMo RL patterns
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
        
        # Log comparison results using NeMo RL logger
        self.logger.log({
            'model_comparison_ranking': ranking,
            'model_comparison_means': model_means
        })
        
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

### Evaluation Configuration with NeMo RL Integration

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

# Real NeMo RL configuration integration
dpo:
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true

grpo:
  val_period: 100
  val_batch_size: 8
  val_at_start: true
  max_val_samples: 100

logger:
  log_dir: "logs"
  wandb_enabled: true
  tensorboard_enabled: true
  wandb:
    project: "model-evaluation"
    name: "evaluation-experiment"
```

### Real NeMo RL Evaluation Setup

```python
# Real NeMo RL evaluation setup example
from nemo_rl.algorithms.dpo import setup as dpo_setup
from nemo_rl.algorithms.grpo import setup as grpo_setup
from nemo_rl.utils.config import load_config

def setup_evaluation_pipeline(config_path: str):
    """Setup evaluation pipeline using real NeMo RL components"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup DPO evaluation
    if 'dpo' in config:
        policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = dpo_setup(
            master_config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        # Create NeMo RL evaluator
        evaluator = NeMoRLEvaluator(config_path)
        
        return {
            'policy': policy,
            'val_dataloader': val_dataloader,
            'loss_fn': loss_fn,
            'evaluator': evaluator,
            'logger': logger
        }
    
    # Setup GRPO evaluation
    elif 'grpo' in config:
        policy, policy_generation, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, save_state, master_config = grpo_setup(
            master_config=config,
            tokenizer=tokenizer,
            dataset=dataset,
            val_dataset=val_dataset
        )
        
        # Create NeMo RL evaluator
        evaluator = NeMoRLEvaluator(config_path)
        
        return {
            'policy': policy,
            'policy_generation': policy_generation,
            'val_dataloader': val_dataloader,
            'loss_fn': loss_fn,
            'evaluator': evaluator,
            'logger': logger
        }
    
    else:
        raise ValueError("No DPO or GRPO configuration found")
```

## Best Practices

### 1. Comprehensive Evaluation with NeMo RL Integration

Implement comprehensive evaluation strategies:

```python
class ComprehensiveEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.evaluators = {}
        
        # Real NeMo RL components
        self.logger = Logger(config.get('logger', {}))
        self.timer = Timer()
    
    def setup_evaluators(self):
        """
        Setup all evaluators with NeMo RL integration
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
        Run comprehensive evaluation with NeMo RL logging
        """
        results = {}
        
        with self.timer.time("comprehensive_evaluation"):
            for evaluator_name, evaluator in self.evaluators.items():
                try:
                    result = evaluator.evaluate()
                    results[evaluator_name] = result
                    
                    # Log individual evaluation results
                    self.logger.log({
                        f'{evaluator_name}_score': result.value,
                        f'{evaluator_name}_metadata': result.metadata
                    })
                    
                except Exception as e:
                    print(f"Error in {evaluator_name} evaluation: {e}")
                    results[evaluator_name] = None
                    
                    # Log evaluation errors
                    self.logger.log({
                        f'{evaluator_name}_error': str(e)
                    })
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(results)
        results['comprehensive_report'] = report
        
        # Log overall evaluation metrics
        valid_results = {k: v for k, v in results.items() if v is not None}
        self.logger.log({
            'total_evaluation_dimensions': len(results),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(results) - len(valid_results),
            'comprehensive_evaluation_time': self.timer.get_timing_metrics().get('comprehensive_evaluation', 0)
        })
        
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

### 2. Statistical Rigor with NeMo RL Integration

Ensure statistical rigor in evaluation:

```python
class StatisticalEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.significance_level = config.get('significance_level', 0.05)
        self.power = config.get('power', 0.8)
        
        # Real NeMo RL components
        self.logger = Logger(config.get('logger', {}))
    
    def calculate_sample_size(self, effect_size: float, alpha: float = None, power: float = None) -> int:
        """
        Calculate required sample size
        """
        if alpha is None:
            alpha = self.significance_level
        if power is None:
            power = self.power
        
        sample_size = stats.power.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power
        )
        
        # Log sample size calculation
        self.logger.log({
            'sample_size_calculation': {
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'required_sample_size': sample_size
            }
        })
        
        return sample_size
    
    def perform_statistical_tests(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests with NeMo RL logging
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
        
        # Log statistical test results
        self.logger.log({
            'statistical_tests': results
        })
        
        return results
```

### 3. Reproducible Evaluation with NeMo RL Integration

Ensure evaluation reproducibility:

```python
class ReproducibleEvaluator:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        self.evaluation_log = []
        
        # Real NeMo RL components
        self.logger = Logger(config.get('logger', {}))
        self.timer = Timer()
    
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Log seed setting
        self.logger.log({
            'random_seed_set': seed
        })
    
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
        
        # Log using NeMo RL logger
        self.logger.log({
            'evaluation_step': log_entry
        })
    
    def save_evaluation_log(self, filename: str):
        """
        Save evaluation log
        """
        with open(filename, 'w') as f:
            json.dump(self.evaluation_log, f, indent=2)
        
        # Log file save
        self.logger.log({
            'evaluation_log_saved': filename
        })
    
    def load_evaluation_log(self, filename: str):
        """
        Load evaluation log
        """
        with open(filename, 'r') as f:
            self.evaluation_log = json.load(f)
        
        # Log file load
        self.logger.log({
            'evaluation_log_loaded': filename
        })
    
    def reproduce_evaluation(self, log_filename: str) -> Dict[str, Any]:
        """
        Reproduce evaluation from log with NeMo RL integration
        """
        self.load_evaluation_log(log_filename)
        
        results = {}
        with self.timer.time("reproduction_evaluation"):
            for log_entry in self.evaluation_log:
                # Set random seed
                self.set_random_seed(log_entry['random_seed'])
                
                # Reproduce step
                step_name = log_entry['step_name']
                parameters = log_entry['parameters']
                
                # Execute step (simplified)
                result = self.execute_evaluation_step(step_name, parameters)
                results[step_name] = result
        
        # Log reproduction results
        self.logger.log({
            'reproduction_completed': True,
            'reproduction_time': self.timer.get_timing_metrics().get('reproduction_evaluation', 0),
            'reproduction_results': results
        })
        
        return results
```

## Troubleshooting

### Common Evaluation Issues

1. **Insufficient Sample Size**: Use power analysis to determine required sample size
2. **Biased Evaluation**: Use proper randomization and blinding
3. **Metric Selection**: Choose metrics that align with evaluation goals

### Debugging Tips with NeMo RL Integration

```python
# Add debugging to model evaluation with NeMo RL logging
def debug_model_evaluation(self):
    """
    Debug model evaluation issues with NeMo RL integration
    """
    print("=== Model Evaluation Debug ===")
    
    # Check model status
    model_loaded = self.model is not None
    model_device = next(self.model.parameters()).device if self.model else None
    
    print(f"Model loaded: {model_loaded}")
    print(f"Model device: {model_device}")
    
    # Check evaluation datasets
    dataset_info = {}
    for dataset_name, dataset_info in self.evaluation_datasets.items():
        dataset_info[dataset_name] = len(dataset_info)
        print(f"Dataset {dataset_name}: {len(dataset_info)} samples")
    
    # Check evaluation results
    num_results = len(self.evaluation_results)
    result_summary = {}
    for metric_name, result in self.evaluation_results.items():
        result_summary[metric_name] = result.value
        print(f"  {metric_name}: {result.value:.4f}")
    
    print("=============================")
    
    # Log debug information using NeMo RL logger
    self.logger.log({
        'debug_model_loaded': model_loaded,
        'debug_model_device': str(model_device),
        'debug_dataset_info': dataset_info,
        'debug_num_results': num_results,
        'debug_result_summary': result_summary
    })
```

## Next Steps

- Learn about [Experimental Design](experimental-design-validation) for rigorous research
- Review [Reproducible Research](reproducible-research-validation) for scientific rigor
- Explore [Algorithm Development](../algorithm-development/index) for advanced training 