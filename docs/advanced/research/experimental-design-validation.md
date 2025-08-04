---
description: "Design controlled experiments and research studies with proper experimental methodology for RL research"
tags: ["research", "experiments", "methodology", "experimental-design", "validation"]
categories: ["research-validation"]
---

# Experimental Design

This comprehensive guide covers how to design controlled experiments and research studies with proper experimental methodology for NeMo RL research, including both research methodology and validation frameworks.

## Overview

Proper experimental design is crucial for conducting rigorous research in reinforcement learning. This guide provides frameworks and methodologies for designing experiments that produce reliable, reproducible results, covering both research methodology and validation approaches.

**Note**: This guide provides **research methodology and theoretical frameworks** for experimental design. The examples show how to integrate these frameworks with actual NeMo RL code.

## Key Principles

### Hypothesis Formulation

#### Clear Research Questions
Define specific, testable research questions:

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

# Real NeMo RL imports for experimental design
from nemo_rl.utils.config import load_config
from nemo_rl.algorithms.dpo import setup as dpo_setup, dpo_train
from nemo_rl.algorithms.grpo import setup as grpo_setup, grpo_train
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer

class ResearchQuestionType(Enum):
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    VALIDATION = "validation"

@dataclass
class ResearchQuestion:
    """Structured research question"""
    question: str
    type: ResearchQuestionType
    hypothesis: str
    null_hypothesis: str
    alternative_hypothesis: str
    significance_level: float = 0.05
    power: float = 0.8

class ResearchQuestionDesigner:
    def __init__(self):
        self.research_questions = []
        
        # Real NeMo RL components
        self.logger = Logger({})
        self.timer = Timer()
    
    def formulate_question(self, 
                          question: str,
                          question_type: ResearchQuestionType,
                          hypothesis: str) -> ResearchQuestion:
        """
        Formulate a structured research question with NeMo RL integration
        """
        # Example research questions for NeMo RL
        examples = {
            ResearchQuestionType.COMPARISON: {
                "question": "Does DPO outperform SFT on preference alignment?",
                "hypothesis": "DPO achieves higher preference alignment scores than SFT",
                "null_hypothesis": "DPO and SFT achieve similar preference alignment scores",
                "alternative_hypothesis": "DPO achieves significantly higher preference alignment scores than SFT"
            },
            ResearchQuestionType.OPTIMIZATION: {
                "question": "What is the optimal beta parameter for DPO training?",
                "hypothesis": "Beta = 0.1 provides optimal performance",
                "null_hypothesis": "All beta values provide similar performance",
                "alternative_hypothesis": "Beta = 0.1 provides significantly better performance"
            },
            ResearchQuestionType.ANALYSIS: {
                "question": "How does model size affect RL training efficiency?",
                "hypothesis": "Larger models require more training steps but achieve better final performance",
                "null_hypothesis": "Model size has no effect on training efficiency",
                "alternative_hypothesis": "Model size significantly affects training efficiency"
            }
        }
        
        example = examples.get(question_type, {})
        
        research_question = ResearchQuestion(
            question=question,
            type=question_type,
            hypothesis=hypothesis,
            null_hypothesis=example.get("null_hypothesis", ""),
            alternative_hypothesis=example.get("alternative_hypothesis", "")
        )
        
        # Log research question formulation
        self.logger.log({
            'research_question_formulated': {
                'question': question,
                'type': question_type.value,
                'hypothesis': hypothesis
            }
        })
        
        return research_question
    
    def validate_question(self, question: ResearchQuestion) -> bool:
        """
        Validate research question design
        """
        # Check if question is specific and measurable
        if not question.question or len(question.question.strip()) == 0:
            return False
        
        # Check if hypothesis is testable
        if not question.hypothesis or len(question.hypothesis.strip()) == 0:
            return False
        
        # Check if significance level is reasonable
        if question.significance_level <= 0 or question.significance_level >= 1:
            return False
        
        # Log validation result
        self.logger.log({
            'research_question_validation': {
                'question': question.question,
                'valid': True,
                'significance_level': question.significance_level
            }
        })
        
        return True
```

### Real NeMo RL Experimental Setup

#### DPO Experiment Setup

```python
class DPOExperimentDesigner:
    """Design DPO experiments using real NeMo RL components"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
    
    def setup_dpo_experiment(self, research_question: ResearchQuestion):
        """Setup DPO experiment using real NeMo RL components"""
        
        with self.timer.time("dpo_experiment_setup"):
            # Setup DPO training components
            policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = dpo_setup(
                master_config=self.config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            
            # Log experiment setup
            self.logger.log({
                'dpo_experiment_setup': {
                    'research_question': research_question.question,
                    'hypothesis': research_question.hypothesis,
                    'significance_level': research_question.significance_level,
                    'setup_time': self.timer.get_timing_metrics().get('dpo_experiment_setup', 0)
                }
            })
            
            return {
                'policy': policy,
                'cluster': cluster,
                'train_dataloader': train_dataloader,
                'val_dataloader': val_dataloader,
                'loss_fn': loss_fn,
                'master_config': master_config,
                'logger': logger,
                'task_spec': task_spec,
                'save_state': save_state
            }
    
    def run_dpo_experiment(self, components: Dict, research_question: ResearchQuestion):
        """Run DPO experiment using real NeMo RL training"""
        
        with self.timer.time("dpo_experiment_execution"):
            # Run DPO training
            dpo_train(
                policy=components['policy'],
                train_dataloader=components['train_dataloader'],
                val_dataloader=components['val_dataloader'],
                tokenizer=tokenizer,
                loss_fn=components['loss_fn'],
                master_config=components['master_config'],
                logger=components['logger'],
                checkpointer=checkpointer,
                dpo_save_state=components['save_state']
            )
            
            # Log experiment execution
            self.logger.log({
                'dpo_experiment_execution': {
                    'research_question': research_question.question,
                    'execution_time': self.timer.get_timing_metrics().get('dpo_experiment_execution', 0)
                }
            })
```

#### GRPO Experiment Setup

```python
class GRPOExperimentDesigner:
    """Design GRPO experiments using real NeMo RL components"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
    
    def setup_grpo_experiment(self, research_question: ResearchQuestion):
        """Setup GRPO experiment using real NeMo RL components"""
        
        with self.timer.time("grpo_experiment_setup"):
            # Setup GRPO training components
            policy, policy_generation, cluster, train_dataloader, val_dataloader, loss_fn, logger, checkpointer, save_state, master_config = grpo_setup(
                master_config=self.config,
                tokenizer=tokenizer,
                dataset=dataset,
                val_dataset=val_dataset
            )
            
            # Log experiment setup
            self.logger.log({
                'grpo_experiment_setup': {
                    'research_question': research_question.question,
                    'hypothesis': research_question.hypothesis,
                    'significance_level': research_question.significance_level,
                    'setup_time': self.timer.get_timing_metrics().get('grpo_experiment_setup', 0)
                }
            })
            
            return {
                'policy': policy,
                'policy_generation': policy_generation,
                'cluster': cluster,
                'train_dataloader': train_dataloader,
                'val_dataloader': val_dataloader,
                'loss_fn': loss_fn,
                'logger': logger,
                'checkpointer': checkpointer,
                'save_state': save_state,
                'master_config': master_config
            }
    
    def run_grpo_experiment(self, components: Dict, research_question: ResearchQuestion):
        """Run GRPO experiment using real NeMo RL training"""
        
        with self.timer.time("grpo_experiment_execution"):
            # Run GRPO training
            grpo_train(
                policy=components['policy'],
                policy_generation=components['policy_generation'],
                dataloader=components['train_dataloader'],
                val_dataloader=components['val_dataloader'],
                tokenizer=tokenizer,
                loss_fn=components['loss_fn'],
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=components['logger'],
                checkpointer=components['checkpointer'],
                grpo_save_state=components['save_state'],
                master_config=components['master_config']
            )
            
            # Log experiment execution
            self.logger.log({
                'grpo_experiment_execution': {
                    'research_question': research_question.question,
                    'execution_time': self.timer.get_timing_metrics().get('grpo_experiment_execution', 0)
                }
            })
```

### Experimental Design Framework

#### Systematic Experiment Design

```python
class SystematicExperimentDesigner:
    """Systematic experiment design with NeMo RL integration"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
        self.experiments = []
    
    def design_comparison_experiment(self, algorithms: List[str], research_question: ResearchQuestion):
        """Design comparison experiment using NeMo RL patterns"""
        
        with self.timer.time("comparison_experiment_design"):
            experiment_design = {
                'type': 'comparison',
                'algorithms': algorithms,
                'research_question': research_question,
                'configs': {}
            }
            
            # Create configurations for each algorithm
            for algorithm in algorithms:
                if algorithm == 'dpo':
                    experiment_design['configs']['dpo'] = self.create_dpo_config()
                elif algorithm == 'grpo':
                    experiment_design['configs']['grpo'] = self.create_grpo_config()
                elif algorithm == 'sft':
                    experiment_design['configs']['sft'] = self.create_sft_config()
            
            # Log experiment design
            self.logger.log({
                'comparison_experiment_design': {
                    'algorithms': algorithms,
                    'research_question': research_question.question,
                    'design_time': self.timer.get_timing_metrics().get('comparison_experiment_design', 0)
                }
            })
            
            return experiment_design
    
    def design_optimization_experiment(self, parameter_name: str, parameter_range: List, research_question: ResearchQuestion):
        """Design optimization experiment using NeMo RL patterns"""
        
        with self.timer.time("optimization_experiment_design"):
            experiment_design = {
                'type': 'optimization',
                'parameter_name': parameter_name,
                'parameter_range': parameter_range,
                'research_question': research_question,
                'configs': []
            }
            
            # Create configurations for each parameter value
            for param_value in parameter_range:
                config = self.create_base_config()
                self.set_parameter_value(config, parameter_name, param_value)
                experiment_design['configs'].append(config)
            
            # Log experiment design
            self.logger.log({
                'optimization_experiment_design': {
                    'parameter_name': parameter_name,
                    'parameter_range': parameter_range,
                    'research_question': research_question.question,
                    'design_time': self.timer.get_timing_metrics().get('optimization_experiment_design', 0)
                }
            })
            
            return experiment_design
    
    def create_dpo_config(self) -> Dict:
        """Create DPO configuration using real NeMo RL patterns"""
        return {
            'policy': {
                'model_name': "microsoft/DialoGPT-medium",
                'max_total_sequence_length': 2048,
                'precision': "bfloat16",
                'optimizer': {
                    'name': "torch.optim.AdamW",
                    'kwargs': {
                        'lr': 1e-5,
                        'weight_decay': 0.01,
                        'betas': [0.9, 0.999],
                        'eps': 1e-8
                    }
                }
            },
            'dpo': {
                'reference_policy_kl_penalty': 0.1,
                'preference_loss_weight': 1.0,
                'sft_loss_weight': 0.1,
                'max_num_epochs': 3,
                'max_num_steps': 1000,
                'val_period': 100,
                'val_batches': 5,
                'val_global_batch_size': 32,
                'val_micro_batch_size': 8
            },
            'logger': {
                'log_dir': "logs",
                'wandb_enabled': True,
                'tensorboard_enabled': True
            }
        }
    
    def create_grpo_config(self) -> Dict:
        """Create GRPO configuration using real NeMo RL patterns"""
        return {
            'policy': {
                'model_name': "microsoft/DialoGPT-medium",
                'max_total_sequence_length': 2048,
                'precision': "bfloat16",
                'optimizer': {
                    'name': "torch.optim.AdamW",
                    'kwargs': {
                        'lr': 3e-4,
                        'weight_decay': 0.01,
                        'betas': [0.9, 0.999],
                        'eps': 1e-8
                    }
                }
            },
            'grpo': {
                'num_prompts_per_step': 4,
                'num_generations_per_prompt': 4,
                'max_num_steps': 1000,
                'max_rollout_turns': 1,
                'normalize_rewards': True,
                'val_period': 100,
                'val_batch_size': 8,
                'val_at_start': True
            },
            'logger': {
                'log_dir': "logs",
                'wandb_enabled': True,
                'tensorboard_enabled': True
            }
        }
    
    def set_parameter_value(self, config: Dict, parameter_name: str, value: Any):
        """Set parameter value in configuration"""
        # Handle nested parameter paths like 'dpo.reference_policy_kl_penalty'
        keys = parameter_name.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
```

### Real NeMo RL Experiment Execution

#### Automated Experiment Runner

```python
class NeMoRLExperimentRunner:
    """Run experiments using real NeMo RL components"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
        self.results = {}
    
    def run_comparison_experiment(self, experiment_design: Dict):
        """Run comparison experiment using real NeMo RL components"""
        
        with self.timer.time("comparison_experiment_execution"):
            results = {}
            
            for algorithm, config in experiment_design['configs'].items():
                self.logger.log({
                    f'{algorithm}_experiment_started': {
                        'algorithm': algorithm,
                        'research_question': experiment_design['research_question'].question
                    }
                })
                
                if algorithm == 'dpo':
                    result = self.run_dpo_experiment(config)
                elif algorithm == 'grpo':
                    result = self.run_grpo_experiment(config)
                elif algorithm == 'sft':
                    result = self.run_sft_experiment(config)
                
                results[algorithm] = result
                
                self.logger.log({
                    f'{algorithm}_experiment_completed': {
                        'algorithm': algorithm,
                        'result': result
                    }
                })
            
            # Perform statistical comparison
            comparison_result = self.statistical_comparison(results)
            
            self.logger.log({
                'comparison_experiment_completed': {
                    'results': results,
                    'comparison': comparison_result,
                    'execution_time': self.timer.get_timing_metrics().get('comparison_experiment_execution', 0)
                }
            })
            
            return results, comparison_result
    
    def run_optimization_experiment(self, experiment_design: Dict):
        """Run optimization experiment using real NeMo RL components"""
        
        with self.timer.time("optimization_experiment_execution"):
            results = {}
            
            for i, config in enumerate(experiment_design['configs']):
                param_value = experiment_design['parameter_range'][i]
                
                self.logger.log({
                    'optimization_experiment_step': {
                        'parameter_name': experiment_design['parameter_name'],
                        'parameter_value': param_value,
                        'step': i + 1,
                        'total_steps': len(experiment_design['configs'])
                    }
                })
                
                # Run experiment with this configuration
                result = self.run_single_experiment(config)
                results[param_value] = result
                
                self.logger.log({
                    'optimization_experiment_step_completed': {
                        'parameter_value': param_value,
                        'result': result
                    }
                })
            
            # Find optimal configuration
            optimal_config = self.find_optimal_configuration(results)
            
            self.logger.log({
                'optimization_experiment_completed': {
                    'results': results,
                    'optimal_configuration': optimal_config,
                    'execution_time': self.timer.get_timing_metrics().get('optimization_experiment_execution', 0)
                }
            })
            
            return results, optimal_config
    
    def run_dpo_experiment(self, config: Dict):
        """Run DPO experiment using real NeMo RL components"""
        
        with self.timer.time("dpo_experiment"):
            # Setup DPO experiment
            components = self.setup_dpo_experiment(config)
            
            # Run DPO training
            dpo_train(
                policy=components['policy'],
                train_dataloader=components['train_dataloader'],
                val_dataloader=components['val_dataloader'],
                tokenizer=tokenizer,
                loss_fn=components['loss_fn'],
                master_config=components['master_config'],
                logger=components['logger'],
                checkpointer=checkpointer,
                dpo_save_state=components['save_state']
            )
            
            # Extract results
            result = {
                'final_loss': components['logger'].get_latest_metric('loss'),
                'final_accuracy': components['logger'].get_latest_metric('accuracy'),
                'training_time': self.timer.get_timing_metrics().get('dpo_experiment', 0)
            }
            
            return result
    
    def run_grpo_experiment(self, config: Dict):
        """Run GRPO experiment using real NeMo RL components"""
        
        with self.timer.time("grpo_experiment"):
            # Setup GRPO experiment
            components = self.setup_grpo_experiment(config)
            
            # Run GRPO training
            grpo_train(
                policy=components['policy'],
                policy_generation=components['policy_generation'],
                dataloader=components['train_dataloader'],
                val_dataloader=components['val_dataloader'],
                tokenizer=tokenizer,
                loss_fn=components['loss_fn'],
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=components['logger'],
                checkpointer=components['checkpointer'],
                grpo_save_state=components['save_state'],
                master_config=components['master_config']
            )
            
            # Extract results
            result = {
                'final_accuracy': components['logger'].get_latest_metric('accuracy'),
                'final_avg_length': components['logger'].get_latest_metric('avg_length'),
                'training_time': self.timer.get_timing_metrics().get('grpo_experiment', 0)
            }
            
            return result
    
    def statistical_comparison(self, results: Dict) -> Dict:
        """Perform statistical comparison of results"""
        # Extract metrics for comparison
        metrics = {}
        for algorithm, result in results.items():
            if 'final_accuracy' in result:
                metrics[algorithm] = result['final_accuracy']
            elif 'final_loss' in result:
                metrics[algorithm] = -result['final_loss']  # Convert loss to score
        
        # Perform statistical tests
        algorithms = list(metrics.keys())
        if len(algorithms) == 2:
            # T-test for two algorithms
            t_stat, p_value = stats.ttest_ind([metrics[algorithms[0]]], [metrics[algorithms[1]]])
            comparison = {
                'test': 't_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            # ANOVA for multiple algorithms
            f_stat, p_value = stats.f_oneway(*[metrics[alg] for alg in algorithms])
            comparison = {
                'test': 'anova',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return comparison
    
    def find_optimal_configuration(self, results: Dict) -> Dict:
        """Find optimal configuration from optimization results"""
        best_value = float('-inf')
        best_config = None
        
        for param_value, result in results.items():
            if 'final_accuracy' in result:
                score = result['final_accuracy']
            elif 'final_loss' in result:
                score = -result['final_loss']  # Convert loss to score
            else:
                continue
            
            if score > best_value:
                best_value = score
                best_config = {
                    'parameter_value': param_value,
                    'score': score,
                    'result': result
                }
        
        return best_config
```

## Configuration

### Real NeMo RL Experiment Configuration

```yaml
# configs/experiment_design.yaml
experiment:
  type: "comparison"  # or "optimization"
  research_question:
    question: "Does DPO outperform SFT on preference alignment?"
    hypothesis: "DPO achieves higher preference alignment scores than SFT"
    significance_level: 0.05
    power: 0.8
  
  # Comparison experiment
  comparison:
    algorithms: ["dpo", "sft", "grpo"]
    metrics: ["accuracy", "loss", "training_time"]
    statistical_test: "anova"
  
  # Optimization experiment
  optimization:
    parameter_name: "dpo.reference_policy_kl_penalty"
    parameter_range: [0.05, 0.1, 0.2, 0.3]
    optimization_metric: "accuracy"
  
  # Real NeMo RL configuration
  dpo:
    reference_policy_kl_penalty: 0.1
    preference_loss_weight: 1.0
    sft_loss_weight: 0.1
    max_num_epochs: 3
    max_num_steps: 1000
    val_period: 100
    val_batches: 5
    val_global_batch_size: 32
    val_micro_batch_size: 8
  
  grpo:
    num_prompts_per_step: 4
    num_generations_per_prompt: 4
    max_num_steps: 1000
    max_rollout_turns: 1
    normalize_rewards: true
    val_period: 100
    val_batch_size: 8
    val_at_start: true
  
  logger:
    log_dir: "logs"
    wandb_enabled: true
    tensorboard_enabled: true
    wandb:
      project: "experiment-design"
      name: "comparison-experiment"
```

## Best Practices

### 1. Systematic Experiment Design with NeMo RL Integration

```python
class SystematicExperimentDesigner:
    """Systematic experiment design with NeMo RL integration"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
        self.experiments = []
    
    def design_comparison_experiment(self, algorithms: List[str], research_question: ResearchQuestion):
        """Design comparison experiment using NeMo RL patterns"""
        
        with self.timer.time("comparison_experiment_design"):
            experiment_design = {
                'type': 'comparison',
                'algorithms': algorithms,
                'research_question': research_question,
                'configs': {}
            }
            
            # Create configurations for each algorithm
            for algorithm in algorithms:
                if algorithm == 'dpo':
                    experiment_design['configs']['dpo'] = self.create_dpo_config()
                elif algorithm == 'grpo':
                    experiment_design['configs']['grpo'] = self.create_grpo_config()
                elif algorithm == 'sft':
                    experiment_design['configs']['sft'] = self.create_sft_config()
            
            # Log experiment design
            self.logger.log({
                'comparison_experiment_design': {
                    'algorithms': algorithms,
                    'research_question': research_question.question,
                    'design_time': self.timer.get_timing_metrics().get('comparison_experiment_design', 0)
                }
            })
            
            return experiment_design
    
    def design_optimization_experiment(self, parameter_name: str, parameter_range: List, research_question: ResearchQuestion):
        """Design optimization experiment using NeMo RL patterns"""
        
        with self.timer.time("optimization_experiment_design"):
            experiment_design = {
                'type': 'optimization',
                'parameter_name': parameter_name,
                'parameter_range': parameter_range,
                'research_question': research_question,
                'configs': []
            }
            
            # Create configurations for each parameter value
            for param_value in parameter_range:
                config = self.create_base_config()
                self.set_parameter_value(config, parameter_name, param_value)
                experiment_design['configs'].append(config)
            
            # Log experiment design
            self.logger.log({
                'optimization_experiment_design': {
                    'parameter_name': parameter_name,
                    'parameter_range': parameter_range,
                    'research_question': research_question.question,
                    'design_time': self.timer.get_timing_metrics().get('optimization_experiment_design', 0)
                }
            })
            
            return experiment_design
```

### 2. Statistical Analysis with NeMo RL Integration

```python
class StatisticalAnalyzer:
    """Statistical analysis with NeMo RL integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.significance_level = config.get('significance_level', 0.05)
        self.power = config.get('power', 0.8)
        
        # Real NeMo RL components
        self.logger = Logger(config.get('logger', {}))
    
    def analyze_experiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results with NeMo RL logging"""
        
        analysis_results = {}
        
        # Basic statistics
        analysis_results['descriptive_stats'] = self.compute_descriptive_statistics(results)
        
        # Statistical tests
        analysis_results['statistical_tests'] = self.perform_statistical_tests(results)
        
        # Effect size analysis
        analysis_results['effect_sizes'] = self.compute_effect_sizes(results)
        
        # Log analysis results
        self.logger.log({
            'experiment_analysis': analysis_results
        })
        
        return analysis_results
    
    def compute_descriptive_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute descriptive statistics"""
        stats = {}
        
        for algorithm, result in results.items():
            if isinstance(result, dict) and 'final_accuracy' in result:
                stats[algorithm] = {
                    'mean': result['final_accuracy'],
                    'std': 0.0,  # Single run
                    'n': 1
                }
        
        return stats
    
    def perform_statistical_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests"""
        tests = {}
        
        # Extract metrics for testing
        metrics = {}
        for algorithm, result in results.items():
            if isinstance(result, dict) and 'final_accuracy' in result:
                metrics[algorithm] = result['final_accuracy']
        
        if len(metrics) >= 2:
            # Perform appropriate statistical test
            if len(metrics) == 2:
                # T-test for two groups
                algorithms = list(metrics.keys())
                t_stat, p_value = stats.ttest_ind([metrics[algorithms[0]]], [metrics[algorithms[1]]])
                tests['comparison'] = {
                    'test': 't_test',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
            else:
                # ANOVA for multiple groups
                f_stat, p_value = stats.f_oneway(*[metrics[alg] for alg in metrics.keys()])
                tests['comparison'] = {
                    'test': 'anova',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
        
        return tests
    
    def compute_effect_sizes(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute effect sizes"""
        effect_sizes = {}
        
        # Extract metrics for effect size calculation
        metrics = {}
        for algorithm, result in results.items():
            if isinstance(result, dict) and 'final_accuracy' in result:
                metrics[algorithm] = result['final_accuracy']
        
        if len(metrics) >= 2:
            algorithms = list(metrics.keys())
            
            # Compute Cohen's d for pairwise comparisons
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    alg1, alg2 = algorithms[i], algorithms[j]
                    
                    # Simplified effect size calculation
                    mean_diff = metrics[alg1] - metrics[alg2]
                    pooled_std = np.sqrt((0 + 0) / 2)  # Simplified for single values
                    
                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                    else:
                        cohens_d = 0
                    
                    effect_sizes[f'{alg1}_vs_{alg2}'] = {
                        'cohens_d': cohens_d,
                        'interpretation': self.interpret_effect_size(cohens_d)
                    }
        
        return effect_sizes
    
    def interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(cohens_d) < 0.2:
            return "small"
        elif abs(cohens_d) < 0.5:
            return "medium"
        else:
            return "large"
```

### 3. Reproducible Experiment Design with NeMo RL Integration

```python
class ReproducibleExperimentDesigner:
    """Reproducible experiment design with NeMo RL integration"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(self.config.get('logger', {}))
        self.timer = Timer()
        self.experiment_log = []
    
    def design_reproducible_experiment(self, research_question: ResearchQuestion):
        """Design reproducible experiment using NeMo RL patterns"""
        
        with self.timer.time("reproducible_experiment_design"):
            # Set random seed for reproducibility
            seed = self.config.get('seed', 42)
            self.set_random_seed(seed)
            
            # Design experiment
            experiment_design = {
                'research_question': research_question,
                'config': self.config,
                'seed': seed,
                'timestamp': time.time(),
                'design_time': self.timer.get_timing_metrics().get('reproducible_experiment_design', 0)
            }
            
            # Log experiment design
            self.logger.log({
                'reproducible_experiment_design': experiment_design
            })
            
            self.experiment_log.append(experiment_design)
            
            return experiment_design
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Log seed setting
        self.logger.log({
            'random_seed_set': seed
        })
    
    def save_experiment_log(self, filename: str):
        """Save experiment log for reproducibility"""
        with open(filename, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        # Log file save
        self.logger.log({
            'experiment_log_saved': filename
        })
    
    def load_experiment_log(self, filename: str):
        """Load experiment log for reproduction"""
        with open(filename, 'r') as f:
            self.experiment_log = json.load(f)
        
        # Log file load
        self.logger.log({
            'experiment_log_loaded': filename
        })
    
    def reproduce_experiment(self, log_filename: str):
        """Reproduce experiment from log"""
        self.load_experiment_log(log_filename)
        
        with self.timer.time("experiment_reproduction"):
            for experiment_design in self.experiment_log:
                # Set random seed
                self.set_random_seed(experiment_design['seed'])
                
                # Reproduce experiment
                result = self.run_experiment(experiment_design)
                
                # Log reproduction
                self.logger.log({
                    'experiment_reproduced': {
                        'original_design': experiment_design,
                        'reproduction_result': result,
                        'reproduction_time': self.timer.get_timing_metrics().get('experiment_reproduction', 0)
                    }
                })
```

## Troubleshooting

### Common Experimental Design Issues

1. **Insufficient Sample Size**: Use power analysis to determine required sample size
2. **Confounding Variables**: Control for confounding variables in experimental design
3. **Multiple Testing**: Use appropriate corrections for multiple comparisons

### Debugging Tips with NeMo RL Integration

```python
# Add debugging to experimental design with NeMo RL logging
def debug_experimental_design(self):
    """
    Debug experimental design issues with NeMo RL integration
    """
    print("=== Experimental Design Debug ===")
    
    # Check configuration
    config_valid = self.validate_config(self.config)
    print(f"Configuration valid: {config_valid}")
    
    # Check research question
    question_valid = self.validate_research_question(self.research_question)
    print(f"Research question valid: {question_valid}")
    
    # Check experiment setup
    setup_valid = self.validate_experiment_setup()
    print(f"Experiment setup valid: {setup_valid}")
    
    print("================================")
    
    # Log debug information using NeMo RL logger
    self.logger.log({
        'experimental_design_debug': {
            'config_valid': config_valid,
            'question_valid': question_valid,
            'setup_valid': setup_valid
        }
    })
```

## Next Steps

- Learn about [Model Evaluation](model-evaluation-validation) for comprehensive assessment
- Review [Reproducible Research](reproducible-research-validation) for scientific rigor
- Explore [Performance Analysis](performance-analysis) for result interpretation 