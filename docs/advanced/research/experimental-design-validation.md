---
description: "Design controlled experiments and research studies with proper experimental methodology for RL research"
tags: ["research", "experiments", "methodology", "experimental-design", "validation"]
categories: ["research-validation"]
---

# Experimental Design

This comprehensive guide covers how to design controlled experiments and research studies with proper experimental methodology for NeMo RL research, including both research methodology and validation frameworks.

## Overview

Proper experimental design is crucial for conducting rigorous research in reinforcement learning. This guide provides frameworks and methodologies for designing experiments that produce reliable, reproducible results, covering both research methodology and validation approaches.

## Key Principles

### Hypothesis Formulation

#### Clear Research Questions
Define specific, testable research questions:

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

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
    
    def formulate_question(self, 
                          question: str,
                          question_type: ResearchQuestionType,
                          hypothesis: str) -> ResearchQuestion:
        """
        Formulate a structured research question
        """
        # Example research questions
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
        
        return ResearchQuestion(
            question=question,
            type=question_type,
            hypothesis=hypothesis,
            null_hypothesis=example.get("null_hypothesis", ""),
            alternative_hypothesis=example.get("alternative_hypothesis", "")
        )
    
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
        
        return True
```

#### Success Criteria
Define measurable success metrics:

- **Primary Metrics**: Preference alignment accuracy, training stability
- **Secondary Metrics**: Training speed, memory efficiency, convergence rate
- **Statistical Thresholds**: p < 0.05, effect size > 0.1

### Control Groups and Baselines

#### Algorithm Baselines
Establish appropriate comparison baselines:

```python
# Example baseline configuration
baseline_configs = {
    "sft": {"algorithm": "sft", "learning_rate": 1e-5},
    "dpo": {"algorithm": "dpo", "beta": 0.2, "learning_rate": 1e-5},
    "grpo": {"algorithm": "grpo", "epsilon": 0.2, "learning_rate": 1e-5}
}
```

#### Hyperparameter Controls
Maintain consistent experimental conditions:

- **Model Architecture**: Same base model across all experiments
- **Dataset**: Identical training and evaluation datasets
- **Hardware**: Consistent GPU configuration and batch sizes
- **Random Seeds**: Fixed seeds for reproducible results

## Experimental Design Framework

### Sample Size Determination

Implement proper sample size calculations:

```python
import scipy.stats as stats
from typing import Tuple

class SampleSizeCalculator:
    def __init__(self):
        self.default_alpha = 0.05
        self.default_power = 0.8
        self.default_effect_size = 0.5
    
    def calculate_sample_size_t_test(self, 
                                   effect_size: float = None,
                                   alpha: float = None,
                                   power: float = None) -> int:
        """
        Calculate sample size for t-test
        """
        if effect_size is None:
            effect_size = self.default_effect_size
        if alpha is None:
            alpha = self.default_alpha
        if power is None:
            power = self.default_power
        
        # Use scipy for power analysis
        result = stats.power.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0  # Equal group sizes
        )
        
        return int(result)
    
    def calculate_sample_size_anova(self,
                                  num_groups: int,
                                  effect_size: float = None,
                                  alpha: float = None,
                                  power: float = None) -> int:
        """
        Calculate sample size for ANOVA
        """
        if effect_size is None:
            effect_size = self.default_effect_size
        if alpha is None:
            alpha = self.default_alpha
        if power is None:
            power = self.default_power
        
        # Use scipy for power analysis
        result = stats.power.f_oneway_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            nobs=num_groups
        )
        
        return int(result)
    
    def calculate_effect_size(self, group1_data: List[float], 
                            group2_data: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        """
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1_data, ddof=1) + 
                             (n2 - 1) * np.var(group2_data, ddof=1)) / (n1 + n2 - 2))
        
        effect_size = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
        return effect_size
    
    def power_analysis(self, sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
        """
        Calculate statistical power
        """
        power = stats.power.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            nobs1=sample_size,
            ratio=1.0
        )
        return power
```

### Randomization and Blocking

Implement proper randomization and blocking strategies:

```python
import random
from typing import List, Dict, Any

class RandomizationManager:
    def __init__(self):
        self.random_seed = None
        self.randomization_scheme = "simple"
    
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def simple_randomization(self, experimental_runs: List[Dict]) -> List[Dict]:
        """
        Simple random assignment
        """
        randomized_runs = experimental_runs.copy()
        random.shuffle(randomized_runs)
        return randomized_runs
    
    def block_randomization(self, experimental_runs: List[Dict], 
                          block_size: int) -> List[Dict]:
        """
        Block randomization
        """
        randomized_runs = []
        
        # Group runs into blocks
        for i in range(0, len(experimental_runs), block_size):
            block = experimental_runs[i:i + block_size]
            random.shuffle(block)
            randomized_runs.extend(block)
        
        return randomized_runs
    
    def stratified_randomization(self, experimental_runs: List[Dict],
                               stratification_factor: str) -> List[Dict]:
        """
        Stratified randomization
        """
        # Group by stratification factor
        strata = {}
        for run in experimental_runs:
            stratum = run['factors'].get(stratification_factor)
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(run)
        
        # Randomize within each stratum
        randomized_runs = []
        for stratum_runs in strata.values():
            random.shuffle(stratum_runs)
            randomized_runs.extend(stratum_runs)
        
        return randomized_runs

class BlockingManager:
    def __init__(self):
        self.blocking_factors = {}
    
    def add_blocking_factor(self, name: str, levels: List[Any]):
        """
        Add blocking factor
        """
        self.blocking_factors[name] = levels
    
    def create_blocks(self, experimental_runs: List[Dict]) -> List[List[Dict]]:
        """
        Create blocks based on blocking factors
        """
        blocks = {}
        
        for run in experimental_runs:
            # Create block key based on blocking factors
            block_key = []
            for factor_name, factor_levels in self.blocking_factors.items():
                if factor_name in run['factors']:
                    block_key.append(run['factors'][factor_name])
                else:
                    block_key.append(None)
            
            block_key = tuple(block_key)
            
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append(run)
        
        return list(blocks.values())
    
    def analyze_block_effects(self, blocks: List[List[Dict]], 
                            response_variable: str) -> Dict[str, float]:
        """
        Analyze block effects
        """
        block_means = {}
        block_variances = {}
        
        for i, block in enumerate(blocks):
            block_values = [run['responses'].get(response_variable, 0) for run in block]
            block_means[f'block_{i}'] = np.mean(block_values)
            block_variances[f'block_{i}'] = np.var(block_values)
        
        return {
            'block_means': block_means,
            'block_variances': block_variances,
            'block_effect_size': np.var(list(block_means.values()))
        }
```

## Experimental Frameworks

### Algorithm Comparison Studies

#### Standardized Evaluation Protocol
```python
# Example evaluation framework
class AlgorithmComparison:
    def __init__(self, algorithms, dataset, metrics):
        self.algorithms = algorithms
        self.dataset = dataset
        self.metrics = metrics
        self.results = {}
    
    def run_comparison(self, n_runs=5):
        """Run multiple independent comparisons"""
        for algorithm in self.algorithms:
            self.results[algorithm] = []
            for run in range(n_runs):
                result = self.evaluate_algorithm(algorithm, run)
                self.results[algorithm].append(result)
    
    def statistical_analysis(self):
        """Perform statistical significance testing"""
        from scipy import stats
        
        # ANOVA for multiple algorithm comparison
        f_stat, p_value = stats.f_oneway(*self.results.values())
        
        # Post-hoc pairwise comparisons
        pairwise_results = {}
        for i, alg1 in enumerate(self.algorithms):
            for j, alg2 in enumerate(self.algorithms[i+1:], i+1):
                t_stat, p_val = stats.ttest_ind(
                    self.results[alg1], 
                    self.results[alg2]
                )
                pairwise_results[f"{alg1}_vs_{alg2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "effect_size": self.calculate_effect_size(alg1, alg2)
                }
        
        return {"anova": (f_stat, p_value), "pairwise": pairwise_results}
```

#### Fair Comparison Methodologies
- **Computational Budget**: Equal training time or steps
- **Hyperparameter Tuning**: Same tuning effort for all algorithms
- **Implementation Quality**: Equivalent implementation quality
- **Evaluation Metrics**: Consistent metric computation

### Hyperparameter Studies

#### Systematic Parameter Exploration
```python
# Example hyperparameter study design
class HyperparameterStudy:
    def __init__(self, base_config, param_ranges):
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.results = {}
    
    def grid_search(self):
        """Systematic grid search over parameter space"""
        from itertools import product
        
        param_combinations = list(product(*self.param_ranges.values()))
        
        for combo in param_combinations:
            config = self.base_config.copy()
            for param, value in zip(self.param_ranges.keys(), combo):
                config[param] = value
            
            result = self.evaluate_config(config)
            self.results[combo] = result
    
    def bayesian_optimization(self, n_trials=50):
        """Bayesian optimization for efficient search"""
        from optuna import create_study
        
        study = create_study(direction="maximize")
        
        def objective(trial):
            config = self.base_config.copy()
            for param, range_def in self.param_ranges.items():
                if isinstance(range_def, tuple):
                    config[param] = trial.suggest_float(param, *range_def)
                elif isinstance(range_def, list):
                    config[param] = trial.suggest_categorical(param, range_def)
            
            return self.evaluate_config(config)
        
        study.optimize(objective, n_trials=n_trials)
        return study
```

#### Efficient Search Strategies
- **Grid Search**: Systematic exploration of parameter space
- **Random Search**: Efficient exploration of large spaces
- **Bayesian Optimization**: Intelligent search with surrogate models
- **Population-based Methods**: Evolutionary algorithms for complex spaces

### Ablation Studies

#### Component Contribution Analysis
```python
# Example ablation study framework
class AblationStudy:
    def __init__(self, full_algorithm, components):
        self.full_algorithm = full_algorithm
        self.components = components
        self.results = {}
    
    def systematic_removal(self):
        """Remove components one by one and measure impact"""
        # Baseline: full algorithm
        self.results["full"] = self.evaluate_algorithm(self.full_algorithm)
        
        # Remove each component
        for component in self.components:
            modified_algorithm = self.remove_component(
                self.full_algorithm, component
            )
            self.results[f"no_{component}"] = self.evaluate_algorithm(
                modified_algorithm
            )
    
    def progressive_removal(self):
        """Remove components progressively and measure cumulative impact"""
        current_algorithm = self.full_algorithm.copy()
        
        for i, component in enumerate(self.components):
            current_algorithm = self.remove_component(
                current_algorithm, component
            )
            self.results[f"removed_{i+1}_components"] = self.evaluate_algorithm(
                current_algorithm
            )
    
    def calculate_contribution(self):
        """Calculate relative contribution of each component"""
        baseline = self.results["full"]
        contributions = {}
        
        for component in self.components:
            key = f"no_{component}"
            if key in self.results:
                performance_loss = baseline - self.results[key]
                contributions[component] = performance_loss / baseline
        
        return contributions
```

#### Impact Quantification Methods
- **Performance Loss**: Measure degradation when component removed
- **Relative Contribution**: Normalize impact by baseline performance
- **Interaction Effects**: Analyze component interactions
- **Statistical Significance**: Test if impact is statistically significant

## Statistical Analysis

### Significance Testing

#### T-Tests for Pairwise Comparisons
```python
from scipy import stats

def pairwise_comparison(algorithm_a_results, algorithm_b_results):
    """Perform t-test for algorithm comparison"""
    t_stat, p_value = stats.ttest_ind(algorithm_a_results, algorithm_b_results)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(algorithm_a_results) - 1) * np.var(algorithm_a_results) +
         (len(algorithm_b_results) - 1) * np.var(algorithm_b_results)) /
        (len(algorithm_a_results) + len(algorithm_b_results) - 2)
    )
    
    effect_size = (np.mean(algorithm_a_results) - np.mean(algorithm_b_results)) / pooled_std
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "significant": p_value < 0.05
    }
```

#### ANOVA for Multiple Comparisons
```python
def multi_algorithm_comparison(algorithm_results):
    """Perform ANOVA for multiple algorithm comparison"""
    f_stat, p_value = stats.f_oneway(*algorithm_results.values())
    
    # Post-hoc analysis with Tukey's HSD
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # Flatten results for Tukey test
    all_results = []
    group_labels = []
    for alg_name, results in algorithm_results.items():
        all_results.extend(results)
        group_labels.extend([alg_name] * len(results))
    
    tukey_result = pairwise_tukeyhsd(all_results, group_labels)
    
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "tukey_results": tukey_result
    }
```

### Effect Size Analysis

#### Cohen's d for Effect Size
```python
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    if abs(d) < 0.2:
        return "small"
    elif abs(d) < 0.5:
        return "medium"
    else:
        return "large"
```

#### Practical Significance
- **Effect Size Thresholds**: d > 0.2 for practical significance
- **Performance Improvement**: >5% improvement for practical relevance
- **Computational Cost**: Balance performance vs. computational overhead

## Advanced Experimental Designs

### Response Surface Methodology

Implement response surface methodology for optimization:

```python
class ResponseSurfaceDesign:
    def __init__(self, factors: Dict[str, List[float]]):
        self.factors = factors
        self.design_points = []
    
    def central_composite_design(self, alpha: float = 1.414) -> List[Dict]:
        """
        Create central composite design
        """
        # Factorial points
        factorial_points = self.full_factorial_design()
        
        # Axial points
        axial_points = self.create_axial_points(alpha)
        
        # Center points
        center_points = self.create_center_points(5)
        
        # Combine all points
        design_points = factorial_points + axial_points + center_points
        
        return design_points
    
    def create_axial_points(self, alpha: float) -> List[Dict]:
        """
        Create axial points for CCD
        """
        axial_points = []
        factor_names = list(self.factors.keys())
        
        for i, factor_name in enumerate(factor_names):
            # High axial point
            high_point = {name: 0 for name in factor_names}
            high_point[factor_name] = alpha
            axial_points.append(high_point)
            
            # Low axial point
            low_point = {name: 0 for name in factor_names}
            low_point[factor_name] = -alpha
            axial_points.append(low_point)
        
        return axial_points
    
    def create_center_points(self, num_center: int) -> List[Dict]:
        """
        Create center points
        """
        center_points = []
        factor_names = list(self.factors.keys())
        
        for _ in range(num_center):
            center_point = {name: 0 for name in factor_names}
            center_points.append(center_point)
        
        return center_points
    
    def analyze_response_surface(self, responses: List[float]) -> Dict[str, Any]:
        """
        Analyze response surface
        """
        # Fit quadratic model
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # Create design matrix
        X = np.array([list(point.values()) for point in self.design_points])
        
        # Add quadratic terms
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, responses)
        
        return {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r_squared': model.score(X_poly, responses),
            'feature_names': poly.get_feature_names_out()
        }
```

### Sequential Experimental Design

Implement sequential experimental design:

```python
class SequentialExperimentalDesign:
    def __init__(self, initial_design: List[Dict], 
                 stopping_criteria: str = "futility_bound"):
        self.initial_design = initial_design
        self.stopping_criteria = stopping_criteria
        self.current_stage = 1
        self.results = []
        self.futility_bound = 0.1
    
    def run_sequential_experiment(self) -> Dict[str, Any]:
        """
        Run sequential experiment
        """
        current_design = self.initial_design.copy()
        
        while not self.should_stop():
            # Run current stage
            stage_results = self.run_stage(current_design)
            self.results.append(stage_results)
            
            # Analyze results
            analysis = self.analyze_stage_results(stage_results)
            
            # Check stopping criteria
            if self.check_stopping_criteria(analysis):
                break
            
            # Update design for next stage
            current_design = self.update_design(current_design, analysis)
            self.current_stage += 1
        
        return self.summarize_results()
    
    def should_stop(self) -> bool:
        """
        Check if experiment should stop
        """
        if self.current_stage > 5:  # Maximum stages
            return True
        
        if len(self.results) > 0:
            # Check futility bound
            latest_result = self.results[-1]
            if latest_result.get('p_value', 1.0) > self.futility_bound:
                return True
        
        return False
    
    def run_stage(self, design: List[Dict]) -> Dict[str, Any]:
        """
        Run experimental stage
        """
        # Simulate running experiments
        results = {
            'stage': self.current_stage,
            'design': design,
            'responses': [],
            'p_value': 0.05,  # Simulated p-value
            'effect_size': 0.3  # Simulated effect size
        }
        
        for run in design:
            # Simulate response
            response = self.simulate_response(run)
            results['responses'].append(response)
        
        return results
    
    def simulate_response(self, run: Dict) -> float:
        """
        Simulate experimental response
        """
        # Simple simulation - replace with actual experiment
        base_response = 0.5
        algorithm_effect = 0.1 if run['factors'].get('algorithm') == 'dpo' else 0
        beta_effect = 0.05 * run['factors'].get('beta', 0.1)
        
        response = base_response + algorithm_effect + beta_effect + np.random.normal(0, 0.1)
        return max(0, min(1, response))  # Clamp to [0, 1]
    
    def analyze_stage_results(self, stage_results: Dict) -> Dict[str, Any]:
        """
        Analyze stage results
        """
        responses = stage_results['responses']
        
        analysis = {
            'mean_response': np.mean(responses),
            'std_response': np.std(responses),
            'effect_size': stage_results.get('effect_size', 0),
            'p_value': stage_results.get('p_value', 1.0),
            'confidence_interval': self.calculate_confidence_interval(responses)
        }
        
        return analysis
    
    def calculate_confidence_interval(self, responses: List[float], 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval
        """
        mean = np.mean(responses)
        std = np.std(responses, ddof=1)
        n = len(responses)
        
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_value * std / np.sqrt(n)
        
        return (mean - margin_of_error, mean + margin_of_error)
```

## Reproducibility Framework

### Environment Management

```python
def setup_reproducible_experiment(seed=42):
    """Setup reproducible environment for NeMo RL experiments"""
    import torch
    import numpy as np
    import random
    
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for Ray
    import os
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return {
        'seed': seed,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count()
    }

def log_experiment_metadata(config, results, output_dir):
    """Log complete experiment metadata for reproducibility"""
    import json
    import datetime
    
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'results': results,
        'system_info': {
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'ray_version': ray.__version__,
            'gpu_info': get_gpu_info()
        }
    }
    
    with open(f"{output_dir}/experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Seed Management

```python
def set_deterministic_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Configuration

### Experimental Design Configuration

```yaml
# configs/experimental_design.yaml
experimental_design:
  research_question:
    question: "Does DPO outperform SFT on preference alignment?"
    type: "comparison"
    hypothesis: "DPO achieves higher preference alignment scores than SFT"
    significance_level: 0.05
    power: 0.8
  
  factors:
    algorithm:
      levels: ["dpo", "sft", "grpo"]
      type: "categorical"
    
    beta:
      levels: [0.05, 0.1, 0.2]
      type: "continuous"
    
    learning_rate:
      levels: [1e-5, 3e-5, 1e-4]
      type: "continuous"
    
    model_size:
      levels: ["1B", "3B", "7B"]
      type: "categorical"
  
  responses:
    - name: "preference_alignment_score"
      type: "continuous"
    
    - name: "training_time"
      type: "continuous"
    
    - name: "convergence_steps"
      type: "continuous"
  
  design_type: "full_factorial"
  randomization: "block"
  blocking_factors:
    - "data_batch"
    - "gpu_id"
  
  sample_size:
    calculation_method: "power_analysis"
    effect_size: 0.5
    alpha: 0.05
    power: 0.8
```

### Advanced Experimental Configuration

```yaml
# configs/advanced_experimental_design.yaml
experimental_design:
  # Response surface methodology
  response_surface:
    enabled: true
    design_type: "central_composite"
    center_points: 5
    alpha: 1.414
  
  # Covariates
  covariates:
    - name: "data_quality_score"
      values: [0.8, 0.9, 1.0]
    
    - name: "hardware_performance"
      values: [0.9, 1.0, 1.1]
  
  # Replication
  replication:
    enabled: true
    num_replicates: 3
    replication_type: "independent"
  
  # Sequential experimentation
  sequential:
    enabled: true
    stopping_criteria: "futility_bound"
    max_stages: 3
```

## Best Practices

### 1. Experimental Control

Implement proper experimental control:

```python
class ExperimentalController:
    def __init__(self):
        self.control_variables = {}
        self.randomization_scheme = None
        self.blocking_scheme = None
    
    def set_control_variables(self, variables: Dict[str, Any]):
        """
        Set control variables
        """
        self.control_variables = variables
    
    def validate_experimental_conditions(self, run: Dict) -> bool:
        """
        Validate experimental conditions
        """
        # Check if all required factors are present
        required_factors = set(self.control_variables.keys())
        actual_factors = set(run['factors'].keys())
        
        if not required_factors.issubset(actual_factors):
            return False
        
        # Check if factor levels are valid
        for factor_name, expected_level in self.control_variables.items():
            if run['factors'].get(factor_name) != expected_level:
                return False
        
        return True
    
    def log_experimental_conditions(self, run: Dict):
        """
        Log experimental conditions for reproducibility
        """
        log_entry = {
            'timestamp': time.time(),
            'run_id': run.get('run_id'),
            'factors': run['factors'],
            'control_variables': self.control_variables,
            'random_seed': self.randomization_scheme.get('seed') if self.randomization_scheme else None
        }
        
        # Save to file
        with open('experimental_log.json', 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
```

### 2. Statistical Analysis Planning

Plan statistical analyses in advance:

```python
class StatisticalAnalysisPlan:
    def __init__(self, research_question: ResearchQuestion):
        self.research_question = research_question
        self.primary_analysis = None
        self.secondary_analyses = []
        self.additional_analyses = []
    
    def define_primary_analysis(self, analysis_type: str, 
                              response_variable: str,
                              factors: List[str]):
        """
        Define primary statistical analysis
        """
        self.primary_analysis = {
            'type': analysis_type,
            'response_variable': response_variable,
            'factors': factors,
            'alpha': self.research_question.significance_level,
            'power': self.research_question.power
        }
    
    def add_secondary_analysis(self, analysis_type: str,
                             response_variable: str,
                             factors: List[str]):
        """
        Add secondary analysis
        """
        self.secondary_analyses.append({
            'type': analysis_type,
            'response_variable': response_variable,
            'factors': factors
        })
    
    def add_additional_analysis(self, analysis_type: str,
                              description: str,
                              parameters: Dict[str, Any]):
        """
        Add additional analysis
        """
        self.additional_analyses.append({
            'type': analysis_type,
            'description': description,
            'parameters': parameters
        })
    
    def execute_analysis_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete analysis plan
        """
        results = {
            'primary_analysis': self.execute_primary_analysis(data),
            'secondary_analyses': [],
            'additional_analyses': []
        }
        
        # Execute secondary analyses
        for analysis in self.secondary_analyses:
            result = self.execute_analysis(analysis, data)
            results['secondary_analyses'].append(result)
        
        # Execute additional analyses
        for analysis in self.additional_analyses:
            result = self.execute_analysis(analysis, data)
            results['additional_analyses'].append(result)
        
        return results
    
    def execute_primary_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute primary analysis
        """
        if not self.primary_analysis:
            return {}
        
        analysis_type = self.primary_analysis['type']
        
        if analysis_type == 't_test':
            return self.perform_t_test(data)
        elif analysis_type == 'anova':
            return self.perform_anova(data)
        elif analysis_type == 'regression':
            return self.perform_regression(data)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def perform_t_test(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform t-test
        """
        # Extract data for t-test
        group1_data = data.get('group1', [])
        group2_data = data.get('group2', [])
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Calculate effect size
        effect_size = self.calculate_effect_size(group1_data, group2_data)
        
        return {
            'test_type': 't_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.primary_analysis['alpha']
        }
```

### 3. Documentation Standards

#### Experiment Logging
```python
import logging
from datetime import datetime

def setup_experiment_logging(experiment_name):
    """Setup comprehensive experiment logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiments/{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(experiment_name)
```

#### Result Reporting Template
```markdown
## Experiment Results

### Experimental Setup
- **Algorithms**: GRPO, DPO, SFT
- **Dataset**: Preference dataset (N=10,000)
- **Hardware**: 4x A100 GPUs
- **Training Time**: 24 hours per algorithm

### Statistical Results
- **Primary Metric**: Preference alignment accuracy
- **Statistical Test**: One-way ANOVA
- **Effect Size**: Cohen's d
- **Significance Level**: α = 0.05

### Key Findings
1. GRPO significantly outperforms DPO (p < 0.001, d = 0.8)
2. Both RL algorithms outperform SFT baseline (p < 0.001)
3. Training stability: GRPO > DPO > SFT

### Limitations
- Single dataset evaluation
- Limited hyperparameter exploration
- Computational constraints
```

## Implementation Examples

### Complete Experiment Workflow
```python
class NeMoRLExperiment:
    def __init__(self, config):
        self.config = config
        self.logger = setup_experiment_logging(config["experiment_name"])
        set_deterministic_seeds(config["seed"])
    
    def run_comparison(self):
        """Run complete algorithm comparison experiment"""
        self.logger.info("Starting algorithm comparison experiment")
        
        # Initialize algorithms
        algorithms = {
            "sft": SFTAlgorithm(self.config["sft_params"]),
            "dpo": DPOAlgorithm(self.config["dpo_params"]),
            "grpo": GRPOAlgorithm(self.config["grpo_params"])
        }
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            self.logger.info(f"Training {alg_name}")
            alg_results = []
            
            for run in range(self.config["n_runs"]):
                self.logger.info(f"Run {run + 1}/{self.config['n_runs']}")
                result = self.train_and_evaluate(algorithm, run)
                alg_results.append(result)
            
            results[alg_name] = alg_results
        
        # Statistical analysis
        statistical_results = self.analyze_results(results)
        
        # Generate report
        self.generate_report(results, statistical_results)
        
        return results, statistical_results
    
    def analyze_results(self, results):
        """Perform comprehensive statistical analysis"""
        # ANOVA for overall comparison
        anova_result = multi_algorithm_comparison(results)
        
        # Pairwise comparisons
        pairwise_results = {}
        algorithms = list(results.keys())
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                comparison = pairwise_comparison(
                    results[alg1], results[alg2]
                )
                pairwise_results[f"{alg1}_vs_{alg2}"] = comparison
        
        return {
            "anova": anova_result,
            "pairwise": pairwise_results
        }
```

## Troubleshooting

### Common Experimental Design Issues

1. **Insufficient Power**: Increase sample size or effect size
2. **Confounding Variables**: Add blocking factors or covariates
3. **Poor Randomization**: Use proper randomization schemes

### Debugging Tips

```python
# Add debugging to experimental design
def debug_experimental_design(self):
    """
    Debug experimental design issues
    """
    print("=== Experimental Design Debug ===")
    
    # Check research question
    print(f"Research question: {self.research_question.question}")
    print(f"Hypothesis: {self.research_question.hypothesis}")
    print(f"Significance level: {self.research_question.significance_level}")
    
    # Check factors
    print(f"Number of factors: {len(self.factors)}")
    for name, info in self.factors.items():
        print(f"  {name}: {len(info['levels'])} levels")
    
    # Check experimental runs
    print(f"Number of experimental runs: {len(self.experimental_runs)}")
    
    # Check randomization
    if self.randomization_scheme:
        print(f"Randomization scheme: {self.randomization_scheme}")
    
    print("================================")
```

## Next Steps

After understanding experimental design principles:

1. **Plan Your Experiment**: Define clear hypotheses and success criteria
2. **Design Controls**: Establish appropriate baselines and controls
3. **Implement Protocols**: Set up reproducible experimental procedures
4. **Analyze Results**: Apply proper statistical analysis
5. **Report Findings**: Document results with appropriate detail

- Learn about [Model Evaluation](model-evaluation-validation) for comprehensive assessment
- Review [Reproducible Research](reproducible-research-validation) for scientific rigor
- Explore [Algorithm Development](../algorithm-development/index) for advanced training

## References

- Fisher, R.A. "The Design of Experiments." Oliver and Boyd (1935).
- Cohen, J. "Statistical Power Analysis for the Behavioral Sciences." Routledge (1988).
- Tukey, J.W. "Comparing Individual Means in the Analysis of Variance." Biometrics (1949).
- Benjamini, Y., & Hochberg, Y. "Controlling the False Discovery Rate." JRSS-B (1995). 