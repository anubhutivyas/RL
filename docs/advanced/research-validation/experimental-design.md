---
description: "Design controlled experiments and research studies with proper experimental methodology for RL research"
tags: ["research", "experiments", "methodology", "experimental-design", "validation"]
categories: ["research-validation"]
---

# Experimental Design

This guide covers how to design controlled experiments and research studies with proper experimental methodology for NeMo RL research.

## Overview

Proper experimental design is crucial for conducting rigorous research in reinforcement learning. This guide provides frameworks and methodologies for designing experiments that produce reliable, reproducible results.

## Key Components

### Research Question Formulation

Start with well-defined research questions:

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

### Experimental Design Framework

Implement a comprehensive experimental design framework:

```python
from typing import List, Dict, Any, Optional
import random
import numpy as np

class ExperimentalDesign:
    def __init__(self, research_question: ResearchQuestion):
        self.research_question = research_question
        self.factors = {}
        self.responses = []
        self.experimental_runs = []
        
    def add_factor(self, name: str, levels: List[Any], factor_type: str = "categorical"):
        """
        Add experimental factor
        """
        self.factors[name] = {
            'levels': levels,
            'type': factor_type,
            'current_level': None
        }
    
    def add_response(self, name: str, metric_type: str = "continuous"):
        """
        Add response variable
        """
        self.responses.append({
            'name': name,
            'type': metric_type,
            'values': []
        })
    
    def design_experiment(self, design_type: str = "full_factorial") -> List[Dict]:
        """
        Design experimental runs
        """
        if design_type == "full_factorial":
            return self.full_factorial_design()
        elif design_type == "fractional_factorial":
            return self.fractional_factorial_design()
        elif design_type == "response_surface":
            return self.response_surface_design()
        elif design_type == "randomized":
            return self.randomized_design()
        else:
            raise ValueError(f"Unknown design type: {design_type}")
    
    def full_factorial_design(self) -> List[Dict]:
        """
        Create full factorial experimental design
        """
        # Generate all combinations of factor levels
        factor_names = list(self.factors.keys())
        factor_levels = [self.factors[name]['levels'] for name in factor_names]
        
        # Create all combinations
        import itertools
        combinations = list(itertools.product(*factor_levels))
        
        experimental_runs = []
        for i, combination in enumerate(combinations):
            run = {
                'run_id': i + 1,
                'factors': dict(zip(factor_names, combination)),
                'responses': {},
                'status': 'pending'
            }
            experimental_runs.append(run)
        
        self.experimental_runs = experimental_runs
        return experimental_runs
    
    def randomized_design(self, num_runs: int = 10) -> List[Dict]:
        """
        Create randomized experimental design
        """
        experimental_runs = []
        
        for i in range(num_runs):
            run = {
                'run_id': i + 1,
                'factors': {},
                'responses': {},
                'status': 'pending'
            }
            
            # Randomly assign factor levels
            for factor_name, factor_info in self.factors.items():
                run['factors'][factor_name] = random.choice(factor_info['levels'])
            
            experimental_runs.append(run)
        
        self.experimental_runs = experimental_runs
        return experimental_runs
    
    def add_blocking_factor(self, blocking_factor: str, blocks: List[str]):
        """
        Add blocking factor to control for nuisance variables
        """
        self.factors[blocking_factor] = {
            'levels': blocks,
            'type': 'blocking',
            'current_level': None
        }
    
    def add_covariate(self, covariate_name: str, covariate_values: List[float]):
        """
        Add covariate for statistical control
        """
        self.factors[covariate_name] = {
            'levels': covariate_values,
            'type': 'covariate',
            'current_level': None
        }
```

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

### 3. Reproducibility Framework

Ensure experimental reproducibility:

```python
class ReproducibilityFramework:
    def __init__(self):
        self.experiment_config = {}
        self.random_seeds = {}
        self.environment_info = {}
    
    def set_experiment_config(self, config: Dict[str, Any]):
        """
        Set experiment configuration
        """
        self.experiment_config = config
    
    def set_random_seeds(self, seeds: Dict[str, int]):
        """
        Set random seeds for reproducibility
        """
        self.random_seeds = seeds
        
        # Set seeds
        if 'python' in seeds:
            random.seed(seeds['python'])
        if 'numpy' in seeds:
            np.random.seed(seeds['numpy'])
        if 'torch' in seeds:
            torch.manual_seed(seeds['torch'])
    
    def capture_environment_info(self):
        """
        Capture environment information
        """
        import platform
        import sys
        
        self.environment_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'torch_version': torch.__version__,
            'gpu_info': self.get_gpu_info()
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information
        """
        if torch.cuda.is_available():
            return {
                'gpu_count': torch.cuda.device_count(),
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'cuda_version': torch.version.cuda
            }
        else:
            return {'gpu_count': 0}
    
    def save_experiment_metadata(self, filename: str):
        """
        Save experiment metadata for reproducibility
        """
        metadata = {
            'experiment_config': self.experiment_config,
            'random_seeds': self.random_seeds,
            'environment_info': self.environment_info,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_experiment_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Load experiment metadata
        """
        with open(filename, 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        self.experiment_config = metadata.get('experiment_config', {})
        self.random_seeds = metadata.get('random_seeds', {})
        self.environment_info = metadata.get('environment_info', {})
        
        # Restore random seeds
        self.set_random_seeds(self.random_seeds)
        
        return metadata
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

- Learn about [Model Evaluation](model-evaluation) for comprehensive assessment
- Review [Reproducible Research](reproducible-research) for scientific rigor
- Explore [Algorithm Customization](../algorithm-customization/index) for advanced training 