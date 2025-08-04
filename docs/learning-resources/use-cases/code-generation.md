# Code Generation

Train NeMo RL models to generate, debug, and optimize code across multiple programming languages. This use case covers architectural patterns for building code generation systems that can handle syntax validation, multi-language support, and training optimization.

## Overview

Code generation with RL involves training models to understand programming languages, generate syntactically correct code, and produce solutions that meet specific requirements. Unlike traditional supervised learning approaches, RL enables models to learn from feedback about code quality, correctness, and efficiency.

## Key Challenges

### Syntax and Semantics
- **Multi-language support**: Training models that can generate code in Python, JavaScript, Java, C++, etc.
- **Syntax validation**: Ensuring generated code compiles and follows language conventions
- **Context awareness**: Understanding project structure, imports, and dependencies

### Code Quality
- **Correctness**: Generating code that produces expected outputs
- **Efficiency**: Optimizing for performance, memory usage, and readability
- **Best practices**: Following language-specific conventions and patterns

### Evaluation Metrics
- **Compilation success rate**: Percentage of generated code that compiles
- **Functional correctness**: Code that produces expected outputs
- **Code quality scores**: Readability, maintainability, and efficiency metrics

## Architecture Patterns

### Multi-Stage Training Pipeline

```python
# Example: Code generation training pipeline using NeMo RL
from nemo_rl.algorithms.sft import setup as sft_setup, sft_train
from nemo_rl.algorithms.grpo import setup as grpo_setup, grpo_train
from nemo_rl.environments.interfaces import EnvironmentInterface

class CodeGenerationTraining:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.policy = None
    
    def train_sft(self, dataset):
        """Stage 1: Supervised Fine-Tuning for code understanding"""
        # Setup SFT training
        policy, cluster, dataloader, loss_fn, master_config, logger, task_spec, save_state = sft_setup(
            master_config=self.config,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            val_dataset=dataset  # Use same dataset for validation
        )
        
        # Run SFT training
        sft_train(
            policy=policy,
            dataloader=dataloader,
            tokenizer=self.tokenizer,
            loss_fn=loss_fn,
            master_config=master_config,
            logger=logger,
            checkpointer=checkpointer,
            sft_save_state=save_state
        )
        
        return policy
    
    def train_grpo(self, policy, environment):
        """Stage 2: GRPO for code quality optimization"""
        # Setup GRPO training
        policy, policy_generation, cluster, dataloader, val_dataloader, loss_fn, logger, checkpointer, grpo_state, master_config = grpo_setup(
            config=self.config,
            tokenizer=self.tokenizer,
            dataset=dataset,
            val_dataset=val_dataset
        )
        
        # Run GRPO training
        grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            master_config=master_config,
            logger=logger,
            checkpointer=checkpointer,
            grpo_state=grpo_state
        )
```

### Environment Design

```python
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from typing import List, Optional, Dict, Any
import torch
import subprocess
import tempfile
import os

class CodeGenerationEnvironment(EnvironmentInterface):
    """Real code generation environment for NeMo RL."""
    
    def __init__(self, target_language="python"):
        self.language = target_language
        self.compiler = self._get_compiler(target_language)
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process code generation batch and return rewards."""
        observations = []
        rewards = []
        terminateds = []
        
        for i, message_log in enumerate(message_log_batch):
            # Extract generated code from assistant response
            code = self._extract_code(message_log)
            
            # Evaluate code quality
            reward = self._evaluate_code(code)
            
            observations.append({"role": "assistant", "content": code})
            rewards.append(reward)
            terminateds.append(True)  # Code generation is single-turn
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards),
            terminateds=torch.tensor(terminateds)
        )
    
    def _extract_code(self, message_log: List[Dict[str, str]]) -> str:
        """Extract code from assistant response."""
        for message in reversed(message_log):
            if message["role"] == "assistant":
                return message["content"]
        return ""
    
    def _evaluate_code(self, code: str) -> float:
        """Evaluate code quality and return reward."""
        if not code.strip():
            return -1.0
        
        reward = 0.0
        
        # Basic syntax check
        if self._check_syntax(code):
            reward += 0.3
        else:
            reward -= 0.5
        
        # Code quality metrics
        reward += self._assess_code_quality(code) * 0.7
        
        return reward
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid syntax."""
        try:
            if self.language == "python":
                compile(code, '<string>', 'exec')
                return True
            # Add other language checks as needed
            return True
        except:
            return False
    
    def _assess_code_quality(self, code: str) -> float:
        """Assess code quality metrics."""
        quality_score = 0.0
        
        # Length penalty (prefer concise code)
        if len(code) < 100:
            quality_score += 0.2
        elif len(code) > 1000:
            quality_score -= 0.2
        
        # Complexity penalty
        if code.count('if') + code.count('for') + code.count('while') < 5:
            quality_score += 0.3
        
        return min(max(quality_score, 0.0), 1.0)
    
    def global_post_process_and_metrics(self, batch: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process batch and compute global metrics."""
        # Calculate global metrics
        avg_reward = batch.get("rewards", torch.tensor([0.0])).mean().item()
        success_rate = (batch.get("rewards", torch.tensor([0.0])) > 0.0).float().mean().item()
        
        metrics = {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "total_samples": len(batch.get("rewards", []))
        }
        
        return batch, metrics
```

## Implementation Considerations

### Data Preparation

```python
class CodeGenerationDataset:
    def __init__(self, language="python"):
        self.language = language
        self.code_examples = self.load_code_examples()
        self.test_cases = self.load_test_cases()
    
    def format_for_training(self):
        """Format code examples for RL training"""
        formatted_data = []
        
        for example in self.code_examples:
            # Create prompt-response pairs
            prompt = f"Write a {self.language} function that: {example['description']}"
            response = example['code']
            
            # Add test cases for evaluation
            test_cases = example.get('tests', [])
            
            formatted_data.append({
                'prompt': prompt,
                'response': response,
                'tests': test_cases,
                'language': self.language
            })
        
        return formatted_data
```

### Reward Function Design

```python
class CodeQualityReward:
    def __init__(self):
        self.metrics = {
            'compilation': CompilationMetric(),
            'functionality': FunctionalityMetric(),
            'readability': ReadabilityMetric(),
            'efficiency': EfficiencyMetric()
        }
    
    def calculate_reward(self, generated_code, test_cases, expected_output):
        total_reward = 0.0
        weights = {
            'compilation': 0.3,
            'functionality': 0.4,
            'readability': 0.2,
            'efficiency': 0.1
        }
        
        for metric_name, metric in self.metrics.items():
            score = metric.evaluate(generated_code, test_cases, expected_output)
            total_reward += score * weights[metric_name]
        
        return total_reward
```

## Training Optimization

### Distributed Training Architecture

```python
class CodeGenerationService:
    def __init__(self, model_path, language="python"):
        self.model = self.load_model(model_path)
        self.language = language
        self.compiler = LanguageCompiler(language)
        self.validator = CodeValidator()
    
    def generate_code(self, prompt, constraints=None):
        """Generate code based on prompt and constraints"""
        try:
            # Generate code using RL model
            generated_code = self.model.generate(prompt)
            
            # Validate and optimize
            if constraints:
                generated_code = self.apply_constraints(generated_code, constraints)
            
            # Compile and test
            compilation_result = self.compiler.compile(generated_code)
            
            return {
                'code': generated_code,
                'compilation_success': compilation_result.success,
                'warnings': compilation_result.warnings,
                'quality_score': self.validator.assess_quality(generated_code)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def apply_constraints(self, code, constraints):
        """Apply constraints like max lines, specific patterns, etc."""
        # Implementation for constraint application
        return code
```

### A/B Testing Framework

```python
class CodeGenerationABTest:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.metrics = CodeGenerationMetrics()
    
    def run_experiment(self, test_cases, traffic_split=0.5):
        """Run A/B test between two code generation models"""
        results = {
            'model_a': {'success_rate': 0, 'quality_score': 0, 'latency': 0},
            'model_b': {'success_rate': 0, 'quality_score': 0, 'latency': 0}
        }
        
        for test_case in test_cases:
            # Randomly assign to model A or B
            if random.random() < traffic_split:
                model = self.model_a
                model_key = 'model_a'
            else:
                model = self.model_b
                model_key = 'model_b'
            
            # Generate and evaluate
            start_time = time.time()
            generated_code = model.generate(test_case['prompt'])
            latency = time.time() - start_time
            
            # Evaluate quality
            quality_score = self.metrics.evaluate(generated_code, test_case)
            
            # Update results
            results[model_key]['success_rate'] += 1 if quality_score > 0.7 else 0
            results[model_key]['quality_score'] += quality_score
            results[model_key]['latency'] += latency
        
        # Calculate averages
        for model_key in results:
            count = len(test_cases) // 2
            results[model_key]['success_rate'] /= count
            results[model_key]['quality_score'] /= count
            results[model_key]['latency'] /= count
        
        return results
```

## Real-World Examples

### Python Function Generation

```python
# Example: Training a model to generate Python functions
config = {
    'task': 'python_function_generation',
    'training_data': 'python_code_dataset',
    'evaluation_metrics': ['compilation_rate', 'test_pass_rate', 'readability_score'],
    'reward_function': 'code_quality_reward',
    'constraints': {
        'max_lines': 50,
        'required_docstring': True,
        'type_hints': True
    }
}

# Training pipeline
pipeline = CodeGenerationPipeline(config)
pipeline.train(dataset)
```

### Multi-Language Support

```python
# Example: Supporting multiple programming languages
languages = ['python', 'javascript', 'java', 'cpp']

for language in languages:
    env = CodeGenerationEnvironment(target_language=language)
    model = train_code_generation_model(env, language)
    
    # Deploy language-specific model
    deploy_model(model, f"code_gen_{language}")
```

## Best Practices

### Data Quality
- **Curate high-quality code examples** with proper documentation and tests
- **Include diverse programming patterns** and use cases
- **Validate code correctness** before using as training data

### Model Training
- **Start with SFT** to establish basic code understanding
- **Use RL for quality optimization** with carefully designed reward functions
- **Implement curriculum learning** starting with simple tasks and progressing to complex ones

### Evaluation
- **Automated testing** for compilation and functionality
- **Human evaluation** for code quality and readability
- **Continuous monitoring** of model performance in production

### Production Considerations
- **Version control** for generated code
- **Security scanning** for potential vulnerabilities
- **Rate limiting** to prevent abuse
- **Fallback mechanisms** for when generation fails

## Monitoring and Observability

```python
class CodeGenerationMonitor:
    def __init__(self):
        self.metrics = {
            'generation_success_rate': 0,
            'compilation_success_rate': 0,
            'average_quality_score': 0,
            'response_time_p95': 0,
            'error_rate': 0
        }
    
    def log_generation(self, prompt, generated_code, quality_score, latency):
        """Log generation metrics for monitoring"""
        # Update metrics
        self.metrics['generation_success_rate'] += 1 if generated_code else 0
        self.metrics['average_quality_score'] += quality_score
        self.metrics['response_time_p95'] = max(self.metrics['response_time_p95'], latency)
        
        # Send to monitoring system
        self.send_metrics(self.metrics)
```

This use case provides a comprehensive framework for building training-optimized code generation systems with NeMo RL, covering everything from data preparation to training optimization and monitoring. 