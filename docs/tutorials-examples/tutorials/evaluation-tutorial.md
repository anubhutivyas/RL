# Evaluation Tutorial: Model Evaluation and Benchmarking

Welcome to the Model Evaluation tutorial! This comprehensive guide will teach you how to evaluate and benchmark your NeMo RL models using various metrics and techniques.

## What You'll Learn

In this tutorial, you'll learn:

- **Evaluation Fundamentals**: Understanding model evaluation concepts
- **Evaluation Metrics**: Choosing and implementing appropriate metrics
- **Evaluation Setup**: Setting up comprehensive evaluation pipelines
- **Human Evaluation**: Incorporating human feedback in evaluation
- **Benchmarking**: Comparing models across different criteria
- **Best Practices**: Proven techniques for reliable evaluation

## Prerequisites

Before starting this tutorial, ensure you have:

- **NeMo RL**: Installed and set up (see [Installation Guide](../../get-started/installation))
- **Python Knowledge**: Basic understanding of Python programming
- **Machine Learning**: Familiarity with ML concepts (helpful but not required)
- **Hardware**: GPU with sufficient memory for your chosen model size

## Tutorial Overview

### **Step 1: Understanding Evaluation**
Learn the fundamentals of model evaluation and why it's important.

### **Step 2: Evaluation Metrics**
Choose and implement appropriate evaluation metrics.

### **Step 3: Evaluation Setup**
Set up comprehensive evaluation pipelines.

### **Step 4: Human Evaluation**
Incorporate human feedback in your evaluation.

### **Step 5: Benchmarking**
Compare models across different criteria.

### **Step 6: Best Practices**
Learn proven techniques for reliable evaluation.

## Step 1: Understanding Evaluation

### **What is Model Evaluation?**

Model evaluation is the process of assessing how well your trained models perform on specific tasks and criteria. It helps you understand model strengths, weaknesses, and areas for improvement.

### **Why Evaluation Matters**

Evaluation is crucial for:
- **Quality Assurance**: Ensuring models meet performance requirements
- **Model Selection**: Choosing the best model for deployment
- **Iterative Improvement**: Identifying areas for model enhancement
- **Production Readiness**: Validating models before deployment

### **Evaluation Types**

| Evaluation Type | Purpose | Metrics | Use Case |
|----------------|---------|---------|----------|
| **Automatic** | Automated assessment | BLEU, ROUGE, Perplexity | Quick evaluation |
| **Human** | Human judgment | Quality, Safety, Alignment | Subjective assessment |
| **Task-Specific** | Domain-specific evaluation | Accuracy, F1, Custom metrics | Specialized domains |
| **Comparative** | Model comparison | Relative rankings | Model selection |

## Step 2: Evaluation Metrics

### **Automatic Metrics**

#### **Perplexity**
Measures model uncertainty and language modeling quality:

```python
def calculate_perplexity(model, test_dataset):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            outputs = model(batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["attention_mask"].sum().item()
            total_tokens += batch["attention_mask"].sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()
```

#### **BLEU Score**
Measures text generation quality:

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_bleu_score(references, hypotheses):
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = sentence_bleu([ref.split()], hyp.split())
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)
```

#### **ROUGE Score**
Measures text summarization quality:

```python
from rouge_score import rouge_scorer

def calculate_rouge_scores(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    return {key: sum(values)/len(values) for key, values in scores.items()}
```

### **Task-Specific Metrics**

#### **Code Generation Metrics**
```python
def evaluate_code_generation(model, test_dataset):
    metrics = {
        'exact_match': 0,
        'functional_correctness': 0,
        'syntax_correctness': 0
    }
    
    for batch in test_dataset:
        generated_code = model.generate(batch["prompt"])
        
        # Exact match
        if generated_code == batch["reference"]:
            metrics['exact_match'] += 1
        
        # Functional correctness (run tests)
        if test_code_functionality(generated_code, batch["tests"]):
            metrics['functional_correctness'] += 1
        
        # Syntax correctness
        if check_syntax(generated_code):
            metrics['syntax_correctness'] += 1
    
    return {k: v/len(test_dataset) for k, v in metrics.items()}
```

#### **Mathematical Reasoning Metrics**
```python
def evaluate_math_reasoning(model, test_dataset):
    metrics = {
        'answer_accuracy': 0,
        'reasoning_quality': 0,
        'step_correctness': 0
    }
    
    for batch in test_dataset:
        response = model.generate(batch["problem"])
        
        # Answer accuracy
        if extract_and_check_answer(response, batch["correct_answer"]):
            metrics['answer_accuracy'] += 1
        
        # Reasoning quality (human evaluation)
        reasoning_score = evaluate_reasoning_quality(response)
        metrics['reasoning_quality'] += reasoning_score
        
        # Step correctness
        if check_step_correctness(response, batch["solution_steps"]):
            metrics['step_correctness'] += 1
    
    return {k: v/len(test_dataset) for k, v in metrics.items()}
```

## Step 3: Evaluation Setup

### **Evaluation Pipeline**

Create a comprehensive evaluation pipeline:

```python
class ModelEvaluator:
    def __init__(self, model, test_dataset, metrics):
        self.model = model
        self.test_dataset = test_dataset
        self.metrics = metrics
        self.results = {}
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        self.model.eval()
        
        # Automatic metrics
        self.results['automatic'] = self._evaluate_automatic_metrics()
        
        # Task-specific metrics
        self.results['task_specific'] = self._evaluate_task_metrics()
        
        # Human evaluation
        self.results['human'] = self._evaluate_human_metrics()
        
        return self.results
    
    def _evaluate_automatic_metrics(self):
        """Evaluate automatic metrics"""
        results = {}
        
        # Perplexity
        results['perplexity'] = calculate_perplexity(self.model, self.test_dataset)
        
        # Generation quality
        references, hypotheses = self._generate_responses()
        results['bleu'] = calculate_bleu_score(references, hypotheses)
        results['rouge'] = calculate_rouge_scores(references, hypotheses)
        
        return results
    
    def _evaluate_task_metrics(self):
        """Evaluate task-specific metrics"""
        # Implement based on your specific task
        pass
    
    def _evaluate_human_metrics(self):
        """Evaluate human metrics"""
        # Implement human evaluation
        pass
    
    def _generate_responses(self):
        """Generate model responses for evaluation"""
        references = []
        hypotheses = []
        
        for batch in self.test_dataset:
            response = self.model.generate(batch["prompt"])
            references.append(batch["reference"])
            hypotheses.append(response)
        
        return references, hypotheses
```

### **Evaluation Configuration**

Create configuration files for evaluation:

```yaml
# evaluation_config.yaml
evaluation:
  metrics:
    - perplexity
    - bleu
    - rouge
    - task_specific
    - human_evaluation
  
  test_dataset:
    path: "path/to/test_data.json"
    max_samples: 1000
  
  generation:
    max_length: 100
    temperature: 0.7
    top_p: 0.9
  
  human_evaluation:
    num_samples: 100
    evaluators: 3
    criteria:
      - helpfulness
      - safety
      - alignment
```

## Step 4: Human Evaluation

### **Human Evaluation Setup**

Set up human evaluation for subjective assessment:

```python
def setup_human_evaluation(model, test_dataset, num_samples=100):
    """Setup human evaluation pipeline"""
    
    # Sample evaluation data
    evaluation_samples = sample_evaluation_data(test_dataset, num_samples)
    
    # Generate model responses
    model_responses = []
    for sample in evaluation_samples:
        response = model.generate(sample["prompt"])
        model_responses.append({
            "prompt": sample["prompt"],
            "response": response,
            "reference": sample.get("reference", "")
        })
    
    return model_responses

def conduct_human_evaluation(evaluation_samples, evaluators=3):
    """Conduct human evaluation"""
    
    evaluation_results = {
        'helpfulness': [],
        'safety': [],
        'alignment': [],
        'overall_quality': []
    }
    
    for sample in evaluation_samples:
        sample_evaluations = []
        
        for evaluator in range(evaluators):
            evaluation = evaluate_single_response(sample, evaluator)
            sample_evaluations.append(evaluation)
        
        # Aggregate evaluator scores
        for criterion in evaluation_results:
            scores = [eval[criterion] for eval in sample_evaluations]
            evaluation_results[criterion].append(sum(scores) / len(scores))
    
    return evaluation_results
```

### **Evaluation Criteria**

Define clear evaluation criteria:

```python
def evaluate_single_response(sample, evaluator_id):
    """Evaluate a single response"""
    
    evaluation = {
        'helpfulness': 0,  # 1-5 scale
        'safety': 0,       # 1-5 scale
        'alignment': 0,    # 1-5 scale
        'overall_quality': 0  # 1-5 scale
    }
    
    # Helpfulness: How helpful is the response?
    # Safety: Is the response safe and appropriate?
    # Alignment: Does the response align with human values?
    # Overall Quality: Overall assessment of response quality
    
    return evaluation
```

## Step 5: Benchmarking

### **Model Comparison**

Compare multiple models across different criteria:

```python
def benchmark_models(models, test_dataset, metrics):
    """Benchmark multiple models"""
    
    benchmark_results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        evaluator = ModelEvaluator(model, test_dataset, metrics)
        results = evaluator.evaluate()
        
        benchmark_results[model_name] = results
    
    return benchmark_results

def create_benchmark_report(benchmark_results):
    """Create comprehensive benchmark report"""
    
    report = {
        'summary': {},
        'detailed_results': benchmark_results,
        'recommendations': []
    }
    
    # Create summary statistics
    for metric in ['perplexity', 'bleu', 'helpfulness']:
        if metric in benchmark_results[list(benchmark_results.keys())[0]]['automatic']:
            values = [results['automatic'][metric] for results in benchmark_results.values()]
            report['summary'][metric] = {
                'best': max(values),
                'worst': min(values),
                'average': sum(values) / len(values)
            }
    
    return report
```

### **Benchmarking Script**

```python
# benchmark_script.py
from nemo_rl.models import load_models
from evaluation import ModelEvaluator, benchmark_models

def main():
    # Load models to benchmark
    models = {
        'sft_model': load_models('path/to/sft_model'),
        'dpo_model': load_models('path/to/dpo_model'),
        'grpo_model': load_models('path/to/grpo_model')
    }
    
    # Load test dataset
    test_dataset = load_test_dataset('path/to/test_data.json')
    
    # Define metrics
    metrics = ['perplexity', 'bleu', 'rouge', 'human_evaluation']
    
    # Run benchmark
    results = benchmark_models(models, test_dataset, metrics)
    
    # Generate report
    report = create_benchmark_report(results)
    
    # Save results
    save_benchmark_results(results, 'benchmark_results.json')
    save_benchmark_report(report, 'benchmark_report.md')

if __name__ == "__main__":
    main()
```

## Step 6: Best Practices

### **Evaluation Design**

1. **Comprehensive Metrics**: Use multiple metrics for different aspects
2. **Representative Data**: Ensure test data is representative of target domain
3. **Consistent Evaluation**: Use consistent evaluation procedures
4. **Statistical Significance**: Ensure sufficient sample sizes

### **Human Evaluation**

1. **Clear Criteria**: Define clear, objective evaluation criteria
2. **Multiple Evaluators**: Use multiple evaluators for reliability
3. **Training**: Train evaluators on evaluation criteria
4. **Quality Control**: Monitor evaluator consistency and quality

### **Automated Evaluation**

1. **Appropriate Metrics**: Choose metrics relevant to your task
2. **Validation**: Validate automated metrics against human judgment
3. **Interpretation**: Understand what metrics actually measure
4. **Limitations**: Be aware of metric limitations and biases

### **Benchmarking**

1. **Fair Comparison**: Ensure fair comparison conditions
2. **Multiple Criteria**: Compare across multiple criteria
3. **Statistical Testing**: Use statistical tests for significance
4. **Reproducibility**: Ensure results are reproducible

## Common Issues and Solutions

### **Metric Misinterpretation**

**Symptoms**: Metrics don't align with human judgment
**Solutions**:
- Validate metrics against human evaluation
- Use multiple complementary metrics
- Understand metric limitations
- Focus on task-specific metrics

### **Evaluation Bias**

**Symptoms**: Evaluation favors certain model types
**Solutions**:
- Use diverse evaluation datasets
- Include various evaluation criteria
- Blind evaluation procedures
- Multiple evaluator perspectives

### **Statistical Issues**

**Symptoms**: Unreliable or inconsistent results
**Solutions**:
- Increase sample sizes
- Use statistical significance tests
- Ensure proper randomization
- Control for confounding variables

## Next Steps

After completing this tutorial:

1. **Implement Evaluation**: Set up evaluation for your specific models
2. **Customize Metrics**: Adapt metrics for your domain
3. **Scale Evaluation**: Scale evaluation to larger datasets
4. **Production Monitoring**: Set up evaluation for production models
5. **Continuous Evaluation**: Implement ongoing evaluation processes

## Related Resources

- **[Evaluation Algorithm Guide](../../guides/training-algorithms/eval)**: Detailed technical reference
- **[SFT Tutorial](sft-tutorial)**: Learn supervised fine-tuning
- **[DPO Tutorial](dpo-tutorial)**: Learn preference optimization
- **[GRPO Tutorial](grpo-tutorial)**: Learn advanced RL

## Summary

In this tutorial, you learned:

- ✅ **Evaluation Fundamentals**: Understanding model evaluation concepts
- ✅ **Evaluation Metrics**: Choosing and implementing appropriate metrics
- ✅ **Evaluation Setup**: Setting up comprehensive evaluation pipelines
- ✅ **Human Evaluation**: Incorporating human feedback in evaluation
- ✅ **Benchmarking**: Comparing models across different criteria
- ✅ **Best Practices**: Proven techniques for reliable evaluation

You're now ready to implement comprehensive evaluation for your NeMo RL models and ensure they meet your quality requirements! 