---
description: "Step-by-step guide to Direct Preference Optimization for training language models using preference learning techniques with NeMo RL"
categories: ["training-algorithms"]
tags: ["dpo", "preference-learning", "model-alignment", "data-preparation", "training-execution", "evaluation"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "universal"
---

# DPO Tutorial: Direct Preference Optimization

Welcome to the Direct Preference Optimization (DPO) tutorial! This step-by-step guide will teach you how to train language models using preference learning techniques with NeMo RL.

## What You'll Learn

In this tutorial, you'll learn:

- **DPO Fundamentals**: Understanding preference-based learning concepts
- **Preference Data**: Preparing and formatting preference data
- **Model Configuration**: Setting up models for DPO training
- **Training Process**: Running and monitoring DPO training
- **Evaluation**: Assessing model alignment and preferences
- **Best Practices**: Proven techniques for successful DPO

## Prerequisites

Before starting this tutorial, ensure you have:

- **NeMo RL**: Installed and set up (see [Installation Guide](../../get-started/installation))
- **Python Knowledge**: Basic understanding of Python programming
- **Machine Learning**: Familiarity with ML concepts (helpful but not required)
- **Hardware**: GPU with sufficient memory for your chosen model size

## Tutorial Overview

### **Step 1: Understanding DPO**
Learn the fundamentals of Direct Preference Optimization and when to use it.

### **Step 2: Preference Data Preparation**
Prepare and format your preference data for DPO training.

### **Step 3: Model Setup**
Configure your model architecture and training parameters.

### **Step 4: Training Execution**
Run the training process and monitor progress.

### **Step 5: Evaluation**
Assess your model's alignment and preference learning.

### **Step 6: Best Practices**
Learn proven techniques for successful DPO.

## Step 1: Understanding DPO

### **What is Direct Preference Optimization?**

Direct Preference Optimization (DPO) is a technique for training language models to align with human preferences. Unlike supervised fine-tuning, DPO uses preference data where humans rank different model responses.

### **When to Use DPO**

DPO is ideal for:
- **Model Alignment**: Aligning models with human values and preferences
- **Quality Improvement**: Improving response quality and helpfulness
- **Safety Enhancement**: Making models safer and more reliable
- **Preference Learning**: Learning from human feedback and rankings

### **DPO vs. Other Techniques**

| Technique | Use Case | Data Requirements | Training Approach |
|-----------|----------|-------------------|-------------------|
| **SFT** | Domain adaptation, task-specific training | Labeled examples | Supervised learning |
| **DPO** | Preference learning, alignment | Human preferences | Preference optimization |
| **GRPO** | Advanced RL, multi-agent scenarios | Reward signals | Reinforcement learning |

### **How DPO Works**

DPO uses a preference loss function that encourages the model to:
1. **Increase probability** of preferred responses
2. **Decrease probability** of non-preferred responses
3. **Maintain balance** to prevent overfitting

## Step 2: Preference Data Preparation

### **Data Format**

DPO requires preference data in a specific format. Each training example should include:

```python
{
    "prompt": "Your instruction or prompt here",
    "chosen": "Preferred model response",
    "rejected": "Non-preferred model response"
}
```

### **Data Sources**

Common data sources for DPO include:
- **Human Feedback**: Manual rankings of model responses
- **Synthetic Preferences**: Generated preference pairs
- **Quality Rankings**: Responses ranked by quality metrics
- **Safety Preferences**: Safe vs. unsafe response pairs

### **Data Quality Guidelines**

1. **Clear Preferences**: Obvious differences between chosen and rejected responses
2. **Diverse Prompts**: Varied prompts to improve generalization
3. **Balanced Preferences**: Representative distribution of preference types
4. **Quality Control**: High-quality, accurate preference judgments

### **Data Preprocessing**

```python
# Example data preprocessing
def preprocess_dpo_data(raw_data):
    processed_data = []
    for item in raw_data:
        processed_item = {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }
        processed_data.append(processed_item)
    return processed_data
```

## Step 3: Model Setup

### **Model Selection**

Choose a pre-trained or SFT model as your starting point:

- **Pre-trained Models**: Start from scratch with preference learning
- **SFT Models**: Build on supervised fine-tuned models
- **Model Size**: Balance performance and computational requirements

### **Configuration Setup**

Create a configuration file for your DPO training:

```yaml
# dpo_config.yaml
model:
  name: "path/to/sft_model"
  max_length: 512

training:
  batch_size: 2
  learning_rate: 1e-5
  num_epochs: 2
  warmup_steps: 100
  beta: 0.1  # DPO temperature parameter

data:
  train_file: "path/to/train.json"
  validation_file: "path/to/val.json"
  max_seq_length: 512
```

### **Model Loading**

```python
from nemo_rl.models.huggingface import HuggingFaceModel

# Load pre-trained or SFT model
model = HuggingFaceModel.from_pretrained("path/to/sft_model")
```

## Step 4: Training Execution

### **Training Script**

Create a training script for DPO:

```python
import torch
from nemo_rl.algorithms import DPOTrainer
from nemo_rl.data import DPODataset

# Load data
train_dataset = DPODataset("path/to/train.json")
val_dataset = DPODataset("path/to/val.json")

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)

# Start training
trainer.train()
```

### **Training Monitoring**

Monitor key metrics during training:

- **Preference Loss**: DPO-specific loss function
- **Chosen/Rejected Logits**: Model confidence in preferences
- **Learning Rate**: Current learning rate schedule
- **Memory Usage**: GPU memory consumption

### **DPO-Specific Parameters**

Key parameters for DPO training:

```python
# DPO configuration
dpo_config = {
    "beta": 0.1,  # Temperature parameter (higher = stronger preferences)
    "loss_type": "sigmoid",  # Loss function type
    "reference_free": False,  # Whether to use reference model
    "gradient_checkpointing": True,  # Memory optimization
}
```

## Step 5: Evaluation

### **Evaluation Metrics**

Assess your model using multiple metrics:

- **Preference Accuracy**: How well the model learns preferences
- **Response Quality**: Human evaluation of response quality
- **Alignment**: Degree of alignment with human values
- **Safety**: Assessment of safety and harmlessness

### **Evaluation Script**

```python
def evaluate_dpo_model(model, test_dataset):
    model.eval()
    preference_accuracy = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            # Get model predictions for chosen and rejected
            chosen_logits = model(batch["chosen_inputs"])
            rejected_logits = model(batch["rejected_inputs"])
            
            # Calculate preference accuracy
            correct = (chosen_logits > rejected_logits).float().mean()
            preference_accuracy += correct.item()
            total_examples += 1
    
    return preference_accuracy / total_examples
```

### **Human Evaluation**

Conduct human evaluation of model outputs:

```python
def human_evaluation(model, test_prompts):
    model.eval()
    evaluations = []
    
    for prompt in test_prompts:
        response = model.generate(prompt, max_length=100)
        
        # Human evaluation criteria
        evaluation = {
            "prompt": prompt,
            "response": response,
            "helpfulness": 0,  # 1-5 scale
            "safety": 0,       # 1-5 scale
            "alignment": 0     # 1-5 scale
        }
        evaluations.append(evaluation)
    
    return evaluations
```

## Step 6: Best Practices

### **Data Quality**

1. **High-Quality Preferences**: Ensure clear, consistent preference judgments
2. **Diverse Prompts**: Cover a wide range of topics and scenarios
3. **Balanced Preferences**: Include various types of preferences (quality, safety, etc.)
4. **Validation Set**: Reserve data for validation and testing

### **Training Configuration**

1. **Beta Parameter**: Start with beta=0.1, adjust based on preference strength
2. **Learning Rate**: Use smaller learning rates (1e-5 to 5e-5)
3. **Batch Size**: Balance memory usage and training stability
4. **Gradient Accumulation**: Use for effective larger batch sizes

### **Model Selection**

1. **Start with SFT**: Begin with a supervised fine-tuned model
2. **Quality Foundation**: Ensure the base model is of good quality
3. **Appropriate Size**: Choose model size based on your requirements
4. **Domain Match**: Use models appropriate for your domain

### **Monitoring and Debugging**

1. **Preference Loss**: Monitor DPO-specific loss function
2. **Logit Differences**: Check chosen vs. rejected logit differences
3. **Sample Outputs**: Regularly review generated outputs
4. **Human Feedback**: Incorporate human evaluation during training

## Common Issues and Solutions

### **Preference Collapse**

**Symptoms**: Model always prefers one type of response
**Solutions**:
- Increase data diversity
- Adjust beta parameter
- Add regularization
- Balance preference types

### **Overfitting to Preferences**

**Symptoms**: Model loses general capabilities
**Solutions**:
- Reduce training epochs
- Increase regularization
- Use larger datasets
- Monitor general performance

### **Memory Issues**

**Symptoms**: Out of memory errors
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use model parallelism

## Next Steps

After completing this tutorial:

1. **Experiment with Beta**: Try different beta values for preference strength
2. **Explore Domains**: Apply DPO to your specific domain
3. **Scale Up**: Move to larger models and datasets
4. **Advanced Techniques**: Learn about GRPO for advanced RL
5. **Production Deployment**: Deploy your preference-aligned models

## Related Resources

- **[DPO Algorithm Guide](../../guides/training-algorithms/dpo)**: Detailed technical reference
- **[SFT Tutorial](sft-tutorial)**: Prerequisite supervised fine-tuning tutorial
- **[GRPO Tutorial](grpo-tutorial)**: Advanced reinforcement learning
- **[Evaluation Tutorial](evaluation-tutorial)**: Comprehensive evaluation techniques

## Summary

In this tutorial, you learned:

- ✅ **DPO Fundamentals**: Understanding preference-based learning concepts
- ✅ **Preference Data**: Preparing and formatting preference data
- ✅ **Model Configuration**: Setting up models for DPO training
- ✅ **Training Process**: Running and monitoring DPO training
- ✅ **Evaluation**: Assessing model alignment and preferences
- ✅ **Best Practices**: Proven techniques for successful DPO

You're now ready to apply DPO to align models with human preferences and explore more advanced RL techniques like GRPO! 