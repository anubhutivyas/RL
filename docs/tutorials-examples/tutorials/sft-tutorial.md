---
description: "Step-by-step guide to supervised fine-tuning of language models using NeMo RL with data preparation, model setup, and training execution"
categories: ["training-algorithms"]
tags: ["sft", "supervised-fine-tuning", "data-preparation", "model-setup", "training-execution", "evaluation"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

# SFT Tutorial: Supervised Fine-Tuning

Welcome to the Supervised Fine-Tuning (SFT) tutorial! This step-by-step guide will teach you how to fine-tune language models using supervised learning techniques with NeMo RL.

## What You'll Learn

In this tutorial, you'll learn:

- **SFT Fundamentals**: Understanding supervised fine-tuning concepts
- **Data Preparation**: Preparing and formatting training data
- **Model Configuration**: Setting up models for fine-tuning
- **Training Process**: Running and monitoring SFT training
- **Evaluation**: Assessing model performance and quality
- **Best Practices**: Proven techniques for successful SFT

## Prerequisites

Before starting this tutorial, ensure you have:

- **NeMo RL**: Installed and set up (see [Installation Guide](../../get-started/installation))
- **Python Knowledge**: Basic understanding of Python programming
- **Machine Learning**: Familiarity with ML concepts (helpful but not required)
- **Hardware**: GPU with sufficient memory for your chosen model size

## Tutorial Overview

### **Step 1: Understanding SFT**
Learn the fundamentals of supervised fine-tuning and when to use it.

### **Step 2: Data Preparation**
Prepare and format your training data for SFT.

### **Step 3: Model Setup**
Configure your model architecture and training parameters.

### **Step 4: Training Execution**
Run the training process and monitor progress.

### **Step 5: Evaluation**
Assess your model's performance and quality.

### **Step 6: Best Practices**
Learn proven techniques for successful SFT.

## Step 1: Understanding SFT

### **What is Supervised Fine-Tuning?**

Supervised Fine-Tuning (SFT) is a technique for adapting pre-trained language models to specific tasks or domains. Unlike reinforcement learning, SFT uses labeled training data where each example has a correct answer or desired output.

### **When to Use SFT**

SFT is ideal for:
- **Domain Adaptation**: Adapting models to specific domains (medical, legal, technical)
- **Task-Specific Training**: Teaching models to perform specific tasks
- **Instruction Following**: Training models to follow instructions and prompts
- **Foundation for RL**: SFT is often the first step before applying RL techniques

### **SFT vs. Other Techniques**

| Technique | Use Case | Data Requirements | Training Approach |
|-----------|----------|-------------------|-------------------|
| **SFT** | Domain adaptation, task-specific training | Labeled examples | Supervised learning |
| **DPO** | Preference learning, alignment | Human preferences | Preference optimization |
| **GRPO** | Advanced RL, multi-agent scenarios | Reward signals | Reinforcement learning |

## Step 2: Data Preparation

### **Data Format**

SFT requires data in a specific format. Each training example should include:

```python
{
    "instruction": "Your instruction or prompt here",
    "input": "Optional input context",
    "output": "Expected model response"
}
```

### **Data Sources**

Common data sources for SFT include:
- **Instruction Datasets**: Alpaca, Dolly, OpenMathInstruct
- **Domain-Specific Data**: Medical, legal, technical documents
- **Custom Datasets**: Your own task-specific data

### **Data Quality Guidelines**

1. **Consistency**: Ensure consistent formatting and style
2. **Quality**: High-quality, accurate responses
3. **Diversity**: Varied examples to improve generalization
4. **Balance**: Representative distribution of topics and difficulty

### **Data Preprocessing**

```python
# Example data preprocessing
def preprocess_sft_data(raw_data):
    processed_data = []
    for item in raw_data:
        processed_item = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }
        processed_data.append(processed_item)
    return processed_data
```

## Step 3: Model Setup

### **Model Selection**

Choose a pre-trained model appropriate for your task:

- **Small Models (1B-7B)**: Good for experimentation and limited resources
- **Medium Models (7B-13B)**: Balance of performance and efficiency
- **Large Models (13B+)**: Best performance but requires more resources

### **Configuration Setup**

Create a configuration file for your SFT training:

```yaml
# sft_config.yaml
model:
  name: "microsoft/DialoGPT-medium"
  max_length: 512

training:
  batch_size: 4
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 100

data:
  train_file: "path/to/train.json"
  validation_file: "path/to/val.json"
  max_seq_length: 512
```

### **Model Loading**

```python
from nemo_rl.models.huggingface import HuggingFaceModel

# Load pre-trained model
model = HuggingFaceModel.from_pretrained("microsoft/DialoGPT-medium")
```

## Step 4: Training Execution

### **Training Script**

Create a training script for SFT:

```python
import torch
from nemo_rl.algorithms import SFTTrainer
from nemo_rl.data import SFTDataset

# Load data
train_dataset = SFTDataset("path/to/train.json")
val_dataset = SFTDataset("path/to/val.json")

# Initialize trainer
trainer = SFTTrainer(
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

- **Loss**: Training and validation loss
- **Learning Rate**: Current learning rate schedule
- **Memory Usage**: GPU memory consumption
- **Throughput**: Training speed (samples/second)

### **Checkpointing**

Save model checkpoints regularly:

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/sft_model_epoch_1.pt")

# Load checkpoint
trainer.load_checkpoint("checkpoints/sft_model_epoch_1.pt")
```

## Step 5: Evaluation

### **Evaluation Metrics**

Assess your model using multiple metrics:

- **Perplexity**: Measure of model uncertainty
- **BLEU Score**: Text generation quality
- **Human Evaluation**: Manual assessment of outputs
- **Task-Specific Metrics**: Domain-relevant evaluation

### **Evaluation Script**

```python
def evaluate_model(model, test_dataset):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            outputs = model(batch)
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(test_dataset)
```

### **Qualitative Assessment**

Manually review model outputs:

```python
def generate_sample_outputs(model, test_prompts):
    model.eval()
    outputs = []
    
    for prompt in test_prompts:
        response = model.generate(prompt, max_length=100)
        outputs.append({
            "prompt": prompt,
            "response": response
        })
    
    return outputs
```

## Step 6: Best Practices

### **Data Quality**

1. **Clean Data**: Remove duplicates, errors, and low-quality examples
2. **Balanced Distribution**: Ensure representative coverage of topics
3. **Consistent Formatting**: Maintain consistent style and structure
4. **Validation Split**: Reserve data for validation and testing

### **Training Configuration**

1. **Learning Rate**: Start with small learning rates (1e-5 to 5e-5)
2. **Batch Size**: Balance memory usage and training stability
3. **Gradient Accumulation**: Use for effective larger batch sizes
4. **Mixed Precision**: Enable for faster training with minimal accuracy loss

### **Model Selection**

1. **Start Small**: Begin with smaller models for experimentation
2. **Domain Match**: Choose models pre-trained on similar domains
3. **Resource Constraints**: Consider available GPU memory and compute
4. **Performance Requirements**: Balance speed vs. quality needs

### **Monitoring and Debugging**

1. **Loss Tracking**: Monitor training and validation loss
2. **Gradient Norms**: Check for gradient explosion or vanishing
3. **Memory Usage**: Monitor GPU memory consumption
4. **Sample Outputs**: Regularly review generated outputs

## Common Issues and Solutions

### **Overfitting**

**Symptoms**: Training loss decreases but validation loss increases
**Solutions**:
- Reduce model capacity
- Increase regularization
- Add more training data
- Early stopping

### **Underfitting**

**Symptoms**: Both training and validation loss are high
**Solutions**:
- Increase model capacity
- Train for more epochs
- Reduce learning rate
- Improve data quality

### **Memory Issues**

**Symptoms**: Out of memory errors
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use model parallelism

## Next Steps

After completing this tutorial:

1. **Try Different Models**: Experiment with different pre-trained models
2. **Explore Domains**: Apply SFT to your specific domain
3. **Scale Up**: Move to larger models and datasets
4. **Advanced Techniques**: Learn about DPO and GRPO for preference learning
5. **Production Deployment**: Deploy your fine-tuned models

## Related Resources

- **[SFT Algorithm Guide](../../guides/training-algorithms/sft)**: Detailed technical reference
- **[SFT Example](../examples/sft-openmathinstruct2)**: Complete working example
- **[DPO Tutorial](dpo-tutorial)**: Next step in preference learning
- **[Evaluation Tutorial](evaluation-tutorial)**: Learn comprehensive evaluation techniques

## Summary

In this tutorial, you learned:

- ✅ **SFT Fundamentals**: Understanding supervised fine-tuning concepts
- ✅ **Data Preparation**: Preparing and formatting training data
- ✅ **Model Configuration**: Setting up models for fine-tuning
- ✅ **Training Process**: Running and monitoring SFT training
- ✅ **Evaluation**: Assessing model performance and quality
- ✅ **Best Practices**: Proven techniques for successful SFT

You're now ready to apply SFT to your own projects and explore more advanced techniques like DPO and GRPO! 