# GRPO Tutorial: Group Relative Policy Optimization

Welcome to the Group Relative Policy Optimization (GRPO) tutorial! This advanced guide will teach you how to train language models using group-based reinforcement learning techniques with NeMo RL.

## What You'll Learn

In this tutorial, you'll learn:

- **GRPO Fundamentals**: Understanding group-based reinforcement learning concepts
- **Group Formation**: Creating and managing training groups
- **Model Configuration**: Setting up models for GRPO training
- **Training Process**: Running and monitoring GRPO training
- **Evaluation**: Assessing model performance in group scenarios
- **Best Practices**: Proven techniques for successful GRPO

## Prerequisites

Before starting this tutorial, ensure you have:

- **NeMo RL**: Installed and set up (see [Installation Guide](../../get-started/installation))
- **Python Knowledge**: Basic understanding of Python programming
- **Machine Learning**: Familiarity with ML concepts (helpful but not required)
- **Hardware**: GPU with sufficient memory for your chosen model size

## Tutorial Overview

### **Step 1: Understanding GRPO**
Learn the fundamentals of Group Relative Policy Optimization and when to use it.

### **Step 2: Group Formation**
Create and manage training groups for GRPO.

### **Step 3: Model Setup**
Configure your model architecture and training parameters.

### **Step 4: Training Execution**
Run the training process and monitor progress.

### **Step 5: Evaluation**
Assess your model's performance in group scenarios.

### **Step 6: Best Practices**
Learn proven techniques for successful GRPO.

## Step 1: Understanding GRPO

### **What is Group Relative Policy Optimization?**

Group Relative Policy Optimization (GRPO) is an advanced reinforcement learning technique that trains models using group-based comparisons. Unlike DPO which uses pairwise preferences, GRPO operates on groups of responses and learns relative preferences within groups.

### **When to Use GRPO**

GRPO is ideal for:
- **Complex Preferences**: Multi-dimensional preference learning
- **Group Comparisons**: Scenarios with multiple response options
- **Advanced RL**: Sophisticated reinforcement learning applications
- **Production Systems**: Large-scale training with complex reward structures

### **GRPO vs. Other Techniques**

| Technique | Use Case | Data Requirements | Training Approach |
|-----------|----------|-------------------|-------------------|
| **SFT** | Domain adaptation, task-specific training | Labeled examples | Supervised learning |
| **DPO** | Preference learning, alignment | Human preferences | Preference optimization |
| **GRPO** | Advanced RL, group scenarios | Group preferences | Group-based RL |

### **How GRPO Works**

GRPO uses group-based loss functions that:
1. **Compare responses** within training groups
2. **Learn relative preferences** between group members
3. **Optimize policy** based on group rankings
4. **Maintain balance** across different preference dimensions

## Step 2: Group Formation

### **Group Structure**

GRPO requires data organized in groups. Each training example should include:

```python
{
    "prompt": "Your instruction or prompt here",
    "group": [
        {"response": "Response option 1", "rank": 1},
        {"response": "Response option 2", "rank": 2},
        {"response": "Response option 3", "rank": 3}
    ]
}
```

### **Group Creation Strategies**

1. **Quality-Based Groups**: Group responses by quality levels
2. **Diversity-Based Groups**: Ensure diverse response types
3. **Domain-Specific Groups**: Group by domain or topic
4. **Multi-Objective Groups**: Balance multiple preference dimensions

### **Data Sources**

Common data sources for GRPO include:
- **Human Rankings**: Manual group rankings by humans
- **Synthetic Groups**: Generated group comparisons
- **Quality Metrics**: Responses grouped by quality scores
- **Multi-Criteria**: Groups based on multiple evaluation criteria

### **Data Preprocessing**

```python
# Example data preprocessing for GRPO
def preprocess_grpo_data(raw_data):
    processed_data = []
    for item in raw_data:
        processed_item = {
            "prompt": item["prompt"],
            "group": sorted(item["group"], key=lambda x: x["rank"])
        }
        processed_data.append(processed_item)
    return processed_data
```

## Step 3: Model Setup

### **Model Selection**

Choose a pre-trained, SFT, or DPO model as your starting point:

- **Pre-trained Models**: Start from scratch with group learning
- **SFT Models**: Build on supervised fine-tuned models
- **DPO Models**: Build on preference-aligned models
- **Model Size**: Larger models generally perform better with GRPO

### **Configuration Setup**

Create a configuration file for your GRPO training:

```yaml
# grpo_config.yaml
model:
  name: "path/to/dpo_model"
  max_length: 512

training:
  batch_size: 1  # Smaller batches due to group processing
  learning_rate: 5e-6
  num_epochs: 3
  warmup_steps: 100
  group_size: 3  # Number of responses per group

data:
  train_file: "path/to/train.json"
  validation_file: "path/to/val.json"
  max_seq_length: 512
```

### **Model Loading**

```python
from nemo_rl.models.huggingface import HuggingFaceModel

# Load pre-trained, SFT, or DPO model
model = HuggingFaceModel.from_pretrained("path/to/dpo_model")
```

## Step 4: Training Execution

### **Training Script**

Create a training script for GRPO:

```python
import torch
from nemo_rl.algorithms import GRPOTrainer
from nemo_rl.data import GRPODataset

# Load data
train_dataset = GRPODataset("path/to/train.json")
val_dataset = GRPODataset("path/to/val.json")

# Initialize trainer
trainer = GRPOTrainer(
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

- **Group Loss**: GRPO-specific loss function
- **Group Rankings**: Model's ability to predict correct rankings
- **Learning Rate**: Current learning rate schedule
- **Memory Usage**: GPU memory consumption (higher for groups)

### **GRPO-Specific Parameters**

Key parameters for GRPO training:

```python
# GRPO configuration
grpo_config = {
    "group_size": 3,  # Number of responses per group
    "temperature": 0.1,  # Temperature for group comparisons
    "loss_type": "group_relative",  # Loss function type
    "gradient_checkpointing": True,  # Memory optimization
}
```

## Step 5: Evaluation

### **Evaluation Metrics**

Assess your model using multiple metrics:

- **Group Ranking Accuracy**: How well the model predicts correct rankings
- **Response Quality**: Human evaluation of individual responses
- **Group Consistency**: Consistency across different group types
- **Preference Alignment**: Alignment with human preferences

### **Evaluation Script**

```python
def evaluate_grpo_model(model, test_dataset):
    model.eval()
    ranking_accuracy = 0
    total_groups = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            # Get model predictions for group members
            group_logits = model(batch["group_inputs"])
            
            # Calculate ranking accuracy
            predicted_ranks = torch.argsort(group_logits, descending=True)
            correct_ranks = batch["true_ranks"]
            
            accuracy = (predicted_ranks == correct_ranks).float().mean()
            ranking_accuracy += accuracy.item()
            total_groups += 1
    
    return ranking_accuracy / total_groups
```

### **Group Analysis**

Analyze model performance across different group types:

```python
def analyze_group_performance(model, test_dataset):
    model.eval()
    group_analysis = {}
    
    for batch in test_dataset:
        group_type = batch["group_type"]
        if group_type not in group_analysis:
            group_analysis[group_type] = []
        
        # Calculate group-specific metrics
        accuracy = calculate_group_accuracy(model, batch)
        group_analysis[group_type].append(accuracy)
    
    return group_analysis
```

## Step 6: Best Practices

### **Data Quality**

1. **High-Quality Groups**: Ensure clear, consistent group rankings
2. **Diverse Groups**: Include various types of group comparisons
3. **Balanced Groups**: Representative distribution of group types
4. **Validation Groups**: Reserve data for validation and testing

### **Training Configuration**

1. **Group Size**: Start with small groups (3-5 responses), increase gradually
2. **Learning Rate**: Use smaller learning rates (5e-6 to 1e-5)
3. **Batch Size**: Smaller batches due to group processing overhead
4. **Gradient Accumulation**: Use for effective larger batch sizes

### **Model Selection**

1. **Start with DPO**: Begin with a preference-aligned model
2. **Quality Foundation**: Ensure the base model is of good quality
3. **Appropriate Size**: Choose model size based on your requirements
4. **Domain Match**: Use models appropriate for your domain

### **Monitoring and Debugging**

1. **Group Loss**: Monitor GRPO-specific loss function
2. **Ranking Accuracy**: Track group ranking performance
3. **Sample Groups**: Regularly review group predictions
4. **Memory Usage**: Monitor GPU memory consumption

## Common Issues and Solutions

### **Group Collapse**

**Symptoms**: Model always ranks responses in the same order
**Solutions**:
- Increase group diversity
- Adjust temperature parameter
- Add regularization
- Balance group types

### **Memory Issues**

**Symptoms**: Out of memory errors due to group processing
**Solutions**:
- Reduce group size
- Use gradient accumulation
- Enable mixed precision training
- Use model parallelism

### **Training Instability**

**Symptoms**: Unstable loss or poor convergence
**Solutions**:
- Reduce learning rate
- Increase warmup steps
- Add gradient clipping
- Use smaller groups initially

## Next Steps

After completing this tutorial:

1. **Experiment with Groups**: Try different group sizes and types
2. **Explore Domains**: Apply GRPO to your specific domain
3. **Scale Up**: Move to larger models and datasets
4. **Production Deployment**: Deploy your group-optimized models
5. **Advanced Techniques**: Explore other RL algorithms

## Related Resources

- **[GRPO Algorithm Guide](../../guides/training-algorithms/grpo)**: Detailed technical reference
- **[SFT Tutorial](sft-tutorial)**: Prerequisite supervised fine-tuning tutorial
- **[DPO Tutorial](dpo-tutorial)**: Prerequisite preference learning tutorial
- **[GRPO Example](../examples/grpo-deepscaler)**: Complete working example

## Summary

In this tutorial, you learned:

- ✅ **GRPO Fundamentals**: Understanding group-based RL concepts
- ✅ **Group Formation**: Creating and managing training groups
- ✅ **Model Configuration**: Setting up models for GRPO training
- ✅ **Training Process**: Running and monitoring GRPO training
- ✅ **Evaluation**: Assessing model performance in group scenarios
- ✅ **Best Practices**: Proven techniques for successful GRPO

You're now ready to apply GRPO to advanced reinforcement learning scenarios and deploy group-optimized models in production! 