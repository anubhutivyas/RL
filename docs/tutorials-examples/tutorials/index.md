---
description: "Step-by-step learning guides to master reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["tutorials", "sft", "dpo", "grpo", "evaluation", "reinforcement-learning", "training-execution"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# Tutorials

Welcome to the NeMo RL Tutorials! These step-by-step learning guides will help you master reinforcement learning with large language models.

## Tutorial Categories

Our tutorials are organized by difficulty level to help you progress from basic concepts to advanced techniques:

### **Beginner Tutorials**
Perfect for newcomers to NeMo RL. Start here to build your foundation with supervised fine-tuning and basic concepts.

### **Intermediate Tutorials**
For users familiar with the basics. Learn preference-based training, evaluation techniques, and custom environment development.

### **Advanced Tutorials**
For experienced practitioners. Master advanced reinforcement learning, distributed training, and performance optimization.

## Tutorials by Level

::::{grid} 1 1 1 1
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Supervised Fine-Tuning (SFT)
:link: sft-tutorial
:link-type: doc

Learn supervised fine-tuning fundamentals with step-by-step guidance.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` Direct Preference Optimization (DPO)
:link: dpo-tutorial
:link-type: doc

Master Direct Preference Optimization for preference learning and alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Model Evaluation and Benchmarking
:link: evaluation-tutorial
:link-type: doc

Learn model evaluation and benchmarking strategies for RL-trained models.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Tutorial
:link: grpo-tutorial
:link-type: doc

Advanced reinforcement learning with Group Relative Policy Optimization.

+++
{bdg-warning}`Advanced`
:::

::::

## Learning Progression

### **Beginner Path** (0-2 weeks)
1. **Supervised Fine-Tuning (SFT)** → Learn supervised fine-tuning fundamentals
2. **Basic Examples** → Apply SFT to real datasets
3. **Model Evaluation and Benchmarking** → Learn to evaluate your models

### **Intermediate Path** (2-4 weeks)
1. **Direct Preference Optimization (DPO)** → Master preference-based training
2. **Model Evaluation and Benchmarking** → Learn comprehensive evaluation
3. **Advanced Examples** → Work with complex datasets

### **Advanced Path** (4+ weeks)
1. **GRPO Tutorial** → Advanced reinforcement learning
2. **Advanced Performance** → [Performance and Optimization](../../advanced/performance/index)
3. **Distributed Training** → [Distributed Training Guide](../../advanced/performance/distributed-training)

## Tutorial Descriptions

### **Supervised Fine-Tuning (SFT)**
- **Beginner-Friendly**: Start with basic supervised fine-tuning concepts
- **Domain Adaptation**: Learn to adapt pre-trained models to specific domains
- **Best Practices**: Understand data preparation, training configuration, and evaluation
- **Real Examples**: Work with actual datasets and model configurations

### **Direct Preference Optimization (DPO)**
- **Preference Learning**: Master the fundamentals of preference-based training
- **Human Feedback**: Learn to incorporate human feedback into model training
- **Alignment Techniques**: Understand how to align models with human preferences
- **Advanced Configurations**: Explore sophisticated DPO setups and optimizations

### **GRPO (Group Relative Policy Optimization) Tutorial**
- **Advanced RL**: Dive deep into group-based reinforcement learning
- **Multi-Agent Scenarios**: Learn to handle complex multi-agent training
- **Performance Optimization**: Master techniques for large-scale GRPO training

For comprehensive performance optimization strategies, see the [Performance and Optimization Guide](../../advanced/performance/index).

### **Model Evaluation and Benchmarking**
- **Model Assessment**: Learn comprehensive evaluation strategies
- **Metrics and Benchmarks**: Understand key performance indicators
- **Human Evaluation**: Incorporate human feedback in evaluation
- **Production Monitoring**: Set up evaluation for production systems

## Prerequisites

### **For Beginner Tutorials**
- Basic Python knowledge
- Familiarity with PyTorch (helpful but not required)
- Understanding of machine learning concepts

### **For Intermediate Tutorials**
- Completion of beginner tutorials
- Understanding of supervised fine-tuning
- Basic knowledge of reinforcement learning concepts

### **For Advanced Tutorials**
- Completion of intermediate tutorials
- Experience with distributed computing (for distributed training)
- Understanding of performance optimization concepts

## Getting Started

1. **Choose Your Level**: Start with beginner tutorials if you're new to NeMo RL
2. **Follow the Progression**: Complete tutorials in order for best results
3. **Practice with Examples**: Apply what you learn to the example projects
4. **Explore Advanced Topics**: Move to advanced tutorials as you gain experience

## Next Steps

After completing the tutorials:

- **Try the Examples**: Apply your knowledge to real-world examples
- **Explore Use Cases**: Learn about specific applications and domains
- **Contribute**: Share your experiences and help improve the tutorials
- **Advanced Topics**: Dive into research and experimentation

For additional learning resources, visit the main [Tutorials and Examples](../index) page.

---

::::{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 2
sft-tutorial
dpo-tutorial
grpo-tutorial
evaluation-tutorial
:::: 