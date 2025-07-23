---
description: "Comprehensive collection of tutorials and examples for learning and mastering reinforcement learning with large language models using NeMo RL"
categories: ["training-algorithms"]
tags: ["tutorials", "examples", "sft", "dpo", "grpo", "reinforcement-learning", "training-execution"]
personas: ["mle-focused", "researcher-focused", "admin-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "universal"
---

# Tutorials and Examples

Welcome to the NeMo RL Tutorials and Examples! This comprehensive collection provides everything you need to learn and master reinforcement learning with large language models using NeMo RL.

## What You'll Find Here

Our tutorials and examples are organized into three main categories to help you progress from basic concepts to real-world applications:

### **Tutorials** 
Step-by-step learning guides that teach you NeMo RL concepts and techniques. Start with beginner tutorials to build your foundation, then progress to advanced topics like distributed training and performance optimization.

### **Examples**
Complete, working examples that demonstrate real-world applications. These examples walk you through entire training workflows using specific datasets and configurations, providing hands-on experience with NeMo RL.

### **Use Cases**
Real-world applications and production patterns for NeMo RL. Learn how to apply reinforcement learning to solve practical problems across different domains including code generation, mathematical reasoning, and conversational AI.

## Learning Path

Follow this structured learning path to master NeMo RL:

### **Beginner Path** (0-2 weeks)
1. **Installation and Setup** → [Installation Guide](../get-started/installation)
2. **First Training Run** → [Quickstart Guide](../get-started/quickstart)
3. **SFT Tutorial** → [Supervised Fine-Tuning Tutorial](tutorials/sft-tutorial)
4. **Basic Examples** → [SFT on OpenMathInstruct-2](examples/sft-openmathinstruct2)

### **Intermediate Path** (2-4 weeks)
1. **DPO Tutorial** → [Direct Preference Optimization Tutorial](tutorials/dpo-tutorial)
2. **Evaluation Tutorial** → [Model Evaluation Tutorial](tutorials/evaluation-tutorial)
3. **Advanced Examples** → [GRPO on DeepScaleR](examples/grpo-deepscaler)
4. **Use Cases** → [Code Generation](use-cases/code-generation) and [Mathematical Reasoning](use-cases/mathematical-reasoning)

### **Advanced Path** (4+ weeks)
1. **GRPO Tutorial** → [Group Relative Policy Optimization Tutorial](tutorials/grpo-tutorial)
2. **Advanced Performance** → [Performance and Optimization](../advanced/performance/index)
3. **Distributed Training** → [Distributed Training Guide](../advanced/performance/distributed-training)
4. **Production Deployment** → [Production and Support](../guides/production-support/index)

## Tutorials

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` SFT Tutorial
:link: tutorials/sft-tutorial
:link-type: doc

Learn supervised fine-tuning fundamentals with step-by-step guidance.

+++
{bdg-primary}`Beginner`
:::

:::{grid-item-card} {octicon}`heart;1.5em;sd-mr-1` DPO Tutorial
:link: tutorials/dpo-tutorial
:link-type: doc

Master Direct Preference Optimization for preference learning and alignment.

+++
{bdg-info}`Intermediate`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` GRPO Tutorial
:link: tutorials/grpo-tutorial
:link-type: doc

Advanced reinforcement learning with Group Relative Policy Optimization.

+++
{bdg-warning}`Advanced`
:::

::::

## Examples

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` SFT on OpenMathInstruct-2
:link: examples/sft-openmathinstruct2
:link-type: doc

Complete example of supervised fine-tuning on math instruction dataset.

+++
{bdg-primary}`Example`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` GRPO on DeepScaleR
:link: examples/grpo-deepscaler
:link-type: doc

Large-scale distributed training example with DeepScaleR integration.

+++
{bdg-secondary}`Cloud`
:::

::::

## Use Cases

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: use-cases/code-generation
:link-type: doc

Train models to generate, debug, and optimize code across multiple programming languages.

+++
{bdg-primary}`Development`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Mathematical Reasoning
:link: use-cases/mathematical-reasoning
:link-type: doc

Build models that can solve complex mathematical problems with step-by-step reasoning.

+++
{bdg-warning}`Advanced`
:::

::::

## Tutorial Descriptions

### **SFT (Supervised Fine-Tuning) Tutorial**
- **Beginner-Friendly**: Start with basic supervised fine-tuning concepts
- **Domain Adaptation**: Learn to adapt pre-trained models to specific domains
- **Best Practices**: Understand data preparation, training configuration, and evaluation
- **Real Examples**: Work with actual datasets and model configurations

### **DPO (Direct Preference Optimization) Tutorial**
- **Preference Learning**: Master the fundamentals of preference-based training
- **Human Feedback**: Learn to incorporate human feedback into model training
- **Alignment Techniques**: Understand how to align models with human preferences
- **Advanced Configurations**: Explore sophisticated DPO setups and optimizations

### **GRPO (Group Relative Policy Optimization) Tutorial**
- **Advanced RL**: Dive deep into group-based reinforcement learning
- **Multi-Agent Scenarios**: Learn to handle complex multi-agent training
- **Performance Optimization**: Master techniques for large-scale GRPO training
- **Production Deployment**: Understand how to deploy GRPO models in production

## Advanced Tutorials

Take your NeMo RL skills to the next level with these advanced tutorials designed for experienced practitioners.

### **Advanced Reinforcement Learning**

#### **Custom Environment Development**
Learn to create sophisticated custom environments for specialized RL tasks:

- **Environment Design**: Design environments that accurately represent your problem domain
- **Reward Engineering**: Craft effective reward functions for complex scenarios
- **Multi-Modal Environments**: Handle environments with multiple input/output modalities
- **Real-Time Environments**: Build environments that interact with real-world systems

#### **Model Architecture Customization**
Master the art of customizing model architectures for specific use cases:

- **Custom Model Backends**: Integrate custom model architectures with NeMo RL
- **Attention Mechanisms**: Implement and experiment with different attention patterns
- **Multi-Head Architectures**: Design models with multiple specialized heads
- **Efficient Architectures**: Optimize models for specific hardware constraints

#### **Advanced Training Techniques**
Explore cutting-edge training techniques and optimizations:

- **Curriculum Learning**: Implement progressive difficulty training schedules
- **Meta-Learning**: Build models that can quickly adapt to new tasks
- **Multi-Task Learning**: Train models that excel at multiple related tasks
- **Continual Learning**: Develop models that learn continuously without forgetting

### **Production and Deployment**

#### **High-Performance Training**
Optimize your training pipeline for maximum efficiency:

- **Memory Optimization**: Techniques for training large models with limited memory
- **Throughput Optimization**: Maximize training speed and GPU utilization
- **Mixed Precision Training**: Use FP16/BF16 for faster training with minimal accuracy loss
- **Gradient Accumulation**: Handle large batch sizes with limited memory

#### **Model Serving and Inference**
Deploy your trained models in production environments:

- **Model Export**: Export models for various deployment targets
- **Inference Optimization**: Optimize models for fast inference
- **Serving Infrastructure**: Set up scalable model serving systems
- **Monitoring and Observability**: Implement comprehensive monitoring for deployed models

#### **Advanced Debugging and Profiling**
Master the tools and techniques for debugging complex training issues:

- **Performance Profiling**: Use NSYS and other tools to identify bottlenecks
- **Memory Profiling**: Debug memory issues and optimize memory usage
- **Distributed Debugging**: Troubleshoot issues in distributed training setups
- **Reproducibility**: Ensure reproducible results across different environments

### **Research and Experimentation**

#### **Novel Algorithm Implementation**
Implement and experiment with cutting-edge RL algorithms:

- **Algorithm Prototyping**: Rapidly prototype new RL algorithms
- **Custom Loss Functions**: Implement and test custom loss functions
- **Multi-Objective Optimization**: Handle multiple competing objectives
- **Hierarchical RL**: Implement hierarchical reinforcement learning approaches

#### **Experimental Design**
Design and conduct rigorous experiments:

- **Hyperparameter Optimization**: Systematic approaches to hyperparameter tuning
- **A/B Testing**: Design experiments to compare different approaches
- **Statistical Analysis**: Proper statistical analysis of experimental results
- **Reproducible Research**: Ensure your research is reproducible and well-documented

### **Integration and Extensions**

#### **Third-Party Integrations**
Integrate NeMo RL with other tools and frameworks:

- **Weights and Biases**: Integrate with W&B for experiment tracking
- **MLflow**: Use MLflow for model lifecycle management
- **Kubernetes**: Deploy NeMo RL on Kubernetes clusters
- **Cloud Platforms**: Deploy on AWS, GCP, or Azure

#### **Custom Extensions**
Extend NeMo RL with custom functionality:

- **Custom Metrics**: Implement custom evaluation metrics
- **Custom Callbacks**: Add custom training callbacks and hooks
- **Plugin Development**: Develop plugins for NeMo RL
- **API Extensions**: Extend the NeMo RL API for your specific needs

## Learning Resources

### Documentation
- **API Reference**: Complete API documentation for all NeMo RL components
- **Configuration Guide**: Detailed configuration options and parameters
- **Architecture Documentation**: Deep dive into system design and internals
- **Best Practices**: Proven patterns and recommendations for production use
- **Migration Guides**: Upgrade paths and compatibility information

### Community Resources
- **GitHub Repository**: Source code, issues, and discussions
- **Community Forum**: Connect with other NeMo RL users and developers
- **Blog Posts**: Technical articles and case studies
- **Video Tutorials**: Visual guides and demonstrations
- **Research Papers**: Academic publications and technical papers
- **Conference Talks**: Presentations from NeMo RL team and community

## Getting Help

### Troubleshooting
- **Common Issues**: Solutions to frequently encountered problems
- **Debugging Guides**: Step-by-step debugging procedures
- **Performance Issues**: Optimization and performance tuning
- **Installation Problems**: Resolving setup and environment issues
- **Configuration Errors**: Fixing configuration and parameter issues

### Support Channels
- **GitHub Issues**: Report bugs and request new features
- **Documentation**: Comprehensive guides and troubleshooting
- **Community Forum**: Get help from the NeMo RL community
- **NVIDIA Support**: Enterprise support for production deployments
- **Stack Overflow**: Technical questions and answers
- **Discord/Slack**: Real-time community support

## Next Steps

After completing the tutorials:

- **Explore Advanced Features**: Dive into distributed computing and cloud execution
- **Build Your Workflows**: Create custom experiment pipelines
- **Optimize Performance**: Learn best practices for production deployment
- **Contribute**: Share your experiences and contribute to the community

For additional learning resources and community support, visit the NeMo RL GitHub repository and documentation.

```{note}
**Note**: The tutorial files referenced in this guide are available in the NeMo RL examples repository. Clone the repository to access the complete tutorial notebooks and scripts.
```

Start with the learning path that matches your experience level, and gradually work your way through the tutorial series. The advanced tutorials will help you master complex scenarios and production deployments.

```{toctree}
:maxdepth: 2
:caption: Tutorials and Examples
:hidden:

tutorials/index
examples/index
use-cases/index
```
