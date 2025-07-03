# Guides

Explore detailed guides for NeMo RL algorithms, training, evaluation, and development:

## Training Algorithms

- [SFT Training](sft.md) - Supervised Fine-Tuning for language models
- [GRPO Training](grpo.md) - Group Relative Policy Optimization for RL training
- [DPO Training](dpo.md) - Direct Preference Optimization for preference learning

## Evaluation & Development

- [Evaluation](eval.md) - Model evaluation and benchmarking strategies
- [Adding New Models](adding-new-models.md) - Extend NeMo RL with custom model architectures
- [Testing](testing.md) - Testing strategies and best practices for RL training
- [Debugging](debugging.md) - Debugging techniques and tools for distributed training
- [NSYS Profiling](nsys-profiling.md) - Performance profiling with NSYS for optimization
- [Documentation](documentation.md) - Contributing to NeMo RL documentation

## Distributed Training

- [Set Up Distributed Training with Ray](distributed-training.md) - Scale RL training across multiple GPUs and nodes

## Advanced Topics

- [Create Custom RL Environments](environment-development.md) - Create custom RL environments
- [Packaging](packaging.md) - Deployment and packaging strategies for production
- [Model Quirks](model-quirks.md) - Model-specific considerations and workarounds
- [GRPO on DeepScaler](grpo-deepscaler.md) - DeepScaler integration for large-scale training
- [SFT on OpenMathInstruct2](sft-openmathinstruct2.md) - Math instruction fine-tuning example 

```{toctree}
:maxdepth: 2
:caption: Guides

sft
grpo
dpo
eval
adding-new-models
testing
debugging
nsys-profiling
documentation
distributed-training
environment-development
packaging
model-quirks
grpo-deepscaler
sft-openmathinstruct2
``` 