---
description: "Supervised Fine-Tuning (SFT) mathematical foundations and negative log-likelihood loss theory for language models."
tags: ["sft", "theory", "mathematics", "supervised learning", "language models"]
categories: ["theory"]
---

# SFT Theory

This document provides the mathematical foundations and theoretical analysis of Supervised Fine-Tuning (SFT), the foundational training method for language models in NeMo RL.

## Overview

SFT is the supervised learning component that trains language models to predict the next token given a sequence of previous tokens. It forms the foundation for all other NeMo RL algorithms and provides the base model for preference-based learning.

## Mathematical Formulation

### Basic SFT Objective

The SFT objective function maximizes the likelihood of the target sequence:

$$L_{SFT}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t|x, y_{<t}) \right]$$

where:
- $x$ is the input prompt
- $y = (y_1, y_2, \ldots, y_T)$ is the target sequence
- $y_t$ is the token at position $t$
- $y_{<t} = (y_1, y_2, \ldots, y_{t-1})$ are the previous tokens
- $\pi_\theta(y_t|x, y_{<t})$ is the model's probability of token $y_t$

### Autoregressive Language Modeling

SFT models language as an autoregressive process:

$$P(y|x) = \prod_{t=1}^{T} P(y_t|x, y_{<t})$$

Taking the negative log-likelihood:

$$-\log P(y|x) = -\sum_{t=1}^{T} \log P(y_t|x, y_{<t})$$

### Token-Level Loss

For each token position, the loss is:

$$L_t(\theta) = -\log \pi_\theta(y_t|x, y_{<t})$$

The total loss is the sum over all tokens:

$$L_{SFT}(\theta) = \sum_{t=1}^{T} L_t(\theta)$$

## Theoretical Analysis

### Maximum Likelihood Estimation

**Theorem**: SFT performs maximum likelihood estimation of the conditional distribution $P(y|x)$.

**Proof Sketch**:
1. The negative log-likelihood is minimized
2. This maximizes the likelihood of the training data
3. Under certain conditions, this converges to the true distribution

### Convergence Properties

**Lemma**: Under certain conditions, SFT converges to a local optimum of the likelihood function.

**Conditions**:
- The model is differentiable
- The data distribution is well-behaved
- Appropriate learning rates are used

### Generalization Bounds

The generalization error depends on:
- Model complexity
- Training data size
- Data distribution properties

**Theorem**: With high probability, the generalization error is bounded by:

$$\mathcal{R}(\theta) \leq \hat{\mathcal{R}}(\theta) + O\left(\sqrt{\frac{\log|\mathcal{H}|}{n}}\right)$$

where $\mathcal{H}$ is the hypothesis class and $n$ is the number of training examples.

## Implementation Details

### Logit Computation

SFT computes logits for each token position:

$$\text{logits}_t = f_\theta(x, y_{<t})$$

where $f_\theta$ is the model's logit function.

### Probability Computation

The token probabilities are computed using softmax:

$$\pi_\theta(y_t|x, y_{<t}) = \frac{\exp(\text{logits}_t[y_t])}{\sum_{v \in \mathcal{V}} \exp(\text{logits}_t[v])}$$

where $\mathcal{V}$ is the vocabulary.

### Loss Implementation

The SFT loss is implemented as:

```python
loss = -torch.sum(log_probs * attention_mask, dim=-1)
```

where `attention_mask` handles padding tokens.

### Numerical Stability

To ensure numerical stability, NeMo RL:

1. **Casts to float32**: `next_token_logits = next_token_logits.to(torch.float32)`
2. **Uses log-space**: Computes log probabilities to avoid underflow
3. **Handles padding**: Masks out padding tokens in loss computation

## Advanced Topics

### Teacher Forcing

SFT uses teacher forcing, where the model is trained to predict the next token given the ground truth previous tokens:

$$L_{teacher}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t|x, y_{<t}^{GT})$$

where $y_{<t}^{GT}$ are the ground truth previous tokens.

### Scheduled Sampling

For better generalization, scheduled sampling can be used:

$$L_{scheduled}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t|x, \hat{y}_{<t})$$

where $\hat{y}_{<t}$ are predicted tokens with some probability.

### Label Smoothing

Label smoothing can improve generalization:

$$L_{smooth}(\theta) = -\sum_{t=1}^{T} \sum_{v \in \mathcal{V}} q(v|y_t) \log \pi_\theta(v|x, y_{<t})$$

where $q(v|y_t)$ is a smoothed target distribution.

## Hyperparameter Analysis

### Learning Rate

The learning rate affects convergence speed:

- **High learning rate**: Faster convergence but potential instability
- **Low learning rate**: Stable but slow convergence
- **Adaptive learning rate**: Best of both worlds

### Batch Size

The batch size affects training stability:

- **Large batch size**: More stable gradients but higher memory usage
- **Small batch size**: Less stable gradients but lower memory usage
- **Optimal batch size**: Balanced approach

### Sequence Length

The sequence length affects training efficiency:

- **Long sequences**: More context but higher memory usage
- **Short sequences**: Less context but lower memory usage
- **Optimal length**: Task-dependent

## Comparison with Other Methods

### vs Pre-training

SFT differs from pre-training in:
- **Task-specific**: Focuses on specific tasks
- **Supervised**: Uses labeled data
- **Fine-tuning**: Starts from pre-trained weights

### vs RL Methods

SFT differs from RL methods in:
- **Supervised**: Uses labeled data instead of rewards
- **Teacher forcing**: Uses ground truth context
- **No exploration**: No exploration during training

## Research Applications

SFT theory enables:

1. **Model Development**: Understanding training dynamics
2. **Hyperparameter Tuning**: Mathematical guidance for parameter selection
3. **Performance Analysis**: Theoretical bounds for model performance
4. **Reproducibility**: Mathematical framework for reproducible experiments

## Advanced Techniques

### Gradient Clipping

To prevent gradient explosion:

$$\text{clip}(\nabla_\theta, \text{max_norm}) = \nabla_\theta \cdot \min\left(1, \frac{\text{max_norm}}{\|\nabla_\theta\|_2}\right)$$

### Learning Rate Scheduling

Common schedules include:

- **Linear decay**: $\eta_t = \eta_0 \cdot (1 - \frac{t}{T})$
- **Cosine decay**: $\eta_t = \eta_0 \cdot \cos(\frac{\pi t}{2T})$
- **Exponential decay**: $\eta_t = \eta_0 \cdot \gamma^t$

### Weight Decay

Regularization through weight decay:

$$L_{reg}(\theta) = L_{SFT}(\theta) + \lambda \|\theta\|_2^2$$

where $\lambda$ is the weight decay coefficient.

## References

- Vaswani, A., et al. "Attention is all you need." NeurIPS (2017).
- Brown, T., et al. "Language models are few-shot learners." NeurIPS (2020).
- Radford, A., et al. "Language models are unsupervised multitask learners." OpenAI blog (2019). 