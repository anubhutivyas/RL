---
description: "Core mathematical foundations and theoretical concepts that underpin NeMo RL algorithms and training methods"
categories: ["research-advanced"]
tags: ["mathematics", "theory", "foundations", "reinforcement-learning", "policy-optimization", "convergence", "sft", "dpo", "grpo"]
personas: ["researcher-focused", "mle-focused"]
difficulty: "advanced"
content_type: "concept"
modality: "universal"
---

# Mathematical Foundations

This document provides the core mathematical foundations that underpin all NeMo RL algorithms. Understanding these concepts is essential for AI scientists and researchers working with reinforcement learning for language models.

## Policy Gradient Methods

### Basic Policy Gradient Theorem

The policy gradient theorem forms the foundation of all policy-based reinforcement learning methods used in NeMo RL:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]$$

where:
- $J(\theta)$ is the objective function (expected return)
- $\pi_\theta$ is the policy parameterized by $\theta$
- $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory
- $R(\tau)$ is the return of trajectory $\tau$

### Advantage Function

The advantage function measures how much better an action is compared to the average action in a given state:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

where:
- $Q^\pi(s, a)$ is the state-action value function
- $V^\pi(s)$ is the state value function

### Policy Gradient with Advantage

Using the advantage function, the policy gradient becomes:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]$$

This formulation reduces variance compared to using raw returns.

## Supervised Fine-Tuning (SFT) Theory

### SFT Objective Function

The SFT objective maximizes the log-likelihood of target sequences:

$$L_{SFT}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t|x, y_{<t}) \right]$$

where:
- $x$ is the input context
- $y = (y_1, y_2, \ldots, y_T)$ is the target sequence
- $\pi_\theta(y_t|x, y_{<t})$ is the model's probability of token $y_t$

### SFT Gradient

The gradient of the SFT loss is:

$$\nabla_\theta L_{SFT}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t|x, y_{<t}) \right]$$

### SFT Convergence Properties

**Theorem**: Under appropriate conditions, SFT converges to the maximum likelihood estimate of the training data distribution.

**Conditions**:
- The model class is expressive enough
- The training data is representative
- The learning rate is appropriately chosen

## Direct Preference Optimization (DPO) Theory

### DPO Objective Function

DPO optimizes the policy to align with human preferences using a preference dataset:

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

where:
- $(x, y_w, y_l)$ is a preference tuple (context, preferred response, less preferred response)
- $\pi_{ref}$ is the reference policy (usually SFT model)
- $\beta$ is the temperature parameter controlling preference learning strength
- $\sigma$ is the sigmoid function

### DPO Gradient

The gradient of the DPO loss is:

$$\nabla_\theta L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \beta \cdot \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \cdot \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]$$

### DPO Theoretical Properties

**Theorem**: DPO converges to a policy that maximizes the expected preference reward under the Bradley-Terry model.

**Proof Sketch**:
1. The DPO objective is equivalent to maximizing the Bradley-Terry preference model
2. Under appropriate conditions, the algorithm converges to the optimal policy
3. The convergence rate depends on the preference data quality and $\beta$ parameter

### DPO vs RLHF

DPO provides several advantages over traditional RLHF:
- **Simplicity**: Direct optimization without reward model training
- **Stability**: More stable training compared to PPO-based RLHF
- **Efficiency**: Single-stage optimization instead of two-stage process

## Group Relative Policy Optimization (GRPO) Theory

### GRPO Objective Function

GRPO extends PPO with group-based relative policy optimization:

$$L_{GRPO}(\theta) = L_{policy}(\theta) + c_1 L_{value}(\theta) + c_2 L_{entropy}(\theta)$$

where the policy loss is:

$$L_{policy}(\theta) = -\mathbb{E}_{(s, a, r) \sim \mathcal{D}} \left[ \min\left( w_t A(s, a), \text{clip}(w_t, 1-\epsilon, 1+\epsilon) A(s, a) \right) \right]$$

### GRPO Policy Loss Components

#### Importance Weight
$$w_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

#### Advantage Function
$$A(s, a) = Q^\pi(s, a) - V^\pi(s)$$

#### Clipping Mechanism
$$\text{clip}(w_t, 1-\epsilon, 1+\epsilon) = \begin{cases}
1-\epsilon & \text{if } w_t < 1-\epsilon \\
w_t & \text{if } 1-\epsilon \leq w_t \leq 1+\epsilon \\
1+\epsilon & \text{if } w_t > 1+\epsilon
\end{cases}$$

### GRPO Value Loss

$$L_{value}(\theta) = \mathbb{E}_{(s, a, r) \sim \mathcal{D}} \left[ \frac{1}{2} (V_\theta(s) - V_{target}(s))^2 \right]$$

### GRPO Entropy Loss

$$L_{entropy}(\theta) = -\mathbb{E}_{s \sim \mathcal{D}} \left[ \sum_a \pi_\theta(a|s) \log \pi_\theta(a|s) \right]$$

### GRPO Theoretical Guarantees

**Theorem**: GRPO provides monotonic improvement guarantees under bounded KL divergence.

**Lemma**: The clipping mechanism ensures bounded policy updates:
$$1 - \epsilon \leq \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \leq 1 + \epsilon$$

## KL Divergence Control

### KL Divergence

The KL divergence between two policies measures how different they are:

$$D_{KL}(\pi_\theta || \pi_{\theta_{old}}) = \mathbb{E}_{s \sim \pi_{\theta_{old}}} \left[ \mathbb{E}_{a \sim \pi_{\theta_{old}}} \left[ \log \frac{\pi_{\theta_{old}}(a|s)}{\pi_\theta(a|s)} \right] \right]$$

### Schulman Approximation

NeMo RL uses the Schulman approximation for KL divergence computation:

$$D_{KL}(\pi_\theta || \pi_{\theta_{old}}) \approx \mathbb{E}_{s \sim \pi_{\theta_{old}}} \left[ \mathbb{E}_{a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} - \log \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} - 1 \right] \right]$$

This approximation is unbiased and guaranteed to be positive.

## Convergence Theory

### Policy Gradient Convergence

Under certain conditions, policy gradient methods converge to a local optimum:

**Theorem**: If the policy is differentiable and the advantage function is bounded, then policy gradient methods converge to a local optimum of the expected return.

**Proof Sketch**:
1. The policy gradient is an unbiased estimate of the true gradient
2. Under appropriate learning rate schedules, the algorithm converges
3. The convergence rate depends on the policy class and advantage estimation

### KL Divergence Bounds

For stable training, we need to bound the KL divergence between consecutive policies:

**Lemma**: If $D_{KL}(\pi_\theta || \pi_{\theta_{old}}) \leq \delta$, then the policy update is stable.

This bound prevents excessive policy updates that could destabilize training.

### Algorithm-Specific Convergence

#### SFT Convergence
**Theorem**: SFT converges to the maximum likelihood estimate under appropriate conditions.

#### DPO Convergence
**Theorem**: DPO converges to the optimal policy under the Bradley-Terry preference model.

#### GRPO Convergence
**Theorem**: GRPO provides monotonic improvement with bounded KL divergence.

## Importance Sampling

### Basic Importance Sampling

When using off-policy data, importance sampling corrects for the distribution mismatch:

$$\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right]$$

where $p(x)$ is the target distribution and $q(x)$ is the sampling distribution.

### Importance Weights in NeMo RL

NeMo RL uses importance weights to correct for off-policy data:

$$w_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

These weights are used in loss function computations to ensure unbiased gradient estimates.

## Numerical Stability

### Log-Space Computations

To avoid numerical underflow, NeMo RL performs computations in log-space:

$$\log \pi_\theta(a|s) = \log \text{softmax}(f_\theta(s))_a$$

### Gradient Clipping

To prevent gradient explosion, gradients are clipped:

$$\text{clip}(\nabla_\theta, \text{max_norm}) = \nabla_\theta \cdot \min\left(1, \frac{\text{max_norm}}{\|\nabla_\theta\|_2}\right)$$

## Implementation Considerations

### Floating-Point Precision

NeMo RL casts logits to float32 before computing losses to ensure numerical stability:

```python
next_token_logits = next_token_logits.to(torch.float32)
```

### Masked Computations

To handle variable-length sequences, masked computations are used:

$$\text{masked_mean}(x, \text{mask}) = \frac{\sum_i x_i \cdot \text{mask}_i}{\sum_i \text{mask}_i}$$

### Distributed Reductions

For distributed training, metrics are reduced across ranks:

$$\text{global_metric} = \frac{1}{N} \sum_{i=1}^N \text{local_metric}_i$$

## Theoretical Guarantees

### Monotonic Improvement

Under certain conditions, NeMo RL algorithms provide monotonic improvement guarantees:

**Theorem**: If the KL divergence is bounded and the advantage function is accurate, then the expected return increases monotonically.

### Sample Complexity

The sample complexity depends on:
- Policy class complexity
- Advantage estimation accuracy
- KL divergence bounds
- Learning rate schedule

### Algorithm Comparison

| Algorithm | Objective | Convergence | Sample Efficiency | Stability |
|-----------|-----------|-------------|-------------------|-----------|
| **SFT** | Maximum Likelihood | Global Optimum | High | Very Stable |
| **DPO** | Preference Alignment | Local Optimum | Medium | Stable |
| **GRPO** | Policy Optimization | Local Optimum | High | Stable |

## Research Applications

These mathematical foundations enable:

1. **Algorithm Development**: Understanding theory for custom algorithm design
2. **Hyperparameter Tuning**: Mathematical guidance for parameter selection
3. **Performance Analysis**: Theoretical bounds for expected performance
4. **Reproducibility**: Mathematical framework for reproducible experiments

## References

- Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347 (2017).
- Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv preprint arXiv:2305.18290 (2023).
- Schulman, J. "Approximating the KL divergence." http://joschu.net/blog/kl-approx.html (2020).
- Sutton, R. S., & Barto, A. G. "Reinforcement learning: An introduction." MIT press (2018). 