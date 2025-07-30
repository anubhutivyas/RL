# Custom Loss Functions

This guide covers creating and implementing custom loss functions in NeMo RL, enabling you to design specialized training objectives for specific use cases and model behaviors.

> **Note**: This guide focuses on **training-specific loss functions**. For domain-specific and novel loss function implementations, see [Algorithm Customization Loss Functions](../algorithm-customization/custom-loss-functions.md).

## Overview

Custom loss functions allow you to define training objectives that go beyond standard loss functions, enabling specialized training for specific tasks, constraints, or model behaviors. NeMo RL provides a flexible framework for implementing and integrating custom loss functions.

## Loss Function Framework

### Loss Function Protocol

NeMo RL uses a `LossFunction` protocol that all custom loss functions must implement:

```python
from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
import torch
import torch.nn.functional as F

class CustomLossFunction(LossFunction):
    def __init__(self, config):
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 0.1)
        self.loss_type = LossType.SEQUENCE_LEVEL  # or LossType.TOKEN_LEVEL
    
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute custom loss.
        
        Args:
            next_token_logits: Model logits [batch_size, seq_len, vocab_size]
            data: BatchedDataDict containing all relevant data
            global_valid_seqs: Number of valid sequences in global batch
            global_valid_toks: Number of valid tokens in global batch
            
        Returns:
            tuple: (loss, metrics)
        """
        # Implement custom loss computation
        loss = self._compute_loss(next_token_logits, data, global_valid_seqs, global_valid_toks)
        
        # Compute additional metrics
        metrics = self._compute_metrics(next_token_logits, data)
        
        return loss, metrics
    
    def _compute_loss(self, next_token_logits, data, global_valid_seqs, global_valid_toks):
        """Implement the actual loss computation."""
        raise NotImplementedError
    
    def _compute_metrics(self, next_token_logits, data):
        """Compute additional metrics for monitoring."""
        return {}
```

### Loss Function Types

NeMo RL supports two types of loss functions:

```python
from nemo_rl.algorithms.interfaces import LossType

class TokenLevelLoss(LossFunction):
    """Loss computed at the token level."""
    def __init__(self):
        self.loss_type = LossType.TOKEN_LEVEL

class SequenceLevelLoss(LossFunction):
    """Loss computed at the sequence level."""
    def __init__(self):
        self.loss_type = LossType.SEQUENCE_LEVEL
```

## Common Custom Loss Functions

### Custom Sequence-Level Loss

Implement a custom sequence-level loss for RL training:

```python
from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
import torch
import torch.nn.functional as F

class CustomSequenceLoss(LossFunction):
    def __init__(self, config):
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 0.1)
        self.loss_type = LossType.SEQUENCE_LEVEL
    
    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Custom sequence-level loss implementation.
        
        Args:
            next_token_logits: Model logits [batch_size, seq_len, vocab_size]
            data: BatchedDataDict containing rewards, advantages, etc.
            global_valid_seqs: Number of valid sequences in global batch
            global_valid_toks: Number of valid tokens in global batch
        """
        # Extract data
        rewards = data.get("rewards", torch.zeros(next_token_logits.size(0)))
        advantages = data.get("advantages", torch.zeros(next_token_logits.size(0)))
        
        # Compute log probabilities
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Custom loss computation
        policy_loss = -(log_probs.mean() * advantages.mean())
        value_loss = F.mse_loss(rewards.mean(), advantages.mean())
        
        # Combined loss
        total_loss = self.alpha * policy_loss + self.beta * value_loss
        
        metrics = {
            "custom_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item()
        }
        
        return total_loss, metrics
```

### Contrastive Loss for Representation Learning

Implement contrastive loss for learning better representations:

```python
class ContrastiveLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = config.get("temperature", 0.1)
        self.margin = config.get("margin", 1.0)
        self.negative_weight = config.get("negative_weight", 1.0)
    
    def forward(self, embeddings, labels, **kwargs):
        """
        Contrastive Loss implementation.
        
        L = (1 - y) * d^2 + y * max(0, margin - d)^2
        where d is the distance between embeddings.
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings)
        
        # Create label matrix (1 if same class, 0 if different)
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # Positive pairs (same class)
        positive_mask = label_matrix & ~torch.eye(batch_size, dtype=bool, device=embeddings.device)
        positive_distances = distances[positive_mask]
        
        # Negative pairs (different class)
        negative_mask = ~label_matrix
        negative_distances = distances[negative_mask]
        
        # Compute losses
        positive_loss = positive_distances.pow(2).mean()
        negative_loss = F.relu(self.margin - negative_distances).pow(2).mean()
        
        # Combined loss
        loss = positive_loss + self.negative_weight * negative_loss
        
        metrics = {
            "contrastive_loss": loss.item(),
            "positive_loss": positive_loss.item(),
            "negative_loss": negative_loss.item(),
            "mean_positive_distance": positive_distances.mean().item(),
            "mean_negative_distance": negative_distances.mean().item()
        }
        
        return loss, metrics
```

### KL Divergence Loss for Knowledge Distillation

Implement KL divergence loss for knowledge distillation:

```python
class KLDivergenceLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.temperature = config.get("temperature", 4.0)
        self.alpha = config.get("alpha", 0.7)  # Weight for KL loss vs task loss
    
    def forward(self, student_logits, teacher_logits, targets=None, **kwargs):
        """
        KL Divergence Loss for knowledge distillation.
        
        L = α * L_task + (1 - α) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
        """
        # Compute softmax with temperature
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            student_probs.log(), 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Task loss (if targets provided)
        if targets is not None:
            task_loss = F.cross_entropy(student_logits, targets)
            total_loss = self.alpha * task_loss + (1 - self.alpha) * kl_loss
        else:
            total_loss = kl_loss
            task_loss = torch.tensor(0.0, device=student_logits.device)
        
        metrics = {
            "kl_loss": kl_loss.item(),
            "task_loss": task_loss.item(),
            "total_loss": total_loss.item(),
            "temperature": self.temperature
        }
        
        return total_loss, metrics
```

### Reward Shaping Loss for RL

Implement reward shaping loss for reinforcement learning:

```python
class RewardShapingLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.shaping_weight = config.get("shaping_weight", 0.1)
        self.entropy_weight = config.get("entropy_weight", 0.01)
        self.value_weight = config.get("value_weight", 0.5)
    
    def forward(self, policy_logits, value_predictions, rewards, advantages, **kwargs):
        """
        Reward Shaping Loss for RL training.
        
        L = L_policy + L_value + L_entropy + L_shaping
        """
        # Policy loss (log probability of actions)
        action_probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Sample actions (in practice, actions would be provided)
        actions = torch.multinomial(action_probs, 1)
        action_log_probs = log_probs.gather(1, actions)
        
        policy_loss = -(action_log_probs * advantages.unsqueeze(1)).mean()
        
        # Value loss
        value_loss = F.mse_loss(value_predictions, rewards)
        
        # Entropy regularization
        entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy
        
        # Reward shaping loss (encourage exploration)
        shaping_loss = self.shaping_weight * self._compute_shaping_loss(
            action_probs, rewards, advantages
        )
        
        # Total loss
        total_loss = policy_loss + self.value_weight * value_loss + entropy_loss + shaping_loss
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "shaping_loss": shaping_loss.item(),
            "total_loss": total_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item()
        }
        
        return total_loss, metrics
    
    def _compute_shaping_loss(self, action_probs, rewards, advantages):
        """Compute reward shaping loss to encourage exploration."""
        # Encourage diversity in action selection
        entropy_bonus = action_probs.entropy().mean()
        
        # Encourage positive advantage actions
        advantage_bonus = F.relu(advantages).mean()
        
        return entropy_bonus + advantage_bonus
```

## Advanced Loss Functions

### Multi-Task Loss

Implement loss functions for multi-task learning:

```python
class MultiTaskLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.task_weights = config.get("task_weights", {})
        self.uncertainty_weighting = config.get("uncertainty_weighting", False)
    
    def forward(self, predictions, targets, task_ids, **kwargs):
        """
        Multi-task loss with uncertainty weighting.
        
        L = Σ (1/σ_i^2) * L_i + log(σ_i^2)
        where σ_i is the uncertainty for task i.
        """
        total_loss = 0.0
        task_losses = {}
        task_metrics = {}
        
        for task_id in task_ids.unique():
            task_mask = task_ids == task_id
            task_pred = predictions[task_mask]
            task_target = targets[task_mask]
            
            # Compute task-specific loss
            if task_id == 0:  # Classification task
                task_loss = F.cross_entropy(task_pred, task_target)
            elif task_id == 1:  # Regression task
                task_loss = F.mse_loss(task_pred, task_target)
            else:  # Custom task
                task_loss = self._compute_custom_task_loss(task_pred, task_target, task_id)
            
            # Apply uncertainty weighting if enabled
            if self.uncertainty_weighting:
                uncertainty = self._estimate_uncertainty(task_pred, task_id)
                weighted_loss = task_loss / (uncertainty ** 2) + torch.log(uncertainty)
                task_loss = weighted_loss
            
            # Apply task weight
            weight = self.task_weights.get(task_id, 1.0)
            weighted_task_loss = weight * task_loss
            
            total_loss += weighted_task_loss
            task_losses[f"task_{task_id}_loss"] = task_loss.item()
            task_metrics[f"task_{task_id}_weight"] = weight
        
        metrics = {
            "total_loss": total_loss.item(),
            **task_losses,
            **task_metrics
        }
        
        return total_loss, metrics
    
    def _estimate_uncertainty(self, predictions, task_id):
        """Estimate uncertainty for a task."""
        # Simple uncertainty estimation based on prediction variance
        if predictions.dim() > 1:
            return predictions.var(dim=-1).mean()
        else:
            return predictions.var()
    
    def _compute_custom_task_loss(self, predictions, targets, task_id):
        """Compute custom loss for specific task."""
        # Implement task-specific loss computation
        return F.mse_loss(predictions, targets)
```

### Adversarial Loss

Implement adversarial loss for robust training:

```python
class AdversarialLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config.get("epsilon", 0.1)
        self.alpha = config.get("alpha", 0.5)
        self.num_steps = config.get("num_steps", 3)
    
    def forward(self, model, inputs, targets, **kwargs):
        """
        Adversarial training loss.
        
        L = L_clean + α * L_adversarial
        """
        # Clean loss
        clean_outputs = model(inputs)
        clean_loss = F.cross_entropy(clean_outputs, targets)
        
        # Generate adversarial examples
        adversarial_inputs = self._generate_adversarial_examples(
            model, inputs, targets
        )
        
        # Adversarial loss
        adversarial_outputs = model(adversarial_inputs)
        adversarial_loss = F.cross_entropy(adversarial_outputs, targets)
        
        # Combined loss
        total_loss = clean_loss + self.alpha * adversarial_loss
        
        metrics = {
            "clean_loss": clean_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "total_loss": total_loss.item(),
            "adversarial_perturbation": (adversarial_inputs - inputs).norm().item()
        }
        
        return total_loss, metrics
    
    def _generate_adversarial_examples(self, model, inputs, targets):
        """Generate adversarial examples using FGSM."""
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Generate perturbation
        perturbation = self.epsilon * inputs.grad.sign()
        
        # Create adversarial examples
        adversarial_inputs = inputs + perturbation
        
        # Clip to valid range
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        return adversarial_inputs.detach()
```

## Loss Function Composition

### Combined Loss Functions

Combine multiple loss functions:

```python
class CombinedLoss(BaseLossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.loss_functions = []
        self.loss_weights = config.get("loss_weights", [])
        
        # Initialize component loss functions
        for loss_config in config.get("loss_functions", []):
            loss_name = loss_config["name"]
            loss_params = loss_config.get("params", {})
            
            loss_fn = self._create_loss_function(loss_name, loss_params)
            self.loss_functions.append(loss_fn)
    
    def forward(self, predictions, targets, **kwargs):
        """Combine multiple loss functions."""
        total_loss = 0.0
        combined_metrics = {}
        
        for i, loss_fn in enumerate(self.loss_functions):
            weight = self.loss_weights[i] if i < len(self.loss_weights) else 1.0
            
            loss, metrics = loss_fn(predictions, targets, **kwargs)
            weighted_loss = weight * loss
            
            total_loss += weighted_loss
            
            # Combine metrics
            for key, value in metrics.items():
                combined_metrics[f"{loss_fn.__class__.__name__}_{key}"] = value
        
        combined_metrics["total_loss"] = total_loss.item()
        
        return total_loss, combined_metrics
    
    def _create_loss_function(self, name, params):
        """Create loss function by name."""
        if name == "cross_entropy":
            return CrossEntropyLoss(params)
        elif name == "mse":
            return MSELoss(params)
        elif name == "focal":
            return FocalLoss(params)
        else:
            raise ValueError(f"Unknown loss function: {name}")
```

## Integration with Training

### Configuration Setup

```yaml
# custom_loss_config.yaml
training:
  loss_function:
    name: "custom_loss"
    params:
      alpha: 0.7
      beta: 0.3
      temperature: 4.0
      focal_gamma: 2.0
      contrastive_margin: 1.0
      kl_temperature: 4.0
      shaping_weight: 0.1
      entropy_weight: 0.01
      value_weight: 0.5
      uncertainty_weighting: true
      adversarial_epsilon: 0.1
      adversarial_alpha: 0.5
      adversarial_steps: 3
      task_weights:
        0: 1.0  # Classification
        1: 0.5  # Regression
        2: 0.3  # Custom task
      loss_weights: [0.6, 0.3, 0.1]
      loss_functions:
        - name: "cross_entropy"
          params:
            weight: 1.0
        - name: "focal"
          params:
            alpha: 1.0
            gamma: 2.0
        - name: "kl_divergence"
          params:
            temperature: 4.0
            alpha: 0.7
```

### Training Loop Integration

```python
from nemo_rl.algorithms import CustomLossTrainer

# Create trainer with custom loss
trainer = CustomLossTrainer(
    model=model,
    optimizer=optimizer,
    loss_function=CustomLossFunction(config["training"]["loss_function"])
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        predictions = model(batch["inputs"])
        
        # Compute custom loss
        loss, metrics = trainer.loss_function(
            predictions=predictions,
            targets=batch["targets"],
            **batch  # Pass additional batch data
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log metrics
        trainer.log_metrics(metrics)
```

## Best Practices

### Loss Function Design

1. **Ensure differentiability** - All operations should be differentiable
2. **Handle edge cases** - Add proper error handling and validation
3. **Normalize appropriately** - Scale loss components to similar ranges
4. **Monitor gradients** - Check for gradient explosion or vanishing
5. **Validate numerically** - Test with known inputs and expected outputs

### Performance Optimization

1. **Use efficient operations** - Leverage PyTorch's optimized functions
2. **Minimize memory usage** - Avoid unnecessary tensor copies
3. **Vectorize computations** - Use batch operations when possible
4. **Profile performance** - Monitor computation time and memory usage
5. **Use mixed precision** - Enable FP16 when appropriate

### Debugging and Monitoring

1. **Add comprehensive logging** - Track loss components and metrics
2. **Implement gradient clipping** - Prevent gradient explosion
3. **Monitor loss values** - Check for NaN or infinite values
4. **Validate inputs** - Ensure proper data types and shapes
5. **Test with small batches** - Verify behavior with minimal data

## Troubleshooting

### Common Issues

1. **Loss Explosion**
   ```python
   # Add gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   # Check for NaN values
   if torch.isnan(loss):
       print("NaN loss detected!")
   ```

2. **Memory Issues**
   ```python
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Clear intermediate tensors
   del intermediate_tensors
   torch.cuda.empty_cache()
   ```

3. **Slow Convergence**
   ```python
   # Adjust learning rate
   optimizer.param_groups[0]['lr'] *= 0.1
   
   # Check loss scaling
   loss_scale = 1.0 / loss.item()
   ```

4. **Unstable Training**
   ```python
   # Add loss smoothing
   loss = loss * 0.9 + previous_loss * 0.1
   
   # Use adaptive weights
   weight = min(weight * 1.1, max_weight)
   ```

## Next Steps

- [Loss Function Design](loss-function-design) - Learn loss function design principles
- [Training Stability](training-stability) - Ensure stable training
- [Multi-Objective Training](multi-objective-training) - Balance multiple objectives
- [Advanced Performance](../performance/index) - Optimize performance 