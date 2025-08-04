---
description: "Extend DPO for specific use cases and domains with custom implementations"
tags: ["dpo", "customization", "algorithms", "reinforcement-learning"]
categories: ["algorithm-customization"]
---

# Custom DPO Implementation

This guide covers how to customize Direct Preference Optimization (DPO) for specific use cases and domains in NeMo RL using the actual codebase patterns.

## Overview

NeMo RL provides a flexible DPO implementation that you can extend for domain-specific requirements. This guide shows how to work with the actual `DPOLossFn` class and `dpo_train` function.

## Core DPO Components

### Loss Function Structure

The main DPO loss function is implemented in `DPOLossFn`:

```python
from nemo_rl.algorithms.loss_functions import DPOLossFn, DPOLossConfig

# Real DPO loss configuration using TypedDict
dpo_config = DPOLossConfig(
    reference_policy_kl_penalty=0.1,  # Beta parameter
    preference_loss_weight=1.0,       # Weight for preference loss
    sft_loss_weight=0.1,              # Weight for SFT loss
    preference_average_log_probs=True, # Average log probs across tokens
    sft_average_log_probs=True        # Average SFT loss across tokens
)

# Initialize DPO loss function
dpo_loss_fn = DPOLossFn(dpo_config)
```

### Training Setup

The DPO training pipeline uses these real components:

```python
from nemo_rl.algorithms.dpo import setup, dpo_train
from nemo_rl.data.datasets import AllTaskProcessedDataset
from transformers import AutoTokenizer

# Setup DPO training
policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = setup(
    master_config=config,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Run DPO training
dpo_train(
    policy=policy,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tokenizer=tokenizer,
    loss_fn=loss_fn,
    master_config=master_config,
    logger=logger,
    checkpointer=checkpointer,
    dpo_save_state=save_state
)
```

## Custom DPO Configurations

### Domain-Specific Beta Values

Adjust the KL penalty (beta) for different domains:

```yaml
# configs/dpo_math_reasoning.yaml
dpo:
  reference_policy_kl_penalty: 0.2  # Higher beta for math reasoning
  preference_loss_weight: 1.0
  sft_loss_weight: 0.2  # Stronger SFT component for math
  preference_average_log_probs: true
  sft_average_log_probs: true
  max_num_epochs: 3
  max_num_steps: 1000
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true
  seed: 42
```

```yaml
# configs/dpo_code_generation.yaml
dpo:
  reference_policy_kl_penalty: 0.05  # Lower beta for code generation
  preference_loss_weight: 1.0
  sft_loss_weight: 0.05  # Weaker SFT component
  preference_average_log_probs: false  # Sum across tokens for code
  sft_average_log_probs: false
  max_num_epochs: 2
  max_num_steps: 800
  val_period: 50
  val_batches: 3
  val_global_batch_size: 16
  val_micro_batch_size: 4
  val_at_start: true
  seed: 42
```

### Custom Loss Weight Configurations

```yaml
# configs/dpo_balanced.yaml
dpo:
  reference_policy_kl_penalty: 0.1
  preference_loss_weight: 1.0
  sft_loss_weight: 0.1  # Balanced preference and SFT
  preference_average_log_probs: true
  sft_average_log_probs: true
  max_num_epochs: 3
  max_num_steps: 1000
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true
  seed: 42
```

```yaml
# configs/dpo_preference_focused.yaml
dpo:
  reference_policy_kl_penalty: 0.1
  preference_loss_weight: 1.0
  sft_loss_weight: 0.0  # Focus only on preference learning
  preference_average_log_probs: true
  sft_average_log_probs: true
  max_num_epochs: 3
  max_num_steps: 1000
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true
  seed: 42
```

## Real NeMo RL Integration Examples

### Custom DPO Loss Function

```python
from nemo_rl.algorithms.loss_functions import DPOLossFn

class CustomDPOLossFn(DPOLossFn):
    """Custom DPO loss function with domain-specific modifications"""
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Add custom modifications
        self.custom_weight = cfg.get("custom_weight", 1.0)
    
    def __call__(self, next_token_logits, data, global_valid_seqs, global_valid_toks, 
                 vocab_parallel_rank=None, vocab_parallel_group=None):
        """
        Custom DPO loss computation
        """
        # Call parent implementation
        loss, metrics = super().__call__(
            next_token_logits, data, global_valid_seqs, global_valid_toks,
            vocab_parallel_rank, vocab_parallel_group
        )
        
        # Add custom loss component
        custom_loss = self.compute_custom_loss(data)
        total_loss = loss + self.custom_weight * custom_loss
        
        # Update metrics
        metrics["custom_loss"] = custom_loss.item()
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics
    
    def compute_custom_loss(self, data):
        """
        Compute custom loss component
        """
        # Example: Add regularization based on response length
        # This is a simplified example
        return torch.tensor(0.0, device=next(self.parameters()).device)
```

### Domain-Specific Training Setup

```python
from nemo_rl.algorithms.dpo import setup, dpo_train
from nemo_rl.data.datasets import AllTaskProcessedDataset
from transformers import AutoTokenizer

def setup_domain_specific_dpo(domain: str, config_path: str):
    """
    Setup domain-specific DPO training
    """
    # Load configuration
    config = load_config(config_path)
    
    # Modify configuration based on domain
    if domain == "math_reasoning":
        config["dpo"]["reference_policy_kl_penalty"] = 0.2
        config["dpo"]["sft_loss_weight"] = 0.2
    elif domain == "code_generation":
        config["dpo"]["reference_policy_kl_penalty"] = 0.05
        config["dpo"]["sft_loss_weight"] = 0.05
        config["dpo"]["preference_average_log_probs"] = False
    elif domain == "creative_writing":
        config["dpo"]["reference_policy_kl_penalty"] = 0.15
        config["dpo"]["sft_loss_weight"] = 0.0  # Focus on preference only
    
    # Setup training components
    policy, cluster, train_dataloader, val_dataloader, loss_fn, master_config, logger, task_spec, save_state = setup(
        master_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    return {
        'policy': policy,
        'cluster': cluster,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'loss_fn': loss_fn,
        'master_config': master_config,
        'logger': logger,
        'task_spec': task_spec,
        'save_state': save_state
    }

def run_domain_specific_training(domain: str, config_path: str):
    """
    Run domain-specific DPO training
    """
    # Setup training
    components = setup_domain_specific_dpo(domain, config_path)
    
    # Run training
    dpo_train(
        policy=components['policy'],
        train_dataloader=components['train_dataloader'],
        val_dataloader=components['val_dataloader'],
        tokenizer=tokenizer,
        loss_fn=components['loss_fn'],
        master_config=components['master_config'],
        logger=components['logger'],
        checkpointer=checkpointer,
        dpo_save_state=components['save_state']
    )
```

## Configuration Examples

### Math Reasoning DPO Configuration

```yaml
# configs/dpo_math_reasoning.yaml
dpo:
  max_num_epochs: 3
  max_num_steps: 1000
  val_period: 100
  val_batches: 5
  val_global_batch_size: 32
  val_micro_batch_size: 8
  val_at_start: true
  seed: 42
  
  # Math-specific parameters
  reference_policy_kl_penalty: 0.2  # Higher beta for precise reasoning
  preference_loss_weight: 1.0
  sft_loss_weight: 0.2  # Stronger SFT for accuracy
  preference_average_log_probs: true
  sft_average_log_probs: true

policy:
  model_name: "microsoft/DialoGPT-medium"
  max_total_sequence_length: 2048
  precision: "bfloat16"
  
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 1e-5
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8
  
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs:
        start_factor: 0.1
        end_factor: 1.0
        total_iters: 50
    - name: "torch.optim.lr_scheduler.ConstantLR"
      kwargs:
        factor: 1.0
        total_iters: 10000000000
    - milestones: [50]
  
  train_global_batch_size: 16
  train_micro_batch_size: 1
  max_grad_norm: 1.0

logger:
  log_dir: "logs"
  wandb_enabled: true
  tensorboard_enabled: true
  wandb:
    project: "dpo-math-reasoning"
    name: "math-reasoning-experiment"
```

### Code Generation DPO Configuration

```yaml
# configs/dpo_code_generation.yaml
dpo:
  max_num_epochs: 2
  max_num_steps: 800
  val_period: 50
  val_batches: 3
  val_global_batch_size: 16
  val_micro_batch_size: 4
  val_at_start: true
  seed: 42
  
  # Code-specific parameters
  reference_policy_kl_penalty: 0.05  # Lower beta for creative code
  preference_loss_weight: 1.0
  sft_loss_weight: 0.05  # Weaker SFT for creativity
  preference_average_log_probs: false  # Sum across tokens
  sft_average_log_probs: false

policy:
  model_name: "microsoft/DialoGPT-medium"
  max_total_sequence_length: 2048
  precision: "bfloat16"
  
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 2e-5  # Slightly higher learning rate
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8
  
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs:
        start_factor: 0.1
        end_factor: 1.0
        total_iters: 40
    - name: "torch.optim.lr_scheduler.ConstantLR"
      kwargs:
        factor: 1.0
        total_iters: 10000000000
    - milestones: [40]
  
  train_global_batch_size: 8
  train_micro_batch_size: 1
  max_grad_norm: 1.0

logger:
  log_dir: "logs"
  wandb_enabled: true
  tensorboard_enabled: true
  wandb:
    project: "dpo-code-generation"
    name: "code-generation-experiment"
```

## Best Practices

### 1. Domain-Specific Beta Selection

Choose beta values based on your domain:

```python
def select_beta_for_domain(domain: str) -> float:
    """
    Select appropriate beta value for domain
    """
    beta_values = {
        "math_reasoning": 0.2,      # Higher for precise reasoning
        "code_generation": 0.05,    # Lower for creative code
        "creative_writing": 0.15,   # Medium for creativity
        "factual_qa": 0.1,          # Standard for Q&A
        "dialogue": 0.1,            # Standard for dialogue
    }
    return beta_values.get(domain, 0.1)
```

### 2. Loss Weight Balancing

Balance preference and SFT loss weights:

```python
def balance_loss_weights(domain: str, config: dict) -> dict:
    """
    Balance loss weights for domain
    """
    if domain == "math_reasoning":
        config["dpo"]["sft_loss_weight"] = 0.2  # Stronger SFT
    elif domain == "creative_writing":
        config["dpo"]["sft_loss_weight"] = 0.0  # Focus on preference
    elif domain == "code_generation":
        config["dpo"]["sft_loss_weight"] = 0.05  # Weak SFT
    
    return config
```

### 3. Validation Strategy

Implement domain-specific validation:

```python
def validate_domain_performance(policy, val_dataloader, domain: str):
    """
    Validate domain-specific performance
    """
    # Run standard validation
    val_metrics = validate(
        policy=policy,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        step=step,
        master_config=master_config,
        val_batches=val_batches,
        val_batch_size=val_batch_size,
        val_mbs=val_mbs
    )
    
    # Add domain-specific metrics
    if domain == "math_reasoning":
        val_metrics["math_accuracy"] = compute_math_accuracy(policy, val_dataloader)
    elif domain == "code_generation":
        val_metrics["code_quality"] = compute_code_quality(policy, val_dataloader)
    
    return val_metrics
```

## Troubleshooting

### Common DPO Customization Issues

1. **Loss Divergence**: Reduce beta value if loss diverges
2. **Poor Convergence**: Increase SFT loss weight for better convergence
3. **Overfitting**: Reduce training steps or increase validation frequency

### Debugging Tips

```python
# Monitor DPO training progress
def monitor_dpo_training(logger):
    """
    Monitor DPO training with real NeMo RL logging
    """
    # Check loss components
    preference_loss = logger.get_latest_metric("preference_loss")
    sft_loss = logger.get_latest_metric("sft_loss")
    total_loss = logger.get_latest_metric("loss")
    
    print(f"Preference Loss: {preference_loss:.4f}")
    print(f"SFT Loss: {sft_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")
    
    # Check accuracy
    accuracy = logger.get_latest_metric("accuracy")
    print(f"Accuracy: {accuracy:.4f}")
```

## Next Steps

- Learn about [Model Evaluation](../research/model-evaluation-validation) for comprehensive assessment
- Review [Experimental Design](../research/experimental-design-validation) for rigorous research
- Explore [Performance Analysis](../research/performance-analysis) for result interpretation 