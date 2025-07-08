# Custom Algorithms

This guide covers implementing custom reinforcement learning algorithms in NeMo RL, including algorithm design, integration, and best practices.

## Overview

NeMo RL provides a flexible framework for implementing custom reinforcement learning algorithms. This guide covers the architecture, implementation patterns, and integration strategies.

## Algorithm Architecture

### Base Algorithm Interface

All algorithms in NeMo RL inherit from a base interface:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

class BaseAlgorithm(ABC):
    """Base class for all RL algorithms in NeMo RL."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.setup()
    
    @abstractmethod
    def setup(self):
        """Initialize algorithm components."""
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss for a batch of data."""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform algorithm update step."""
        pass
    
    @abstractmethod
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate algorithm performance."""
        pass
    
    def save_checkpoint(self, path: str):
        """Save algorithm state."""
        torch.save(self.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load algorithm state."""
        self.load_state_dict(torch.load(path, map_location=self.device))
```

### Algorithm Components

Custom algorithms typically consist of several components:

```python
class CustomAlgorithm(BaseAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        
        # Policy network
        self.policy = self.create_policy()
        
        # Value function (if applicable)
        self.value_function = self.create_value_function()
        
        # Optimizers
        self.policy_optimizer = self.create_optimizer(self.policy)
        self.value_optimizer = self.create_optimizer(self.value_function)
        
        # Algorithm-specific components
        self.buffer = self.create_buffer()
        self.scheduler = self.create_scheduler()
    
    def create_policy(self):
        """Create policy network."""
        policy_config = self.config['policy']
        return PolicyNetwork(
            input_size=policy_config['input_size'],
            hidden_size=policy_config['hidden_size'],
            output_size=policy_config['output_size']
        ).to(self.device)
    
    def create_value_function(self):
        """Create value function network."""
        value_config = self.config['value']
        return ValueNetwork(
            input_size=value_config['input_size'],
            hidden_size=value_config['hidden_size']
        ).to(self.device)
    
    def create_optimizer(self, network):
        """Create optimizer for network."""
        optimizer_config = self.config['optimizer']
        return torch.optim.AdamW(
            network.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )
    
    def create_buffer(self):
        """Create experience replay buffer."""
        buffer_config = self.config['buffer']
        return ReplayBuffer(
            capacity=buffer_config['capacity'],
            device=self.device
        )
    
    def create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_config = self.config['scheduler']
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=scheduler_config['max_steps']
        )
```

## Implementation Patterns

### Policy Gradient Methods

Implement policy gradient algorithms:

```python
class CustomPolicyGradient(BaseAlgorithm):
    """Custom policy gradient algorithm implementation."""
    
    def compute_loss(self, batch):
        """Compute policy gradient loss."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        advantages = batch['advantages']
        
        # Get policy distribution
        policy_dist = self.policy(states)
        
        # Compute log probabilities
        log_probs = policy_dist.log_prob(actions)
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Entropy regularization
        entropy = policy_dist.entropy().mean()
        entropy_loss = -self.config['entropy_coef'] * entropy
        
        total_loss = policy_loss + entropy_loss
        
        return total_loss
    
    def update(self, batch):
        """Perform policy gradient update."""
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config['max_grad_norm']
        )
        
        # Update parameters
        self.policy_optimizer.step()
        self.scheduler.step()
        
        return {
            'policy_loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self, batch):
        """Evaluate policy performance."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        
        with torch.no_grad():
            policy_dist = self.policy(states)
            log_probs = policy_dist.log_prob(actions)
            
            # Compute metrics
            avg_reward = rewards.mean().item()
            avg_log_prob = log_probs.mean().item()
            entropy = policy_dist.entropy().mean().item()
            
            return {
                'avg_reward': avg_reward,
                'avg_log_prob': avg_log_prob,
                'entropy': entropy
            }
```

### Actor-Critic Methods

Implement actor-critic algorithms:

```python
class CustomActorCritic(BaseAlgorithm):
    """Custom actor-critic algorithm implementation."""
    
    def compute_losses(self, batch):
        """Compute actor and critic losses."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Actor (policy) loss
        policy_dist = self.policy(states)
        log_probs = policy_dist.log_prob(actions)
        
        # Compute advantages
        current_values = self.value_function(states)
        next_values = self.value_function(next_states)
        
        # TD(0) advantage estimation
        advantages = rewards + self.config['gamma'] * next_values * (1 - dones) - current_values
        
        # Policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(current_values, rewards + self.config['gamma'] * next_values * (1 - dones))
        
        # Entropy regularization
        entropy = policy_dist.entropy().mean()
        entropy_loss = -self.config['entropy_coef'] * entropy
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss
        }
    
    def update(self, batch):
        """Perform actor-critic update."""
        losses = self.compute_losses(batch)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss = losses['policy_loss'] + losses['entropy_loss']
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        self.policy_optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        losses['value_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.config['max_grad_norm'])
        self.value_optimizer.step()
        
        # Update schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        return {
            'policy_loss': losses['policy_loss'].item(),
            'value_loss': losses['value_loss'].item(),
            'entropy_loss': losses['entropy_loss'].item(),
            'policy_lr': self.policy_scheduler.get_last_lr()[0],
            'value_lr': self.value_scheduler.get_last_lr()[0]
        }
```

### Off-Policy Methods

Implement off-policy algorithms:

```python
class CustomOffPolicy(BaseAlgorithm):
    """Custom off-policy algorithm implementation."""
    
    def compute_loss(self, batch):
        """Compute off-policy loss."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Current Q-values
        current_q_values = self.q_function(states, actions)
        
        # Next Q-values (with target network)
        with torch.no_grad():
            next_actions = self.policy(next_states)
            next_q_values = self.target_q_function(next_states, next_actions)
            target_q_values = rewards + self.config['gamma'] * next_q_values * (1 - dones)
        
        # Q-function loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Policy loss (for continuous actions)
        policy_actions = self.policy(states)
        policy_q_values = self.q_function(states, policy_actions)
        policy_loss = -policy_q_values.mean()
        
        return {
            'q_loss': q_loss,
            'policy_loss': policy_loss
        }
    
    def update(self, batch):
        """Perform off-policy update."""
        losses = self.compute_loss(batch)
        
        # Update Q-function
        self.q_optimizer.zero_grad()
        losses['q_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.q_function.parameters(), self.config['max_grad_norm'])
        self.q_optimizer.step()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        losses['policy_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        self.policy_optimizer.step()
        
        # Update target network
        if self.update_step % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        self.update_step += 1
        
        return {
            'q_loss': losses['q_loss'].item(),
            'policy_loss': losses['policy_loss'].item()
        }
    
    def update_target_network(self):
        """Update target network using soft update."""
        tau = self.config['tau']
        for target_param, param in zip(self.target_q_function.parameters(), self.q_function.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## Integration with NeMo RL

### Algorithm Registration

Register custom algorithms with the framework:

```python
from nemo_rl.algorithms import register_algorithm

class CustomAlgorithm(BaseAlgorithm):
    """Custom algorithm implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        # Implementation details...
    
    @classmethod
    def get_config_schema(cls):
        """Return configuration schema for the algorithm."""
        return {
            'type': 'object',
            'properties': {
                'learning_rate': {'type': 'number', 'minimum': 0},
                'batch_size': {'type': 'integer', 'minimum': 1},
                'gamma': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'entropy_coef': {'type': 'number', 'minimum': 0},
                'max_grad_norm': {'type': 'number', 'minimum': 0}
            },
            'required': ['learning_rate', 'batch_size', 'gamma']
        }

# Register the algorithm
register_algorithm('custom_algorithm', CustomAlgorithm)
```

### Configuration Management

Define algorithm-specific configurations:

```python
def get_custom_algorithm_config():
    """Get configuration for custom algorithm."""
    return {
        'algorithm': {
            'type': 'custom_algorithm',
            'learning_rate': 1e-4,
            'batch_size': 64,
            'gamma': 0.99,
            'entropy_coef': 0.01,
            'max_grad_norm': 1.0,
            'target_update_freq': 1000,
            'tau': 0.005
        },
        'policy': {
            'type': 'mlp',
            'hidden_sizes': [256, 256],
            'activation': 'relu'
        },
        'value_function': {
            'type': 'mlp',
            'hidden_sizes': [256, 256],
            'activation': 'relu'
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01
        },
        'buffer': {
            'type': 'replay_buffer',
            'capacity': 100000
        }
    }
```

### Training Integration

Integrate with NeMo RL training loop:

```python
from nemo_rl.training import Trainer

def train_custom_algorithm():
    """Train custom algorithm using NeMo RL framework."""
    
    # Create algorithm
    config = get_custom_algorithm_config()
    algorithm = CustomAlgorithm(config['algorithm'])
    
    # Create trainer
    trainer = Trainer(
        algorithm=algorithm,
        config=config,
        data_loader=create_data_loader(config),
        logger=create_logger(config)
    )
    
    # Train
    trainer.train(
        num_epochs=1000,
        save_checkpoint_every=100,
        evaluate_every=50
    )
    
    return trainer
```

## Advanced Features

### Custom Loss Functions

Implement custom loss functions:

```python
class CustomLossFunction:
    """Custom loss function implementation."""
    
    def __init__(self, config):
        self.config = config
    
    def compute_loss(self, predictions, targets, weights=None):
        """Compute custom loss."""
        # Base loss
        base_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Apply weights if provided
        if weights is not None:
            base_loss = base_loss * weights
        
        # Custom loss modifications
        if self.config.get('huber_loss', False):
            delta = self.config['huber_delta']
            base_loss = torch.where(
                base_loss < delta,
                0.5 * base_loss,
                delta * (base_loss - 0.5 * delta)
            )
        
        return base_loss.mean()
```

### Custom Networks

Implement custom network architectures:

```python
class CustomNetwork(torch.nn.Module):
    """Custom neural network architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_size = config['input_size']
        
        for hidden_size in config['hidden_sizes']:
            layers.extend([
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.get('dropout_rate', 0.0))
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(torch.nn.Linear(input_size, config['output_size']))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through network."""
        return self.network(x)
    
    def get_features(self, x):
        """Extract intermediate features."""
        features = []
        for layer in self.network[:-1]:  # Exclude output layer
            x = layer(x)
            features.append(x)
        return features
```

### Custom Metrics

Implement custom evaluation metrics:

```python
class CustomMetrics:
    """Custom evaluation metrics."""
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.metrics = {}
    
    def update(self, batch, predictions):
        """Update metrics with new batch."""
        # Compute various metrics
        self.update_accuracy(batch, predictions)
        self.update_loss(batch, predictions)
        self.update_custom_metric(batch, predictions)
    
    def update_accuracy(self, batch, predictions):
        """Update accuracy metric."""
        targets = batch['targets']
        correct = (predictions.argmax(dim=-1) == targets).float()
        accuracy = correct.mean()
        
        if 'accuracy' not in self.metrics:
            self.metrics['accuracy'] = []
        self.metrics['accuracy'].append(accuracy.item())
    
    def update_loss(self, batch, predictions):
        """Update loss metric."""
        targets = batch['targets']
        loss = F.cross_entropy(predictions, targets)
        
        if 'loss' not in self.metrics:
            self.metrics['loss'] = []
        self.metrics['loss'].append(loss.item())
    
    def update_custom_metric(self, batch, predictions):
        """Update custom metric."""
        # Implement custom metric computation
        custom_value = self.compute_custom_metric(batch, predictions)
        
        if 'custom_metric' not in self.metrics:
            self.metrics['custom_metric'] = []
        self.metrics['custom_metric'].append(custom_value)
    
    def compute_custom_metric(self, batch, predictions):
        """Compute custom metric."""
        # Example custom metric
        return predictions.std().item()
    
    def get_metrics(self):
        """Get current metric values."""
        return {
            name: np.mean(values) if values else 0.0
            for name, values in self.metrics.items()
        }
```

## Testing and Validation

### Unit Tests

Write comprehensive unit tests:

```python
import unittest
import torch
import numpy as np

class TestCustomAlgorithm(unittest.TestCase):
    """Unit tests for custom algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'gamma': 0.99,
            'entropy_coef': 0.01,
            'max_grad_norm': 1.0
        }
        self.algorithm = CustomAlgorithm(self.config)
    
    def test_initialization(self):
        """Test algorithm initialization."""
        self.assertIsNotNone(self.algorithm.policy)
        self.assertIsNotNone(self.algorithm.optimizer)
        self.assertEqual(self.algorithm.device, torch.device('cuda'))
    
    def test_loss_computation(self):
        """Test loss computation."""
        batch = self.create_test_batch()
        loss = self.algorithm.compute_loss(batch)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)
    
    def test_update_step(self):
        """Test update step."""
        batch = self.create_test_batch()
        metrics = self.algorithm.update(batch)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertGreater(metrics['loss'], 0)
    
    def create_test_batch(self):
        """Create test batch."""
        batch_size = 32
        state_dim = 10
        action_dim = 4
        
        return {
            'states': torch.randn(batch_size, state_dim),
            'actions': torch.randint(0, action_dim, (batch_size,)),
            'rewards': torch.randn(batch_size),
            'advantages': torch.randn(batch_size)
        }
```

### Integration Tests

Test algorithm integration:

```python
def test_algorithm_integration():
    """Test algorithm integration with NeMo RL."""
    
    # Create configuration
    config = get_custom_algorithm_config()
    
    # Create algorithm
    algorithm = CustomAlgorithm(config['algorithm'])
    
    # Create data loader
    data_loader = create_test_data_loader(config)
    
    # Training loop
    for epoch in range(10):
        for batch in data_loader:
            # Update algorithm
            metrics = algorithm.update(batch)
            
            # Check metrics
            assert 'loss' in metrics
            assert metrics['loss'] > 0
            
            # Check model parameters changed
            old_params = [p.clone() for p in algorithm.policy.parameters()]
            algorithm.update(batch)
            new_params = [p.clone() for p in algorithm.policy.parameters()]
            
            # Parameters should have changed
            for old_p, new_p in zip(old_params, new_params):
                assert not torch.allclose(old_p, new_p)
    
    print("Integration test passed!")
```

## Best Practices

### Algorithm Design

1. **Modularity**
   - Separate concerns (policy, value function, etc.)
   - Use clear interfaces
   - Enable easy testing

2. **Efficiency**
   - Minimize computational overhead
   - Use vectorized operations
   - Profile performance

3. **Robustness**
   - Handle edge cases
   - Validate inputs
   - Provide meaningful error messages

### Implementation Guidelines

1. **Code Organization**
   - Follow NeMo RL conventions
   - Use consistent naming
   - Document complex logic

2. **Configuration Management**
   - Use structured configurations
   - Validate parameters
   - Provide sensible defaults

3. **Testing Strategy**
   - Write unit tests
   - Test edge cases
   - Validate numerical stability

## Next Steps

After implementing custom algorithms:

1. **Validate Implementation**: Test against known baselines
2. **Optimize Performance**: Profile and optimize bottlenecks
3. **Document Usage**: Create clear documentation and examples
4. **Contribute**: Share useful algorithms with the community

For more advanced topics, see:
- [Performance Analysis](performance-analysis.md) - Analyzing algorithm performance
- [Research Methodologies](index.md) - Research best practices
- [Hyperparameter Optimization](hyperparameter-optimization.md) - Optimizing algorithm parameters 