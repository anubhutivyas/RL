# Create Custom RL Environments

This guide covers how to create custom reinforcement learning environments for NeMo RL, from basic interfaces to advanced features.

## Overview

NeMo RL provides a flexible environment system that allows you to create custom environments for specific tasks. Whether you're working on mathematical problem solving, game playing, or domain-specific applications, this guide will help you build environments that integrate seamlessly with NeMo RL's training pipeline.

## Environment Interface

All NeMo RL environments must implement the `EnvironmentInterface` class:

```python
from nemo_rl.environments.interfaces import EnvironmentInterface
from typing import Dict, Any, Tuple, Optional
import torch

class CustomEnvironment(EnvironmentInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your environment state
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        # Return initial observation
        return {"observation": initial_state}
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Process action, update state, calculate reward
        return observation, reward, done, info
    
    def get_action_space(self) -> Any:
        """Return the action space definition."""
        pass
    
    def get_observation_space(self) -> Any:
        """Return the observation space definition."""
        pass
```

## Basic Environment Example

Here's a simple example of a custom environment for a text-based task:

```python
import random
from typing import Dict, Any, Tuple
from nemo_rl.environments.interfaces import EnvironmentInterface

class SimpleMathEnvironment(EnvironmentInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_steps = config.get("max_steps", 10)
        self.current_step = 0
        self.score = 0
        self.generate_problem()
    
    def generate_problem(self):
        """Generate a new math problem."""
        self.a = random.randint(1, 100)
        self.b = random.randint(1, 100)
        self.operation = random.choice(['+', '-', '*'])
        
        if self.operation == '+':
            self.correct_answer = self.a + self.b
        elif self.operation == '-':
            self.correct_answer = self.a - self.b
        else:
            self.correct_answer = self.a * self.b
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.score = 0
        self.generate_problem()
        
        return {
            "observation": f"What is {self.a} {self.operation} {self.b}?",
            "problem": {
                "a": self.a,
                "b": self.b,
                "operation": self.operation,
                "correct_answer": self.correct_answer
            }
        }
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process the agent's action."""
        self.current_step += 1
        
        # Parse the action (assuming it's a string with the answer)
        try:
            answer = int(action.strip())
            correct = (answer == self.correct_answer)
        except ValueError:
            correct = False
        
        # Calculate reward
        reward = 1.0 if correct else -0.1
        
        # Update score
        if correct:
            self.score += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Generate new problem if correct or episode done
        if correct or done:
            self.generate_problem()
        
        observation = {
            "observation": f"What is {self.a} {self.operation} {self.b}?",
            "previous_correct": correct,
            "score": self.score,
            "step": self.current_step
        }
        
        info = {
            "correct": correct,
            "expected": self.correct_answer,
            "received": action,
            "score": self.score
        }
        
        return observation, reward, done, info
    
    def get_action_space(self):
        """Define the action space."""
        # For text-based environments, actions are strings
        return "text"
    
    def get_observation_space(self):
        """Define the observation space."""
        # For text-based environments, observations are strings
        return "text"
```

## Getting Help

- [Environment Interfaces](../reference/api.md#environments) - Complete API documentation
- [Training Configuration](../reference/configuration.md) - Environment configuration options
- [Troubleshooting](../reference/troubleshooting.md) - Common environment issues
- [Community Support](https://github.com/NVIDIA-NeMo/RL/issues) - GitHub issues and discussions 