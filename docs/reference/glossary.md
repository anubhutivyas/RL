# Glossary

This glossary defines key terms and concepts used throughout the NeMo RL documentation.

## A

### Actor
A component in the RL system that performs a specific function (e.g., policy model, environment, reward computation). See [RL Actors](../about/architecture.md#rl-actors).

### Alignment
The process of making a language model's behavior align with human preferences and values. NeMo RL supports alignment through algorithms like DPO and GRPO.

## B

### Backend
The underlying training framework used by NeMo RL. Supports Hugging Face, Megatron, and DTensor backends.

### Batch Size
The number of training examples processed together in a single forward/backward pass. Includes micro batch size and gradient accumulation steps.

## C

### Checkpoint
A saved state of the model and optimizer that can be used to resume training or for inference.

### Context Length
The maximum number of tokens that can be processed in a single forward pass. Important for memory usage and training efficiency.

### Controller
The central process that coordinates all RL Actors in the NeMo RL system.

## D

### DPO (Direct Preference Optimization)
A RL-free alignment algorithm that operates on preference data to improve model behavior.

### DTensor
PyTorch's next-generation distributed training framework, providing improved memory efficiency and performance.

## E

### Environment
In RL, the system that provides feedback (rewards) to the agent based on its actions. In NeMo RL, environments can be mathematical problems, games, or other tasks.

### Epoch
A complete pass through the training dataset. One epoch processes all training examples once.

## F

### FSDP (Fully Sharded Data Parallel)
A distributed training technique that shards model parameters across multiple GPUs to reduce memory usage.

### Forward Pass
The computation of model outputs given inputs, without computing gradients.

## G

### Generation
The process of producing text outputs from a language model given input prompts.

### Gradient Accumulation
A technique to simulate larger batch sizes by accumulating gradients over multiple forward passes before updating model parameters.

### GRPO (Group Relative Policy Optimization)
An advanced RL algorithm for preference learning that groups similar examples for more stable training.

## H

### Hugging Face Backend
The default training backend in NeMo RL, supporting models up to 32B parameters with easy integration.

### Hyperparameters
Configurable parameters that control the training process (e.g., learning rate, batch size, number of epochs).

## I

### Inference
The process of using a trained model to generate predictions or responses.

### IPC (Inter-Process Communication)
The mechanism used by NeMo RL to share data and model weights between different processes.

## L

### Learning Rate
The step size used in gradient descent to update model parameters. Critical for training stability and convergence.

### Loss Function
A function that measures how well the model is performing. NeMo RL uses specially designed loss functions for correct gradient accumulation.

## M

### Megatron Backend
NVIDIA's high-performance training framework for scaling to large models (>100B parameters).

### Micro Batch Size
The number of examples processed in a single forward pass before gradient accumulation.

### Mixed Precision
Training technique that uses both FP16 and FP32 precision to speed up training and reduce memory usage.

## N

### NCCL (NVIDIA Collective Communications Library)
High-performance communication library used for GPU-to-GPU communication in distributed training.

### Normalization Factor
A value used in loss functions to ensure correct gradient accumulation across microbatches.

## O

### Optimizer
Algorithm used to update model parameters based on computed gradients (e.g., Adam, AdamW).

## P

### Padding
Adding special tokens to sequences to make them the same length. NeMo RL uses right padding for LLM compatibility.

### Pipeline Parallelism
Distributing model layers across multiple GPUs to handle very large models.

### Policy
The model being trained in RL, which learns to map inputs to actions (text generation in NeMo RL).

### Preference Data
Data consisting of pairs of responses where one is preferred over the other, used in alignment algorithms.

## R

### Ray
Distributed computing framework used by NeMo RL for resource management and coordination.

### Reinforcement Learning (RL)
A machine learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards.

### Reward
A numerical signal that indicates how well the agent is performing. Used to guide RL training.

### Right Padding
Padding strategy where padding tokens are added to the end of sequences, used consistently in NeMo RL.

## S

### SFT (Supervised Fine-Tuning)
Traditional supervised learning approach where the model learns from labeled examples.

### Sharding
Distributing model parameters or data across multiple devices for parallel processing.

### Slurm
Cluster management system commonly used for HPC environments.

### Step
A single forward and backward pass through the model, including parameter updates.

## T

### Tensor Parallelism
Distributing model layers across multiple GPUs to handle large models.

### Token
The basic unit of text processing in language models. Tokens can represent words, subwords, or characters.

### Training Backend
The underlying framework used for model training (Hugging Face, Megatron, or DTensor).

## V

### Value Function
In RL, a function that estimates the expected future reward for a given state or action.

### vLLM
High-performance inference engine used by NeMo RL for fast text generation.

## W

### WandB (Weights & Biases)
Popular experiment tracking platform integrated with NeMo RL for monitoring training progress.

### Worker
A process that performs specific computations in the distributed NeMo RL system.

### Worker Group
A collection of workers that perform similar tasks, managed by NeMo RL for efficient resource allocation.

## Y

### YAML Configuration
Human-readable configuration format used by NeMo RL for specifying training parameters and settings. 