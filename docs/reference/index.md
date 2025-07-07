---
author: lawrence lane
description: "Access comprehensive reference documentation including API specifications, configuration options, and technical details."
tags: ["reference", "api", "configuration", "specifications"]
categories: ["reference", "onboarding"]
---

# Reference

Welcome to the NeMo RL Reference section! This comprehensive reference provides detailed documentation for all APIs, configuration options, and technical details.

## What You'll Find Here

Our reference documentation is organized into four main categories to help you find the information you need:

### 📚 **API Documentation**
Complete API reference for all NeMo RL components, including algorithms, data processing, models, and utilities. Each API is documented with parameters, return values, and usage examples.

### ⚙️ **Configuration Guide**
Detailed configuration options for all training parameters, model settings, and system configurations. Learn how to customize your training runs and optimize performance.

### 🖥️ **Command Line Interface**
Complete CLI reference for all NeMo RL commands, including training scripts, evaluation tools, and utility functions. Each command includes examples and parameter descriptions.

### 🔧 **Troubleshooting & Support**
Common issues, error messages, and solutions for NeMo RL. Find quick answers to frequently encountered problems and learn debugging strategies.

## API Documentation

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="book" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> API Reference
:link: api
:link-type: doc

Complete API documentation for all NeMo RL components and functions.

+++
{bdg-primary}`Reference`
:::

::::

## Configuration Guide

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="gear" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Configuration
:link: configuration
:link-type: doc

Detailed configuration options for training parameters and system settings.

+++
{bdg-info}`Setup`
:::

::::

## Command Line Interface

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="terminal" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> CLI Reference
:link: cli
:link-type: doc

Complete command-line interface reference for all NeMo RL tools.

+++
{bdg-secondary}`Tools`
:::

::::

## Troubleshooting & Support

::::{grid} 1 1 1 2
:gutter: 2 2 2 2

:::{grid-item-card} <span class="octicon" data-icon="bug" data-size="1.5em" style="font-size: 1.5em; margin-right: 0.25em;"></span> Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues, error messages, and solutions for NeMo RL.

+++
{bdg-warning}`Support`
:::

::::

## Quick Reference

### Essential Commands
- **Training**: `python -m nemo_rl.algorithms.grpo` - Start GRPO training
- **Evaluation**: `python -m nemo_rl.evals.eval` - Evaluate trained models
- **Configuration**: `python -m nemo_rl.utils.config` - Validate configuration files

### Key Configuration Files
- **Training Config**: YAML files defining model, data, and training parameters
- **Environment Config**: Ray cluster and distributed training settings
- **Model Config**: Model architecture and tokenizer specifications

### Common Parameters
- **Model**: Model name, size, and architecture settings
- **Data**: Dataset paths, preprocessing, and batch configurations
- **Training**: Learning rates, optimization, and scheduling parameters
- **Hardware**: GPU configuration, memory settings, and distributed training

## Getting Help

- **API Documentation**: Complete reference for all functions and classes
- **Configuration Guide**: Detailed parameter descriptions and examples
- **CLI Reference**: Command-line tool documentation and usage
- **Troubleshooting**: Common issues and debugging strategies
- **Community Support**: GitHub issues and community discussions

```{toctree}
:caption: Reference
:maxdepth: 2
:expanded:

api
configuration
cli
troubleshooting
```


