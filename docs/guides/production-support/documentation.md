# Documentation Guide

This guide covers how to write and maintain documentation for NeMo RL projects.

## Overview

Good documentation is essential for any software project. This guide provides standards and best practices for writing clear, comprehensive, and maintainable documentation for NeMo RL.

## Documentation Structure

### Core Documentation

The main documentation is organized into these sections:

1. **Getting Started** - Quick start guides and installation
2. **Guides** - Tutorials and how-to guides
3. **API Reference** - Complete API documentation
4. **Advanced Topics** - Deep dives into specific features
5. **Examples** - Working examples and use cases

### Documentation Types

#### User Documentation

- **Installation Guides** - How to install and set up NeMo RL
- **Quick Start** - Get up and running quickly
- **Tutorials** - Step-by-step guides for common tasks
- **How-to Guides** - Solutions to specific problems

#### Developer Documentation

- **API Reference** - Complete API documentation
- **Architecture** - System design and architecture
- **Contributing** - How to contribute to the project
- **Development Setup** - Setting up development environment

#### Reference Documentation

- **Configuration Reference** - All configuration options
- **CLI Reference** - Command-line interface documentation
- **Examples** - Code examples and use cases

## Writing Guidelines

### Style and Tone

1. **Clear and Concise**
   - Use simple, direct language
   - Avoid jargon when possible
   - Write for your audience's level

2. **Consistent Formatting**
   - Use consistent heading levels
   - Follow the established style guide
   - Use proper markdown formatting

3. **Action-Oriented**
   - Focus on what users can do
   - Provide practical examples
   - Include step-by-step instructions

### Content Structure

#### Introduction

Start each document with a brief introduction:

```markdown
# Document Title

Brief description of what this document covers and why it's important.

## Overview

More detailed explanation of the topic and its context within NeMo RL.
```

#### Main Content

Organize content logically:

1. **Prerequisites** - What users need to know before starting
2. **Concepts** - Explain key concepts and terminology
3. **Examples** - Provide working examples
4. **Best Practices** - Share recommended approaches
5. **Troubleshooting** - Common issues and solutions

#### Conclusion

End with next steps or related resources:

```markdown
## Next Steps

- [Related Topic 1](link1) - Learn about related concepts
- [Related Topic 2](link2) - Explore advanced features
- [Examples](../examples/) - See more examples
```

### Code Examples

#### Python Code

```python
# Good example
from nemo_rl.algorithms import DPOTrainer
from nemo_rl.utils.config import load_config

# Load configuration
config = load_config("configs/dpo.yaml")

# Initialize trainer
trainer = DPOTrainer.from_config(config)

# Start training
trainer.train()
```

#### Configuration Examples

```yaml
# Example configuration
algorithm:
  name: "dpo"
  beta: 0.1
  loss_type: "sigmoid"

training:
  batch_size: 4
  learning_rate: 5e-5
  max_steps: 1000

model:
  name: "llama2-7b"
  max_length: 2048
```

#### Command Line Examples

```bash
# Basic training
python -m nemo_rl.train --config configs/dpo.yaml

# With overrides
python -m nemo_rl.train --config configs/dpo.yaml \
  --training.batch_size 8 \
  --training.learning_rate 1e-4
```

### Cross-References

Use proper cross-references to link related content:

```markdown
# Internal links
See the [Configuration Reference](../configuration-cli/configuration-reference.md) for details.

# API references
Use the [DPOTrainer](../../api-docs/nemo_rl.algorithms.dpo.md) class.

# External links
Visit the [NeMo RL repository](https://github.com/your-repo/nemo-rl).
```

## Documentation Tools

### MyST Markdown

NeMo RL uses MyST Markdown for enhanced documentation features:

#### Directives

```markdown
```{note}
This is a note with important information.
```

```{warning}
This is a warning about potential issues.
```

```{tip}
This is a helpful tip for users.
```
```

#### Grid Layouts

```markdown
```{grid} 1 2
:gutter: 2 2 2 2

:::{grid-item}
# Column 1
Content for the first column.
:::

:::{grid-item}
# Column 2
Content for the second column.
:::

::::
```

#### Tabs

```markdown
```{tab} Python
```python
# Python code here
```
```

```{tab} YAML
```yaml
# YAML configuration here
```
```
```

### Sphinx Extensions

#### Code Highlighting

```markdown
```python
# Python code with syntax highlighting
def train_model(config):
    trainer = DPOTrainer.from_config(config)
    trainer.train()
```
```

#### Math Equations

```markdown
The DPO loss function is defined as:

$$
L_{DPO} = -\mathbb{E}_{(x,y_w,y_l)\sim D} \left[\log\sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$
```

#### Admonitions

```markdown
```{admonition} Important
This is important information that users should pay attention to.
```

```{admonition} Note
This is additional information that might be helpful.
```
```

## API Documentation

### Docstring Standards

Follow consistent docstring formatting:

```python
def train_model(config: Dict[str, Any], 
                checkpoint_path: Optional[str] = None) -> Trainer:
    """
    Train a model using the specified configuration.
    
    Args:
        config: Training configuration dictionary
        checkpoint_path: Optional path to resume from checkpoint
        
    Returns:
        Trainer: The trained model trainer
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
        
    Example:
        >>> config = load_config("configs/dpo.yaml")
        >>> trainer = train_model(config)
        >>> trainer.save("checkpoints/model.pt")
    """
    pass
```

### Type Hints

Use type hints consistently:

```python
from typing import Dict, List, Optional, Union
import torch

def process_data(
    data: List[Dict[str, Union[str, int]]],
    tokenizer: Optional[object] = None
) -> torch.Tensor:
    """Process input data for training."""
    pass
```

### Parameter Documentation

Document all parameters clearly:

```python
def create_model(
    model_name: str,
    max_length: int = 2048,
    trust_remote_code: bool = False,
    **kwargs
) -> torch.nn.Module:
    """
    Create a model for training.
    
    Args:
        model_name: Name or path of the model to load
        max_length: Maximum sequence length for the model
        trust_remote_code: Whether to trust custom model code
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        The initialized model
    """
    pass
```

## Maintenance

### Keeping Documentation Current

1. **Update with Code Changes**
   - Update documentation when APIs change
   - Add examples for new features
   - Remove references to deprecated functionality

2. **Regular Reviews**
   - Schedule regular documentation reviews
   - Check for broken links
   - Verify examples still work

3. **User Feedback**
   - Collect feedback from users
   - Address common questions in documentation
   - Improve unclear sections

### Documentation Workflow

#### For New Features

1. **Plan Documentation**
   - Identify what needs to be documented
   - Determine the appropriate location
   - Plan the structure and content

2. **Write Documentation**
   - Write clear, comprehensive documentation
   - Include examples and use cases
   - Add cross-references to related content

3. **Review and Test**
   - Have others review the documentation
   - Test all code examples
   - Verify links and references

4. **Publish and Maintain**
   - Publish with the feature
   - Monitor for issues
   - Update based on feedback

#### For Existing Documentation

1. **Regular Audits**
   - Check for outdated information
   - Verify all links work
   - Update examples as needed

2. **User-Driven Updates**
   - Address user questions
   - Add missing information
   - Clarify confusing sections

## Best Practices

### Writing Clear Documentation

1. **Start with the Problem**
   - Explain what problem the feature solves
   - Provide context for why it's needed

2. **Show, Don't Just Tell**
   - Include working examples
   - Provide before/after comparisons
   - Use diagrams when helpful

3. **Be Consistent**
   - Use consistent terminology
   - Follow established patterns
   - Maintain consistent formatting

### Making Documentation Accessible

1. **Use Clear Language**
   - Avoid unnecessary jargon
   - Define technical terms
   - Use simple, direct sentences

2. **Provide Multiple Entry Points**
   - Quick start for beginners
   - Detailed guides for advanced users
   - Reference material for experts

3. **Include Visual Aids**
   - Screenshots for UI elements
   - Diagrams for complex concepts
   - Code examples with output

### Testing Documentation

1. **Test Code Examples**
   ```bash
   # Test all code examples
   python -m pytest docs/examples/
   ```

2. **Validate Links**
   ```bash
   # Check for broken links
   python -m linkcheck docs/
   ```

3. **Build Documentation**
   ```bash
   # Build and check for errors
   make docs
   ```

## Documentation Standards

### File Naming

- Use lowercase with hyphens: `getting-started.md`
- Be descriptive but concise
- Group related files in directories

### Headers

- Use title case for headers
- Keep headers concise and descriptive
- Use consistent header levels

### Code Blocks

- Specify language for syntax highlighting
- Include complete, runnable examples
- Show expected output when helpful

### Links

- Use relative links for internal references
- Use descriptive link text
- Verify all links work

## Tools and Resources

### Documentation Tools

1. **MyST Markdown** - Enhanced markdown for Sphinx
2. **Sphinx** - Documentation generator
3. **GitHub Pages** - Hosting documentation
4. **Read the Docs** - Documentation hosting

### Useful Extensions

1. **myst-parser** - MyST markdown parser
2. **sphinx-autodoc** - Auto-generate API docs
3. **sphinx-copybutton** - Copy code blocks
4. **sphinx-design** - Enhanced design elements

### Quality Assurance

1. **Spell Checkers** - Check for typos
2. **Link Checkers** - Verify all links work
3. **Linters** - Check markdown formatting
4. **Build Tests** - Ensure documentation builds

For more information about documentation tools and standards, see the [Sphinx Documentation](https://www.sphinx-doc.org/) and [MyST Markdown Guide](https://myst-parser.readthedocs.io/). 