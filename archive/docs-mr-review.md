# NeMo RL: Documentation Files for GitHub MR Review

This document organizes all files in the `/docs` directory by logical groupings for efficient GitHub MR assignments and developer review.

## Table of Contents

- [MR Group 1: Core Setup & Installation](#mr-group-1-core-setup--installation)
- [MR Group 2: Training Algorithms Core](#mr-group-2-training-algorithms-core)
- [MR Group 3: Tutorials & Learning Paths](#mr-group-3-tutorials--learning-paths)
- [MR Group 4: Model Development & Integration](#mr-group-4-model-development--integration)
- [MR Group 5: Production Support & Operations](#mr-group-5-production-support--operations)
- [MR Group 6: Core Architecture & Design](#mr-group-6-core-architecture--design)
- [MR Group 7: Performance & Advanced Optimization](#mr-group-7-performance--advanced-optimization)
- [MR Group 8: Research & Theory](#mr-group-8-research--theory)
- [MR Group 9: Advanced Training Techniques](#mr-group-9-advanced-training-techniques)
- [MR Group 10: API Documentation](#mr-group-10-api-documentation)
- [MR Group 11: Project Overview & About](#mr-group-11-project-overview--about)
- [Total Count Summary](#total-count-summary)

---

## MR Group 1: Core Setup & Installation
**Focus:** Essential installation guides, environment setup, and first-time user onboarding
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/index.md` | Main documentation landing page and navigation hub |
| `docs/get-started/index.md` | Get started landing page with setup overview |
| `docs/get-started/installation.md` | Step-by-step installation guide for NeMo RL |
| `docs/get-started/quickstart.md` | Quick start tutorial for first-time users |
| `docs/get-started/local-workstation.md` | Local development environment setup guide |
| `docs/get-started/cluster.md` | Multi-node cluster configuration and setup |
| `docs/get-started/docker.md` | Containerized deployment with Docker |

---

## MR Group 2: Training Algorithms Core
**Focus:** Core RLHF algorithms (SFT, DPO, GRPO) implementation guides and evaluation methods
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/guides/training-algorithms/index.md` | Overview of all supported RLHF training algorithms |
| `docs/guides/training-algorithms/sft.md` | Supervised Fine-Tuning implementation and configuration |
| `docs/guides/training-algorithms/dpo.md` | Direct Preference Optimization algorithm guide |
| `docs/guides/training-algorithms/grpo.md` | Group Relative Policy Optimization implementation |
| `docs/guides/training-algorithms/eval.md` | Model evaluation metrics and assessment methods |

---

## MR Group 3: Tutorials & Learning Paths
**Focus:** Hands-on tutorials, practical examples, and real-world use cases for RLHF training
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/tutorials-examples/index.md` | Tutorials and examples landing page |
| `docs/tutorials-examples/tutorials/index.md` | Complete tutorials overview and navigation |
| `docs/tutorials-examples/tutorials/sft-tutorial.md` | Step-by-step SFT training tutorial |
| `docs/tutorials-examples/tutorials/dpo-tutorial.md` | Complete DPO training walkthrough |
| `docs/tutorials-examples/tutorials/grpo-tutorial.md` | GRPO algorithm implementation tutorial |
| `docs/tutorials-examples/tutorials/evaluation-tutorial.md` | Model evaluation and assessment tutorial |
| `docs/tutorials-examples/examples/index.md` | Real-world examples and case studies |
| `docs/tutorials-examples/examples/sft-openmathinstruct2.md` | SFT training on OpenMathInstruct dataset |
| `docs/tutorials-examples/examples/grpo-deepscaler.md` | GRPO training on DeepScaler model |
| `docs/tutorials-examples/use-cases/index.md` | Practical use cases and applications |
| `docs/tutorials-examples/use-cases/mathematical-reasoning.md` | Mathematical reasoning RLHF application |
| `docs/tutorials-examples/use-cases/code-generation.md` | Code generation with RLHF training |

---

## MR Group 4: Model Development & Integration
**Focus:** Custom model integration, environment development, debugging tools, and performance profiling
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/guides/model-development/index.md` | Model development and integration overview |
| `docs/guides/model-development/adding-new-models.md` | Guide for integrating custom models into NeMo RL |
| `docs/guides/model-development/model-quirks.md` | Known model-specific behaviors and workarounds |
| `docs/guides/environment-data/index.md` | Environment and data management overview |
| `docs/guides/environment-data/environment-development.md` | Custom environment development guide |
| `docs/guides/environment-data/debugging.md` | Environment debugging and troubleshooting |
| `docs/guides/environment-data/nsys-profiling.md` | Performance profiling with NVIDIA NSight Systems |

---

## MR Group 5: Production Support & Operations
**Focus:** Production deployment workflows, testing strategies, CLI tools, and operational troubleshooting
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/guides/production-support/index.md` | Production deployment and support overview |
| `docs/guides/production-support/testing.md` | Testing strategies and best practices |
| `docs/guides/production-support/debugging.md` | Production debugging and monitoring |
| `docs/guides/production-support/documentation.md` | Documentation standards and guidelines |
| `docs/guides/production-support/packaging.md` | Model packaging and distribution |
| `docs/guides/production-support/troubleshooting.md` | Common issues and solutions |
| `docs/configuration-cli/index.md` | Configuration management and CLI tools |
| `docs/configuration-cli/cli-reference.md` | Complete CLI command reference |
| `docs/configuration-cli/configuration-reference.md` | Configuration file format and options |
| `docs/configuration-cli/troubleshooting.md` | CLI tool troubleshooting and debugging |

---

## MR Group 6: Core Architecture & Design
**Focus:** System design philosophy, data management, computational backends, and development infrastructure
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/design-docs/index.md` | System design documentation landing page |
| `docs/design-docs/core-architecture/index.md` | Core system architecture and components |
| `docs/design-docs/core-architecture/design-and-philosophy.md` | System design principles and philosophy |
| `docs/design-docs/core-architecture/generation.md` | Text generation architecture and flow |
| `docs/design-docs/core-architecture/fsdp2-parallel-plan.md` | FSDP2 distributed training architecture |
| `docs/design-docs/data-management/index.md` | Data processing and management systems |
| `docs/design-docs/data-management/padding.md` | Data padding strategies and implementation |
| `docs/design-docs/data-management/chat-datasets.md` | Chat dataset processing and formatting |
| `docs/design-docs/computational-systems/index.md` | Computational infrastructure overview |
| `docs/design-docs/computational-systems/training-backends.md` | Training backend systems and engines |
| `docs/design-docs/computational-systems/uv.md` | UV package management system |
| `docs/design-docs/computational-systems/logger.md` | Logging and monitoring infrastructure |
| `docs/design-docs/development-infrastructure/index.md` | Development tools and infrastructure |
| `docs/design-docs/development-infrastructure/loss-functions.md` | Loss function implementations and design |
| `docs/design-docs/development-infrastructure/checkpointing.md` | Model checkpointing and recovery |

---

## MR Group 7: Performance & Advanced Optimization
**Focus:** Distributed training, memory optimization, profiling tools, and performance benchmarking
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/advanced/index.md` | Advanced topics and optimization landing page |
| `docs/advanced/performance/index.md` | Performance optimization overview |
| `docs/advanced/performance/distributed-training.md` | Multi-GPU and multi-node training |
| `docs/advanced/performance/profiling.md` | Performance profiling and analysis tools |
| `docs/advanced/performance/monitoring.md` | Real-time performance monitoring |
| `docs/advanced/performance/memory-optimization.md` | Memory usage optimization strategies |
| `docs/advanced/performance/benchmarking.md` | Performance benchmarking and comparison |
| `docs/advanced/performance/mixed-precision.md` | Mixed precision training for efficiency |

---

## MR Group 8: Research & Theory
**Focus:** Reproducible research practices, experimental design, theoretical foundations, and algorithm analysis
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/advanced/research/index.md` | Research methodologies and practices overview |
| `docs/advanced/research/reproducibility.md` | Reproducible research practices and standards |
| `docs/advanced/research/performance-analysis.md` | Performance analysis methodologies |
| `docs/advanced/research/hyperparameter-optimization.md` | Hyperparameter tuning and optimization |
| `docs/advanced/research/experimental-design.md` | Experimental design and methodology |
| `docs/advanced/research/ablation-studies.md` | Ablation study design and analysis |
| `docs/advanced/research/custom-algorithms.md` | Custom algorithm development and testing |
| `docs/advanced/theory/index.md` | Theoretical foundations and concepts |
| `docs/advanced/theory/sft-theory.md` | Supervised Fine-Tuning theoretical background |
| `docs/advanced/theory/dpo-theory.md` | Direct Preference Optimization theory |
| `docs/advanced/theory/grpo-theory.md` | Group Relative Policy Optimization theory |
| `docs/advanced/theory/loss-functions.md` | Loss function theory and design principles |
| `docs/advanced/theory/mathematical-foundations.md` | Mathematical foundations and proofs |

---

## MR Group 9: Advanced Training Techniques
**Focus:** Advanced optimization techniques, curriculum learning, multi-objective training, and custom loss functions
**Priority:** Low

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/advanced/training/index.md` | Advanced training techniques overview |
| `docs/advanced/training/custom-loss-functions.md` | Custom loss function development |
| `docs/advanced/training/pareto-optimization.md` | Pareto optimization for multi-objective problems |
| `docs/advanced/training/multi-objective-tuning.md` | Multi-objective hyperparameter tuning |
| `docs/advanced/training/hyperparameter-optimization.md` | Advanced hyperparameter optimization |
| `docs/advanced/training/learning-rate-scheduling.md` | Learning rate scheduling strategies |
| `docs/advanced/training/training-stability.md` | Training stability and convergence |
| `docs/advanced/training/loss-function-design.md` | Loss function design principles |
| `docs/advanced/training/adaptive-curriculum.md` | Adaptive curriculum learning strategies |
| `docs/advanced/training/curriculum-learning.md` | Curriculum learning implementation |
| `docs/advanced/training/multi-objective-training.md` | Multi-objective training methodologies |

---

## MR Group 10: API Documentation
**Focus:** Complete API reference, distributed computing interfaces, and auto-generated documentation
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/api-docs/index.md` | Complete API documentation overview |
| `docs/api-docs/index.rst` | API documentation in RST format |
| `docs/api-docs/auto-generated.md` | Auto-generated API documentation |
| `docs/api-docs/models.md` | Model API interfaces and classes |
| `docs/api-docs/distributed.md` | Distributed computing API reference |
| `docs/api-docs/nemo_rl/` | Complete NeMo RL API reference (81 files) |

---

## MR Group 11: Project Overview & About
**Focus:** Project introduction, key features, architectural overview, and documentation infrastructure
**Priority:** Low

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/about/index.md` | Project overview and introduction |
| `docs/about/why-nemo-rl.md` | Why choose NeMo RL framework |
| `docs/about/key-features.md` | Key features and capabilities |
| `docs/about/architecture.md` | High-level system architecture |
| `docs/README.md` | Documentation setup and maintenance |
| `docs/test_json_output.py` | Documentation testing utilities |

---

## Total Count Summary

### Total Count: 92 files
Breakdown:
- get-started/: 6 files
- guides/: 16 files
- design-docs/: 14 files
- tutorials-examples/: 11 files
- advanced/: 31 files
- api-docs/: 6 files (5 main + 1 nemo_rl directory)
- configuration-cli/: 4 files
- about/: 4 files

**Total: 92 files in the 8 nested subdirectories of docs/**