# Documentation Files by Logical Groupings for GitHub MR Review

This document organizes all files in the `/docs` directory by logical groupings for efficient GitHub MR assignments and developer review.

## MR Group 1: Core Setup & Installation
**Focus:** Getting started, installation, and basic setup
**Reviewer:** DevOps/Infrastructure Engineer
**Priority:** High

### Files:
- `docs/index.md` - Main documentation landing page
- `docs/get-started/index.md` - Get started landing page
- `docs/get-started/installation.md` - Installation guide
- `docs/get-started/quickstart.md` - Quick start tutorial
- `docs/get-started/local-workstation.md` - Local workstation setup
- `docs/get-started/cluster.md` - Cluster setup guide
- `docs/get-started/docker.md` - Docker setup guide
- `docs/BUILD_INSTRUCTIONS.md` - Build instructions
- `docs/conf.py` - Sphinx configuration
- `docs/project.json` - Project configuration
- `docs/versions1.json` - Version configuration

---

## MR Group 2: Training Algorithms Core
**Focus:** Core training algorithm documentation
**Reviewer:** ML Engineer / Algorithm Specialist
**Priority:** High

### Files:
- `docs/guides/training-algorithms/index.md` - Training algorithms overview
- `docs/guides/training-algorithms/sft.md` - Supervised Fine-Tuning guide
- `docs/guides/training-algorithms/dpo.md` - Direct Preference Optimization guide
- `docs/guides/training-algorithms/grpo.md` - Group Relative Policy Optimization guide
- `docs/guides/training-algorithms/eval.md` - Evaluation guide

---

## MR Group 3: Tutorials & Learning Paths
**Focus:** Step-by-step tutorials and learning content
**Reviewer:** Technical Writer / ML Engineer
**Priority:** High

### Files:
- `docs/tutorials-examples/index.md` - Tutorials landing page
- `docs/tutorials-examples/tutorials/index.md` - Tutorials overview
- `docs/tutorials-examples/tutorials/sft-tutorial.md` - SFT tutorial
- `docs/tutorials-examples/tutorials/dpo-tutorial.md` - DPO tutorial
- `docs/tutorials-examples/tutorials/grpo-tutorial.md` - GRPO tutorial
- `docs/tutorials-examples/tutorials/evaluation-tutorial.md` - Evaluation tutorial
- `docs/tutorials-examples/examples/index.md` - Examples overview
- `docs/tutorials-examples/examples/sft-openmathinstruct2.md` - SFT example
- `docs/tutorials-examples/examples/grpo-deepscaler.md` - GRPO example
- `docs/tutorials-examples/use-cases/index.md` - Use cases overview
- `docs/tutorials-examples/use-cases/mathematical-reasoning.md` - Math reasoning use case
- `docs/tutorials-examples/use-cases/code-generation.md` - Code generation use case

---

## MR Group 4: Model Development & Integration
**Focus:** Model development, integration, and customization
**Reviewer:** ML Engineer / Model Specialist
**Priority:** High

### Files:
- `docs/guides/model-development/index.md` - Model development overview
- `docs/guides/model-development/adding-new-models.md` - Adding new models guide
- `docs/guides/model-development/model-quirks.md` - Model quirks documentation
- `docs/guides/environment-data/index.md` - Environment and data overview
- `docs/guides/environment-data/environment-development.md` - Environment development
- `docs/guides/environment-data/debugging.md` - Environment debugging
- `docs/guides/environment-data/nsys-profiling.md` - NSYS profiling guide

---

## MR Group 5: Production Support & Operations
**Focus:** Production deployment, testing, and maintenance
**Reviewer:** DevOps Engineer / Production Engineer
**Priority:** Medium

### Files:
- `docs/guides/production-support/index.md` - Production support overview
- `docs/guides/production-support/testing.md` - Testing guide
- `docs/guides/production-support/debugging.md` - Debugging guide
- `docs/guides/production-support/documentation.md` - Documentation guide
- `docs/guides/production-support/packaging.md` - Packaging guide
- `docs/guides/production-support/troubleshooting.md` - Troubleshooting guide
- `docs/configuration-cli/index.md` - Configuration and CLI overview
- `docs/configuration-cli/cli-reference.md` - CLI reference
- `docs/configuration-cli/configuration-reference.md` - Configuration reference
- `docs/configuration-cli/troubleshooting.md` - CLI troubleshooting

---

## MR Group 6: Core Architecture & Design
**Focus:** System architecture and design principles
**Reviewer:** System Architect / Senior Engineer
**Priority:** Medium

### Files:
- `docs/design-docs/index.md` - Design docs landing page
- `docs/design-docs/core-architecture/index.md` - Core architecture overview
- `docs/design-docs/core-architecture/design-and-philosophy.md` - Design philosophy
- `docs/design-docs/core-architecture/generation.md` - Generation architecture
- `docs/design-docs/core-architecture/fsdp2-parallel-plan.md` - FSDP2 parallel plan
- `docs/design-docs/data-management/index.md` - Data management overview
- `docs/design-docs/data-management/padding.md` - Data padding
- `docs/design-docs/data-management/chat-datasets.md` - Chat datasets
- `docs/design-docs/computational-systems/index.md` - Computational systems overview
- `docs/design-docs/computational-systems/training-backends.md` - Training backends
- `docs/design-docs/computational-systems/uv.md` - UV package manager
- `docs/design-docs/computational-systems/logger.md` - Logging system
- `docs/design-docs/development-infrastructure/index.md` - Development infrastructure overview
- `docs/design-docs/development-infrastructure/loss-functions.md` - Loss functions
- `docs/design-docs/development-infrastructure/checkpointing.md` - Checkpointing

---

## MR Group 7: Performance & Advanced Optimization
**Focus:** Performance optimization and advanced techniques
**Reviewer:** Performance Engineer / ML Engineer
**Priority:** Medium

### Files:
- `docs/advanced/index.md` - Advanced topics landing page
- `docs/advanced/performance/index.md` - Performance overview
- `docs/advanced/performance/distributed-training.md` - Distributed training
- `docs/advanced/performance/profiling.md` - Performance profiling
- `docs/advanced/performance/monitoring.md` - Performance monitoring
- `docs/advanced/performance/memory-optimization.md` - Memory optimization
- `docs/advanced/performance/benchmarking.md` - Performance benchmarking
- `docs/advanced/performance/mixed-precision.md` - Mixed precision training

---

## MR Group 8: Research & Theory
**Focus:** Research methodologies and theoretical foundations
**Reviewer:** Research Scientist / ML Researcher
**Priority:** Medium

### Files:
- `docs/advanced/research/index.md` - Research overview
- `docs/advanced/research/reproducibility.md` - Reproducibility guide
- `docs/advanced/research/performance-analysis.md` - Performance analysis
- `docs/advanced/research/hyperparameter-optimization.md` - Hyperparameter optimization
- `docs/advanced/research/experimental-design.md` - Experimental design
- `docs/advanced/research/ablation-studies.md` - Ablation studies
- `docs/advanced/research/custom-algorithms.md` - Custom algorithms
- `docs/advanced/theory/index.md` - Theory overview
- `docs/advanced/theory/sft-theory.md` - SFT theory
- `docs/advanced/theory/dpo-theory.md` - DPO theory
- `docs/advanced/theory/grpo-theory.md` - GRPO theory
- `docs/advanced/theory/loss-functions.md` - Loss function theory
- `docs/advanced/theory/mathematical-foundations.md` - Mathematical foundations

---

## MR Group 9: Advanced Training Techniques
**Focus:** Advanced training methodologies and techniques
**Reviewer:** Senior ML Engineer / Training Specialist
**Priority:** Low

### Files:
- `docs/advanced/training/index.md` - Advanced training overview
- `docs/advanced/training/custom-loss-functions.md` - Custom loss functions
- `docs/advanced/training/pareto-optimization.md` - Pareto optimization
- `docs/advanced/training/multi-objective-tuning.md` - Multi-objective tuning
- `docs/advanced/training/hyperparameter-optimization.md` - Hyperparameter optimization
- `docs/advanced/training/learning-rate-scheduling.md` - Learning rate scheduling
- `docs/advanced/training/training-stability.md` - Training stability
- `docs/advanced/training/loss-function-design.md` - Loss function design
- `docs/advanced/training/adaptive-curriculum.md` - Adaptive curriculum
- `docs/advanced/training/curriculum-learning.md` - Curriculum learning
- `docs/advanced/training/multi-objective-training.md` - Multi-objective training

---

## MR Group 10: API Documentation
**Focus:** API reference documentation
**Reviewer:** API Developer / Backend Engineer
**Priority:** Medium

### Files:
- `docs/api-docs/index.md` - API documentation overview
- `docs/api-docs/index.rst` - API documentation RST
- `docs/api-docs/auto-generated.md` - Auto-generated API docs
- `docs/api-docs/models.md` - Models API
- `docs/api-docs/distributed.md` - Distributed API
- `docs/api-docs/nemo_rl/` - All nemo_rl API files (81 files)

---

## MR Group 11: Project Overview & About
**Focus:** Project overview, features, and architecture
**Reviewer:** Product Manager / Technical Writer
**Priority:** Low

### Files:
- `docs/about/index.md` - About page
- `docs/about/why-nemo-rl.md` - Why NeMo RL
- `docs/about/key-features.md` - Key features
- `docs/about/architecture.md` - Architecture overview
- `docs/README.md` - Documentation README
- `docs/test_json_output.py` - Test file

---

## MR Group 12: Assets & Static Files
**Focus:** Images, static files, and build artifacts
**Reviewer:** UI/UX Designer / Frontend Developer
**Priority:** Low

### Files:
- `docs/assets/` - All image assets (10 PNG files)
- `docs/_static/` - Static files
- `docs/_extensions/` - Documentation extensions
- `docs/_build/` - Build artifacts

---

## Summary Statistics

**Total Files:** 200+ files across 12 logical groupings

**File Distribution:**
- Core Setup & Installation: 11 files
- Training Algorithms Core: 5 files
- Tutorials & Learning Paths: 12 files
- Model Development & Integration: 7 files
- Production Support & Operations: 10 files
- Core Architecture & Design: 16 files
- Performance & Advanced Optimization: 8 files
- Research & Theory: 13 files
- Advanced Training Techniques: 11 files
- API Documentation: 85+ files
- Project Overview & About: 6 files
- Assets & Static Files: 10+ files

**Review Priority:**
- **High Priority:** Groups 1-4 (Core setup, algorithms, tutorials, model development)
- **Medium Priority:** Groups 5-8 (Production, architecture, performance, research)
- **Low Priority:** Groups 9-12 (Advanced training, API docs, overview, assets)

**Recommended Review Timeline:**
- Week 1-2: High priority groups (1-4)
- Week 3-4: Medium priority groups (5-8)
- Week 5-6: Low priority groups (9-12)

Each MR group can be assigned to different developers based on their expertise and the content focus area. 