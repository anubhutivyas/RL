# Documentation File Mapping: Archive to New Structure

This document maps source files from `archive/docs/` to their new locations in `docs/` and describes the changes made for developer review.

## File Mapping Table

| Archive Source | New Location | Status | Changes Description |
|----------------|--------------|--------|-------------------|
| `index.md` | `docs/index.md` | **Major Restructure** | Complete rewrite with new taxonomy frontmatter, learning paths, navigation cards, and user-centric organization. Original was simple toctree structure. |
| `guides/sft.md` | `docs/guides/training-algorithms/sft.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths (../../examples/ → ../../../examples/), same core content |
| `guides/dpo.md` | `docs/guides/training-algorithms/dpo.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, same core content |
| `guides/grpo.md` | `docs/guides/training-algorithms/grpo.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, same core content |
| `guides/eval.md` | `docs/guides/training-algorithms/eval.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, same core content |
| `guides/sft-openmathinstruct2.md` | `docs/tutorials-examples/examples/sft-openmathinstruct2.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved from guides to examples section |
| `guides/grpo-deepscaler.md` | `docs/tutorials-examples/examples/grpo-deepscaler.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved from guides to examples section |
| `adding-new-models.md` | `docs/guides/model-development/adding-new-models.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to model-development section |
| `cluster.md` | `docs/get-started/cluster.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to get-started section |
| `docker.md` | `docs/get-started/docker.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to get-started section |
| `local-workstation.md` | `docs/get-started/local-workstation.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to get-started section |
| `design-docs/design-and-philosophy.md` | `docs/design-docs/core-architecture/design-and-philosophy.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to core-architecture section |
| `design-docs/padding.md` | `docs/design-docs/data-management/padding.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to data-management section |
| `design-docs/logger.md` | `docs/design-docs/computational-systems/logger.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to computational-systems section |
| `design-docs/uv.md` | `docs/design-docs/computational-systems/uv.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to computational-systems section |
| `design-docs/chat-datasets.md` | `docs/design-docs/data-management/chat-datasets.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to data-management section |
| `design-docs/generation.md` | `docs/design-docs/core-architecture/generation.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to core-architecture section |
| `design-docs/checkpointing.md` | `docs/design-docs/development-infrastructure/checkpointing.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to development-infrastructure section |
| `design-docs/loss-functions.md` | `docs/design-docs/development-infrastructure/loss-functions.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to development-infrastructure section |
| `design-docs/fsdp2-parallel-plan.md` | `docs/design-docs/core-architecture/fsdp2-parallel-plan.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to core-architecture section |
| `design-docs/training-backends.md` | `docs/design-docs/computational-systems/training-backends.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to computational-systems section |
| `testing.md` | `docs/guides/production-support/testing.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to production-support section |
| `debugging.md` | `docs/guides/production-support/debugging.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to production-support section |
| `documentation.md` | `docs/guides/production-support/documentation.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to production-support section |
| `conf.py` | `docs/conf.py` | **Enhanced** | Updated configuration for new directory structure, enhanced Sphinx settings |
| `project.json` | `docs/project.json` | **Enhanced** | Updated project configuration for new structure |
| `versions1.json` | `docs/versions1.json` | **Enhanced** | Updated version configuration for new structure |

## New Files Created (Not in Archive)

| New File | Purpose | Review Focus |
|----------|---------|--------------|
| `docs/get-started/index.md` | Get started landing page with navigation | Navigation structure, learning paths, user flow |
| `docs/get-started/installation.md` | Installation guide | Installation steps, requirements, platform support |
| `docs/get-started/quickstart.md` | Quick start tutorial | Step-by-step tutorial accuracy, reproducibility |
| `docs/tutorials-examples/index.md` | Tutorials landing page | Navigation, learning progression, content organization |
| `docs/tutorials-examples/tutorials/index.md` | Tutorials overview | Tutorial descriptions, learning paths |
| `docs/tutorials-examples/tutorials/sft-tutorial.md` | SFT tutorial | Tutorial accuracy, code examples, step-by-step instructions |
| `docs/tutorials-examples/tutorials/dpo-tutorial.md` | DPO tutorial | Tutorial accuracy, code examples, step-by-step instructions |
| `docs/tutorials-examples/tutorials/grpo-tutorial.md` | GRPO tutorial | Tutorial accuracy, code examples, step-by-step instructions |
| `docs/tutorials-examples/tutorials/evaluation-tutorial.md` | Evaluation tutorial | Tutorial accuracy, code examples, step-by-step instructions |
| `docs/tutorials-examples/examples/index.md` | Examples overview | Example descriptions, use case coverage |
| `docs/tutorials-examples/use-cases/index.md` | Use cases overview | Use case descriptions, real-world applicability |
| `docs/tutorials-examples/use-cases/mathematical-reasoning.md` | Math reasoning use case | Real-world applicability, code examples |
| `docs/tutorials-examples/use-cases/code-generation.md` | Code generation use case | Real-world applicability, code examples |
| `docs/guides/index.md` | Guides landing page | Navigation structure, content organization |
| `docs/guides/training-algorithms/index.md` | Training algorithms overview | Algorithm descriptions, navigation |
| `docs/guides/model-development/index.md` | Model development overview | Development workflow, navigation |
| `docs/guides/model-development/model-quirks.md` | Model quirks documentation | Model-specific issues, troubleshooting |
| `docs/guides/production-support/index.md` | Production support overview | Production workflow, navigation |
| `docs/guides/production-support/packaging.md` | Packaging guide | Packaging process, deployment instructions |
| `docs/guides/production-support/troubleshooting.md` | Troubleshooting guide | Common issues, solutions, debugging |
| `docs/guides/environment-data/index.md` | Environment and data overview | Data handling, environment setup |
| `docs/guides/environment-data/environment-development.md` | Environment development | Environment creation, customization |
| `docs/guides/environment-data/nsys-profiling.md` | NSYS profiling guide | Performance profiling, optimization |
| `docs/design-docs/index.md` | Design docs landing page | Architecture overview, navigation |
| `docs/design-docs/core-architecture/index.md` | Core architecture overview | System design, architecture principles |
| `docs/design-docs/data-management/index.md` | Data management overview | Data architecture, flow |
| `docs/design-docs/computational-systems/index.md` | Computational systems overview | System architecture, performance |
| `docs/design-docs/development-infrastructure/index.md` | Development infrastructure overview | Dev tools, infrastructure |
| `docs/advanced/index.md` | Advanced topics landing page | Advanced content organization, navigation |
| `docs/advanced/performance/index.md` | Performance optimization overview | Performance content, optimization techniques |
| `docs/advanced/performance/distributed-training.md` | Distributed training guide | Distributed training, scaling |
| `docs/advanced/performance/profiling.md` | Performance profiling guide | Profiling techniques, optimization |
| `docs/advanced/performance/monitoring.md` | Performance monitoring guide | Monitoring, observability |
| `docs/advanced/performance/memory-optimization.md` | Memory optimization guide | Memory management, optimization |
| `docs/advanced/performance/benchmarking.md` | Performance benchmarking guide | Benchmarking, evaluation |
| `docs/advanced/performance/mixed-precision.md` | Mixed precision training guide | Mixed precision, optimization |
| `docs/advanced/research/index.md` | Research topics overview | Research content, methodologies |
| `docs/advanced/research/reproducibility.md` | Reproducibility guide | Research reproducibility, best practices |
| `docs/advanced/research/performance-analysis.md` | Performance analysis guide | Analysis techniques, evaluation |
| `docs/advanced/research/hyperparameter-optimization.md` | Hyperparameter optimization guide | Optimization techniques, tuning |
| `docs/advanced/research/experimental-design.md` | Experimental design guide | Research design, methodology |
| `docs/advanced/research/ablation-studies.md` | Ablation studies guide | Research methodology, analysis |
| `docs/advanced/research/custom-algorithms.md` | Custom algorithms guide | Algorithm development, implementation |
| `docs/advanced/theory/index.md` | Theoretical foundations overview | Theory content, mathematical foundations |
| `docs/advanced/theory/sft-theory.md` | SFT theory guide | Theoretical foundations, mathematical concepts |
| `docs/advanced/theory/dpo-theory.md` | DPO theory guide | Theoretical foundations, mathematical concepts |
| `docs/advanced/theory/grpo-theory.md` | GRPO theory guide | Theoretical foundations, mathematical concepts |
| `docs/advanced/theory/loss-functions.md` | Loss function theory guide | Mathematical foundations, loss functions |
| `docs/advanced/theory/mathematical-foundations.md` | Mathematical foundations guide | Mathematical concepts, theory |
| `docs/advanced/training/index.md` | Advanced training overview | Advanced training techniques, navigation |
| `docs/advanced/training/custom-loss-functions.md` | Custom loss functions guide | Loss function development, implementation |
| `docs/advanced/training/pareto-optimization.md` | Pareto optimization guide | Multi-objective optimization, Pareto frontiers |
| `docs/advanced/training/multi-objective-tuning.md` | Multi-objective tuning guide | Multi-objective optimization, tuning |
| `docs/advanced/training/hyperparameter-optimization.md` | Hyperparameter optimization guide | Optimization techniques, tuning |
| `docs/advanced/training/learning-rate-scheduling.md` | Learning rate scheduling guide | Scheduling strategies, optimization |
| `docs/advanced/training/training-stability.md` | Training stability guide | Stability techniques, best practices |
| `docs/advanced/training/loss-function-design.md` | Loss function design guide | Loss function design, implementation |
| `docs/advanced/training/adaptive-curriculum.md` | Adaptive curriculum guide | Curriculum learning, adaptation |
| `docs/advanced/training/curriculum-learning.md` | Curriculum learning guide | Curriculum design, implementation |
| `docs/advanced/training/multi-objective-training.md` | Multi-objective training guide | Multi-objective training, optimization |
| `docs/api-docs/index.md` | API documentation overview | API organization, navigation |
| `docs/api-docs/index.rst` | API documentation RST | RST structure, API organization |
| `docs/api-docs/auto-generated.md` | Auto-generated API docs | Auto-generation, API coverage |
| `docs/api-docs/models.md` | Models API | Model API documentation, interfaces |
| `docs/api-docs/distributed.md` | Distributed API | Distributed computing API, interfaces |
| `docs/configuration-cli/index.md` | Configuration and CLI overview | CLI organization, navigation |
| `docs/configuration-cli/cli-reference.md` | CLI reference | CLI commands, options, usage |
| `docs/configuration-cli/configuration-reference.md` | Configuration reference | Configuration options, settings |
| `docs/configuration-cli/troubleshooting.md` | CLI troubleshooting | CLI issues, solutions, debugging |
| `docs/about/index.md` | About page | Project overview, introduction |
| `docs/about/why-nemo-rl.md` | Why NeMo RL | Project rationale, benefits |
| `docs/about/key-features.md` | Key features | Feature descriptions, capabilities |
| `docs/about/architecture.md` | Architecture overview | System architecture, design |
| `docs/README.md` | Documentation README | Documentation overview, structure |
| `docs/BUILD_INSTRUCTIONS.md` | Build instructions | Build process, requirements |

## Key Changes Summary

### **Frontmatter Addition**
All markdown files now include standardized frontmatter with:
- **Description**: 1-2 sentence content summary
- **Categories**: Single primary category (e.g., "training-algorithms")
- **Tags**: 2-8 relevant tags for search/discovery
- **Personas**: Target audience (e.g., "mle-focused", "researcher-focused")
- **Difficulty**: beginner/intermediate/advanced/reference
- **Content Type**: tutorial/concept/reference/troubleshooting/example
- **Modality**: text-only/image-only/video-only/multimodal/universal

### **Path Updates**
All internal links updated to reflect new directory structure:
- `../../examples/` → `../../../examples/`
- `../cluster.md` → `../../get-started/cluster.md`
- `../design-docs/` → `../../design-docs/[section]/`

### **Directory Reorganization**
Files moved from flat structure to user-centric organization:
- **Training guides** → `guides/training-algorithms/`
- **Setup files** → `get-started/`
- **Design docs** → organized by topic (`core-architecture/`, `data-management/`, etc.)
- **Examples** → `tutorials-examples/examples/`
- **Use cases** → `tutorials-examples/use-cases/`

### **New Landing Pages**
Added index pages for each major section with:
- Navigation cards and learning paths
- Overview content and key resources
- Logical progression and user workflows

## Developer Review Priorities

### **High Priority (Critical)**
1. **Path Updates** - Verify all internal links work correctly
2. **Frontmatter Accuracy** - Ensure metadata matches content
3. **Navigation Structure** - Check that users can find content
4. **Code Examples** - Verify all code paths and examples work

### **Medium Priority (Important)**
1. **Content Organization** - Ensure logical grouping
2. **Learning Progression** - Verify learning paths make sense
3. **Cross-references** - Check that related content is properly linked
4. **Asset References** - Verify images and other assets load correctly

### **Low Priority (Polish)**
1. **Frontmatter Completeness** - Ensure all metadata is appropriate
2. **Navigation Consistency** - Check for consistent navigation patterns
3. **Content Completeness** - Verify no content was lost in migration
4. **Style Consistency** - Ensure consistent formatting and style

## Review Checklist for Each File

### **For Moved Files:**
- [ ] Frontmatter added and accurate
- [ ] Internal links updated correctly
- [ ] External links still work
- [ ] Code examples reference correct paths
- [ ] Asset references updated
- [ ] Content matches new location context

### **For New Files:**
- [ ] Content is accurate and complete
- [ ] Navigation structure is logical
- [ ] Links point to correct locations
- [ ] Learning progression makes sense
- [ ] Overview content is helpful

### **For All Files:**
- [ ] Frontmatter taxonomy is appropriate
- [ ] Difficulty level matches content
- [ ] Personas are correctly identified
- [ ] Tags are relevant and helpful
- [ ] Content type classification is accurate 