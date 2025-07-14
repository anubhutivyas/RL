# Documentation Structure Changes

This document compares the documentation structure between the archive/docs directory and the current docs directory, focusing on the reorganization of content across 8 main subdirectories.

## Overview

The documentation has been significantly reorganized from a flat structure in archive/docs to a more hierarchical and organized structure in the current docs directory. The main changes involve:

1. **Reorganization**: Content has been moved from flat directories to nested subdirectories with better categorization
2. **New Structure**: Added new top-level directories and improved navigation
3. **Content Consolidation**: Related content has been grouped together under logical categories
4. **Enhanced Organization**: Added index files and better content hierarchy

## Detailed Changes

| Old Location | New Location | Change Description |
|-------------|-------------|-------------------|
| **archive/docs/guides/** | **docs/guides/** | **Major reorganization**: Content moved from flat structure to organized subdirectories |
| `guides/dpo.md` | `guides/training-algorithms/dpo.md` | Moved to training-algorithms subdirectory |
| `guides/eval.md` | `guides/training-algorithms/eval.md` | Moved to training-algorithms subdirectory |
| `guides/grpo.md` | `guides/training-algorithms/grpo.md` | Moved to training-algorithms subdirectory |
| `guides/sft.md` | `guides/training-algorithms/sft.md` | Moved to training-algorithms subdirectory |
| `guides/grpo-deepscaler.md` | `tutorials-examples/examples/grpo-deepscaler.md` | Moved to examples subdirectory |
| `guides/sft-openmathinstruct2.md` | `tutorials-examples/examples/sft-openmathinstruct2.md` | Moved to examples subdirectory |
| - | `guides/model-development/` | **New**: Added subdirectory for model development content |
| - | `guides/environment-data/` | **New**: Added subdirectory for environment and data content |
| - | `guides/production-support/` | **New**: Added subdirectory for production support content |
| - | `guides/index.md` | **New**: Added index file for guides navigation |
| **archive/docs/design-docs/** | **docs/design-docs/** | **Major reorganization**: Split into logical subdirectories |
| `design-docs/design-and-philosophy.md` | `design-docs/core-architecture/design-and-philosophy.md` | Moved to core-architecture subdirectory |
| `design-docs/generation.md` | `design-docs/core-architecture/generation.md` | Moved to core-architecture subdirectory |
| `design-docs/fsdp2-parallel-plan.md` | `design-docs/core-architecture/fsdp2-parallel-plan.md` | Moved to core-architecture subdirectory |
| `design-docs/chat-datasets.md` | `design-docs/data-management/chat-datasets.md` | Moved to data-management subdirectory |
| `design-docs/padding.md` | `design-docs/data-management/padding.md` | Moved to data-management subdirectory |
| `design-docs/loss-functions.md` | `design-docs/development-infrastructure/loss-functions.md` | Moved to development-infrastructure subdirectory |
| `design-docs/checkpointing.md` | `design-docs/development-infrastructure/checkpointing.md` | Moved to development-infrastructure subdirectory |
| `design-docs/training-backends.md` | `design-docs/computational-systems/training-backends.md` | Moved to computational-systems subdirectory |
| `design-docs/logger.md` | `design-docs/computational-systems/logger.md` | Moved to computational-systems subdirectory |
| `design-docs/uv.md` | `design-docs/computational-systems/uv.md` | Moved to computational-systems subdirectory |
| - | `design-docs/index.md` | **New**: Added index file for design docs navigation |
| - | `design-docs/core-architecture/index.md` | **New**: Added subdirectory index |
| - | `design-docs/data-management/index.md` | **New**: Added subdirectory index |
| - | `design-docs/development-infrastructure/index.md` | **New**: Added subdirectory index |
| - | `design-docs/computational-systems/index.md` | **New**: Added subdirectory index |
| **archive/docs/assets/** | **docs/assets/** | **No structural changes**: Same files maintained |
| `assets/*.png` | `assets/*.png` | All image files preserved without changes |
| **archive/docs/ (root files)** | **docs/** | **Major reorganization**: Root files moved to appropriate locations |
| `adding-new-models.md` | `guides/model-development/adding-new-models.md` | Moved to model development guides |
| `cluster.md` | `get-started/cluster.md` | Moved to get-started section |
| `docker.md` | `get-started/docker.md` | Moved to get-started section |
| `local-workstation.md` | `get-started/local-workstation.md` | Moved to get-started section |
| `debugging.md` | `guides/environment-data/debugging.md` | Moved to environment-data guides |
| `testing.md` | `guides/production-support/testing.md` | Moved to production-support guides |
| `documentation.md` | `guides/production-support/documentation.md` | Moved to production-support guides |
| - | **docs/get-started/** | **New**: Complete new section for getting started |
| - | `get-started/index.md` | **New**: Added get-started index |
| - | `get-started/installation.md` | **New**: Added installation guide |
| - | `get-started/quickstart.md` | **New**: Added quickstart guide |
| - | **docs/configuration-cli/** | **New**: Complete new section for CLI and configuration |
| - | `configuration-cli/index.md` | **New**: Added CLI index |
| - | `configuration-cli/cli-reference.md` | **New**: Added CLI reference |
| - | `configuration-cli/configuration-reference.md` | **New**: Added configuration reference |
| - | `configuration-cli/troubleshooting.md` | **New**: Added troubleshooting guide |
| - | **docs/tutorials-examples/** | **New**: Complete new section for tutorials and examples |
| - | `tutorials-examples/index.md` | **New**: Added tutorials index |
| - | `tutorials-examples/tutorials/` | **New**: Added tutorials subdirectory |
| - | `tutorials-examples/examples/` | **New**: Added examples subdirectory |
| - | `tutorials-examples/use-cases/` | **New**: Added use-cases subdirectory |
| - | **docs/advanced/** | **New**: Complete new section for advanced topics |
| - | `advanced/index.md` | **New**: Added advanced topics index |
| - | `advanced/training/` | **New**: Added training subdirectory |
| - | `advanced/theory/` | **New**: Added theory subdirectory |
| - | `advanced/research/` | **New**: Added research subdirectory |
| - | `advanced/performance/` | **New**: Added performance subdirectory |
| - | **docs/about/** | **New**: Complete new section for project information |
| - | `about/index.md` | **New**: Added about index |
| - | `about/architecture.md` | **New**: Added architecture overview |
| - | `about/key-features.md` | **New**: Added key features |
| - | `about/why-nemo-rl.md` | **New**: Added project rationale |
| - | **docs/api-docs/** | **New**: Complete new section for API documentation |
| - | `api-docs/index.md` | **New**: Added API docs index |
| - | `api-docs/auto-generated.md` | **New**: Added auto-generated docs info |
| - | `api-docs/models.md` | **New**: Added models documentation |
| - | `api-docs/distributed.md` | **New**: Added distributed documentation |
| - | `api-docs/nemo_rl/` | **New**: Added detailed API reference |

## Summary of Major Changes

### 1. **Structural Reorganization**
- **Flat → Hierarchical**: Moved from flat directory structure to organized subdirectories
- **Logical Grouping**: Related content grouped under appropriate categories
- **Navigation Enhancement**: Added index files for better navigation

### 2. **New Top-Level Sections**
- **get-started/**: Installation, setup, and quickstart guides
- **configuration-cli/**: CLI reference and configuration documentation
- **tutorials-examples/**: Tutorials, examples, and use cases
- **advanced/**: Advanced topics, theory, and research
- **about/**: Project overview and architecture information
- **api-docs/**: Comprehensive API documentation

### 3. **Content Consolidation**
- **Training Algorithms**: All algorithm guides moved to `guides/training-algorithms/`
- **Design Documents**: Split into logical categories (core-architecture, data-management, etc.)
- **Development Guides**: Organized into model-development, environment-data, and production-support

### 4. **Enhanced Documentation**
- **Index Files**: Added navigation index files throughout
- **Better Organization**: Clear separation of concerns and topics
- **Improved Discoverability**: Content easier to find and navigate

## File Preservation
All original content has been preserved and reorganized rather than removed. The new structure provides better organization while maintaining all existing documentation. 