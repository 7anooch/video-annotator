# Refactoring Guide for Video Annotator

This document provides guidelines for ongoing refactoring efforts in the Video Annotator project.

## Code Organization

### Directory Structure

The project follows this directory structure:

```
video-annotator/
├── src/                      # Source code
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── video_player.py
│   │   ├── annotation_manager.py
│   │   └── config.py
│   ├── ui/                   # User interface components
│   │   ├── __init__.py
│   │   ├── ui_controller.py
│   │   └── config_editor.py
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── error_handling.py
│   │   ├── export.py
│   │   └── annotation/        # Annotation utilities
│   │       ├── __init__.py
│   │       └── funcs.py
│   ├── tools/                # Standalone tools
│   │   ├── __init__.py
│   │   ├── plot.py
│   │   ├── analyze.py
│   │   ├── gen_ground_truth.py
│   │   └── ...
├── tests/                    # Test files
├── docs/                     # Documentation
├── examples/                 # Example files
└── scripts/                  # Scripts for running the application
```

### Import Guidelines

- Use absolute imports from the `src` directory
- Example: `from src.utils.logger import setup_logger`
- Avoid relative imports to prevent confusion

### Refactoring Legacy Code

When refactoring legacy code:

1. Identify utility functions in the root directory
2. Move them to appropriate modules in the `src` directory
3. Update imports in all files that use these functions
4. Add proper docstrings to the functions
5. Add unit tests for the functions

## Completed Refactoring

### April 2024

- Updated imports in `annotator.py` to use utility functions from the `src` directory
- Updated imports in `src/core/config.py` to use the logger from `src/utils/logger.py`
- Removed duplicate `get_csv_file_path` function from `annotator_modular.py` and imported it from `src/utils/annotation/funcs.py`

## Pending Refactoring

- Complete migration from legacy `annotator.py` to modular structure
- Move remaining utility functions to appropriate modules
- Add comprehensive docstrings to all functions
- Expand test coverage for utility functions
- Standardize error handling across the codebase
