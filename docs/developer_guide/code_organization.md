# Code Organization

This document describes how the code is organized in the Video Annotator project.

## Directory Structure

The Video Annotator project follows this directory structure:

```text
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
│   │   ├── export_gui.py
│   │   ├── visualization.py
│   │   ├── patch_gaps.py
│   │   ├── performance_profiler.py
│   │   └── convert_tools/    # Conversion tools
│   │       ├── __init__.py
│   │       ├── convert_kindata.py
│   │       ├── convert_pmtx.py
│   │       └── convert_pmtx_pc.py
│   └── __init__.py
├── tests/                    # Test files
│   ├── __init__.py
│   ├── test_annotator.py
│   └── test_modular.py
├── scripts/                  # Scripts for running the application
│   ├── run.sh
│   └── run.bat
├── docs/                     # Documentation
│   ├── user_guide/
│   ├── developer_guide/
│   └── api/
├── examples/                 # Example files and notebooks
│   └── testing_headcast.ipynb
├── .github/                  # GitHub configuration
│   └── workflows/
│       └── tests.yml
├── annotator.py              # Legacy entry point
├── annotator_modular.py      # Modular entry point
├── main.py                   # Main entry point
├── setup.py                  # Setup script
├── environment.yml           # Conda environment file
├── config.json               # Default configuration
├── README.md                 # Project README
└── LICENSE                   # License file
```

## Source Code Organization

### Core Module

The `src/core` directory contains the core functionality of the application:

- `video_player.py`: Handles video loading and playback
- `annotation_manager.py`: Manages annotations
- `config.py`: Manages configuration settings

### UI Module

The `src/ui` directory contains the user interface components:

- `ui_controller.py`: Manages the main user interface
- `config_editor.py`: Provides a graphical interface for editing configuration settings

### Utils Module

The `src/utils` directory contains utility functions:

- `logger.py`: Provides logging functionality
- `error_handling.py`: Provides error handling utilities
- `export.py`: Provides utilities for exporting annotations
- `annotation/funcs.py`: Provides utility functions for working with annotations

### Tools Module

The `src/tools` directory contains standalone tools:

- `plot.py`: Provides tools for plotting annotations
- `analyze.py`: Provides tools for analyzing annotations
- `gen_ground_truth.py`: Provides tools for generating ground truth from multiple annotations
- `export_gui.py`: Provides a graphical interface for exporting annotations
- `visualization.py`: Provides tools for visualizing annotations
- `patch_gaps.py`: Provides tools for patching gaps in annotations
- `performance_profiler.py`: Provides tools for profiling application performance
- `convert_tools/`: Contains tools for converting data from other formats

## Entry Points

The application has three main entry points:

- `main.py`: The unified entry point for all components
- `annotator_modular.py`: The modular entry point for the annotator
- `annotator.py`: The legacy entry point for the annotator

## Tests

The `tests` directory contains test files:

- `test_annotator.py`: Tests for the legacy annotator
- `test_modular.py`: Tests for the modular annotator

## Scripts

The `scripts` directory contains scripts for running the application:

- `run.sh`: Shell script for running the application on macOS/Linux
- `run.bat`: Batch script for running the application on Windows

## Documentation

The `docs` directory contains documentation:

- `user_guide/`: User documentation
- `developer_guide/`: Developer documentation
- `api/`: API documentation

## Examples

The `examples` directory contains example files and notebooks:

- `testing_headcast.ipynb`: Jupyter notebook for testing the application with headcast data

## GitHub Configuration

The `.github` directory contains GitHub configuration:

- `workflows/tests.yml`: GitHub Actions workflow for running tests

## Configuration

The `config.json` file contains the default configuration for the application.

## Dependencies

The `environment.yml` file contains the Conda environment definition for the application.

## Module Dependencies

The module dependencies in the application follow this pattern:

- `main.py` depends on all other modules
- `annotator_modular.py` depends on the core, ui, and utils modules
- Core modules depend on utils modules
- UI modules depend on core and utils modules
- Tools modules depend on core and utils modules
- Utils modules have no dependencies on other modules

This dependency structure ensures that:

- Utils modules can be used independently
- Core modules can be used without UI modules
- Tools modules can be used independently
- UI modules require core modules

## Coding Conventions

The Video Annotator project follows these coding conventions:

- **File Names**: Snake case (e.g., `video_player.py`)
- **Class Names**: Pascal case (e.g., `VideoPlayer`)
- **Function Names**: Snake case (e.g., `load_frame`)
- **Variable Names**: Snake case (e.g., `frame_number`)
- **Constant Names**: Upper case with underscores (e.g., `MAX_CACHE_SIZE`)
- **Module Names**: Snake case (e.g., `error_handling`)

## Refactoring Guide

The `src/utils/refactoring_guide.md` file contains guidelines for ongoing refactoring efforts in the project. It includes:

- Guidelines for code organization
- Import conventions
- Steps for refactoring legacy code
- Record of completed refactoring tasks
- List of pending refactoring tasks

Refer to this guide when working on refactoring tasks to ensure consistency across the codebase.

## Next Steps

For more detailed information about setting up a development environment, see the [Development Setup](development_setup.md) guide.
