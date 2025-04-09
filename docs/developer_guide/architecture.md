# Architecture Overview

This document provides an overview of the Video Annotator application architecture.

## High-Level Architecture

The Video Annotator follows a modular architecture with clear separation of concerns between different components. The application is divided into four main modules:

1. **Core**: Core functionality for video playback and annotation management
2. **UI**: User interface components for interacting with the application
3. **Utils**: Utility functions for logging, error handling, and other common tasks
4. **Tools**: Standalone tools for visualization, analysis, and export

![Architecture Diagram](../images/architecture.png)

## Component Interactions

The components interact with each other in the following ways:

- **Main Entry Point** (`main.py`): Provides a unified entry point for all components
- **Core Components**: Handle video playback and annotation management
- **UI Components**: Provide the user interface for interacting with the core components
- **Utils Components**: Provide utility functions used by all other components
- **Tools Components**: Provide standalone tools that use the core and utils components

## Core Components

### VideoPlayer

The `VideoPlayer` class is responsible for loading and playing videos. It provides methods for:

- Loading video files
- Navigating through frames
- Caching frames for better performance
- Resizing frames for display

### AnnotationManager

The `AnnotationManager` class is responsible for managing annotations. It provides methods for:

- Loading and saving annotations
- Adding, updating, and removing annotations
- Querying annotations

### Config

The `Config` class is responsible for managing configuration settings. It provides methods for:

- Loading and saving configuration files
- Accessing configuration settings
- Updating configuration settings

## UI Components

### UIController

The `UIController` class is responsible for managing the user interface. It provides methods for:

- Setting up the UI elements
- Handling user interactions
- Updating the UI based on changes in the core components

### ConfigEditor

The `ConfigEditor` class provides a graphical interface for editing configuration settings.

## Utils Components

### Logger

The `Logger` module provides logging functionality for the application.

### ErrorHandling

The `ErrorHandling` module provides error handling utilities for the application.

### Export

The `Export` module provides utilities for exporting annotations to different formats.

## Tools Components

### Visualization

The `Visualization` module provides tools for visualizing annotations.

### Analysis

The `Analysis` module provides tools for analyzing annotations.

### GenGroundTruth

The `GenGroundTruth` module provides tools for generating ground truth from multiple annotations.

### ExportGUI

The `ExportGUI` module provides a graphical interface for exporting annotations.

## Data Flow

The data flow in the application follows this pattern:

1. User interacts with the UI
2. UI sends commands to the core components
3. Core components process the commands and update their state
4. Core components notify the UI of state changes
5. UI updates to reflect the new state

## Configuration

The application uses a JSON configuration file (`config.json`) to store settings. The configuration includes:

- Label definitions (name, key, value, color)
- UI settings (window size, controls position)
- Video settings (cache size, playback speed)
- Annotation settings (auto-save, auto-advance)

## Error Handling

The application uses a centralized error handling system:

- Exceptions are caught and logged
- User-friendly error messages are displayed
- The application continues running when possible

## Logging

The application uses a centralized logging system:

- Log messages are written to a log file
- Log messages include timestamp, level, and message
- Log levels can be configured (INFO, WARNING, ERROR, DEBUG)

## Next Steps

For more detailed information about the code organization, see the [Code Organization](code_organization.md) guide.
