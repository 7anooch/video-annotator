# API Reference

This API reference provides detailed information about the Video Annotator API.

## Overview

The Video Annotator API is organized into four main modules:

- [Core API](core.md): Core functionality for video playback and annotation management
- [UI API](ui.md): User interface components for interacting with the application
- [Utils API](utils.md): Utility functions for logging, error handling, and other common tasks
- [Tools API](tools.md): Standalone tools for visualization, analysis, and export

## Using the API

The Video Annotator API can be used to:

- Load and play videos
- Manage annotations
- Visualize annotations
- Analyze annotations
- Export annotations

### Example: Loading a Video

```python
from src.core.video_player import VideoPlayer

# Create a video player
video_player = VideoPlayer('path/to/video.mp4')

# Load a frame
frame = video_player.load_frame(0)
```

### Example: Managing Annotations

```python
from src.core.annotation_manager import AnnotationManager

# Create an annotation manager
annotation_manager = AnnotationManager('path/to/annotations.csv')

# Add an annotation
annotation_manager.annotate_frame(0, 1)

# Save annotations
annotation_manager.save_annotations()
```

### Example: Visualizing Annotations

```python
from src.tools.visualization import VisualizationTool

# Create a visualization tool
visualization_tool = VisualizationTool()

# Load annotations
annotations = visualization_tool.load_annotations('path/to/annotations.csv')

# Plot an ethogram
visualization_tool.plot_ethogram(annotations)
```

## API Stability

The Video Annotator API is still under development and may change in future versions. The following stability levels are used:

- **Stable**: The API is stable and will not change in a backward-incompatible way
- **Experimental**: The API is experimental and may change in a backward-incompatible way
- **Deprecated**: The API is deprecated and will be removed in a future version

The current stability level for each module is:

- **Core API**: Stable
- **UI API**: Experimental
- **Utils API**: Stable
- **Tools API**: Experimental

## Contributing to the API

If you want to contribute to the Video Annotator API, please see the [Contributing Guidelines](../developer_guide/contributing.md).
