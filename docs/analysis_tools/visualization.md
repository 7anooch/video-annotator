# Visualization Documentation

The visualization module provides enhanced visualization functions for annotation data that complement and extend the functionality in the existing visualization tools.

## Overview

The `VisualizationManager` class is the core of the visualization module. It provides methods for creating various visualizations of annotation data, including timeline plots, label distribution plots, transition heatmaps, and interactive visualizations.

## Key Features

- **Basic Visualizations**: Create basic visualizations such as timeline plots, label distribution plots, and transition heatmaps.
- **Interactive Visualizations**: Create interactive visualizations using Plotly, including interactive timelines and 3D visualizations.
- **Comprehensive Reports**: Generate comprehensive HTML reports with multiple visualizations.
- **Customization Options**: Customize visualizations with various options.
- **Export Capabilities**: Export visualizations in various formats (PNG, HTML).
- **Integration**: Seamlessly integrates with existing visualization tools through adapter classes.

## Usage

### Basic Visualizations

```python
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a visualization manager
viz = VisualizationManager()

# Create a timeline plot
fig = viz.timeline_plot(data)
fig.show()

# Create a label distribution plot
fig = viz.label_distribution_plot(data)
fig.show()

# Create a transition heatmap
fig = viz.transition_heatmap(data)
fig.show()
```

### Interactive Visualizations

```python
# Create an interactive timeline visualization
interactive_timeline_path = viz.create_interactive_timeline(data, output_path='interactive_timeline.html')
print(f"Interactive timeline saved to: {interactive_timeline_path}")

# Create a 3D visualization
visualization_3d_path = viz.create_3d_visualization(data, output_path='3d_visualization.html')
print(f"3D visualization saved to: {visualization_3d_path}")
```

### Comprehensive Reports

```python
# Create a comprehensive report
output_files = viz.create_report(data, output_dir='report')
print(f"Report files: {output_files}")
```

## Class Reference

### VisualizationManager

```python
class VisualizationManager:
    """
    Enhanced visualization manager for annotation data that complements existing tools.
    
    This class provides enhanced visualization functions for annotation data that
    complement and extend the functionality in the existing visualization tools.
    
    Attributes:
        logger: The logger instance
        plot_adapter: Adapter for the plot.py module
        visualization_adapter: Adapter for the visualization.py module
    """
```

#### Methods

- `timeline_plot(data: AnnotationData, figsize: Tuple[int, int] = (12, 6), title: str = 'Annotation Timeline', save_path: Optional[str] = None) -> plt.Figure`: Create a timeline plot of annotations.
- `label_distribution_plot(data: AnnotationData, figsize: Tuple[int, int] = (10, 6), title: str = 'Label Distribution', save_path: Optional[str] = None) -> plt.Figure`: Create a bar plot of label distribution.
- `duration_boxplot(data: AnnotationData, figsize: Tuple[int, int] = (12, 6), title: str = 'Label Duration Distribution', save_path: Optional[str] = None) -> plt.Figure`: Create a box plot of label durations.
- `transition_heatmap(data: AnnotationData, figsize: Tuple[int, int] = (10, 8), title: str = 'Label Transitions', save_path: Optional[str] = None) -> plt.Figure`: Create a heatmap of label transitions.
- `time_series_plot(data: AnnotationData, figsize: Tuple[int, int] = (12, 6), title: str = 'Label Frequency Over Time', save_path: Optional[str] = None) -> plt.Figure`: Create a time series plot of label frequencies.
- `comparison_plot(data1: AnnotationData, data2: AnnotationData, figsize: Tuple[int, int] = (12, 6), title: str = 'Annotation Comparison', save_path: Optional[str] = None) -> plt.Figure`: Create a comparison plot of two annotation sets.
- `create_interactive_timeline(data: AnnotationData, output_path: str = 'interactive_timeline.html') -> str`: Create an interactive timeline visualization using Plotly.
- `create_3d_visualization(data: AnnotationData, output_path: str = '3d_visualization.html') -> str`: Create a 3D visualization of annotation patterns using Plotly.
- `create_report(data: AnnotationData, output_dir: str, prefix: str = 'report') -> List[str]`: Create a comprehensive report with multiple visualizations.
