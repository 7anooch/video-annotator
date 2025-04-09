# Enhanced Analysis Module for Video Annotator

This module provides enhanced tools for analyzing and visualizing annotation data in the Video Annotator application. It builds upon and extends the existing analysis tools in the `src/tools` directory.

## Overview

The enhanced analysis module includes the following components:

- **Core Analysis**: Enhanced analysis functionality that integrates with existing tools
- **Data Model**: Compatible data model for annotations that works with existing formats
- **Adapters**: Adapter classes that interface with existing analysis and visualization tools
- **Statistical Analysis**: Advanced statistical analysis functions that complement existing analysis
- **Visualization**: Enhanced visualization functions that work alongside existing visualizations
- **Utilities**: Utility functions that extend existing functionality

## Usage

### Using Adapters with Existing Tools

```python
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter
from src.analysis.adapters.visualization_adapter import VisualizationAdapter

# Create adapters
analyze_adapter = AnalyzeAdapter()
plot_adapter = PlotAdapter()
visualization_adapter = VisualizationAdapter()

# Use existing analysis functions through adapters
precision_recall = analyze_adapter.calculate_precision_recall(ground_truth, predictions)
sequence_analysis = analyze_adapter.analyze_sequence(annotations)

# Use existing plotting functions through adapters
fig = plot_adapter.plot_ethogram(annotations)
fig.show()

# Use existing visualization functions through adapters
visualization_adapter.visualize_annotations(annotations)
```

### Loading Annotations with Compatible Data Model

```python
from src.analysis.data_model import AnnotationData

# Load annotations from a file
data = AnnotationData()
data.load_from_file('annotations.csv')

# Get basic information
print(f"Total annotations: {data.get_annotation_count()}")
print(f"Labels: {data.get_labels()}")

# Convert to format compatible with existing tools
frames = data.get_frames()
labels = [data.get_annotation(frame).get('label') for frame in frames if 'label' in data.get_annotation(frame)]
```

### Enhanced Analysis Functions

```python
from src.analysis.statistics import StatisticalAnalysis

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Get enhanced statistics
basic_stats = stats.basic_statistics(data)
print(f"Total annotations: {basic_stats['total_annotations']}")
print(f"Unique labels: {basic_stats['unique_labels']}")

# Get advanced analysis results
correlation = stats.correlation_analysis(data1, data2)
time_series = stats.time_series_analysis(data)
anomalies = stats.detect_anomalies(data)
```

### Enhanced Visualization Functions

```python
from src.analysis.visualization import VisualizationManager

# Create a visualization manager
viz = VisualizationManager()

# Create enhanced visualizations
fig = viz.timeline_plot(data)
fig.show()

fig = viz.transition_heatmap(data)
fig.show()

# Create a comprehensive report that includes both basic and advanced visualizations
output_files = viz.create_report(data, 'output_dir')
print(f"Report files: {output_files}")
```

### Enhanced Utility Functions

```python
from src.analysis.utils import interpolate_annotations, smooth_annotations, calculate_cohen_kappa

# Interpolate missing annotations
interpolated = interpolate_annotations(data.annotations, method='nearest')

# Smooth annotations
smoothed = smooth_annotations(data.annotations, window_size=5)

# Calculate advanced metrics
kappa = calculate_cohen_kappa(annotations1, annotations2)
```

## Dependencies

- NumPy: For numerical computations
- Pandas: For data manipulation
- Matplotlib: For visualizations
- SciPy: For scientific computations
- scikit-learn: For machine learning analysis (optional)
- Existing Video Annotator tools: analyze.py, plot.py, visualization.py

## Integration with Existing Tools

The enhanced analysis module is designed to work seamlessly with the existing analysis tools in the Video Annotator. It provides:

1. **Adapter Classes**: Interface with existing tools to use their functionality
2. **Compatible Data Model**: Works with existing annotation formats
3. **Enhanced Functionality**: Adds new features that complement existing tools
4. **Unified Interface**: Provides a consistent interface for all analysis functions

## Future Improvements

- Interactive visualizations using Plotly that complement existing visualizations
- 3D visualizations for complex behavioral data
- Plugin system that works with the existing architecture
- Enhanced integration with the main application
- Comprehensive documentation and tutorials for both new and existing tools
