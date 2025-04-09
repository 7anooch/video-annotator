# Integration Guide

This guide provides information on how to integrate the enhanced analysis tools with the existing tools in the Video Annotator application.

## Overview

The enhanced analysis tools are designed to work seamlessly with the existing tools in the Video Annotator application. They provide a set of adapter classes that interface with the existing tools, allowing them to work together.

## Integration Approaches

There are several approaches to integrating the enhanced analysis tools with the existing tools:

1. **Direct Integration**: Use the enhanced tools directly alongside the existing tools.
2. **Adapter-Based Integration**: Use the adapter classes to interface with the existing tools.
3. **Data Model Integration**: Use the compatible data model to convert between different annotation formats.

## Direct Integration

The enhanced analysis tools can be used directly alongside the existing tools. For example, you can use the enhanced statistical analysis functions to analyze annotation data, and then use the existing visualization tools to visualize the results.

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.tools import visualization

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Use enhanced statistical analysis
stats = StatisticalAnalysis()
basic_stats = stats.basic_statistics(data)
print(f"Total annotations: {basic_stats['total_annotations']}")

# Convert to the format used by the existing visualization tools
frames, labels = data.to_frame_label_lists()

# Use existing visualization tools
visualization.visualize_annotations(frames, labels)
```

## Adapter-Based Integration

The adapter classes provide a bridge between the enhanced analysis tools and the existing tools. They allow you to use the existing tools through a consistent interface.

```python
from src.analysis.data_model import AnnotationData
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter
from src.analysis.adapters.visualization_adapter import VisualizationAdapter

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create adapters
analyze_adapter = AnalyzeAdapter()
plot_adapter = PlotAdapter()
visualization_adapter = VisualizationAdapter()

# Use analyze adapter
sequence_analysis = analyze_adapter.analyze_sequence(data.annotations)
print(f"Sequences: {sequence_analysis['sequences']}")

# Use plot adapter
fig = plot_adapter.plot_ethogram(data.annotations)
fig.savefig('ethogram.png')

# Use visualization adapter
visualization_adapter.visualize_annotations(data.annotations)
```

## Data Model Integration

The compatible data model provides methods for converting between different annotation formats. This allows you to use the enhanced analysis tools with existing annotation data, and vice versa.

```python
from src.analysis.data_model import AnnotationData
from src.tools import analyze, plot, visualization

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Convert to the format used by the existing analysis tools
frames, labels = data.to_analyze_format()

# Use existing analysis tools
sequences, counts = analyze.analyze_sequences(labels)
print(f"Sequences: {sequences}")
print(f"Counts: {counts}")

# Convert to the format used by the existing plot tools
frames, labels = data.to_plot_format()

# Use existing plot tools
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot.plot_ethogram(ax, frames, labels)
fig.savefig('ethogram.png')

# Convert to the format used by the existing visualization tools
visualization_data = data.to_visualization_format()

# Use existing visualization tools
visualization.visualize_annotations(visualization_data['frames'], visualization_data['labels'])
```

## Integration Examples

### Example 1: Combining Enhanced Analysis with Existing Visualization

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.tools import visualization

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Use enhanced statistical analysis
stats = StatisticalAnalysis()
anomalies = stats.detect_anomalies(data, method='zscore', threshold=3.0)
print(f"Detected {len(anomalies)} anomalies")

# Highlight anomalies in the existing visualization
frames, labels = data.to_frame_label_lists()
anomaly_frames = list(anomalies.keys())
visualization.visualize_annotations_with_highlights(frames, labels, anomaly_frames)
```

### Example 2: Using Existing Analysis with Enhanced Visualization

```python
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager
from src.tools import analyze

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Use existing analysis tools
frames, labels = data.to_analyze_format()
precision, recall, f1, confusion_matrix = analyze.calculate_precision_recall(
    frames, labels, frames, labels  # Compare with itself for demonstration
)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Use enhanced visualization
viz = VisualizationManager()
fig = viz.transition_heatmap(data)
fig.savefig('transition_heatmap.png')

# Create an interactive visualization
interactive_path = viz.create_interactive_timeline(data)
print(f"Interactive visualization saved to: {interactive_path}")
```

### Example 3: Complete Integration Workflow

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create analysis objects
stats = StatisticalAnalysis()
viz = VisualizationManager()
analyze_adapter = AnalyzeAdapter()
plot_adapter = PlotAdapter()

# Step 1: Basic Analysis with Enhanced Tools
basic_stats = stats.basic_statistics(data)
print(f"Total annotations: {basic_stats['total_annotations']}")

# Step 2: Advanced Analysis with Enhanced Tools
sequence_analysis = stats.advanced_sequence_analysis(data)
time_series = stats.time_series_analysis(data)

# Step 3: Analysis with Existing Tools via Adapters
existing_sequence_analysis = analyze_adapter.analyze_sequence(data.annotations)
print(f"Existing sequences: {existing_sequence_analysis['sequences']}")

# Step 4: Visualization with Enhanced Tools
fig = viz.timeline_plot(data)
fig.savefig('timeline.png')

# Step 5: Visualization with Existing Tools via Adapters
fig = plot_adapter.plot_ethogram(data.annotations)
fig.savefig('ethogram.png')

# Step 6: Generate Comprehensive Report
output_files = viz.create_report(data, output_dir='report')
print(f"Report files: {output_files}")
```

## Best Practices

1. **Use the Compatible Data Model**: Always use the `AnnotationData` class to work with annotation data. It provides methods for converting between different annotation formats.

2. **Use Adapters for Existing Tools**: Use the adapter classes to interface with the existing tools. This provides a consistent interface and ensures compatibility.

3. **Combine Enhanced and Existing Tools**: Don't be afraid to combine the enhanced analysis tools with the existing tools. They are designed to work together.

4. **Convert Data Formats as Needed**: Use the conversion methods in the `AnnotationData` class to convert between different annotation formats as needed.

5. **Handle Errors Gracefully**: Always handle errors gracefully, especially when working with different annotation formats and tools.

## Troubleshooting

### Common Issues

1. **Incompatible Annotation Formats**: If you encounter issues with incompatible annotation formats, use the conversion methods in the `AnnotationData` class to convert between formats.

2. **Missing Dependencies**: Some enhanced features require additional dependencies. Make sure you have installed all the required dependencies.

3. **Integration Errors**: If you encounter errors when integrating the enhanced tools with the existing tools, check that you are using the correct adapter classes and conversion methods.

### Solutions

1. **Check Annotation Formats**: Make sure you are using the correct annotation formats for the tools you are using. Use the conversion methods in the `AnnotationData` class to convert between formats.

2. **Install Dependencies**: Install all the required dependencies for the enhanced features you are using.

3. **Use Adapters**: Use the adapter classes to interface with the existing tools. This provides a consistent interface and ensures compatibility.

4. **Check Documentation**: Refer to the documentation for the enhanced analysis tools and the existing tools for more information on how to use them together.

5. **Ask for Help**: If you encounter issues that you cannot resolve, ask for help from the development team.
