# Usage Examples

This document provides detailed examples of how to use the enhanced analysis tools for common tasks.

## Table of Contents

1. [Basic Analysis](#basic-analysis)
2. [Advanced Analysis](#advanced-analysis)
3. [Visualization](#visualization)
4. [Integration with Existing Tools](#integration-with-existing-tools)
5. [Complete Workflow](#complete-workflow)

## Basic Analysis

### Loading and Analyzing Annotations

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Calculate basic statistics
basic_stats = stats.basic_statistics(data)
print(f"Total annotations: {basic_stats['total_annotations']}")
print(f"Unique labels: {basic_stats['unique_labels']}")
print(f"Label counts: {basic_stats['label_counts']}")
print(f"Frame range: {basic_stats['frame_min']} - {basic_stats['frame_max']}")
print(f"Total gaps: {basic_stats['total_gaps']}")
print(f"Total gap frames: {basic_stats['total_gap_frames']}")
print(f"Average gap size: {basic_stats['avg_gap_size']:.2f}")
print(f"Maximum gap size: {basic_stats['max_gap_size']}")

# Calculate label transitions
transitions = stats.label_transitions(data)
print("Label transitions:")
for (from_label, to_label), count in transitions.items():
    print(f"  {from_label} -> {to_label}: {count}")

# Calculate label durations
durations = stats.label_durations(data)
print("Label durations:")
for label, duration_list in durations.items():
    print(f"  {label}: {duration_list}")

# Calculate duration statistics
duration_stats = stats.duration_statistics(data)
print("Duration statistics:")
for label, stats_dict in duration_stats.items():
    print(f"  {label}:")
    print(f"    Count: {stats_dict['count']}")
    print(f"    Total frames: {stats_dict['total_frames']}")
    print(f"    Min: {stats_dict['min']}")
    print(f"    Max: {stats_dict['max']}")
    print(f"    Mean: {stats_dict['mean']:.2f}")
    print(f"    Median: {stats_dict['median']:.2f}")
    print(f"    Std: {stats_dict['std']:.2f}")
```

### Comparing Annotations

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

# Load annotations
data1 = AnnotationData()
data1.load_from_file('annotations1.csv')

data2 = AnnotationData()
data2.load_from_file('annotations2.csv')

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Compare annotations
comparison = stats.compare_annotations(data1, data2)
print(f"Total annotations in set 1: {comparison['total_annotations_1']}")
print(f"Total annotations in set 2: {comparison['total_annotations_2']}")
print(f"Common frames: {comparison['common_frames']}")
print(f"Agreements: {comparison['agreements']}")
print(f"Disagreements: {comparison['disagreements']}")
print(f"Agreement rate: {comparison['agreement_rate']:.2f}")
print(f"Unique to set 1: {comparison['unique_to_1']}")
print(f"Unique to set 2: {comparison['unique_to_2']}")

# Calculate correlation
correlation = stats.correlation_analysis(data1, data2)
print(f"Pearson correlation: {correlation['pearson_correlation']:.2f}")
print(f"Spearman correlation: {correlation['spearman_correlation']:.2f}")
print(f"Cohen's kappa: {correlation['cohen_kappa']:.2f}")
```

## Advanced Analysis

### Advanced Sequence Analysis

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Perform advanced sequence analysis
sequence_analysis = stats.advanced_sequence_analysis(data)

# Get basic analysis results
basic_results = sequence_analysis['basic_analysis']
print("Basic sequence analysis:")
print(f"  Sequences: {basic_results['sequences']}")
print(f"  Counts: {basic_results['counts']}")

# Get transition probabilities
transition_probs = sequence_analysis['transition_probabilities']
print("Transition probabilities:")
for (from_label, to_label), prob in transition_probs.items():
    print(f"  {from_label} -> {to_label}: {prob:.2f}")

# Get n-grams
ngrams = sequence_analysis['ngrams']
print("N-grams:")
for n, ngram_counts in ngrams.items():
    print(f"  {n}-grams:")
    for ngram, count in sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {ngram}: {count}")

# Get complexity measures
complexity = sequence_analysis['complexity']
print("Complexity measures:")
print(f"  Label entropy: {complexity['label_entropy']:.2f}")
print(f"  Bigram entropy: {complexity['bigram_entropy']:.2f}")
print(f"  Conditional entropy: {complexity['conditional_entropy']:.2f}")

# Get recurring patterns
patterns = sequence_analysis['recurring_patterns']
print("Recurring patterns:")
for pattern in patterns:
    print(f"  Pattern: {pattern['pattern']}")
    print(f"    Length: {pattern['length']}")
    print(f"    Occurrences: {pattern['occurrences']}")
    print(f"    Positions: {pattern['positions']}")
```

### Time Series Analysis

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Perform time series analysis
time_series = stats.time_series_analysis(data)

# Get windows
windows = time_series['windows']
print(f"Time windows: {windows}")

# Get frequencies
frequencies = time_series['frequencies']
print("Label frequencies:")
for label, freq in frequencies.items():
    print(f"  {label}: {freq}")

# Get trends
trends = time_series['trends']
print("Trends:")
for label, trend in trends.items():
    print(f"  {label}:")
    print(f"    Slope: {trend['slope']:.2f}")
    print(f"    Intercept: {trend['intercept']:.2f}")
    print(f"    Increasing: {trend['increasing']}")
    print(f"    Decreasing: {trend['decreasing']}")

# Get autocorrelations
autocorrelations = time_series['autocorrelations']
print("Autocorrelations:")
for label, autocorr in autocorrelations.items():
    print(f"  {label}: {autocorr}")
```

### Anomaly Detection

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a statistical analysis object
stats = StatisticalAnalysis()

# Detect anomalies using Z-score method
anomalies = stats.detect_anomalies(data, method='zscore', threshold=3.0)
print("Anomalies (Z-score method):")
for frame, anomaly in anomalies.items():
    print(f"  Frame {frame}:")
    print(f"    Label: {anomaly['label']}")
    print(f"    Duration: {anomaly['duration']}")
    print(f"    Z-score: {anomaly['zscore']:.2f}")
    print(f"    Mean duration: {anomaly['mean_duration']:.2f}")
    print(f"    Std duration: {anomaly['std_duration']:.2f}")

# Detect anomalies using IQR method
anomalies = stats.detect_anomalies(data, method='iqr', threshold=1.5)
print("Anomalies (IQR method):")
for frame, anomaly in anomalies.items():
    print(f"  Frame {frame}:")
    print(f"    Label: {anomaly['label']}")
    print(f"    Duration: {anomaly['duration']}")
    print(f"    Q1: {anomaly['q1']:.2f}")
    print(f"    Q3: {anomaly['q3']:.2f}")
    print(f"    IQR: {anomaly['iqr']:.2f}")
    print(f"    Lower bound: {anomaly['lower_bound']:.2f}")
    print(f"    Upper bound: {anomaly['upper_bound']:.2f}")

# Detect anomalies using Isolation Forest method
anomalies = stats.detect_anomalies(data, method='isolation_forest')
print("Anomalies (Isolation Forest method):")
for frame, anomaly in anomalies.items():
    print(f"  Frame {frame}:")
    print(f"    Label: {anomaly['label']}")
    print(f"    Duration: {anomaly['duration']}")
    print(f"    Anomaly score: {anomaly['anomaly_score']:.2f}")
```

## Visualization

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
fig.savefig('timeline.png')

# Create a label distribution plot
fig = viz.label_distribution_plot(data)
fig.savefig('label_distribution.png')

# Create a duration boxplot
fig = viz.duration_boxplot(data)
fig.savefig('duration_boxplot.png')

# Create a transition heatmap
fig = viz.transition_heatmap(data)
fig.savefig('transition_heatmap.png')

# Create a time series plot
fig = viz.time_series_plot(data)
fig.savefig('time_series.png')
```

### Interactive Visualizations

```python
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a visualization manager
viz = VisualizationManager()

# Create an interactive timeline visualization
interactive_timeline_path = viz.create_interactive_timeline(data, output_path='interactive_timeline.html')
print(f"Interactive timeline saved to: {interactive_timeline_path}")

# Create a 3D visualization
visualization_3d_path = viz.create_3d_visualization(data, output_path='3d_visualization.html')
print(f"3D visualization saved to: {visualization_3d_path}")
```

### Comprehensive Reports

```python
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a visualization manager
viz = VisualizationManager()

# Create a comprehensive report
output_files = viz.create_report(data, output_dir='report')
print(f"Report files: {output_files}")
```

## Integration with Existing Tools

### Using Adapters

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
print(f"Counts: {sequence_analysis['counts']}")

# Use plot adapter
fig = plot_adapter.plot_ethogram(data.annotations)
fig.savefig('ethogram.png')

# Use visualization adapter
visualization_adapter.visualize_annotations(data.annotations)
```

### Converting Between Formats

```python
from src.analysis.data_model import AnnotationData

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Convert to the format used by the existing analysis tools
frames, labels = data.to_analyze_format()
print(f"Frames: {frames[:10]}...")
print(f"Labels: {labels[:10]}...")

# Convert to the format used by the existing visualization tools
visualization_data = data.to_visualization_format()
print(f"Visualization data: {visualization_data}")

# Convert to the format used by the existing plot tools
frames, labels = data.to_plot_format()
print(f"Frames: {frames[:10]}...")
print(f"Labels: {labels[:10]}...")
```

## Complete Workflow

### End-to-End Analysis

```python
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager
from src.analysis.utils import calculate_annotation_quality

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create analysis objects
stats = StatisticalAnalysis()
viz = VisualizationManager()

# Step 1: Basic Analysis
basic_stats = stats.basic_statistics(data)
print(f"Total annotations: {basic_stats['total_annotations']}")
print(f"Unique labels: {basic_stats['unique_labels']}")
print(f"Label counts: {basic_stats['label_counts']}")

# Step 2: Advanced Analysis
sequence_analysis = stats.advanced_sequence_analysis(data)
time_series = stats.time_series_analysis(data)
anomalies = stats.detect_anomalies(data, method='zscore', threshold=3.0)

# Step 3: Visualization
fig = viz.timeline_plot(data)
fig.savefig('timeline.png')

fig = viz.transition_heatmap(data)
fig.savefig('transition_heatmap.png')

interactive_timeline_path = viz.create_interactive_timeline(data, output_path='interactive_timeline.html')
visualization_3d_path = viz.create_3d_visualization(data, output_path='3d_visualization.html')

# Step 4: Generate Report
output_files = viz.create_report(data, output_dir='report')
print(f"Report files: {output_files}")

# Step 5: Quality Assessment (if ground truth is available)
ground_truth = AnnotationData()
ground_truth.load_from_file('ground_truth.csv')

quality = calculate_annotation_quality(data.annotations, ground_truth.annotations)
print(f"Coverage: {quality['coverage']:.2f}")
print(f"Accuracy: {quality['accuracy']:.2f}")
print(f"Kappa: {quality['kappa']:.2f}")
```
