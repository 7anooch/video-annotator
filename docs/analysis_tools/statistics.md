# Statistical Analysis Documentation

The statistical analysis module provides enhanced statistical analysis functions for annotation data that complement and extend the functionality in the existing analysis tools.

## Overview

The `StatisticalAnalysis` class is the core of the statistical analysis module. It provides methods for performing various statistical analyses on annotation data, including basic statistics, correlation analysis, time series analysis, and anomaly detection.

## Key Features

- **Basic Statistics**: Calculate basic statistics for annotation data, including label counts, frame statistics, and gap statistics.
- **Advanced Sequence Analysis**: Analyze sequences of annotations with transition probabilities, n-grams, entropy-based complexity measures, and pattern recognition.
- **Correlation Analysis**: Calculate correlation between two annotation sets using various correlation metrics.
- **Time Series Analysis**: Perform time series analysis on annotation data, including trend analysis and autocorrelation.
- **Anomaly Detection**: Detect anomalies in annotation data using various methods (Z-score, IQR, Isolation Forest).
- **Integration**: Seamlessly integrates with existing analysis tools through adapter classes.

## Usage

### Basic Statistics

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
print(f"Frame range: {basic_stats['frame_min']} - {basic_stats['frame_max']}")
```

### Advanced Sequence Analysis

```python
# Perform advanced sequence analysis
sequence_analysis = stats.advanced_sequence_analysis(data)

# Get transition probabilities
transition_probs = sequence_analysis['transition_probabilities']
for (from_label, to_label), prob in transition_probs.items():
    print(f"Transition from {from_label} to {to_label}: {prob:.2f}")

# Get recurring patterns
patterns = sequence_analysis['recurring_patterns']
for pattern in patterns:
    print(f"Pattern: {pattern['pattern']}, Occurrences: {pattern['occurrences']}")
```

### Anomaly Detection

```python
# Detect anomalies using Z-score method
anomalies = stats.detect_anomalies(data, method='zscore', threshold=3.0)
for frame, anomaly in anomalies.items():
    print(f"Anomaly at frame {frame}: {anomaly['label']} (Z-score: {anomaly['zscore']:.2f})")

# Detect anomalies using IQR method
anomalies = stats.detect_anomalies(data, method='iqr', threshold=1.5)
for frame, anomaly in anomalies.items():
    print(f"Anomaly at frame {frame}: {anomaly['label']} (Duration: {anomaly['duration']})")

# Detect anomalies using Isolation Forest method
anomalies = stats.detect_anomalies(data, method='isolation_forest')
for frame, anomaly in anomalies.items():
    print(f"Anomaly at frame {frame}: {anomaly['label']} (Score: {anomaly['anomaly_score']:.2f})")
```

## Class Reference

### StatisticalAnalysis

```python
class StatisticalAnalysis:
    """
    Enhanced statistical analysis for annotations that complements existing tools.
    
    This class provides enhanced statistical analysis functions for annotation data
    that complement and extend the functionality in the existing analysis tools.
    
    Attributes:
        logger: The logger instance
        analyze_adapter: Adapter for the analyze.py module
    """
```

#### Methods

- `basic_statistics(data: AnnotationData) -> Dict[str, Any]`: Calculate basic statistics for annotation data.
- `label_transitions(data: AnnotationData) -> Dict[Tuple[str, str], int]`: Calculate transitions between labels.
- `label_durations(data: AnnotationData) -> Dict[str, List[int]]`: Calculate durations of continuous label segments.
- `duration_statistics(data: AnnotationData) -> Dict[str, Dict[str, float]]`: Calculate statistics for label durations.
- `advanced_sequence_analysis(data: AnnotationData) -> Dict[str, Any]`: Perform advanced sequence analysis on annotation data.
- `correlation_analysis(data1: AnnotationData, data2: AnnotationData) -> Dict[str, float]`: Calculate correlation between two annotation sets.
- `compare_annotations(data1: AnnotationData, data2: AnnotationData) -> Dict[str, Any]`: Compare two sets of annotations.
- `time_series_analysis(data: AnnotationData) -> Dict[str, Any]`: Perform time series analysis on annotation data.
- `detect_anomalies(data: AnnotationData, method: str = 'zscore', threshold: float = 3.0) -> Dict[int, Dict[str, Any]]`: Detect anomalies in annotation data.
