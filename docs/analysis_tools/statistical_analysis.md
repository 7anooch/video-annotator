# Statistical Analysis Documentation

This document provides comprehensive documentation for the statistical analysis methods available in the enhanced analysis tools.

## Table of Contents

1. [Basic Statistics](#basic-statistics)
2. [Label Transitions](#label-transitions)
3. [Label Durations](#label-durations)
4. [Duration Statistics](#duration-statistics)
5. [Advanced Sequence Analysis](#advanced-sequence-analysis)
6. [Correlation Analysis](#correlation-analysis)
7. [Comparison Analysis](#comparison-analysis)
8. [Time Series Analysis](#time-series-analysis)
9. [Anomaly Detection](#anomaly-detection)
10. [Change Point Detection](#change-point-detection)
11. [Hidden Markov Model Analysis](#hidden-markov-model-analysis)
12. [Markov Chain Monte Carlo Simulation](#markov-chain-monte-carlo-simulation)

## Basic Statistics

The `basic_statistics` method calculates basic statistics for annotation data, including label counts, frame statistics, and gap statistics.

### Method Signature

```python
def basic_statistics(data: AnnotationData) -> Dict[str, Any]:
    """
    Calculate basic statistics for annotation data.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[str, Any]: Dictionary of basic statistics
    """
```

### Example Usage

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
```

### Return Value

The method returns a dictionary with the following keys:

- `total_annotations`: Total number of annotations
- `unique_labels`: Number of unique labels
- `label_counts`: Dictionary of label counts
- `frame_min`: Minimum frame number
- `frame_max`: Maximum frame number
- `frame_range`: Range of frames (max - min)
- `total_gaps`: Total number of gaps in the annotation sequence
- `total_gap_frames`: Total number of frames in gaps
- `avg_gap_size`: Average gap size
- `max_gap_size`: Maximum gap size

## Label Transitions

The `label_transitions` method calculates transitions between labels in the annotation data.

### Method Signature

```python
def label_transitions(data: AnnotationData) -> Dict[Tuple[str, str], int]:
    """
    Calculate transitions between labels.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[Tuple[str, str], int]: Dictionary of transitions with (from_label, to_label) tuples as keys
    """
```

### Example Usage

```python
# Calculate label transitions
transitions = stats.label_transitions(data)
for (from_label, to_label), count in sorted(transitions.items()):
    print(f"  {from_label} -> {to_label}: {count}")
```

### Return Value

The method returns a dictionary with tuples of (from_label, to_label) as keys and transition counts as values.

## Label Durations

The `label_durations` method calculates durations of continuous label segments in the annotation data.

### Method Signature

```python
def label_durations(data: AnnotationData) -> Dict[str, List[int]]:
    """
    Calculate durations of continuous label segments.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[str, List[int]]: Dictionary of durations with labels as keys
    """
```

### Example Usage

```python
# Calculate label durations
durations = stats.label_durations(data)
for label, duration_list in durations.items():
    print(f"  {label}: {duration_list}")
```

### Return Value

The method returns a dictionary with labels as keys and lists of durations as values.

## Duration Statistics

The `duration_statistics` method calculates statistics for label durations in the annotation data.

### Method Signature

```python
def duration_statistics(data: AnnotationData) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for label durations.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of duration statistics with labels as keys
    """
```

### Example Usage

```python
# Calculate duration statistics
duration_stats = stats.duration_statistics(data)
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

### Return Value

The method returns a dictionary with labels as keys and dictionaries of statistics as values. Each statistics dictionary contains the following keys:

- `count`: Number of segments with this label
- `total_frames`: Total number of frames with this label
- `min`: Minimum duration
- `max`: Maximum duration
- `mean`: Mean duration
- `median`: Median duration
- `std`: Standard deviation of durations

## Advanced Sequence Analysis

The `advanced_sequence_analysis` method performs advanced sequence analysis on annotation data, including transition probabilities, n-grams, entropy-based complexity measures, and pattern recognition.

### Method Signature

```python
def advanced_sequence_analysis(data: AnnotationData) -> Dict[str, Any]:
    """
    Perform advanced sequence analysis on annotation data.
    This extends the existing sequence analysis functionality with more sophisticated techniques.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[str, Any]: Advanced sequence analysis results
    """
```

### Example Usage

```python
# Perform advanced sequence analysis
sequence_analysis = stats.advanced_sequence_analysis(data)

# Get transition probabilities
transition_probs = sequence_analysis['transition_probabilities']
for (from_label, to_label), prob in sorted(transition_probs.items()):
    print(f"  {from_label} -> {to_label}: {prob:.2f}")

# Get n-grams
ngrams = sequence_analysis['ngrams']
for n, ngram_counts in ngrams.items():
    print(f"  {n}-grams:")
    for ngram, count in sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {ngram}: {count}")

# Get complexity measures
complexity = sequence_analysis['complexity']
print(f"  Label entropy: {complexity['label_entropy']:.2f}")
print(f"  Bigram entropy: {complexity['bigram_entropy']:.2f}")
print(f"  Conditional entropy: {complexity['conditional_entropy']:.2f}")

# Get recurring patterns
patterns = sequence_analysis['recurring_patterns']
for pattern in patterns:
    print(f"  Pattern: {pattern['pattern']}")
    print(f"    Length: {pattern['length']}")
    print(f"    Occurrences: {pattern['occurrences']}")
    print(f"    Positions: {pattern['positions']}")
```

### Return Value

The method returns a dictionary with the following keys:

- `basic_analysis`: Results from the existing sequence analysis functionality
- `transition_probabilities`: Dictionary of transition probabilities with (from_label, to_label) tuples as keys
- `ngrams`: Dictionary of n-gram counts with n as keys
- `complexity`: Dictionary of complexity measures (label_entropy, bigram_entropy, conditional_entropy)
- `recurring_patterns`: List of recurring patterns with pattern, length, occurrences, and positions

## Correlation Analysis

The `correlation_analysis` method calculates correlation between two annotation sets using various correlation metrics.

### Method Signature

```python
def correlation_analysis(data1: AnnotationData, data2: AnnotationData) -> Dict[str, float]:
    """
    Calculate correlation between two annotation sets.
    
    Args:
        data1 (AnnotationData): First annotation data
        data2 (AnnotationData): Second annotation data
        
    Returns:
        Dict[str, float]: Correlation results
    """
```

### Example Usage

```python
# Calculate correlation
correlation = stats.correlation_analysis(data1, data2)
print(f"Pearson correlation: {correlation['pearson_correlation']:.2f}")
print(f"Spearman correlation: {correlation['spearman_correlation']:.2f}")
print(f"Cohen's kappa: {correlation['cohen_kappa']:.2f}")
```

### Return Value

The method returns a dictionary with the following keys:

- `pearson_correlation`: Pearson correlation coefficient
- `spearman_correlation`: Spearman correlation coefficient
- `cohen_kappa`: Cohen's kappa coefficient

## Comparison Analysis

The `compare_annotations` method compares two sets of annotations and calculates various comparison metrics.

### Method Signature

```python
def compare_annotations(data1: AnnotationData, data2: AnnotationData) -> Dict[str, Any]:
    """
    Compare two sets of annotations.
    
    Args:
        data1 (AnnotationData): First annotation data
        data2 (AnnotationData): Second annotation data
        
    Returns:
        Dict[str, Any]: Comparison results
    """
```

### Example Usage

```python
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
```

### Return Value

The method returns a dictionary with the following keys:

- `total_annotations_1`: Total number of annotations in the first set
- `total_annotations_2`: Total number of annotations in the second set
- `common_frames`: Number of frames that are present in both sets
- `agreements`: Number of frames where the labels agree
- `disagreements`: Number of frames where the labels disagree
- `agreement_rate`: Proportion of common frames where the labels agree
- `unique_to_1`: Number of frames that are only in the first set
- `unique_to_2`: Number of frames that are only in the second set

## Time Series Analysis

The `time_series_analysis` method performs time series analysis on annotation data, including trend analysis and autocorrelation.

### Method Signature

```python
def time_series_analysis(data: AnnotationData) -> Dict[str, Any]:
    """
    Perform enhanced time series analysis on annotation data.
    This extends the existing functionality by providing more detailed time series analysis.
    
    Args:
        data (AnnotationData): Annotation data
        
    Returns:
        Dict[str, Any]: Time series analysis results
    """
```

### Example Usage

```python
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

### Return Value

The method returns a dictionary with the following keys:

- `windows`: List of time windows (start, end) used for analysis
- `frequencies`: Dictionary of label frequencies in each window
- `trends`: Dictionary of trend information for each label
- `autocorrelations`: Dictionary of autocorrelation values for each label

## Anomaly Detection

The `detect_anomalies` method detects anomalies in annotation data using various advanced statistical methods.

### Method Signature

```python
def detect_anomalies(data: AnnotationData, method: str = 'zscore', threshold: float = 3.0) -> Dict[int, Dict[str, Any]]:
    """
    Detect anomalies in annotation data using advanced statistical methods.
    This extends the existing functionality by providing more sophisticated anomaly detection.
    
    Args:
        data (AnnotationData): Annotation data
        method (str, optional): Anomaly detection method ('zscore', 'iqr', 'isolation_forest', 'dbscan', or 'lof'). 
                                Defaults to 'zscore'.
        threshold (float, optional): Anomaly threshold. Defaults to 3.0.
        
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary of anomalies with frame numbers as keys
    """
```

### Example Usage

```python
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

# Detect anomalies using DBSCAN method
anomalies = stats.detect_anomalies(data, method='dbscan', threshold=0.5)
print("Anomalies (DBSCAN method):")
for frame, anomaly in anomalies.items():
    print(f"  Frame {frame}:")
    print(f"    Label: {anomaly['label']}")
    print(f"    Duration: {anomaly['duration']}")
    print(f"    Cluster: {anomaly['cluster']}")

# Detect anomalies using LOF method
anomalies = stats.detect_anomalies(data, method='lof')
print("Anomalies (LOF method):")
for frame, anomaly in anomalies.items():
    print(f"  Frame {frame}:")
    print(f"    Label: {anomaly['label']}")
    print(f"    Duration: {anomaly['duration']}")
    print(f"    Outlier factor: {anomaly['outlier_factor']:.2f}")
```

### Return Value

The method returns a dictionary with frame numbers as keys and dictionaries of anomaly information as values. The anomaly information depends on the method used:

#### Z-score Method

- `label`: Label of the anomaly
- `duration`: Duration of the anomaly
- `zscore`: Z-score of the anomaly
- `mean_duration`: Mean duration of the label
- `std_duration`: Standard deviation of the label duration
- `method`: 'zscore'

#### IQR Method

- `label`: Label of the anomaly
- `duration`: Duration of the anomaly
- `q1`: First quartile of the label duration
- `q3`: Third quartile of the label duration
- `iqr`: Interquartile range of the label duration
- `lower_bound`: Lower bound for anomaly detection
- `upper_bound`: Upper bound for anomaly detection
- `method`: 'iqr'

#### Isolation Forest Method

- `label`: Label of the anomaly
- `duration`: Duration of the anomaly
- `anomaly_score`: Anomaly score from the Isolation Forest algorithm
- `method`: 'isolation_forest'

#### DBSCAN Method

- `label`: Label of the anomaly
- `duration`: Duration of the anomaly
- `cluster`: Cluster label (-1 indicates an outlier)
- `method`: 'dbscan'

#### LOF Method

- `label`: Label of the anomaly
- `duration`: Duration of the anomaly
- `outlier_factor`: Outlier factor from the LOF algorithm
- `method`: 'lof'

## Change Point Detection

The `detect_change_points` method detects change points in annotation data using advanced statistical methods.

### Method Signature

```python
def detect_change_points(data: AnnotationData, method: str = 'binary_segmentation', penalty: str = 'bic') -> Dict[str, Any]:
    """
    Detect change points in annotation data using advanced statistical methods.
    This method identifies points where the statistical properties of the annotation sequence change.
    
    Args:
        data (AnnotationData): Annotation data
        method (str, optional): Change point detection method ('binary_segmentation', 'window', or 'pelt'). 
                                Defaults to 'binary_segmentation'.
        penalty (str, optional): Penalty for adding change points ('bic', 'aic', or 'manual'). Defaults to 'bic'.
        
    Returns:
        Dict[str, Any]: Change point detection results including change points and segments
    """
```

### Example Usage

```python
# Detect change points using binary segmentation method
change_points = stats.detect_change_points(data, method='binary_segmentation', penalty='bic')
print("Change points (Binary Segmentation method):")
print(f"  Change points: {change_points['change_points']}")
print(f"  Change point frames: {change_points['change_point_frames']}")
print("Segments:")
for i, segment in enumerate(change_points['segments']):
    print(f"  Segment {i+1}:")
    print(f"    Start frame: {segment['start_frame']}")
    print(f"    End frame: {segment['end_frame']}")
    print(f"    Length: {segment['length']}")
    print(f"    Most common label: {segment.get('most_common_label', 'N/A')}")
    print(f"    Label counts: {segment.get('label_counts', {})}")

# Detect change points using window method
change_points = stats.detect_change_points(data, method='window', penalty='aic')
print("Change points (Window method):")
print(f"  Change points: {change_points['change_points']}")
print(f"  Change point frames: {change_points['change_point_frames']}")

# Detect change points using PELT method
change_points = stats.detect_change_points(data, method='pelt', penalty='manual')
print("Change points (PELT method):")
print(f"  Change points: {change_points['change_points']}")
print(f"  Change point frames: {change_points['change_point_frames']}")
```

### Return Value

The method returns a dictionary with the following keys:

- `method`: Change point detection method used
- `penalty`: Penalty used for adding change points
- `change_points`: List of change points (indices in the sequence)
- `change_point_frames`: List of change point frames
- `segments`: List of segments between change points

Each segment is a dictionary with the following keys:

- `start_frame`: Start frame of the segment
- `end_frame`: End frame of the segment
- `start_index`: Start index of the segment in the sequence
- `end_index`: End index of the segment in the sequence
- `length`: Length of the segment
- `most_common_label`: Most common label in the segment
- `label_counts`: Dictionary of label counts in the segment

## Hidden Markov Model Analysis

The `hmm_analysis` method performs Hidden Markov Model (HMM) analysis on annotation data.

### Method Signature

```python
def hmm_analysis(data: AnnotationData, n_states: int = 3) -> Dict[str, Any]:
    """
    Perform Hidden Markov Model (HMM) analysis on annotation data.
    This advanced method can identify hidden states and transitions in the annotation sequence.
    
    Args:
        data (AnnotationData): Annotation data
        n_states (int, optional): Number of hidden states to identify. Defaults to 3.
        
    Returns:
        Dict[str, Any]: HMM analysis results including hidden states, transition matrix, and emission matrix
    """
```

### Example Usage

```python
# Perform HMM analysis
hmm_results = stats.hmm_analysis(data, n_states=3)
print("HMM Analysis:")
print(f"  Number of states: {hmm_results['n_states']}")
print(f"  Hidden states: {hmm_results['hidden_states'][:10]}...")
print("Transition matrix:")
for i, row in enumerate(hmm_results['transition_matrix']):
    print(f"  State {i}: {row}")
print("Most likely labels for each state:")
for state, label in hmm_results['most_likely_labels'].items():
    print(f"  State {state}: {label}")
```

### Return Value

The method returns a dictionary with the following keys:

- `n_states`: Number of hidden states
- `hidden_states`: List of hidden states for each frame
- `transition_matrix`: Transition matrix between hidden states
- `emission_means`: Means of the Gaussian emissions
- `emission_covars`: Covariances of the Gaussian emissions
- `state_frames`: Dictionary of frames for each state
- `state_labels`: Dictionary of label counts for each state
- `most_likely_labels`: Dictionary of most likely labels for each state

## Markov Chain Monte Carlo Simulation

The `mcmc_simulation` method performs Markov Chain Monte Carlo (MCMC) simulation based on annotation data.

### Method Signature

```python
def mcmc_simulation(data: AnnotationData, n_samples: int = 1000) -> Dict[str, Any]:
    """
    Perform Markov Chain Monte Carlo (MCMC) simulation based on annotation data.
    This advanced method can generate synthetic annotation sequences based on the observed data.
    
    Args:
        data (AnnotationData): Annotation data
        n_samples (int, optional): Number of samples to generate. Defaults to 1000.
        
    Returns:
        Dict[str, Any]: MCMC simulation results including transition matrix and generated sequence
    """
```

### Example Usage

```python
# Perform MCMC simulation
mcmc_results = stats.mcmc_simulation(data, n_samples=1000)
print("MCMC Simulation:")
print(f"  Number of samples: {mcmc_results['n_samples']}")
print(f"  Unique labels: {mcmc_results['unique_labels']}")
print("Transition matrix:")
for i, row in enumerate(mcmc_results['transition_matrix']):
    print(f"  {mcmc_results['unique_labels'][i]}: {row}")
print("Generated label counts:")
for label, count in mcmc_results['generated_label_counts'].items():
    print(f"  {label}: {count}")
```

### Return Value

The method returns a dictionary with the following keys:

- `n_samples`: Number of samples generated
- `unique_labels`: List of unique labels
- `transition_matrix`: Transition matrix between labels
- `generated_labels`: List of generated labels
- `generated_label_counts`: Dictionary of label counts in the generated sequence
- `generated_transition_counts`: Transition counts in the generated sequence
