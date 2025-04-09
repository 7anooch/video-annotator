# Statistics API Reference

This document provides a reference for the statistics API.

## StatisticalAnalysis Class

The `StatisticalAnalysis` class provides methods for analyzing annotation data.

### Methods

#### basic_statistics

```python
def basic_statistics(self, data: AnnotationData) -> Dict[str, Any]:
```

Calculates basic statistics for the annotation data.

**Parameters:**
- `data`: AnnotationData object

**Returns:**
- Dictionary containing basic statistics:
  - `total_annotations`: Total number of annotations
  - `unique_labels`: List of unique labels
  - `label_counts`: Dictionary of label counts
  - `frame_min`: Minimum frame number
  - `frame_max`: Maximum frame number
  - `total_gaps`: Total number of gaps
  - `total_gap_frames`: Total number of frames in gaps
  - `avg_gap_size`: Average gap size
  - `max_gap_size`: Maximum gap size

#### detect_change_points

```python
def detect_change_points(self, data: AnnotationData, method: str = 'binary_segmentation', penalty: str = 'bic') -> Dict[str, Any]:
```

Detects change points in the annotation data.

**Parameters:**
- `data`: AnnotationData object
- `method`: Change point detection method (default: 'binary_segmentation')
- `penalty`: Penalty method (default: 'bic')

**Returns:**
- Dictionary containing change point detection results:
  - `method`: Method used
  - `penalty`: Penalty used
  - `change_points`: List of change points
  - `change_point_frames`: List of change point frames
  - `segments`: List of segments
  - `visualization`: Base64-encoded visualization

#### segment_transition_matrix

```python
def segment_transition_matrix(self, data: AnnotationData) -> Dict[str, Any]:
```

Calculates transition matrix for grouped segments.

**Parameters:**
- `data`: AnnotationData object

**Returns:**
- Dictionary containing transition matrix results:
  - `unique_labels`: List of unique labels
  - `transition_counts`: Matrix of transition counts
  - `transition_matrix`: Matrix of transition probabilities
  - `segments`: List of segments
  - `visualization`: Base64-encoded visualization

#### hmm_analysis

```python
def hmm_analysis(self, data: AnnotationData, n_states: int = 3) -> Dict[str, Any]:
```

Performs Hidden Markov Model analysis on the annotation data.

**Parameters:**
- `data`: AnnotationData object
- `n_states`: Number of states (default: 3)

**Returns:**
- Dictionary containing HMM analysis results:
  - `n_states`: Number of states
  - `most_likely_labels`: Dictionary of most likely labels for each state
  - `transition_matrix`: Matrix of transition probabilities

#### detect_anomalies

```python
def detect_anomalies(self, data: AnnotationData, method: str = 'zscore', threshold: float = 3.0) -> Dict[int, Dict[str, Any]]:
```

Detects anomalies in the annotation data.

**Parameters:**
- `data`: AnnotationData object
- `method`: Anomaly detection method (default: 'zscore')
- `threshold`: Threshold for anomaly detection (default: 3.0)

**Returns:**
- Dictionary of anomalies with frame numbers as keys
