# Adapters API Reference

This document provides a reference for the adapters API.

## AnalyzeAdapter Class

The `AnalyzeAdapter` class provides methods for adapting the analyze module to other modules.

### Methods

#### load_annotations

```python
def load_annotations(self, csv_path: str) -> Dict[int, Any]:
```

Loads annotations from a CSV file.

**Parameters:**
- `csv_path`: Path to the CSV file

**Returns:**
- Dictionary of annotations with frame numbers as keys

#### analyze_sequence

```python
def analyze_sequence(self, annotations: Dict[int, Any]) -> Dict[str, Any]:
```

Analyzes a sequence of annotations.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys

**Returns:**
- Dictionary containing analysis results:
  - `total_frames`: Total number of frames
  - `total_annotations`: Total number of annotations
  - `unique_labels`: List of unique labels
  - `label_counts`: Dictionary of label counts
  - `sequences`: List of sequences
  - `transitions`: Dictionary of transitions

## PlotAdapter Class

The `PlotAdapter` class provides methods for adapting the plot module to other modules.

### Methods

#### plot_annotations

```python
def plot_annotations(self, annotations: Dict[int, Any], title: str = 'Annotation Plot') -> None:
```

Plots annotations as an ethogram.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the plot (default: 'Annotation Plot')

#### plot_comparison

```python
def plot_comparison(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any], title: str = 'Annotation Comparison') -> None:
```

Plots a comparison of two sets of annotations.

**Parameters:**
- `annotations1`: First set of annotations
- `annotations2`: Second set of annotations
- `title`: Title for the plot (default: 'Annotation Comparison')

## VisualizationAdapter Class

The `VisualizationAdapter` class provides methods for adapting the visualization module to other modules.

### Methods

#### visualize_annotations

```python
def visualize_annotations(self, annotations: Dict[int, Any], title: str = 'Annotation Visualization') -> None:
```

Visualizes annotations using the visualization module.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the visualization (default: 'Annotation Visualization')

#### visualize_comparison

```python
def visualize_comparison(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any], title: str = 'Annotation Comparison') -> None:
```

Visualizes a comparison of two sets of annotations.

**Parameters:**
- `annotations1`: First set of annotations
- `annotations2`: Second set of annotations
- `title`: Title for the visualization (default: 'Annotation Comparison')
