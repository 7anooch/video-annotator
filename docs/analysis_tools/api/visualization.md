# Visualization API Reference

This document provides a reference for the visualization API.

## VisualizationTool Class

The `VisualizationTool` class provides methods for visualizing annotation data.

### Methods

#### visualize_annotations

```python
def visualize_annotations(self, annotations: Dict[int, Any], title: str = 'Annotation Visualization') -> None:
```

Visualizes annotations as an ethogram.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the visualization (default: 'Annotation Visualization')

#### visualize_label_distribution

```python
def visualize_label_distribution(self, annotations: Dict[int, Any], title: str = 'Label Distribution') -> None:
```

Visualizes the distribution of labels in the annotations.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the visualization (default: 'Label Distribution')

#### visualize_label_timeline

```python
def visualize_label_timeline(self, annotations: Dict[int, Any], title: str = 'Label Timeline') -> None:
```

Visualizes the timeline of labels in the annotations.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the visualization (default: 'Label Timeline')

#### visualize_transition_matrix

```python
def visualize_transition_matrix(self, annotations: Dict[int, Any], title: str = 'Transition Matrix') -> None:
```

Visualizes the transition matrix of labels in the annotations.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `title`: Title for the visualization (default: 'Transition Matrix')

#### compare_annotations

```python
def compare_annotations(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any], title: str = 'Annotation Comparison') -> None:
```

Compares two sets of annotations.

**Parameters:**
- `annotations1`: First set of annotations
- `annotations2`: Second set of annotations
- `title`: Title for the visualization (default: 'Annotation Comparison')

#### analyze_agreement

```python
def analyze_agreement(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any], title: str = 'Agreement Analysis') -> None:
```

Analyzes the agreement between two sets of annotations.

**Parameters:**
- `annotations1`: First set of annotations
- `annotations2`: Second set of annotations
- `title`: Title for the visualization (default: 'Agreement Analysis')
