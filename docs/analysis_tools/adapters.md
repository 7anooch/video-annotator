# Adapters Documentation

The adapters module provides adapter classes that interface with the existing analysis tools in the Video Annotator application.

## Overview

The adapters module includes three main adapter classes:

- `AnalyzeAdapter`: Interfaces with the existing analyze.py module.
- `PlotAdapter`: Interfaces with the existing plot.py module.
- `VisualizationAdapter`: Interfaces with the existing visualization.py module.

These adapter classes provide a bridge between the enhanced analysis tools and the existing tools, allowing them to work together seamlessly.

## Key Features

- **Seamless Integration**: Provides a seamless integration between the enhanced analysis tools and the existing tools.
- **Compatibility**: Ensures compatibility with the existing annotation formats and analysis workflows.
- **Extensibility**: Allows for easy extension of the existing tools with new functionality.

## Usage

### AnalyzeAdapter

```python
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.data_model import AnnotationData

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create an analyze adapter
analyze_adapter = AnalyzeAdapter()

# Calculate precision and recall
ground_truth = data.annotations
predictions = another_data.annotations
precision_recall = analyze_adapter.calculate_precision_recall(ground_truth, predictions)
print(f"Precision: {precision_recall['precision']:.2f}")
print(f"Recall: {precision_recall['recall']:.2f}")
print(f"F1 Score: {precision_recall['f1_score']:.2f}")

# Analyze sequences
sequence_analysis = analyze_adapter.analyze_sequence(data.annotations)
print(f"Sequences: {sequence_analysis['sequences']}")
print(f"Counts: {sequence_analysis['counts']}")
```

### PlotAdapter

```python
from src.analysis.adapters.plot_adapter import PlotAdapter
from src.analysis.data_model import AnnotationData

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a plot adapter
plot_adapter = PlotAdapter()

# Plot an ethogram
fig = plot_adapter.plot_ethogram(data.annotations)
fig.show()

# Get a color map
labels = data.get_labels()
color_map = plot_adapter.get_color_map(labels)
print(f"Color map: {color_map}")
```

### VisualizationAdapter

```python
import tkinter as tk
from src.analysis.adapters.visualization_adapter import VisualizationAdapter
from src.analysis.data_model import AnnotationData

# Load annotations
data = AnnotationData()
data.load_from_file('annotations.csv')

# Create a visualization adapter
visualization_adapter = VisualizationAdapter()

# Visualize annotations
visualization_adapter.visualize_annotations(data.annotations)

# Compare annotations
visualization_adapter.compare_annotations(data.annotations, another_data.annotations)

# Create a visualization GUI
root = tk.Tk()
gui = visualization_adapter.create_visualization_gui(root, data.annotations)
root.mainloop()
```

## Class Reference

### AnalyzeAdapter

```python
class AnalyzeAdapter:
    """
    Adapter for the analyze.py module.
    
    This class provides an interface to the functionality in the existing analyze.py module.
    It allows the enhanced analysis tools to use the existing analysis functions.
    
    Attributes:
        logger: The logger instance
    """
```

#### Methods

- `calculate_precision_recall(ground_truth: Dict[int, Dict[str, Any]], predictions: Dict[int, Dict[str, Any]]) -> Dict[str, Any]`: Calculate precision and recall using the existing analyze.py module.
- `analyze_sequence(annotations: Dict[int, Dict[str, Any]]) -> Dict[str, Any]`: Analyze sequence patterns using the existing analyze.py module.
- `calculate_confusion_matrix(ground_truth: Dict[int, Dict[str, Any]], predictions: Dict[int, Dict[str, Any]]) -> np.ndarray`: Calculate confusion matrix using the existing analyze.py module.
- `align_sequences(sequence1: List[str], sequence2: List[str]) -> Dict[str, Any]`: Align two sequences using the existing analyze.py module.

### PlotAdapter

```python
class PlotAdapter:
    """
    Adapter for the plot.py module.
    
    This class provides an interface to the functionality in the existing plot.py module.
    It allows the enhanced analysis tools to use the existing plotting functions.
    
    Attributes:
        logger: The logger instance
    """
```

#### Methods

- `plot_ethogram(annotations: Dict[int, Dict[str, Any]], figsize: Tuple[int, int] = (12, 6), title: str = 'Ethogram', save_path: Optional[str] = None) -> plt.Figure`: Plot an ethogram using the existing plot.py module.
- `plot_confusion_matrix(confusion_matrix: np.ndarray, labels: List[str], figsize: Tuple[int, int] = (10, 8), title: str = 'Confusion Matrix', save_path: Optional[str] = None) -> plt.Figure`: Plot a confusion matrix using the existing plot.py module.
- `plot_precision_recall(precision: float, recall: float, f1: float, figsize: Tuple[int, int] = (8, 6), title: str = 'Precision and Recall', save_path: Optional[str] = None) -> plt.Figure`: Plot precision and recall metrics using the existing plot.py module.
- `get_color_map(labels: List[str]) -> Dict[str, str]`: Get a color map for labels using the existing plot.py module.

### VisualizationAdapter

```python
class VisualizationAdapter:
    """
    Adapter for the visualization.py module.
    
    This class provides an interface to the functionality in the existing visualization.py module.
    It allows the enhanced analysis tools to use the existing visualization functions.
    
    Attributes:
        logger: The logger instance
    """
```

#### Methods

- `create_visualization_gui(master: tk.Tk, annotations: Dict[int, Dict[str, Any]]) -> visualization.VisualizationGUI`: Create a visualization GUI using the existing visualization.py module.
- `visualize_annotations(annotations: Dict[int, Dict[str, Any]]) -> None`: Visualize annotations using the existing visualization.py module.
- `compare_annotations(annotations1: Dict[int, Dict[str, Any]], annotations2: Dict[int, Dict[str, Any]], labels1: str = 'Set 1', labels2: str = 'Set 2') -> None`: Compare two sets of annotations using the existing visualization.py module.
