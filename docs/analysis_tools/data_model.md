# Data Model Documentation

The data model provides a common representation for annotation data that is compatible with the existing annotation formats used in the Video Annotator application.

## Overview

The `AnnotationData` class is the core of the data model. It provides methods for loading, saving, and manipulating annotation data. The class is designed to be compatible with the existing annotation formats used in the Video Annotator application, including CSV, JSON, and Excel formats.

## Key Features

- **Format Compatibility**: Works with existing annotation formats (CSV, JSON, Excel).
- **Conversion Methods**: Provides methods for converting between different annotation formats.
- **Data Manipulation**: Includes methods for filtering, merging, and transforming annotation data.
- **Integration**: Seamlessly integrates with existing analysis tools through adapter classes.

## Usage

### Loading Annotations

```python
from src.analysis.data_model import AnnotationData

# Create an AnnotationData object
data = AnnotationData()

# Load annotations from a file
data.load_from_file('annotations.csv')

# Get basic information
print(f"Total annotations: {data.get_annotation_count()}")
print(f"Labels: {data.get_labels()}")
```

### Converting Between Formats

```python
# Convert to the format used by the existing analysis tools
frames, labels = data.to_analyze_format()

# Convert to the format used by the existing visualization tools
visualization_data = data.to_visualization_format()

# Convert to the format used by the existing plot tools
frames, labels = data.to_plot_format()
```

### Filtering and Merging

```python
# Filter annotations by label
filtered_data = data.filter_by_label('walking')

# Filter annotations by frame range
filtered_data = data.filter_by_frames(100, 200)

# Merge two annotation sets
merged_data = data1.merge(data2)
```

## Class Reference

### AnnotationData

```python
class AnnotationData:
    """
    Data model for annotations that is compatible with existing tools.
    
    This class provides a common data model for working with annotations,
    including loading, saving, and manipulating annotation data. It is designed
    to be compatible with the existing annotation formats used in the Video Annotator.
    
    Attributes:
        logger: The logger instance
        annotations: Dictionary of annotations with frame numbers as keys
        metadata: Dictionary of metadata about the annotations
    """
```

#### Methods

- `load_from_file(path: str, column: int = 1) -> bool`: Load annotations from a file.
- `save_to_file(path: str) -> bool`: Save annotations to a file.
- `get_frames() -> List[int]`: Get a list of all frame numbers.
- `get_labels() -> List[str]`: Get a list of all unique labels.
- `get_annotation(frame: int) -> Optional[Dict[str, Any]]`: Get the annotation for a specific frame.
- `set_annotation(frame: int, annotation: Dict[str, Any]) -> None`: Set the annotation for a specific frame.
- `delete_annotation(frame: int) -> bool`: Delete the annotation for a specific frame.
- `clear_annotations() -> None`: Clear all annotations.
- `get_annotation_count() -> int`: Get the total number of annotations.
- `get_label_counts() -> Dict[str, int]`: Get counts of each label.
- `filter_by_label(label: str) -> AnnotationData`: Filter annotations by label.
- `filter_by_frames(start_frame: int, end_frame: int) -> AnnotationData`: Filter annotations by frame range.
- `merge(other: AnnotationData, overwrite: bool = False) -> AnnotationData`: Merge with another AnnotationData object.
- `to_dataframe() -> pd.DataFrame`: Convert annotations to a pandas DataFrame.
- `from_dataframe(df: pd.DataFrame) -> bool`: Load annotations from a pandas DataFrame.
- `to_frame_label_lists() -> Tuple[List[int], List[str]]`: Convert annotations to separate lists of frames and labels.
- `from_frame_label_lists(frames: List[int], labels: List[str]) -> bool`: Load annotations from separate lists of frames and labels.
- `to_visualization_format() -> Dict[str, Any]`: Convert annotations to the format used by the existing visualization tools.
- `from_visualization_format(data: Dict[str, Any]) -> bool`: Load annotations from the format used by the existing visualization tools.
- `to_analyze_format() -> Tuple[List[int], List[str]]`: Convert annotations to the format used by the existing analyze.py module.
- `from_analyze_format(frames: List[int], labels: List[str]) -> bool`: Load annotations from the format used by the existing analyze.py module.
- `to_plot_format() -> Tuple[List[int], List[str]]`: Convert annotations to the format used by the existing plot.py module.
- `from_plot_format(frames: List[int], labels: List[str]) -> bool`: Load annotations from the format used by the existing plot.py module.
