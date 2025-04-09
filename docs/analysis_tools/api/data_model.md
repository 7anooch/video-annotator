# Data Model API Reference

## AnnotationData

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

### Constructor

```python
def __init__(self, annotations: Optional[Dict[int, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Initialize the AnnotationData.
    
    Args:
        annotations (Dict[int, Any], optional): Dictionary of annotations. Defaults to None.
        metadata (Dict[str, Any], optional): Dictionary of metadata. Defaults to None.
    """
```

### File Operations

```python
def load_from_file(self, path: str, column: int = 1) -> bool:
    """
    Load annotations from a file.
    
    Args:
        path (str): Path to the annotation file
        column (int, optional): Column index for the label in CSV files. Defaults to 1.
        
    Returns:
        bool: True if successful, False otherwise
    """
```

```python
def save_to_file(self, path: str) -> bool:
    """
    Save annotations to a file.
    
    Args:
        path (str): Path to save the annotations
        
    Returns:
        bool: True if successful, False otherwise
    """
```

### Data Access

```python
def get_frames(self) -> List[int]:
    """
    Get a list of all frame numbers.
    
    Returns:
        List[int]: List of frame numbers
    """
```

```python
def get_labels(self) -> List[str]:
    """
    Get a list of all unique labels.
    
    Returns:
        List[str]: List of unique labels
    """
```

```python
def get_annotation(self, frame: int) -> Optional[Dict[str, Any]]:
    """
    Get the annotation for a specific frame.
    
    Args:
        frame (int): Frame number
        
    Returns:
        Optional[Dict[str, Any]]: Annotation for the frame, or None if not found
    """
```

```python
def set_annotation(self, frame: int, annotation: Dict[str, Any]) -> None:
    """
    Set the annotation for a specific frame.
    
    Args:
        frame (int): Frame number
        annotation (Dict[str, Any]): Annotation data
    """
```

```python
def delete_annotation(self, frame: int) -> bool:
    """
    Delete the annotation for a specific frame.
    
    Args:
        frame (int): Frame number
        
    Returns:
        bool: True if the annotation was deleted, False if it didn't exist
    """
```

```python
def clear_annotations(self) -> None:
    """Clear all annotations."""
```

```python
def get_annotation_count(self) -> int:
    """
    Get the total number of annotations.
    
    Returns:
        int: Total number of annotations
    """
```

```python
def get_label_counts(self) -> Dict[str, int]:
    """
    Get counts of each label.
    
    Returns:
        Dict[str, int]: Dictionary of label counts
    """
```

### Data Manipulation

```python
def filter_by_label(self, label: str) -> 'AnnotationData':
    """
    Filter annotations by label.
    
    Args:
        label (str): Label to filter by
        
    Returns:
        AnnotationData: New AnnotationData object with filtered annotations
    """
```

```python
def filter_by_frames(self, start_frame: int, end_frame: int) -> 'AnnotationData':
    """
    Filter annotations by frame range.
    
    Args:
        start_frame (int): Start frame (inclusive)
        end_frame (int): End frame (inclusive)
        
    Returns:
        AnnotationData: New AnnotationData object with filtered annotations
    """
```

```python
def merge(self, other: 'AnnotationData', overwrite: bool = False) -> 'AnnotationData':
    """
    Merge with another AnnotationData object.
    
    Args:
        other (AnnotationData): Other AnnotationData object
        overwrite (bool, optional): Whether to overwrite existing annotations. Defaults to False.
        
    Returns:
        AnnotationData: New AnnotationData object with merged annotations
    """
```

### Data Conversion

```python
def to_dataframe(self) -> pd.DataFrame:
    """
    Convert annotations to a pandas DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame representation of the annotations
    """
```

```python
def from_dataframe(self, df: pd.DataFrame) -> bool:
    """
    Load annotations from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing annotations
        
    Returns:
        bool: True if successful, False otherwise
    """
```

```python
def to_frame_label_lists(self) -> Tuple[List[int], List[str]]:
    """
    Convert annotations to separate lists of frames and labels.
    This format is compatible with the existing analysis tools.
    
    Returns:
        Tuple[List[int], List[str]]: Tuple of (frames, labels)
    """
```

```python
def from_frame_label_lists(self, frames: List[int], labels: List[str]) -> bool:
    """
    Load annotations from separate lists of frames and labels.
    This format is used by the existing analysis tools.
    
    Args:
        frames (List[int]): List of frame numbers
        labels (List[str]): List of labels
        
    Returns:
        bool: True if successful, False otherwise
    """
```

```python
def to_visualization_format(self) -> Dict[str, Any]:
    """
    Convert annotations to the format used by the existing visualization tools.
    
    Returns:
        Dict[str, Any]: Dictionary with 'frames' and 'labels' keys
    """
```

```python
def from_visualization_format(self, data: Dict[str, Any]) -> bool:
    """
    Load annotations from the format used by the existing visualization tools.
    
    Args:
        data (Dict[str, Any]): Dictionary with 'frames' and 'labels' keys
        
    Returns:
        bool: True if successful, False otherwise
    """
```

```python
def to_analyze_format(self) -> Tuple[List[int], List[str]]:
    """
    Convert annotations to the format used by the existing analyze.py module.
    
    Returns:
        Tuple[List[int], List[str]]: Tuple of (frames, labels)
    """
```

```python
def from_analyze_format(self, frames: List[int], labels: List[str]) -> bool:
    """
    Load annotations from the format used by the existing analyze.py module.
    
    Args:
        frames (List[int]): List of frame numbers
        labels (List[str]): List of labels
        
    Returns:
        bool: True if successful, False otherwise
    """
```

```python
def to_plot_format(self) -> Tuple[List[int], List[str]]:
    """
    Convert annotations to the format used by the existing plot.py module.
    
    Returns:
        Tuple[List[int], List[str]]: Tuple of (frames, labels)
    """
```

```python
def from_plot_format(self, frames: List[int], labels: List[str]) -> bool:
    """
    Load annotations from the format used by the existing plot.py module.
    
    Args:
        frames (List[int]): List of frame numbers
        labels (List[str]): List of labels
        
    Returns:
        bool: True if successful, False otherwise
    """
```
