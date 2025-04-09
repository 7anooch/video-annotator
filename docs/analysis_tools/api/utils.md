# Utils API Reference

This document provides a reference for the utils API.

## FileUtils Class

The `FileUtils` class provides utility methods for file operations.

### Methods

#### load_csv

```python
def load_csv(self, csv_path: str) -> pd.DataFrame:
```

Loads a CSV file into a pandas DataFrame.

**Parameters:**
- `csv_path`: Path to the CSV file

**Returns:**
- pandas DataFrame containing the CSV data

#### save_csv

```python
def save_csv(self, data: pd.DataFrame, csv_path: str) -> None:
```

Saves a pandas DataFrame to a CSV file.

**Parameters:**
- `data`: pandas DataFrame to save
- `csv_path`: Path to save the CSV file

#### load_annotations

```python
def load_annotations(self, csv_path: str) -> Dict[int, Any]:
```

Loads annotations from a CSV file.

**Parameters:**
- `csv_path`: Path to the CSV file

**Returns:**
- Dictionary of annotations with frame numbers as keys

#### save_annotations

```python
def save_annotations(self, annotations: Dict[int, Any], csv_path: str) -> None:
```

Saves annotations to a CSV file.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys
- `csv_path`: Path to save the CSV file

## DataUtils Class

The `DataUtils` class provides utility methods for data operations.

### Methods

#### convert_to_dataframe

```python
def convert_to_dataframe(self, annotations: Dict[int, Any]) -> pd.DataFrame:
```

Converts annotations to a pandas DataFrame.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys

**Returns:**
- pandas DataFrame containing the annotations

#### convert_to_dict

```python
def convert_to_dict(self, df: pd.DataFrame) -> Dict[int, Any]:
```

Converts a pandas DataFrame to a dictionary of annotations.

**Parameters:**
- `df`: pandas DataFrame containing the annotations

**Returns:**
- Dictionary of annotations with frame numbers as keys

#### merge_annotations

```python
def merge_annotations(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any]) -> Dict[int, Any]:
```

Merges two sets of annotations.

**Parameters:**
- `annotations1`: First set of annotations
- `annotations2`: Second set of annotations

**Returns:**
- Merged dictionary of annotations

## ValidationUtils Class

The `ValidationUtils` class provides utility methods for validation.

### Methods

#### validate_annotations

```python
def validate_annotations(self, annotations: Dict[int, Any]) -> bool:
```

Validates annotations.

**Parameters:**
- `annotations`: Dictionary of annotations with frame numbers as keys

**Returns:**
- True if annotations are valid, False otherwise

#### validate_csv

```python
def validate_csv(self, csv_path: str) -> bool:
```

Validates a CSV file.

**Parameters:**
- `csv_path`: Path to the CSV file

**Returns:**
- True if CSV file is valid, False otherwise
