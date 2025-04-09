# Utils API

The Utils API provides utility functions for logging, error handling, and other common tasks.

## Logger

The `logger` module provides logging functionality.

### Functions

#### setup_logger

```python
setup_logger(name, level=logging.INFO, log_file='video_annotator.log')
```

Set up a logger with the specified name and level.

- `name` (str): The name of the logger
- `level` (int, optional): The logging level. Defaults to logging.INFO.
- `log_file` (str, optional): The path to the log file. Defaults to 'video_annotator.log'.
- Returns: logging.Logger: The configured logger

## Error Handling

The `error_handling` module provides error handling utilities.

### Functions

#### exception_handler

```python
exception_handler(func)
```

Decorator for handling exceptions in functions.

- `func` (function): The function to decorate
- Returns: function: The decorated function

#### show_error_message

```python
show_error_message(message, title="Error", parent=None)
```

Show an error message dialog.

- `message` (str): The error message
- `title` (str, optional): The dialog title. Defaults to "Error".
- `parent` (tk.Widget, optional): The parent widget. Defaults to None.

#### show_warning_message

```python
show_warning_message(message, title="Warning", parent=None)
```

Show a warning message dialog.

- `message` (str): The warning message
- `title` (str, optional): The dialog title. Defaults to "Warning".
- `parent` (tk.Widget, optional): The parent widget. Defaults to None.

#### show_info_message

```python
show_info_message(message, title="Information", parent=None)
```

Show an information message dialog.

- `message` (str): The information message
- `title` (str, optional): The dialog title. Defaults to "Information".
- `parent` (tk.Widget, optional): The parent widget. Defaults to None.

## Export

The `export` module provides utilities for exporting annotations.

### Classes

#### AnnotationExporter

The `AnnotationExporter` class is responsible for exporting annotations to different formats.

##### Constructor

```python
AnnotationExporter()
```

##### Methods

###### export_csv

```python
export_csv(annotations, output_path)
```

Export annotations to CSV format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- Returns: bool: True if successful, False otherwise

###### export_json

```python
export_json(annotations, output_path)
```

Export annotations to JSON format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- Returns: bool: True if successful, False otherwise

###### export_txt

```python
export_txt(annotations, output_path)
```

Export annotations to TXT format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- Returns: bool: True if successful, False otherwise

###### export_matlab

```python
export_matlab(annotations, output_path)
```

Export annotations to MATLAB format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- Returns: bool: True if successful, False otherwise

###### export_excel

```python
export_excel(annotations, output_path)
```

Export annotations to Excel format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- Returns: bool: True if successful, False otherwise

###### export

```python
export(annotations, output_path, format_type)
```

Export annotations to the specified format.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_path` (str): Path to the output file
- `format_type` (str): Format type (csv, json, txt, matlab, excel)
- Returns: bool: True if successful, False otherwise

## Annotation

The `annotation` module provides utilities for working with annotations.

### Functions

#### save_annotations

```python
save_annotations(annotations, output_csv_path)
```

Save annotations to a CSV file.

- `annotations` (dict): Dictionary of frame numbers to labels
- `output_csv_path` (str): Path to the output CSV file

#### get_csv_file_path

```python
get_csv_file_path(video_path, output_csv_name=None)
```

Get the path to the CSV file for annotations.

- `video_path` (str): Path to the video file
- `output_csv_name` (str, optional): Name of the output CSV file. Defaults to None.
- Returns: str: Path to the CSV file

#### get_common_substring

```python
get_common_substring(strs)
```

Get the longest common substring from a list of strings.

- `strs` (list): List of strings
- Returns: str: The longest common substring

#### format_frames_and_ranges

```python
format_frames_and_ranges(frames)
```

Format a list of frames as a string of ranges.

- `frames` (list): List of frame numbers
- Returns: str: Formatted string of ranges
