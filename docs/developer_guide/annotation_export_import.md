# Annotation Export and Import

This document describes the annotation export and import capabilities of the Video Annotator application and how to extend them.

## Overview

The Video Annotator includes modules for exporting annotations to different formats and importing annotations from various sources. These modules are:

- `src/utils/annotation_exporter.py`: Provides functionality for exporting annotations
- `src/utils/annotation_importer.py`: Provides functionality for importing annotations
- `src/ui/annotation_export_dialog.py`: Provides a user interface for exporting annotations
- `src/ui/annotation_import_dialog.py`: Provides a user interface for importing annotations

## AnnotationExporter Class

The `AnnotationExporter` class in `src/utils/annotation_exporter.py` provides the core annotation export functionality.

### Key Methods

#### export_annotations

```python
def export_annotations(self, annotations: Dict[int, Dict[str, Any]], 
                      output_path: str, format: str = None) -> bool:
    """
    Export annotations to a file.
    
    Args:
        annotations (Dict[int, Dict[str, Any]]): The annotations to export
        output_path (str): Path to save the exported annotations
        format (str, optional): Export format. Defaults to None (inferred from output_path).
        
    Returns:
        bool: True if export was successful, False otherwise
    """
```

This method exports annotations to a file in the specified format. If no format is specified, it is inferred from the output path extension.

### Supported Formats

The `AnnotationExporter` class supports the following export formats:

- `csv`: Comma-Separated Values format
- `json`: JavaScript Object Notation format
- `excel`: Microsoft Excel format
- `matlab`: MATLAB data format
- `numpy`: NumPy data format

The list of supported formats is stored in the `supported_formats` attribute of the `AnnotationExporter` class.

### Format-Specific Export Methods

The `AnnotationExporter` class includes private methods for exporting to specific formats:

- `_export_to_csv`: Export to CSV format
- `_export_to_json`: Export to JSON format
- `_export_to_excel`: Export to Excel format
- `_export_to_matlab`: Export to MATLAB format
- `_export_to_numpy`: Export to NumPy format

## AnnotationImporter Class

The `AnnotationImporter` class in `src/utils/annotation_importer.py` provides the core annotation import functionality.

### Key Methods

#### import_annotations

```python
def import_annotations(self, input_path: str, format: str = None) -> Dict[int, Dict[str, Any]]:
    """
    Import annotations from a file.
    
    Args:
        input_path (str): Path to the file to import
        format (str, optional): Import format. Defaults to None (inferred from input_path).
        
    Returns:
        Dict[int, Dict[str, Any]]: The imported annotations
    """
```

This method imports annotations from a file in the specified format. If no format is specified, it is inferred from the input path extension.

### Supported Formats

The `AnnotationImporter` class supports the same formats as the `AnnotationExporter` class:

- `csv`: Comma-Separated Values format
- `json`: JavaScript Object Notation format
- `excel`: Microsoft Excel format
- `matlab`: MATLAB data format
- `numpy`: NumPy data format

The list of supported formats is stored in the `supported_formats` attribute of the `AnnotationImporter` class.

### Format-Specific Import Methods

The `AnnotationImporter` class includes private methods for importing from specific formats:

- `_import_from_csv`: Import from CSV format
- `_import_from_json`: Import from JSON format
- `_import_from_excel`: Import from Excel format
- `_import_from_matlab`: Import from MATLAB format
- `_import_from_numpy`: Import from NumPy format

## AnnotationExportDialog Class

The `AnnotationExportDialog` class in `src/ui/annotation_export_dialog.py` provides a user interface for exporting annotations.

### Key Methods

#### export_annotations

```python
def export_annotations(self):
    """Export the annotations."""
```

This method exports the annotations using the options specified in the dialog.

## AnnotationImportDialog Class

The `AnnotationImportDialog` class in `src/ui/annotation_import_dialog.py` provides a user interface for importing annotations.

### Key Methods

#### import_annotations

```python
def import_annotations(self):
    """Import the annotations."""
```

This method imports the annotations using the options specified in the dialog and calls the callback function with the imported annotations.

## Integration with UI Controller

The annotation export and import dialogs are integrated with the UI controller in `src/ui/ui_controller.py`. The `open_export_dialog` and `open_import_dialog` methods open the respective dialogs:

```python
@exception_handler
def open_export_dialog(self):
    """Open the annotation export dialog."""
    # Get the annotations from the annotation manager
    annotations = self.annotation_manager.get_annotations()
    
    # Check if there are any annotations
    if not annotations:
        show_error_message("No annotations to export")
        return
    
    # Create the export dialog
    export_dialog = AnnotationExportDialog(self.master, annotations, self.theme_manager)
    
    # Make the dialog modal
    export_dialog.transient(self.master)
    export_dialog.grab_set()
    
    # Wait for the dialog to close
    self.master.wait_window(export_dialog)

@exception_handler
def open_import_dialog(self):
    """Open the annotation import dialog."""
    # Create a callback function to handle imported annotations
    def import_callback(annotations):
        # Set the annotations in the annotation manager
        self.annotation_manager.set_annotations(annotations)
        
        # Update the UI
        self.update_annotations_listbox()
        
        # Update status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status(f"Imported {len(annotations)} annotations")
    
    # Create the import dialog
    import_dialog = AnnotationImportDialog(self.master, import_callback, self.theme_manager)
    
    # Make the dialog modal
    import_dialog.transient(self.master)
    import_dialog.grab_set()
    
    # Wait for the dialog to close
    self.master.wait_window(import_dialog)
```

The dialogs are accessible from the "File" > "Export/Import" menu in the main application.

## Extending Export/Import Capabilities

### Adding New Export Formats

To add support for a new export format:

1. Add the format to the `supported_formats` list in the `AnnotationExporter` class:

```python
self.supported_formats = ['csv', 'json', 'excel', 'matlab', 'numpy', 'new_format']
```

2. Add a private method to the `AnnotationExporter` class to handle the new format:

```python
@exception_handler
def _export_to_new_format(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
    """
    Export annotations to the new format.
    
    Args:
        annotations (Dict[int, Dict[str, Any]]): The annotations to export
        output_path (str): Path to save the exported annotations
        
    Returns:
        bool: True if export was successful, False otherwise
    """
    # Implementation goes here
    pass
```

3. Update the `export_annotations` method to handle the new format:

```python
# Export annotations based on format
if format == 'csv':
    return self._export_to_csv(annotations, output_path)
elif format == 'json':
    return self._export_to_json(annotations, output_path)
elif format == 'excel':
    return self._export_to_excel(annotations, output_path)
elif format == 'matlab':
    return self._export_to_matlab(annotations, output_path)
elif format == 'numpy':
    return self._export_to_numpy(annotations, output_path)
elif format == 'new_format':
    return self._export_to_new_format(annotations, output_path)
else:
    self.logger.error(f"Unsupported export format: {format}")
    return False
```

### Adding New Import Formats

To add support for a new import format:

1. Add the format to the `supported_formats` list in the `AnnotationImporter` class:

```python
self.supported_formats = ['csv', 'json', 'excel', 'matlab', 'numpy', 'new_format']
```

2. Add a private method to the `AnnotationImporter` class to handle the new format:

```python
@exception_handler
def _import_from_new_format(self, input_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Import annotations from the new format.
    
    Args:
        input_path (str): Path to the file to import
        
    Returns:
        Dict[int, Dict[str, Any]]: The imported annotations
    """
    # Implementation goes here
    pass
```

3. Update the `import_annotations` method to handle the new format:

```python
# Import annotations based on format
if format == 'csv':
    return self._import_from_csv(input_path)
elif format == 'json':
    return self._import_from_json(input_path)
elif format == 'excel':
    return self._import_from_excel(input_path)
elif format == 'matlab':
    return self._import_from_matlab(input_path)
elif format == 'numpy':
    return self._import_from_numpy(input_path)
elif format == 'new_format':
    return self._import_from_new_format(input_path)
else:
    self.logger.error(f"Unsupported import format: {format}")
    return {}
```

### Updating the UI

To update the UI for the new format:

1. Update the format description in the `update_format_description` method of the `AnnotationExportDialog` class:

```python
# Set the description based on the format
if format == 'csv':
    description = "CSV (Comma-Separated Values) format. Simple text format that can be opened in Excel or other spreadsheet software."
elif format == 'json':
    description = "JSON (JavaScript Object Notation) format. Structured text format that can be parsed by many programming languages."
elif format == 'excel':
    description = "Excel format. Can be opened directly in Microsoft Excel or other compatible spreadsheet software."
elif format == 'matlab':
    description = "MATLAB format. Can be loaded directly into MATLAB for analysis."
elif format == 'numpy':
    description = "NumPy format. Can be loaded directly into Python using NumPy for analysis."
elif format == 'new_format':
    description = "Description of the new format."
else:
    description = ""
```

2. Update the format description in the `update_format_description` method of the `AnnotationImportDialog` class:

```python
# Set the description based on the format
if format == 'csv':
    description = "CSV (Comma-Separated Values) format. Simple text format that can be exported from Excel or other spreadsheet software."
elif format == 'json':
    description = "JSON (JavaScript Object Notation) format. Structured text format that can be generated by many programming languages."
elif format == 'excel':
    description = "Excel format. Can be exported from Microsoft Excel or other compatible spreadsheet software."
elif format == 'matlab':
    description = "MATLAB format. Can be exported from MATLAB for use in the Video Annotator."
elif format == 'numpy':
    description = "NumPy format. Can be exported from Python using NumPy for use in the Video Annotator."
elif format == 'new_format':
    description = "Description of the new format."
else:
    description = ""
```

## Best Practices

### Error Handling

Export and import operations can fail for various reasons. To ensure robust error handling:

- Use the `@exception_handler` decorator for all methods
- Check input parameters for validity
- Handle edge cases (e.g., empty annotations, corrupted files)
- Provide meaningful error messages
- Log errors for debugging

### Performance

Export and import operations can be slow for large annotation sets. To ensure good performance:

- Use efficient data structures
- Avoid unnecessary memory allocations
- Consider using streaming approaches for large files
- Provide progress feedback for long-running operations

### User Interface

The export and import dialogs should provide a user-friendly interface:

- Use clear and concise labels
- Provide tooltips for complex options
- Show progress indicators for long-running operations
- Validate user input before processing
- Provide feedback on success or failure

## Future Improvements

Potential improvements to the annotation export and import modules:

- Add support for more formats (e.g., XML, HDF5, SQLite)
- Add batch export and import capabilities
- Add preview functionality
- Add more validation and error recovery options
- Add support for annotation metadata
- Add support for annotation hierarchies
- Add support for annotation relationships
- Add support for annotation versioning
