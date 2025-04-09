# Annotation Export and Import

This guide explains how to export and import annotations in the Video Annotator.

## Overview

The Video Annotator allows you to export annotations to different formats and import annotations from various sources. This functionality is useful for:

- Sharing annotations with colleagues
- Analyzing annotations in other software
- Backing up annotations
- Transferring annotations between projects

## Supported Formats

The Video Annotator supports the following formats for both export and import:

- **CSV** (Comma-Separated Values): Simple text format that can be opened in Excel or other spreadsheet software
- **JSON** (JavaScript Object Notation): Structured text format that can be parsed by many programming languages
- **Excel**: Microsoft Excel format (.xlsx)
- **MATLAB**: MATLAB data format (.mat)
- **NumPy**: NumPy data format (.npz)

## Exporting Annotations

To export annotations:

1. Open a video file and its annotations in the Video Annotator
2. Select "File" > "Export/Import" > "Export Annotations" from the menu bar
3. In the Export Annotations dialog:
   - Specify the output file path by clicking "Browse"
   - Select the export format from the dropdown menu
   - Click "Export" to save the annotations

![Export Annotations Dialog](../images/export_annotations_dialog.png)

### Export Format Details

#### CSV Format

The CSV format exports annotations as a table with the following columns:

- `frame`: The frame number
- Additional columns for each annotation property (e.g., `name`, `value`, `color`)

Example:
```
frame,name,value,color
10,Stop,0,red
25,Run,1,green
40,Turn,2,blue
```

#### JSON Format

The JSON format exports annotations as a structured array of objects:

```json
[
    {
        "frame": 10,
        "name": "Stop",
        "value": 0,
        "color": "red"
    },
    {
        "frame": 25,
        "name": "Run",
        "value": 1,
        "color": "green"
    },
    {
        "frame": 40,
        "name": "Turn",
        "value": 2,
        "color": "blue"
    }
]
```

#### Excel Format

The Excel format exports annotations as a table similar to the CSV format, but in a native Excel file (.xlsx) that can be opened directly in Microsoft Excel or other compatible spreadsheet software.

#### MATLAB Format

The MATLAB format exports annotations as a MATLAB data file (.mat) with the following variables:

- `frame`: Array of frame numbers
- Additional variables for each annotation property

#### NumPy Format

The NumPy format exports annotations as a NumPy data file (.npz) with the following arrays:

- `frame`: Array of frame numbers
- Additional arrays for each annotation property

## Importing Annotations

To import annotations:

1. Open a video file in the Video Annotator
2. Select "File" > "Export/Import" > "Import Annotations" from the menu bar
3. In the Import Annotations dialog:
   - Specify the input file path by clicking "Browse"
   - Select the import format from the dropdown menu
   - Click "Import" to load the annotations

![Import Annotations Dialog](../images/import_annotations_dialog.png)

### Import Format Requirements

For successful import, the annotation file should have the following structure:

#### CSV Format

The CSV file should have a header row and include a `frame` column:

```
frame,name,value,color
10,Stop,0,red
25,Run,1,green
40,Turn,2,blue
```

#### JSON Format

The JSON file should contain an array of objects, each with a `frame` property:

```json
[
    {
        "frame": 10,
        "name": "Stop",
        "value": 0,
        "color": "red"
    },
    {
        "frame": 25,
        "name": "Run",
        "value": 1,
        "color": "green"
    },
    {
        "frame": 40,
        "name": "Turn",
        "value": 2,
        "color": "blue"
    }
]
```

#### Excel Format

The Excel file should have a header row and include a `frame` column, similar to the CSV format.

#### MATLAB Format

The MATLAB file should include a `frame` variable containing frame numbers, along with variables for annotation properties.

#### NumPy Format

The NumPy file should include a `frame` array containing frame numbers, along with arrays for annotation properties.

## Batch Export/Import

### Batch Export

To export annotations for multiple videos:

1. Create a directory to store the exported annotation files
2. Open each video in the Video Annotator and export its annotations to the directory
3. Use a consistent naming convention for the exported files

### Batch Import

To import annotations for multiple videos:

1. Organize the annotation files in a directory
2. Open each video in the Video Annotator and import its corresponding annotation file

## Tips for Export/Import

### File Naming

Use a consistent naming convention for annotation files to make them easy to identify and manage. For example:

- `video_name_annotations.csv`
- `video_name_annotations.json`
- `video_name_annotations.xlsx`

### Data Validation

After importing annotations, verify that they were imported correctly by:

- Checking the number of annotations
- Viewing some annotations in the Video Annotator
- Comparing the imported annotations with the original source

### Compatibility

When sharing annotations with others, consider using a format that is widely supported:

- CSV is the most universally compatible format
- JSON is well-supported by most programming languages
- Excel is useful for sharing with non-technical users

### Backup

Regularly export annotations as a backup to prevent data loss. Store backups in a different location from the original files.

## Troubleshooting

### "Failed to export annotations"

This error can occur if:

- The output directory does not exist and could not be created
- You don't have write permissions for the output directory
- The annotations are in an invalid format

Check that you have write permissions for the output directory and that the annotations are valid.

### "Failed to import annotations or no annotations found"

This error can occur if:

- The input file does not exist
- The input file is in an incorrect format
- The input file does not contain any annotations
- The input file does not have a `frame` column or property

Check that the input file exists, is in the correct format, and contains annotations with frame numbers.

### "No annotations to export"

This error occurs when you try to export annotations but there are no annotations to export. Create some annotations before exporting.

## Programmatic Export/Import

Advanced users can use the Annotation Exporter and Importer APIs to export and import annotations programmatically:

```python
from src.utils.annotation_exporter import AnnotationExporter
from src.utils.annotation_importer import AnnotationImporter

# Export annotations
exporter = AnnotationExporter()
exporter.export_annotations(annotations, 'path/to/output.csv', 'csv')

# Import annotations
importer = AnnotationImporter()
annotations = importer.import_annotations('path/to/input.csv', 'csv')
```
