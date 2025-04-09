# Export Guide

This guide provides detailed information about exporting annotations using the Video Annotator application.

## Export Options

The Video Annotator provides several options for exporting annotations:

1. **CSV**: Comma-separated values format for use in spreadsheet software
2. **JSON**: JavaScript Object Notation format for use in web applications
3. **TXT**: Plain text format for human readability
4. **MATLAB**: MAT-file format for use in MATLAB
5. **Excel**: XLSX format for use in Microsoft Excel

## Launching the Export Tool

You can launch the export tool using one of the following methods:

### Using the Run Scripts

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

Select "7. Run Export Tool" from the menu.

### Using the Command Line

```bash
# Run the export tool
python main.py --mode export
```

## Using the Export Tool

### 1. Select an Annotation File

When you launch the export tool, you'll need to select an annotation file (CSV) to export.

1. Click the "Browse" button
2. Navigate to your annotation file
3. Select the file and click "Open"

![Select Annotation File](../images/select_export_file.png)

### 2. Choose Export Format

The export tool allows you to choose which format to export to:

- **CSV**: For use in spreadsheet software
- **JSON**: For use in web applications
- **TXT**: For human readability
- **MATLAB**: For use in MATLAB
- **Excel**: For use in Microsoft Excel

Select the desired format from the dropdown menu.

![Choose Export Format](../images/choose_export_format.png)

### 3. Specify Output File

Specify the path for the output file:

1. Click the "Browse" button
2. Navigate to the desired location
3. Enter a file name
4. Click "Save"

The file extension will be automatically added based on the selected format.

![Specify Output File](../images/specify_output_file.png)

### 4. Export Annotations

Click the "Export" button to export the annotations to the selected format.

![Export Annotations](../images/export_annotations.png)

## Understanding Export Formats

### CSV Format

The CSV format is a simple comma-separated values format:

```
frame,label
0,0
1,0
2,1
3,1
...
```

Where:
- `frame`: The frame number (0-indexed)
- `label`: The label value (0 = Stop, 1 = Run, 2 = Turn, or your custom values)

### JSON Format

The JSON format is a structured format:

```json
{
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1,
    ...
}
```

Where:
- The key is the frame number
- The value is the label value

### TXT Format

The TXT format is a simple text format:

```
frame,label
0,0
1,0
2,1
3,1
...
```

### MATLAB Format

The MATLAB format is a binary format that can be loaded in MATLAB:

```matlab
% In MATLAB
data = load('annotations.mat');
frames = data.frames;
labels = data.labels;
```

### Excel Format

The Excel format is a spreadsheet format:

| frame | label |
|-------|-------|
| 0     | 0     |
| 1     | 0     |
| 2     | 1     |
| 3     | 1     |
| ...   | ...   |

## Batch Export

To export multiple annotation files at once, you can use a script:

```python
from src.utils.export import AnnotationExporter
from src.utils.annotation.funcs import load_annotations
import os

# Create an exporter
exporter = AnnotationExporter()

# Define the directory containing annotation files
annotation_dir = 'path/to/annotations'

# Define the output directory
output_dir = 'path/to/output'

# Define the export format
format_type = 'csv'  # or 'json', 'txt', 'matlab', 'excel'

# Export all annotation files
for file_name in os.listdir(annotation_dir):
    if file_name.endswith('.csv'):
        # Load annotations
        annotation_path = os.path.join(annotation_dir, file_name)
        annotations = load_annotations(annotation_path)
        
        # Define output path
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.{format_type}")
        
        # Export annotations
        exporter.export(annotations, output_path, format_type)
        print(f"Exported {annotation_path} to {output_path}")
```

## Programmatic Export

You can also export annotations programmatically using the Python API:

```python
from src.utils.export import AnnotationExporter
from src.utils.annotation.funcs import load_annotations

# Load annotations
annotations = load_annotations('path/to/annotations.csv')

# Create an exporter
exporter = AnnotationExporter()

# Export to CSV
exporter.export_csv(annotations, 'path/to/output.csv')

# Export to JSON
exporter.export_json(annotations, 'path/to/output.json')

# Export to TXT
exporter.export_txt(annotations, 'path/to/output.txt')

# Export to MATLAB
exporter.export_matlab(annotations, 'path/to/output.mat')

# Export to Excel
exporter.export_excel(annotations, 'path/to/output.xlsx')
```

## Next Steps

After exporting your annotations, you can:

- Use them in other applications
- Share them with collaborators
- Analyze them using custom scripts
- Visualize them using custom visualization tools
