# Analysis Guide

This guide provides detailed information about analyzing annotations using the Video Annotator application.

## Analysis Tools

The Video Annotator provides several tools for analyzing annotations:

1. **Annotation Statistics**: Calculate statistics about annotations
2. **Annotation Comparison**: Compare annotations from multiple annotators
3. **Ground Truth Generation**: Generate ground truth from multiple annotations

## Launching the Analysis Tool

You can launch the analysis tool using one of the following methods:

### Using the Run Scripts

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

Select "5. Run Analysis Tool" from the menu.

### Using the Command Line

```bash
# Run the analysis tool
python main.py --mode analyze
```

## Using the Analysis Tool

### 1. Select Annotation Files

When you launch the analysis tool, you'll need to select one or more annotation files (CSV) to analyze.

1. Click the "Browse" button or wait for the file dialog to appear
2. Navigate to your annotation files
3. Select one or more files and click "Open"

![Select Annotation Files](../images/select_annotation_files.png)

### 2. Analyze Annotations

The analysis tool will automatically analyze the selected annotation files and display the results.

## Understanding the Analysis Results

### Annotation Statistics

The analysis tool calculates several statistics for each annotation file:

- **Number of Annotations**: The total number of annotated frames
- **Number of Frames**: The total number of frames in the video
- **Annotation Density**: The percentage of frames that are annotated
- **Label Distribution**: The distribution of labels in the annotations

![Annotation Statistics](../images/annotation_statistics.png)

### Annotation Comparison

When multiple annotation files are selected, the analysis tool compares them and calculates:

- **Agreement Percentage**: The percentage of frames where all annotators agree
- **Cohen's Kappa**: A measure of inter-annotator agreement
- **Confusion Matrix**: A matrix showing how often annotators agree or disagree

![Annotation Comparison](../images/annotation_comparison.png)

### Frame-by-Frame Comparison

The analysis tool also provides a frame-by-frame comparison of annotations:

- **Agreement Frames**: Frames where all annotators agree
- **Disagreement Frames**: Frames where annotators disagree
- **Annotation Gaps**: Frames that are not annotated by all annotators

![Frame-by-Frame Comparison](../images/frame_comparison.png)

## Generating Ground Truth

To generate ground truth from multiple annotations, you can use the ground truth generator:

```bash
# Using the run scripts
# Select "6. Run Ground Truth Generator" from the menu

# Using the command line
python main.py --mode ground_truth
```

The ground truth generator will:

1. Load multiple annotation files
2. Identify frames where annotators agree
3. Use a majority vote for frames where annotators disagree
4. Generate a ground truth annotation file

## Patching Gaps in Annotations

To patch gaps in annotations, you can use the patch gaps tool:

```bash
# Using the run scripts
# Select "10. Run Patch Gaps Tool" from the menu

# Using the command line
python main.py --mode patch_gaps
```

The patch gaps tool will:

1. Identify gaps in annotations
2. Fill in the gaps based on surrounding annotations
3. Generate a patched annotation file

## Exporting Analysis Results

The analysis tool can export results to various formats:

1. **CSV**: For further analysis in spreadsheet software
2. **JSON**: For use in other applications
3. **Text**: For human-readable reports

To export results, click the "Export" button and select the desired format.

## Advanced Analysis

### Custom Analysis Scripts

For advanced analysis, you can use the provided Python API:

```python
from src.tools.analyze import load_annotations, analyze_sequence

# Load annotations
annotations = load_annotations('path/to/annotations.csv')

# Analyze the sequence
sequence_summary = analyze_sequence(annotations)

# Print the results
for label, duration, frames in sequence_summary:
    print(f"Label: {label}, Duration: {duration}, Frames: {len(frames)}")
```

### Batch Analysis

For batch analysis of multiple videos, you can create a script that:

1. Finds all annotation files in a directory
2. Analyzes each file
3. Aggregates the results

## Next Steps

After analyzing your annotations, you can:

- [Visualize](visualization_guide.md) your annotations to identify patterns
- [Export](export_guide.md) your annotations for use in other applications
- Generate ground truth for training machine learning models
