# Enhanced Visualization Guide

This guide explains how to use the enhanced visualization tools in the Video Annotator.

## Overview

The enhanced visualization tools provide advanced visualization capabilities for annotation data, including:

- **Heatmaps**: Visualize annotation density across video frames
- **Comparison Visualizations**: Compare annotations from multiple annotators
- **Timeline Visualizations**: View annotations as a timeline
- **Statistics Visualizations**: Analyze annotation statistics

## Running the Enhanced Visualization Tool

You can run the enhanced visualization tool using the main.py script:

```bash
python main.py --mode visualize_enhanced
```

Or you can run it directly:

```bash
python src/tools/visualization_enhanced.py
```

## Command-line Options

The enhanced visualization tool supports the following command-line options:

```bash
python src/tools/visualization_enhanced.py --csv [csv_file1] [csv_file2] --bin_size 100 --mode all
```

- `--csv`: Specify one or more CSV files to visualize
- `--bin_size`: Specify the bin size for heatmap visualization (default: 100)
- `--mode`: Specify the visualization mode (default: all)
  - `all`: Show all visualizations
  - `heatmap`: Show only heatmap visualization
  - `comparison`: Show only comparison visualization
  - `timeline`: Show only timeline visualization
  - `statistics`: Show only statistics visualization

## Visualization Types

### Heatmap Visualization

The heatmap visualization shows the density of annotations across video frames. It divides the video into bins and displays the proportion of each label within each bin.

![Heatmap Visualization](../images/heatmap_visualization.png)

This visualization is useful for:
- Identifying patterns in annotation density
- Detecting regions with high annotation variability
- Comparing annotation patterns across multiple files

### Comparison Visualization

The comparison visualization shows annotations from multiple files side by side, making it easy to compare annotations from different annotators.

![Comparison Visualization](../images/comparison_visualization.png)

This visualization is useful for:
- Comparing annotations from multiple annotators
- Identifying discrepancies between annotators
- Validating annotation consistency

### Timeline Visualization

The timeline visualization shows annotations as a timeline, with different colors representing different labels.

![Timeline Visualization](../images/timeline_visualization.png)

This visualization is useful for:
- Viewing the sequence of annotations
- Identifying transitions between labels
- Analyzing the duration of different behaviors

### Statistics Visualization

The statistics visualization shows counts and percentages of each label across multiple files.

![Statistics Visualization](../images/statistics_visualization.png)

This visualization is useful for:
- Comparing label distributions across files
- Identifying biases in annotation
- Analyzing annotation consistency

## Examples

### Visualizing a Single Annotation File

```bash
python src/tools/visualization_enhanced.py --csv path/to/annotation.csv
```

This will show all visualization types for the specified file.

### Comparing Multiple Annotation Files

```bash
python src/tools/visualization_enhanced.py --csv path/to/annotation1.csv path/to/annotation2.csv --mode comparison
```

This will show a comparison visualization for the specified files.

### Creating a Heatmap with Custom Bin Size

```bash
python src/tools/visualization_enhanced.py --csv path/to/annotation.csv --bin_size 50 --mode heatmap
```

This will show a heatmap visualization with a bin size of 50 frames.

## Next Steps

For more information about other visualization tools, see the [Visualization Guide](visualization_guide.md).
