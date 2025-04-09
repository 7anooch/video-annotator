# Enhanced Visualization Tools

The enhanced visualization tools provide advanced visualization capabilities for annotation data. These tools have been updated to include more interactive features and better integration with the analysis tools.

## Features

- Heatmap visualization: Visualize the distribution of annotations over time
- Comparison visualization: Compare multiple annotation sets
- Timeline visualization: Visualize annotations on a timeline
- Statistics visualization: Visualize statistical analyses of annotations
- Change point visualization: Visualize change points detected in annotations
- Segment transition matrix visualization: Visualize transitions between segments

## Usage

The enhanced visualization tools can be accessed through the main.py interface:

```bash
python main.py --mode visualize_enhanced
```

Or directly:

```bash
python -m src.tools.analysis.visualization_enhanced
```

## API Reference

The enhanced visualization tools are implemented in the `src.tools.analysis.visualization_enhanced` module.

### EnhancedVisualizer Class

The `EnhancedVisualizer` class provides methods for generating various visualizations:

- `generate_heatmap()`: Generates a heatmap visualization of annotation distribution
- `generate_comparison()`: Generates a comparison visualization of multiple annotation sets
- `generate_timeline()`: Generates a timeline visualization of annotations
- `generate_statistics()`: Generates a statistics visualization of annotations
- `generate_change_points()`: Generates a visualization of change points detected in annotations
- `generate_segment_transitions()`: Generates a visualization of transitions between segments
- `generate_all()`: Generates all available visualizations

## Examples

```python
from src.tools.analysis.visualization_enhanced import EnhancedVisualizer

# Create a visualizer
visualizer = EnhancedVisualizer()

# Load annotation data
visualizer.load_data('path/to/annotation.csv')

# Generate visualizations
visualizer.generate_heatmap()
visualizer.generate_comparison()
visualizer.generate_timeline()
visualizer.generate_statistics()
visualizer.generate_change_points()
visualizer.generate_segment_transitions()

# Or generate all visualizations at once
visualizer.generate_all()
```

## Integration with Analysis Tools

The enhanced visualization tools can be integrated with the analysis tools to provide a comprehensive analysis of annotation data. The new integration features allow you to visualize the results of advanced analyses such as change point detection and segment transition matrix analysis.

```python
from src.tools.analysis.analyze import load_annotations, analyze_sequence
from src.analysis.statistics import StatisticalAnalysis
from src.tools.analysis.visualization_enhanced import EnhancedVisualizer

# Load annotation data
annotations = load_annotations('path/to/annotation.csv')

# Perform basic analysis
results = analyze_sequence(annotations)

# Perform advanced analysis
stats = StatisticalAnalysis()
data = stats.convert_to_annotation_data(annotations)
change_points = stats.detect_change_points(data)
segment_transitions = stats.segment_transition_matrix(data)

# Create a visualizer
visualizer = EnhancedVisualizer()

# Set the annotation data
visualizer.set_data(annotations)

# Set the analysis results
visualizer.set_analysis_results(results)
visualizer.set_change_points(change_points)
visualizer.set_segment_transitions(segment_transitions)

# Generate visualizations
visualizer.generate_all()
```

## Command Line Options

The enhanced visualization tools can be run from the command line with various options:

```bash
python -m src.tools.analysis.visualization_enhanced --csv path/to/annotation.csv --mode all
```

Options:

- `--csv`: Specify one or more CSV files to visualize
- `--mode`: Specify the visualization mode (default: all)
  - `all`: Generate all visualizations
  - `heatmap`: Generate heatmap visualization
  - `comparison`: Generate comparison visualization
  - `timeline`: Generate timeline visualization
  - `statistics`: Generate statistics visualization
  - `change_points`: Generate change point visualization
  - `segment_transitions`: Generate segment transition matrix visualization
- `--advanced`: Run advanced analyses and visualize the results
