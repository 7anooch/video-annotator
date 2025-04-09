# Analysis Tools

This directory contains tools for analyzing and visualizing annotation data.

## Modules

- `analyze.py`: Functions for analyzing annotation data
- `visualization.py`: Basic visualization tools
- `visualization_enhanced.py`: Enhanced visualization tools

## Usage

These modules can be imported and used in your code:

```python
from src.tools.analysis.analyze import analysis
from src.tools.analysis.visualization import VisualizationTool
from src.tools.analysis.visualization_enhanced import EnhancedVisualizationTool
```

Or they can be run directly from the command line:

```bash
# Run the analysis tools
python -m src.tools.analysis.analyze --csv path/to/your/annotation.csv

# Run the visualization tools
python -m src.tools.analysis.visualization

# Run the enhanced visualization tools
python -m src.tools.analysis.visualization_enhanced
```

You can also run them through the main.py interface:

```bash
# Run the analysis tools
python main.py --mode analyze --csv path/to/your/annotation.csv

# Run the analysis tools with advanced analyses
python main.py --mode analyze --csv path/to/your/annotation.csv --advanced
```
