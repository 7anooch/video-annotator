# Video Annotator

A tool for annotating video frames with behavioral labels, primarily designed for neuroscience research.

## Features

- Frame-by-frame video navigation
- Keyboard shortcuts for quick annotation
- Annotation visualization with ethograms
- Advanced visualization tools (heatmaps, comparisons, timelines)
- Support for multiple annotation sets
- Ground truth generation from multiple annotators
- Analysis tools for comparing annotations
- Advanced statistical analyses including:
  - Change point detection with visualization
  - Segment transition matrix with visualization
  - Hidden Markov Model analysis
  - Anomaly detection

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.12 or higher

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator

# Create and activate conda environment
conda env create -f environment.yml
conda activate video-annotator
```

### Optional Dependencies for Advanced Analyses

For the full functionality of advanced analyses, you need to install the following packages:

```bash
# Install ruptures for change point detection
conda install -c conda-forge ruptures

# Install hmmlearn for Hidden Markov Model analysis
conda install -c conda-forge hmmlearn

# Install seaborn for enhanced visualizations
conda install -c conda-forge seaborn

# Install other optional dependencies
conda install -c conda-forge plotly scikit-learn scipy statsmodels
```

For detailed installation instructions, troubleshooting, and configuration options, please see the [Installation Guide](INSTALL.md).

## Usage

### Video Annotation

To start the annotation tool:

```bash
python annotator.py
```

This will open a file dialog to select a video file. Once selected, the annotation interface will appear.

#### Command-line Options

```bash
python annotator.py --csv [annotation_file_name] --side_controls --video [video_path]
```

- `--csv`: Specify the name of the annotation CSV file (default: video_name_annotation.csv)
- `--side_controls`: Place controls on the right side of the interface
- `--video`: Specify the path to the video file (optional)

#### Keyboard Shortcuts

- **Left Arrow**: Previous frame
- **Right Arrow**: Next frame
- **Spacebar**: Play/Pause
- **S**: Annotate as "Stop"
- **R**: Annotate as "Run"
- **T**: Annotate as "Turn"

### Visualization

To visualize annotations as ethograms:

```bash
python main.py --mode plot
```

This will open a file dialog to select one or more annotation CSV files. Once selected, ethograms will be displayed.

#### Plot Command-line Options

```bash
python -m src.tools.analysis.visualization --use_cols 2,3
```

- `--use_cols`: Specify which columns to use from the CSV file (default: 2)

### Enhanced Visualization

To use the enhanced visualization tools:

```bash
python main.py --mode visualize_enhanced
```

This will open a file dialog to select one or more annotation CSV files. Once selected, various visualizations will be displayed, including heatmaps, comparison visualizations, timeline visualizations, and statistics visualizations.

#### Enhanced Visualization Command-line Options

```bash
python -m src.tools.analysis.visualization_enhanced --csv [csv_file1] [csv_file2] --bin_size 100 --mode all
```

- `--csv`: Specify one or more CSV files to visualize
- `--bin_size`: Specify the bin size for heatmap visualization (default: 100)
- `--mode`: Specify the visualization mode (default: all)
  - `all`: Show all visualizations
  - `heatmap`: Show only heatmap visualization
  - `comparison`: Show only comparison visualization
  - `timeline`: Show only timeline visualization
  - `statistics`: Show only statistics visualization

### Ground Truth Generation

To generate ground truth from multiple annotations:

```bash
python gen_ground_truth.py
```

This will open a file dialog to select multiple annotation CSV files. The tool will then generate a ground truth file based on the consensus of the annotations.

### Analysis

To analyze and compare annotations:

```bash
python main.py --mode analyze
```

This will open a file dialog to select annotation files for analysis. The tool will generate statistics and comparisons between the annotations.

#### Basic Analysis

```bash
# Run basic analysis with a file dialog
python main.py --mode analyze

# Run basic analysis with specific CSV files
python main.py --mode analyze --csv [csv_file1] [csv_file2]

# Using the module directly
python -m src.tools.analysis.analyze --csv [csv_file1] [csv_file2]
```

#### Advanced Analysis

```bash
# Run advanced analysis with a file dialog
python main.py --mode analyze --advanced

# Run advanced analysis with specific CSV files
python main.py --mode analyze --csv [csv_file1] [csv_file2] --advanced

# Using the module directly
python -m src.tools.analysis.analyze --csv [csv_file1] [csv_file2] --advanced
```

The `--advanced` option enables additional analyses including:

- Change point detection with visualization
- Segment transition matrix with visualization
- Hidden Markov Model analysis (requires hmmlearn package)
- Anomaly detection (Z-score and IQR methods)

## File Formats

### Annotation CSV

The annotation CSV file has the following format:

```csv
frame,label
0,0
1,0
2,1
3,1
...
```

- `frame`: Frame number (0-indexed)
- `label`: Annotation label (0 = Stop, 1 = Run, 2 = Turn)

## Examples

### Basic Annotation Workflow

1. Run `python annotator.py`
2. Select a video file
3. Use keyboard shortcuts (S, R, T) to annotate frames
4. Annotations are automatically saved to a CSV file

### Generating Ground Truth from Multiple Annotators

1. Have multiple annotators annotate the same video
2. Run `python gen_ground_truth.py`
3. Select all annotation CSV files
4. A ground truth file will be generated based on the consensus

### Visualizing Multiple Annotations

1. Run `python main.py --mode plot`
2. Select multiple annotation CSV files
3. Ethograms will be displayed for each annotation file

### Running Advanced Analyses

1. Run `python main.py --mode analyze --advanced`
2. Select annotation CSV files
3. The tool will generate basic statistics, change point detection, segment transition matrix, and anomaly detection
4. Visualizations will be displayed in your web browser

## Project Structure

The project is organized into the following directories:

```bash
.
├── data/                  # Sample data and test files
├── docs/                  # Documentation
│   ├── analysis_tools/    # Documentation for analysis tools
│   ├── developer_guide/   # Guide for developers
│   └── ...                # Other documentation
├── src/                   # Source code
│   ├── analysis/          # Advanced analysis modules
│   │   ├── adapters/      # Adapters for integrating with other modules
│   │   ├── data_model.py  # Data models for analysis
│   │   └── statistics.py  # Statistical analysis tools
│   ├── examples/          # Example scripts
│   ├── tools/             # Core tools
│   │   ├── analysis/      # Analysis tools
│   │   │   ├── analyze.py            # Analysis functions
│   │   │   ├── visualization.py      # Basic visualization tools
│   │   │   └── visualization_enhanced.py # Enhanced visualization tools
│   │   ├── analyze.py     # Legacy analysis module (for backward compatibility)
│   │   └── ...            # Other tools
│   ├── ui/                # User interface components
│   └── utils/             # Utility functions
└── tests/                 # Test files
    └── tools/             # Tests for tools
```

## Documentation

For more detailed documentation, please see the [docs](docs/) directory.

- [User Guide](docs/user_guide.md): Guide for users
- [Developer Guide](docs/developer_guide/index.md): Guide for developers
- [API Reference](docs/api_reference.md): API reference
- [Analysis Tools](docs/analysis_tools/README.md): Documentation for analysis tools
- [Statistical Analysis](docs/statistical_analysis.md): Documentation for statistical analysis tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development

If you're interested in contributing to the development of Video Annotator, please check out the following resources:

- [Comprehensive Improvement Plan](docs/comprehensive_improvement_plan.md): Detailed plan for improving the project
- [Developer Guide](docs/developer_guide/index.md): Guide for developers
- [Refactoring Guide](src/utils/refactoring_guide.md): Guidelines for refactoring the codebase

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool was developed for neuroscience research at UCSB
- Thanks to all contributors and users for their feedback and suggestions
