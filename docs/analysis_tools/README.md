# Enhanced Analysis Tools Documentation

This documentation provides a comprehensive guide to the enhanced analysis tools for the Video Annotator application. These tools complement and extend the existing analysis functionality, providing more advanced analysis capabilities, interactive visualizations, and improved data handling.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [Usage Examples](#usage-examples)
5. [Integration with Existing Tools](#integration-with-existing-tools)
6. [API Reference](#api-reference)

## Overview

The enhanced analysis tools provide a set of advanced analysis and visualization capabilities for annotation data. They are designed to work seamlessly with the existing analysis tools in the Video Annotator application, extending their functionality rather than replacing them.

Key features include:

- **Advanced Statistical Analysis**: Enhanced statistical analysis functions including correlation analysis, time series analysis, and anomaly detection.
- **Interactive Visualizations**: Interactive timeline and 3D visualizations using Plotly.
- **Comprehensive Quality Assessment**: Advanced metrics for assessing annotation quality.
- **Seamless Integration**: Adapters that interface with existing analysis tools.

## Installation

The enhanced analysis tools are included in the Video Annotator application. However, some features require additional dependencies:

```bash
# Install required dependencies
conda install numpy pandas matplotlib

# Install optional dependencies for advanced features
conda install -c conda-forge plotly scikit-learn scipy statsmodels

# Install dependencies for change point detection and HMM analysis
conda install -c conda-forge ruptures hmmlearn
```

## Components

The enhanced analysis tools consist of the following components:

- [**Data Model**](data_model.md): A compatible data model for annotations that works with existing formats.
- [**Statistical Analysis**](statistics.md): Enhanced statistical analysis functions.
- [**Visualization**](visualization.md): Enhanced visualization functions with interactive capabilities.
- [**Enhanced Visualization**](visualization_enhanced.md): Advanced visualization tools with interactive capabilities.
- [**Adapters**](adapters.md): Adapter classes that interface with existing analysis tools.
- [**Utilities**](utils.md): Utility functions that extend existing functionality.

## Usage Examples

### Running the Enhanced Analysis Tools

#### Using the Main Interface

You can now run the enhanced analysis tools through the main.py interface:

```bash
# Run advanced analyses with a file dialog
python main.py --mode analyze --advanced

# Run advanced analyses with specific CSV files
python main.py --mode analyze --csv path/to/your/annotation.csv --advanced
```

#### Using the Module Directly

You can also run the enhanced analysis tools directly:

```bash
# Run the analysis tools with a file dialog
python -m src.tools.analysis.analyze --advanced

# Run the analysis tools with specific CSV files
python -m src.tools.analysis.analyze --csv path/to/your/annotation.csv --advanced
```

#### Using the Example Scripts

You can also run the example scripts:

```bash
# Run the advanced statistical analysis example
python src/examples/advanced_statistical_analysis_example.py <annotation_file>

# Run the basic analysis example
python src/examples/basic_analysis_example.py <annotation_file>

# Run the visualization example
python src/examples/visualization_example.py <annotation_file>

# Run the comparison example
python src/examples/comparison_example.py <annotation_file_1> <annotation_file_2>

# Run the integration demo
python src/examples/integration_demo.py <annotation_file>
```

### Using the Existing Analysis Tools

The existing analysis tools can be accessed through the new module structure:

```bash
# Run the analysis tools with a file dialog
python -m src.tools.analysis.analyze
```

This will open a file dialog for you to select the CSV file(s) you want to analyze. If you prefer to specify the files directly in the command line, you can use the `--csv` argument:

```bash
# Run the analysis tools with specific CSV files
python -m src.tools.analysis.analyze --csv path/to/your/annotation.csv

# You can specify multiple CSV files
python -m src.tools.analysis.analyze --csv file1.csv file2.csv file3.csv
```

You can also use the main.py interface:

```bash
# Run the analysis tools with a file dialog
python main.py --mode analyze

# Run the analysis tools with specific CSV files
python main.py --mode analyze --csv path/to/your/annotation.csv
```

### Running Advanced Analyses

You can run advanced analyses using the `--advanced` option with the new module structure:

```bash
# Run advanced analyses with a file dialog
python -m src.tools.analysis.analyze --advanced

# Run advanced analyses with specific CSV files
python -m src.tools.analysis.analyze --csv path/to/your/annotation.csv --advanced
```

The advanced analyses include:

- **Change Point Detection**: Detects significant changes in the annotation patterns with graphical visualization
- **Segment Transition Matrix**: Generates a transition matrix for grouped segments with visualization
- **Hidden Markov Model Analysis**: Models the annotation data as a hidden Markov model (requires hmmlearn package)
- **Anomaly Detection**: Detects anomalies in the annotation data using various methods (Z-score, IQR)

You can also run advanced analyses using the main.py interface:

```bash
# Run advanced analyses with a file dialog
python main.py --mode analyze --advanced

# Run advanced analyses with specific CSV files
python main.py --mode analyze --csv path/to/your/annotation.csv --advanced
```

#### Required Packages for Advanced Analyses

For the full functionality of advanced analyses, you need to install the following packages using conda:

```bash
# Install ruptures for change point detection
conda install -c conda-forge ruptures

# Install hmmlearn for Hidden Markov Model analysis
conda install -c conda-forge hmmlearn

# Install seaborn for visualization
conda install -c conda-forge seaborn
```

Similarly, you can use other tools directly:

```bash
# Run the existing visualization tools (opens a file dialog)
python src/tools/visualization.py

# Run the existing plot tools (opens a file dialog)
python src/tools/plot.py
```

See the [Usage Examples](examples.md) document for detailed examples of how to use the enhanced analysis tools.

## Integration with Existing Tools

The enhanced analysis tools are designed to work seamlessly with the existing analysis tools in the Video Annotator application. See the [Integration Guide](integration.md) for details on how to integrate the enhanced tools with existing workflows.

## API Reference

- [Data Model API](api/data_model.md)
- [Statistical Analysis API](api/statistics.md)
- [Visualization API](api/visualization.md)
- [Adapters API](api/adapters.md)
- [Utilities API](api/utils.md)
