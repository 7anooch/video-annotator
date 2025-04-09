# Enhanced Analysis Tools Examples

This directory contains example scripts that demonstrate how to use the enhanced analysis tools for the Video Annotator application.

## Example Scripts

1. **Basic Analysis Example**: Demonstrates how to use the enhanced analysis tools for basic analysis tasks.
2. **Advanced Analysis Example**: Demonstrates how to use the enhanced analysis tools for advanced analysis tasks.
3. **Advanced Statistical Analysis Example**: Demonstrates how to use the enhanced analysis tools for advanced statistical analysis tasks.
4. **Visualization Example**: Demonstrates how to use the enhanced analysis tools for visualization tasks.
5. **Comparison Example**: Demonstrates how to use the enhanced analysis tools to compare two annotation sets.
6. **Integration Demo**: Demonstrates how the enhanced analysis tools can be integrated with the existing analysis tools.

## Usage

### Running the Example Scripts

The example scripts are designed to be run directly from the command line. Make sure you're in the root directory of the project when running these commands.

### Basic Analysis Example

```bash
python src/examples/basic_analysis_example.py <annotation_file>
```

This script demonstrates how to:

- Load annotations from a file
- Calculate basic statistics
- Calculate label transitions
- Calculate label durations
- Calculate duration statistics

### Advanced Analysis Example

```bash
python src/examples/advanced_analysis_example.py <annotation_file>
```

This script demonstrates how to:

- Perform advanced sequence analysis
- Analyze transition probabilities
- Analyze n-grams
- Calculate complexity measures
- Identify recurring patterns
- Perform time series analysis
- Detect anomalies using various methods

### Advanced Statistical Analysis Example

```bash
python src/examples/advanced_statistical_analysis_example.py <annotation_file>
```

This script demonstrates how to:

- Perform change point detection using different methods
- Perform Hidden Markov Model (HMM) analysis
- Perform Markov Chain Monte Carlo (MCMC) simulation
- Detect anomalies using various advanced methods (Z-score, IQR, Isolation Forest, DBSCAN, LOF)

### Visualization Example

```bash
python src/examples/visualization_example.py <annotation_file>
```

This script demonstrates how to:

- Create a timeline plot
- Create a label distribution plot
- Create a duration boxplot
- Create a transition heatmap
- Create a time series plot
- Create an interactive timeline
- Create a 3D visualization
- Generate a comprehensive report

### Comparison Example

```bash
python src/examples/comparison_example.py <annotation_file_1> <annotation_file_2>
```

This script demonstrates how to:

- Compare two annotation sets
- Calculate agreement
- Calculate Cohen's kappa
- Calculate correlation
- Create a comparison plot
- Create a confusion matrix

### Integration Demo

```bash
python src/examples/integration_demo.py <annotation_file>
```

This script demonstrates how:

- The enhanced analysis tools can be used with existing data formats
- The adapter classes can be used to bridge enhanced and existing tools
- Enhanced and existing tools can be combined
- Interactive visualizations can be created
- Comprehensive reports can be generated

## Sample Data

You can use the sample annotation files in the `data` directory to test the example scripts:

```bash
python src/examples/basic_analysis_example.py data/sample_annotations.csv
```

## Dependencies

The example scripts require the following dependencies:

- NumPy: For numerical computations
- Pandas: For data manipulation
- Matplotlib: For visualizations
- SciPy: For scientific computations
- scikit-learn: For machine learning analysis (optional)
- Plotly: For interactive visualizations (optional)
- hmmlearn: For Hidden Markov Model analysis (optional)
- ruptures: For change point detection (optional)
- statsmodels: For advanced statistical analysis (optional)

You can install the required dependencies using pip:

```bash
pip install numpy pandas matplotlib scipy scikit-learn plotly hmmlearn ruptures statsmodels
```

Or using conda:

```bash
conda install numpy pandas matplotlib scipy scikit-learn plotly statsmodels
conda install -c conda-forge hmmlearn ruptures
```

Alternatively, you can install the dependencies from the requirements file:

```bash
pip install -r requirements_analysis.txt
```
