# Video Annotator

A tool for annotating video frames with behavioral labels, primarily designed for neuroscience research.

## Features

- Frame-by-frame video navigation
- Keyboard shortcuts for quick annotation
- Annotation visualization with ethograms
- Support for multiple annotation sets
- Ground truth generation from multiple annotators
- Analysis tools for comparing annotations

## Installation

### Prerequisites

- Python 3.12 or higher
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Pillow (PIL)

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator

# Create and activate conda environment
conda env create -f environment.yml
conda activate annotator
```

## Usage

### Video Annotation

To start the annotation tool:

```bash
python annotator.py
```

This will open a file dialog to select a video file. Once selected, the annotation interface will appear.

#### Command-line Options

```bash
python annotator.py --csv [annotation_file_name] --side_controls
```

- `--csv`: Specify the name of the annotation CSV file (default: video_name_annotation.csv)
- `--side_controls`: Place controls on the right side of the interface

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
python plot.py
```

This will open a file dialog to select one or more annotation CSV files. Once selected, ethograms will be displayed.

#### Command-line Options

```bash
python plot.py --use_cols 2,3
```

- `--use_cols`: Specify which columns to use from the CSV file (default: 2)

### Ground Truth Generation

To generate ground truth from multiple annotations:

```bash
python gen_ground_truth.py
```

This will open a file dialog to select multiple annotation CSV files. The tool will then generate a ground truth file based on the consensus of the annotations.

### Analysis

To analyze and compare annotations:

```bash
python analyze.py
```

This will open a file dialog to select annotation files for analysis. The tool will generate statistics and comparisons between the annotations.

## File Formats

### Annotation CSV

The annotation CSV file has the following format:

```
frame,label
0,0
1,0
2,1
3,1
...
```

- `frame`: Frame number (0-indexed)
- `label`: Annotation label (0 = Stop, 1 = Run, 2 = Turn)

### Configuration

You can customize the labels and other settings by creating a `config.json` file:

```json
{
  "labels": [
    {"name": "Stop", "key": "s", "value": 0, "color": "red"},
    {"name": "Run", "key": "r", "value": 1, "color": "green"},
    {"name": "Turn", "key": "t", "value": 2, "color": "blue"}
  ],
  "default_fps": 30,
  "ui": {
    "controls_right": false,
    "window_width": 1200
  }
}
```

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

1. Run `python plot.py`
2. Select multiple annotation CSV files
3. Ethograms will be displayed for each annotation file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool was developed for neuroscience research at UCSB
- Thanks to all contributors and users for their feedback and suggestions
