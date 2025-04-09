# Installation Guide for Video Annotator

This guide provides instructions for installing and setting up the Video Annotator application.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone or Download the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator

# Or download and extract the ZIP file
```

### 2. Create and Activate the Conda Environment

```bash
# Create the conda environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate video-annotator
```

### 3. Verify the Installation

```bash
# Run the main application
python annotator_modular.py
```

## Configuration

The application uses a configuration file (`config.json`) to customize various aspects:

- Labels and their keyboard shortcuts
- UI settings
- Video playback settings
- Annotation behavior

You can edit this file directly or use the configuration editor:

```bash
python config_editor.py
```

## Troubleshooting

### Common Issues

#### OpenCV Installation Problems

If you encounter issues with OpenCV, try installing it via pip:

```bash
pip install opencv-python
```

#### Tkinter Issues

Tkinter should be included with Python, but if you encounter issues:

```bash
# On Ubuntu/Debian
sudo apt-get install python3-tk

# On macOS with Homebrew
brew install python-tk

# On Windows
# Tkinter is included with Python installations
```

#### Path Issues

If the application cannot find modules, ensure your PYTHONPATH is set correctly:

```bash
# Add the repository root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/video-annotator
```

## Updating

To update the application:

```bash
# Pull the latest changes
git pull

# Update the conda environment
conda env update -f environment.yml
```

## Uninstallation

To remove the application:

```bash
# Remove the conda environment
conda deactivate
conda env remove -n video-annotator

# Delete the repository
rm -rf /path/to/video-annotator
```
