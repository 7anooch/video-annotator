# Installation Guide

This guide provides detailed instructions for installing the Video Annotator application.

## Prerequisites

Before installing Video Annotator, ensure you have the following prerequisites:

- **Python 3.12 or higher**: The application is built with Python and requires version 3.12 or higher.
- **Conda**: We recommend using Conda for managing dependencies. You can install either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
- **Git** (optional): If you want to clone the repository from GitHub.

## Installation Methods

There are two main methods for installing Video Annotator:

1. **Using the provided scripts**: The easiest way to install and run the application.
2. **Manual installation**: For more control over the installation process.

### Method 1: Using the Provided Scripts

#### Step 1: Get the Code

Download the Video Annotator code from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator
```

Alternatively, you can download the ZIP file from the [GitHub repository](https://github.com/yourusername/video-annotator) and extract it.

#### Step 2: Run the Setup Script

Run the setup script to create the Conda environment and install all dependencies:

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
chmod +x scripts/run.sh
./scripts/run.sh
```

When prompted, select the option to update the environment.

#### Step 3: Verify the Installation

To verify that the installation was successful, run the application:

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

Select the option to run the annotator.

### Method 2: Manual Installation

#### Step 1: Get the Code

Download the Video Annotator code from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator
```

Alternatively, you can download the ZIP file from the [GitHub repository](https://github.com/yourusername/video-annotator) and extract it.

#### Step 2: Create the Conda Environment

Create a Conda environment from the provided environment.yml file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate video-annotator
```

#### Step 3: Run the Application

Run the application using the main.py script:

```bash
# Run the annotator
python main.py --mode annotator
```

## Installation Options

### Custom Configuration

You can customize the installation by editing the `config.json` file. See the [Configuration Guide](configuration_guide.md) for more information.

### Development Installation

If you plan to contribute to the Video Annotator project, see the [Development Setup](../developer_guide/development_setup.md) guide for instructions on setting up a development environment.

## Troubleshooting

If you encounter issues during installation, check the following:

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

### Getting Help

If you continue to experience issues, please open an issue on the [GitHub repository](https://github.com/yourusername/video-annotator/issues) with details about the problem and your system configuration.
