# Development Setup

This guide provides instructions for setting up a development environment for the Video Annotator project.

## Prerequisites

Before setting up the development environment, ensure you have the following prerequisites:

- [Git](https://git-scm.com/downloads)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- A code editor or IDE (e.g., [Visual Studio Code](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/))

## Clone the Repository

First, clone the repository to your local machine:

```bash
# Clone the repository
git clone https://github.com/yourusername/video-annotator.git
cd video-annotator
```

## Create the Development Environment

Create a conda environment for development:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate video-annotator
```

## Install Development Dependencies

Install additional dependencies for development:

```bash
# Install development dependencies
conda install -c conda-forge pytest pytest-cov flake8 black sphinx
```

## Set Up Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install the pre-commit hooks
pre-commit install
```

## Configure the IDE

### Visual Studio Code

If you're using Visual Studio Code, create a `.vscode/settings.json` file with the following content:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length",
        "88"
    ],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.linting.flake8Args": [
        "--max-line-length=88",
        "--extend-ignore=E203"
    ]
}
```

### PyCharm

If you're using PyCharm:

1. Open the project in PyCharm
2. Go to File > Settings > Project: video-annotator > Python Interpreter
3. Click the gear icon and select "Add..."
4. Select "Conda Environment" and choose the existing environment "video-annotator"
5. Click "OK" to save the settings

## Directory Structure

The project follows this directory structure:

```
video-annotator/
├── src/                      # Source code
│   ├── core/                 # Core functionality
│   ├── ui/                   # User interface components
│   ├── utils/                # Utility functions
│   └── tools/                # Standalone tools
├── tests/                    # Test files
├── scripts/                  # Scripts for running the application
├── docs/                     # Documentation
├── examples/                 # Example files and notebooks
├── main.py                   # Main entry point
└── ...
```

## Running the Application

To run the application in development mode:

```bash
# Run the main application
python main.py --mode annotator

# Run a specific component
python main.py --mode [component]
```

## Running Tests

To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run a specific test file
pytest tests/test_modular.py

# Run a specific test
pytest tests/test_modular.py::TestModularImplementation::test_video_player
```

## Code Style

The project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with some modifications:

- Line length: 88 characters (Black default)
- Docstrings: Google style
- Imports: Grouped by standard library, third-party, and local

To format your code:

```bash
# Format a file
black src/core/video_player.py

# Format all files
black src
```

To check your code for style issues:

```bash
# Check a file
flake8 src/core/video_player.py

# Check all files
flake8 src
```

## Documentation

The project uses Markdown for documentation. To build the documentation:

```bash
# Install mkdocs
pip install mkdocs mkdocs-material

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Debugging

### Using Visual Studio Code

1. Open the project in Visual Studio Code
2. Set breakpoints in your code
3. Press F5 to start debugging
4. Select "Python: Current File" as the debug configuration

### Using PyCharm

1. Open the project in PyCharm
2. Set breakpoints in your code
3. Right-click on the file you want to debug and select "Debug"

### Using pdb

You can also use the Python debugger (pdb) directly:

```python
import pdb

# Add this line where you want to break
pdb.set_trace()
```

## Profiling

To profile the application:

```bash
# Run the performance profiler
python main.py --mode profile
```

## Next Steps

After setting up your development environment, you can:

- Read the [Architecture Overview](architecture.md) to understand the application structure
- Check the [Code Organization](code_organization.md) guide for details on the codebase
- Review the [Contributing Guidelines](contributing.md) for information on how to contribute
- Look at the [Testing Guide](testing.md) for details on testing the application
