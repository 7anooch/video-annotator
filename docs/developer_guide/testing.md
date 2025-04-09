# Testing Guide

This guide provides information about testing the Video Annotator application.

## Testing Framework

The Video Annotator project uses [pytest](https://docs.pytest.org/) as its testing framework. Tests are located in the `tests` directory.

## Test Structure

The tests are organized to mirror the structure of the source code:

```text
tests/
├── __init__.py
├── run_tests.py            # Test runner script
├── test_annotator.py       # Tests for the legacy annotator
├── test_modular.py         # Tests for the modular annotator
├── test_annotation_utils.py # Tests for annotation utility functions
├── test_error_handling.py  # Tests for error handling utilities
├── test_visualization_enhanced.py # Tests for enhanced visualization
└── ...
```

## Running Tests

### Running All Tests

To run all tests, use the provided test runner scripts:

```bash
# On macOS/Linux
./scripts/run_tests.sh

# On Windows
scripts\run_tests.bat
```

You can also run tests with additional options:

```bash
# Run tests with verbose output
./scripts/run_tests.sh --verbosity 3

# Run specific test patterns
./scripts/run_tests.sh --pattern "test_error_*.py"
```

Alternatively, you can use pytest directly:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src

# Run tests with coverage and generate a report
pytest --cov=src --cov-report=html
```

### Running Specific Tests

To run specific tests:

```bash
# Run tests in a specific file
pytest tests/test_modular.py

# Run a specific test class
pytest tests/test_modular.py::TestModularImplementation

# Run a specific test method
pytest tests/test_modular.py::TestModularImplementation::test_video_player
```

### Running Tests with Different Python Versions

To run tests with different Python versions:

```bash
# Create a new conda environment with a different Python version
conda create -n video-annotator-py311 python=3.11
conda activate video-annotator-py311
conda env update -f environment.yml

# Run the tests
pytest
```

## Writing Tests

### Test File Naming

Test files should be named `test_*.py` to be discovered by pytest.

### Test Class Naming

Test classes should be named `Test*` and should inherit from `unittest.TestCase`.

### Test Method Naming

Test methods should be named `test_*` to be discovered by pytest.

### Test Pattern

Tests should follow the Arrange-Act-Assert (AAA) pattern:

1. **Arrange**: Set up the test data and environment
2. **Act**: Perform the action being tested
3. **Assert**: Check that the expected outcome occurred

Example:

```python
def test_video_player_load_frame(self):
    # Arrange
    video_path = "path/to/test/video.mp4"
    player = VideoPlayer(video_path)

    # Act
    frame = player.load_frame(0)

    # Assert
    self.assertIsNotNone(frame)
    self.assertEqual(frame.shape[2], 3)  # RGB image
```

### Mocking

Use the `unittest.mock` module to mock external dependencies:

```python
from unittest.mock import patch, MagicMock

@patch('cv2.VideoCapture')
def test_video_player_init(self, mock_video_capture):
    # Arrange
    mock_video_capture.return_value.isOpened.return_value = True
    mock_video_capture.return_value.get.return_value = 100  # Total frames

    # Act
    player = VideoPlayer("path/to/test/video.mp4")

    # Assert
    self.assertEqual(player.total_frames, 100)
```

### Fixtures

Use pytest fixtures to set up common test data:

```python
import pytest

@pytest.fixture
def video_player():
    """Create a VideoPlayer for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4') as video_file:
        player = VideoPlayer(video_file.name)
        yield player

def test_video_player_next_frame(video_player):
    # Arrange
    initial_frame = video_player.frame_number

    # Act
    video_player.next_frame()

    # Assert
    self.assertEqual(video_player.frame_number, initial_frame + 1)
```

### Parameterized Tests

Use pytest's parameterize feature to run the same test with different inputs:

```python
import pytest

@pytest.mark.parametrize("frame_number,expected_label", [
    (0, 0),  # Frame 0 has label 0
    (10, 1),  # Frame 10 has label 1
    (20, 2),  # Frame 20 has label 2
])
def test_get_annotation(self, frame_number, expected_label):
    # Arrange
    annotations = {0: 0, 10: 1, 20: 2}
    manager = AnnotationManager("path/to/test/annotations.csv")
    manager.annotations = annotations

    # Act
    label = manager.get_annotation(frame_number)

    # Assert
    self.assertEqual(label, expected_label)
```

## Test Coverage

The project aims for high test coverage. To check the test coverage:

```bash
# Run tests with coverage
pytest --cov=src

# Generate a coverage report
pytest --cov=src --cov-report=html
```

The coverage report will be generated in the `htmlcov` directory. Open `htmlcov/index.html` in a web browser to view the report.

## Continuous Integration

The project uses GitHub Actions for continuous integration. The CI pipeline runs tests on multiple platforms (Windows, macOS, Linux) and Python versions.

The CI configuration is located in `.github/workflows/tests.yml`.

## Test Data

Test data is stored in the `tests/data` directory. This includes:

- Sample videos
- Sample annotation files
- Sample configuration files

## Integration Tests

Integration tests test the interaction between multiple components. These tests are located in the `tests` directory and are named `test_integration_*.py`.

## End-to-End Tests

End-to-end tests test the entire application from the user's perspective. These tests are located in the `tests` directory and are named `test_e2e_*.py`.

## Performance Tests

Performance tests measure the performance of the application. These tests are located in the `tests` directory and are named `test_performance_*.py`.

## Test Environment

Tests should be run in a clean environment to avoid interference from other tests. Use pytest's fixture system to set up and tear down the test environment.

## Debugging Tests

To debug tests:

```bash
# Run tests with the debugger
pytest --pdb

# Run tests with verbose output
pytest -v

# Run tests with print statements
pytest -s
```

## Test Documentation

Tests should be well-documented:

- Each test file should have a docstring explaining what it tests
- Each test class should have a docstring explaining what it tests
- Each test method should have a docstring explaining what it tests

Example:

```python
"""Tests for the VideoPlayer class."""

class TestVideoPlayer(unittest.TestCase):
    """Tests for the VideoPlayer class."""

    def test_load_frame(self):
        """Test that load_frame returns a valid frame."""
        # Test code here
```

## Next Steps

After reading this guide, you should be able to:

- Run existing tests
- Write new tests
- Check test coverage
- Debug tests

For more information, see the [pytest documentation](https://docs.pytest.org/).
