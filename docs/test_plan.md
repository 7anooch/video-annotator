# Test Plan for Video Annotator

This document outlines the testing strategy for the Video Annotator project to ensure that all components work correctly after refactoring and improvements.

## 1. Unit Tests

### 1.1 VideoPlayer Class

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_init` | Test initialization with valid video path | VideoPlayer object created with correct attributes |
| `test_load_frame` | Test loading a specific frame | Frame is loaded correctly |
| `test_next_frame` | Test moving to the next frame | Frame number incremented and correct frame loaded |
| `test_prev_frame` | Test moving to the previous frame | Frame number decremented and correct frame loaded |
| `test_frame_cache` | Test frame caching mechanism | Frames are cached and retrieved from cache when requested again |
| `test_invalid_video_path` | Test initialization with invalid video path | Appropriate error raised |
| `test_out_of_bounds_frame` | Test loading a frame beyond video length | Appropriate error raised |

### 1.2 AnnotationManager Class

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_init` | Test initialization with valid annotation path | AnnotationManager object created with correct attributes |
| `test_load_annotations` | Test loading annotations from CSV | Annotations loaded correctly |
| `test_save_annotations` | Test saving annotations to CSV | Annotations saved correctly |
| `test_annotate_frame` | Test annotating a frame | Frame annotation added to annotations dictionary |
| `test_annotate_frame_range` | Test annotating a range of frames | Range of frames annotated correctly |
| `test_invalid_annotation_path` | Test initialization with invalid annotation path | Appropriate error raised |
| `test_invalid_annotation_format` | Test loading malformed CSV | Appropriate error raised |

### 1.3 Config Class

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_init` | Test initialization with default config | Config object created with default values |
| `test_load_config` | Test loading config from file | Config loaded correctly |
| `test_save_config` | Test saving config to file | Config saved correctly |
| `test_get_labels` | Test retrieving labels | Correct labels returned |
| `test_invalid_config_file` | Test loading malformed config file | Default config used |

## 2. Integration Tests

### 2.1 VideoApp Integration

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_video_player_integration` | Test VideoPlayer integration with VideoApp | VideoApp correctly uses VideoPlayer for frame navigation |
| `test_annotation_manager_integration` | Test AnnotationManager integration with VideoApp | VideoApp correctly uses AnnotationManager for annotations |
| `test_config_integration` | Test Config integration with VideoApp | VideoApp correctly uses Config for settings |

### 2.2 UI Integration

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_ui_controls` | Test UI controls functionality | UI controls correctly interact with VideoPlayer and AnnotationManager |
| `test_keyboard_shortcuts` | Test keyboard shortcuts | Keyboard shortcuts correctly trigger actions |
| `test_progress_indicators` | Test progress indicators | Progress indicators correctly show progress for long operations |

### 2.3 End-to-End Workflow

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_annotation_workflow` | Test complete annotation workflow | Video loaded, frames annotated, annotations saved |
| `test_visualization_workflow` | Test visualization workflow | Annotations loaded and visualized correctly |
| `test_ground_truth_workflow` | Test ground truth generation workflow | Ground truth generated correctly from multiple annotations |

## 3. Performance Tests

### 3.1 Video Loading and Playback

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_video_loading_performance` | Test video loading performance | Video loads within acceptable time |
| `test_frame_navigation_performance` | Test frame navigation performance | Frame navigation is responsive |
| `test_playback_performance` | Test playback performance | Playback is smooth at different speeds |

### 3.2 Annotation Performance

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_annotation_performance` | Test annotation performance | Annotation operations complete within acceptable time |
| `test_range_annotation_performance` | Test range annotation performance | Range annotation completes within acceptable time |
| `test_large_annotation_set_performance` | Test performance with large annotation sets | Application remains responsive with large annotation sets |

## 4. Error Handling Tests

### 4.1 File Operations

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_missing_video_file` | Test behavior when video file is missing | Appropriate error message displayed |
| `test_missing_annotation_file` | Test behavior when annotation file is missing | New annotation file created |
| `test_corrupted_video_file` | Test behavior with corrupted video file | Appropriate error message displayed |
| `test_corrupted_annotation_file` | Test behavior with corrupted annotation file | Appropriate error message displayed |

### 4.2 User Input

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_invalid_frame_number` | Test entering invalid frame number | Appropriate error message displayed |
| `test_invalid_range` | Test entering invalid frame range | Appropriate error message displayed |
| `test_invalid_label` | Test selecting invalid label | Appropriate error message displayed |

## 5. UI Tests

### 5.1 Layout and Appearance

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_ui_layout` | Test UI layout | UI elements correctly positioned |
| `test_ui_appearance` | Test UI appearance | UI elements have correct appearance |
| `test_ui_responsiveness` | Test UI responsiveness | UI remains responsive during operations |

### 5.2 Accessibility

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_keyboard_navigation` | Test keyboard navigation | UI can be navigated using keyboard |
| `test_screen_reader_compatibility` | Test screen reader compatibility | UI elements have appropriate labels for screen readers |

## 6. Cross-Platform Tests

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| `test_windows_compatibility` | Test on Windows | Application works correctly on Windows |
| `test_macos_compatibility` | Test on macOS | Application works correctly on macOS |
| `test_linux_compatibility` | Test on Linux | Application works correctly on Linux |

## Test Implementation

### Unit Test Implementation

Unit tests will be implemented using the Python `unittest` framework. Each test case will be implemented as a method in a test class.

Example:

```python
import unittest
from video_player import VideoPlayer

class TestVideoPlayer(unittest.TestCase):
    def setUp(self):
        self.video_player = VideoPlayer('test_video.mp4')
    
    def test_init(self):
        self.assertEqual(self.video_player.video_path, 'test_video.mp4')
        self.assertIsNotNone(self.video_player.cap)
        self.assertGreater(self.video_player.total_frames, 0)
        self.assertEqual(self.video_player.frame_number, 0)
    
    def test_load_frame(self):
        frame = self.video_player.load_frame(0)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[2], 3)  # RGB image
```

### Integration Test Implementation

Integration tests will be implemented using the Python `unittest` framework with mocks for external dependencies.

Example:

```python
import unittest
from unittest.mock import patch, MagicMock
from video_app import VideoApp

class TestVideoAppIntegration(unittest.TestCase):
    @patch('tkinter.Tk')
    @patch('video_player.VideoPlayer')
    @patch('annotation_manager.AnnotationManager')
    def setUp(self, mock_annotation_manager, mock_video_player, mock_tk):
        self.mock_video_player = mock_video_player.return_value
        self.mock_annotation_manager = mock_annotation_manager.return_value
        self.video_app = VideoApp(mock_tk(), 'test_video.mp4', 'test_annotation.csv')
    
    def test_video_player_integration(self):
        self.video_app.next_frame()
        self.mock_video_player.next_frame.assert_called_once()
```

### Performance Test Implementation

Performance tests will be implemented using the Python `timeit` module to measure execution time.

Example:

```python
import timeit
import unittest
from video_player import VideoPlayer

class TestVideoPlayerPerformance(unittest.TestCase):
    def setUp(self):
        self.video_player = VideoPlayer('test_video.mp4')
    
    def test_frame_navigation_performance(self):
        def navigate_frames():
            for _ in range(100):
                self.video_player.next_frame()
        
        execution_time = timeit.timeit(navigate_frames, number=1)
        self.assertLess(execution_time, 1.0)  # Should complete in less than 1 second
```

## Test Execution

Tests will be executed using the following command:

```bash
python -m unittest discover -s tests
```

This will discover and run all tests in the `tests` directory.

## Continuous Integration

Tests will be integrated into a CI/CD pipeline to ensure that they are run automatically on each commit.

Example GitHub Actions workflow:

```yaml
name: Run Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m unittest discover -s tests
```

## Test Coverage

Test coverage will be measured using the `coverage` tool:

```bash
coverage run -m unittest discover -s tests
coverage report
```

The goal is to achieve at least 80% test coverage for the codebase.
