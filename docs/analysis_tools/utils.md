# Utilities Documentation

The utilities module provides utility functions for working with annotation data that complement and extend the functionality in the existing analysis tools.

## Overview

The utilities module includes various utility functions for working with annotation data, including file operations, data manipulation, and quality assessment.

## Key Features

- **File Operations**: Functions for finding annotation files and working with video files.
- **Data Manipulation**: Functions for interpolating, smoothing, and transforming annotation data.
- **Quality Assessment**: Functions for calculating agreement, Cohen's kappa, and other quality metrics.
- **Time Conversion**: Functions for converting between frames and time.

## Usage

### File Operations

```python
from src.analysis.utils import find_annotation_files, get_video_frame_count, get_video_fps

# Find annotation files in a directory
annotation_files = find_annotation_files('data', recursive=True)
print(f"Found {len(annotation_files)} annotation files")

# Get the number of frames in a video
frame_count = get_video_frame_count('video.mp4')
print(f"Video has {frame_count} frames")

# Get the frames per second of a video
fps = get_video_fps('video.mp4')
print(f"Video has {fps} FPS")
```

### Data Manipulation

```python
from src.analysis.utils import interpolate_annotations, smooth_annotations

# Interpolate missing annotations
interpolated = interpolate_annotations(annotations, method='nearest')
print(f"Interpolated {len(interpolated) - len(annotations)} frames")

# Smooth annotations
smoothed = smooth_annotations(annotations, window_size=5)
print(f"Smoothed annotations")
```

### Quality Assessment

```python
from src.analysis.utils import calculate_agreement, calculate_cohen_kappa, calculate_fleiss_kappa, calculate_annotation_quality

# Calculate agreement between two annotation sets
agreement = calculate_agreement(annotations1, annotations2)
print(f"Agreement: {agreement:.2f}")

# Calculate Cohen's kappa
kappa = calculate_cohen_kappa(annotations1, annotations2)
print(f"Cohen's kappa: {kappa:.2f}")

# Calculate Fleiss' kappa for multiple annotators
fleiss_kappa = calculate_fleiss_kappa([annotations1, annotations2, annotations3])
print(f"Fleiss' kappa: {fleiss_kappa:.2f}")

# Calculate comprehensive quality metrics
quality = calculate_annotation_quality(annotations, ground_truth)
print(f"Coverage: {quality['coverage']:.2f}")
print(f"Accuracy: {quality['accuracy']:.2f}")
print(f"Kappa: {quality['kappa']:.2f}")
```

### Time Conversion

```python
from src.analysis.utils import frames_to_time, time_to_frames

# Convert frames to time
time_str = frames_to_time(1000, 30.0)
print(f"Frame 1000 at 30 FPS is at {time_str}")

# Convert time to frames
frame = time_to_frames('00:00:33.333', 30.0)
print(f"Time 00:00:33.333 at 30 FPS is frame {frame}")
```

## Function Reference

### File Operations

- `find_annotation_files(directory: str, recursive: bool = False) -> List[str]`: Find annotation files in a directory.
- `get_video_frame_count(video_path: str) -> int`: Get the number of frames in a video.
- `get_video_fps(video_path: str) -> float`: Get the frames per second of a video.

### Data Manipulation

- `interpolate_annotations(annotations: Dict[int, Dict[str, Any]], method: str = 'nearest') -> Dict[int, Dict[str, Any]]`: Interpolate missing annotations.
- `smooth_annotations(annotations: Dict[int, Dict[str, Any]], window_size: int = 3) -> Dict[int, Dict[str, Any]]`: Smooth annotations using a sliding window.

### Quality Assessment

- `calculate_agreement(annotations1: Dict[int, Dict[str, Any]], annotations2: Dict[int, Dict[str, Any]]) -> float`: Calculate agreement between two sets of annotations.
- `calculate_cohen_kappa(annotations1: Dict[int, Dict[str, Any]], annotations2: Dict[int, Dict[str, Any]]) -> float`: Calculate Cohen's kappa coefficient between two sets of annotations.
- `calculate_fleiss_kappa(annotations_list: List[Dict[int, Dict[str, Any]]]) -> float`: Calculate Fleiss' kappa coefficient for multiple annotators.
- `calculate_annotation_quality(annotations: Dict[int, Dict[str, Any]], ground_truth: Dict[int, Dict[str, Any]]) -> Dict[str, float]`: Calculate various quality metrics for annotations compared to ground truth.

### Time Conversion

- `frames_to_time(frames: int, fps: float) -> str`: Convert frames to time string (HH:MM:SS.mmm).
- `time_to_frames(time_str: str, fps: float) -> int`: Convert time string (HH:MM:SS.mmm) to frames.
