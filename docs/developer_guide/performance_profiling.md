# Performance Profiling

This document describes the performance profiling tools available in the Video Annotator project.

## Overview

The Video Annotator includes a performance profiler that can be used to measure and analyze the performance of various aspects of the application. The profiler can help identify bottlenecks and optimize the application for better performance.

## Performance Profiler

The `PerformanceProfiler` class in `src/tools/performance_profiler.py` provides tools for profiling the performance of the Video Annotator application. It can profile:

- Video loading performance
- Annotation loading and saving performance
- Memory usage

### Video Loading Performance

The profiler can measure the time it takes to load frames from a video file. This can help identify bottlenecks in the video loading process.

```python
from src.tools.performance_profiler import PerformanceProfiler

# Create a profiler
profiler = PerformanceProfiler()

# Profile video loading
metrics = profiler.profile_video_loading('path/to/video.mp4', num_frames=100)

# Print metrics
print(f"Average frame time: {metrics['avg_frame_time']:.4f} seconds")
print(f"Frames per second: {metrics['fps']:.2f}")
```

### Annotation Performance

The profiler can measure the time it takes to load and save annotations. This can help identify bottlenecks in the annotation process.

```python
# Profile annotation loading
metrics = profiler.profile_annotation_loading('path/to/annotations.csv')
print(f"Annotation loading time: {metrics['load_time']:.4f} seconds")
print(f"Number of annotations: {metrics['num_annotations']}")

# Profile annotation saving
metrics = profiler.profile_annotation_saving('path/to/annotations.csv')
print(f"Annotation saving time: {metrics['save_time']:.4f} seconds")
```

### Memory Usage Tracking

The profiler can track memory usage over time. This can help identify memory leaks and optimize memory usage.

```python
# Track memory usage for 10 seconds with measurements every 0.5 seconds
metrics = profiler.track_memory_usage(interval=0.5, duration=10.0)

# Print memory metrics
print(f"Minimum memory usage: {metrics['min_memory_mb']:.2f} MB")
print(f"Maximum memory usage: {metrics['max_memory_mb']:.2f} MB")
print(f"Average memory usage: {metrics['avg_memory_mb']:.2f} MB")
print(f"Memory usage at start: {metrics['start_memory_mb']:.2f} MB")
print(f"Memory usage at end: {metrics['end_memory_mb']:.2f} MB")
print(f"Memory usage difference: {metrics['diff_memory_mb']:.2f} MB")

# Plot memory usage
fig = profiler.plot_memory_usage(metrics)
plt.show()
```

### Comprehensive Performance Report

The profiler can generate a comprehensive performance report that includes all of the above metrics.

```python
# Generate a performance report
metrics = profiler.generate_performance_report(
    video_path='path/to/video.mp4',
    annotation_path='path/to/annotations.csv',
    output_path='path/to/report.csv'
)
```

This will generate a CSV file with all the performance metrics and save plots of the performance data.

## Performance Profiler GUI

The Video Annotator includes a graphical user interface for the performance profiler. You can launch it using:

```bash
python main.py --mode profile
```

The GUI allows you to:

1. Select a video file
2. Select an annotation file
3. Configure profiling options
4. Profile memory usage
5. Generate a comprehensive performance report

## Best Practices

### When to Profile

Profile your code when:

- You suspect a performance bottleneck
- You've made changes that might affect performance
- You're optimizing a specific part of the application
- You want to establish a performance baseline

### What to Look For

When analyzing profiling results, look for:

- Operations that take a long time
- Operations that use a lot of memory
- Unexpected memory growth over time
- Large variations in performance

### Optimizing Performance

Based on profiling results, you can optimize performance by:

- Caching frequently accessed data
- Reducing unnecessary computations
- Using more efficient algorithms
- Reducing memory allocations
- Implementing lazy loading for large data

## Frame Caching

The Video Annotator includes a frame caching system that can improve performance when working with videos. The frame cache is implemented in `src/utils/cache_manager.py`.

### Using the Frame Cache

```python
from src.utils.cache_manager import get_frame_cache, cache_frame, get_cached_frame

# Get the global frame cache
frame_cache = get_frame_cache()

# Cache a frame
cache_frame(frame_number=10, frame=frame_data)

# Get a cached frame
frame = get_cached_frame(frame_number=10)

# Configure the frame cache
from src.utils.cache_manager import configure_frame_cache
configure_frame_cache(capacity=200, max_size_bytes=2 * 1024 * 1024 * 1024)  # 2 GB
```

### Frame Cache in the Video Player

The `VideoPlayer` class in `src/core/video_player.py` uses the frame cache to improve performance when loading frames. The cache is particularly useful when:

- Scrubbing through the video
- Playing the video at different speeds
- Jumping to specific frames

## Memory Tracking

The Video Annotator includes a memory tracking system that can help identify memory leaks and optimize memory usage. The memory tracker is implemented in `src/utils/memory_tracker.py`.

### Using the Memory Tracker

```python
from src.utils.memory_tracker import get_memory_tracker, take_memory_snapshot

# Get the global memory tracker
memory_tracker = get_memory_tracker()

# Take a memory snapshot
snapshot = take_memory_snapshot()
print(snapshot)

# Start continuous memory tracking
from src.utils.memory_tracker import start_memory_tracking, stop_memory_tracking
start_memory_tracking(interval=0.5)  # Take a snapshot every 0.5 seconds

# Do some work...

# Stop memory tracking
stop_memory_tracking()

# Get a summary of memory usage
from src.utils.memory_tracker import get_memory_summary, print_memory_summary
summary = get_memory_summary()
print_memory_summary()
```

### Memory Usage Decorator

The memory tracker includes a decorator that can be used to track memory usage during function execution.

```python
from src.utils.memory_tracker import memory_usage_decorator

@memory_usage_decorator
def my_function():
    # This function's memory usage will be tracked
    pass
```

## Conclusion

Performance profiling is an important part of developing and maintaining the Video Annotator application. By using the profiling tools described in this document, you can identify and fix performance issues, leading to a better user experience.
