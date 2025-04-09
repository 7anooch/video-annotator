# Video Processing

This document describes the video processing capabilities of the Video Annotator application and how to extend them.

## Overview

The Video Annotator includes a video processing module that provides functionality for:

- Getting information about video files
- Extracting frames from videos
- Trimming videos to specific frame ranges
- Supporting multiple video formats

The video processing functionality is implemented in the `src/utils/video_processor.py` module and exposed to users through the `src/ui/video_processing_dialog.py` dialog.

## VideoProcessor Class

The `VideoProcessor` class in `src/utils/video_processor.py` provides the core video processing functionality.

### Key Methods

#### get_video_info

```python
def get_video_info(self, video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Dict[str, Any]: Dictionary containing video information
    """
```

This method returns a dictionary with information about a video file, including:

- `path`: The path to the video file
- `filename`: The name of the video file
- `width`: The width of the video in pixels
- `height`: The height of the video in pixels
- `fps`: The frames per second of the video
- `total_frames`: The total number of frames in the video
- `duration`: The duration of the video in seconds
- `codec`: The codec used to encode the video
- `file_size`: The size of the video file in bytes
- `file_size_mb`: The size of the video file in megabytes

#### extract_frames

```python
def extract_frames(self, video_path: str, output_dir: str, 
                  start_frame: int = 0, end_frame: int = None,
                  step: int = 1, format: str = 'jpg',
                  quality: int = 95) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        start_frame (int, optional): First frame to extract. Defaults to 0.
        end_frame (int, optional): Last frame to extract. Defaults to None (last frame).
        step (int, optional): Step between frames. Defaults to 1.
        format (str, optional): Output image format. Defaults to 'jpg'.
        quality (int, optional): Image quality (0-100). Defaults to 95.
        
    Returns:
        List[str]: List of paths to the extracted frames
    """
```

This method extracts frames from a video file and saves them to the specified output directory. It returns a list of paths to the extracted frames.

#### trim_video

```python
def trim_video(self, input_path: str, output_path: str, 
              start_frame: int, end_frame: int) -> bool:
    """
    Trim a video to a specific frame range.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the output video file
        start_frame (int): First frame to include
        end_frame (int): Last frame to include
        
    Returns:
        bool: True if trimming was successful, False otherwise
    """
```

This method trims a video to a specific frame range and saves the result to the specified output path. It returns `True` if the trimming was successful, `False` otherwise.

#### get_frame

```python
def get_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Get a specific frame from a video file.
    
    Args:
        video_path (str): Path to the video file
        frame_number (int): Frame number to get
        
    Returns:
        Optional[np.ndarray]: The frame as a NumPy array, or None if the frame could not be read
    """
```

This method gets a specific frame from a video file and returns it as a NumPy array. It returns `None` if the frame could not be read.

### Supported Formats

The `VideoProcessor` class supports the following video formats:

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- M4V (.m4v)
- MPEG (.mpg, .mpeg)
- 3GP (.3gp, .3g2)

The list of supported formats is stored in the `supported_formats` attribute of the `VideoProcessor` class.

## VideoProcessingDialog Class

The `VideoProcessingDialog` class in `src/ui/video_processing_dialog.py` provides a user interface for the video processing functionality.

### Key Methods

#### load_video_info

```python
def load_video_info(self):
    """Load information about the video."""
```

This method loads information about the video and displays it in the dialog.

#### extract_frames

```python
def extract_frames(self):
    """Extract frames from the video."""
```

This method extracts frames from the video using the options specified in the dialog.

#### trim_video

```python
def trim_video(self):
    """Trim the video."""
```

This method trims the video using the options specified in the dialog.

## Integration with UI Controller

The video processing dialog is integrated with the UI controller in `src/ui/ui_controller.py`. The `open_video_processing` method opens the video processing dialog:

```python
@exception_handler
def open_video_processing(self):
    """Open the video processing dialog."""
    # Get the current video path
    video_path = self.video_player.video_path if hasattr(self.video_player, 'video_path') else None
    
    # Create the video processing dialog
    video_processing_dialog = VideoProcessingDialog(self.master, video_path, self.theme_manager)
    
    # Make the dialog modal
    video_processing_dialog.transient(self.master)
    video_processing_dialog.grab_set()
    
    # Wait for the dialog to close
    self.master.wait_window(video_processing_dialog)
```

The dialog is accessible from the "Tools" menu in the main application.

## Extending Video Processing Capabilities

### Adding New Video Formats

To add support for a new video format, add the format's extension to the `supported_formats` list in the `VideoProcessor` class:

```python
self.supported_formats = [
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
    '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2',
    '.new_format'  # Add the new format here
]
```

### Adding New Processing Methods

To add a new video processing method:

1. Add the method to the `VideoProcessor` class in `src/utils/video_processor.py`
2. Add a new tab to the `VideoProcessingDialog` class in `src/ui/video_processing_dialog.py`
3. Add a method to the `VideoProcessingDialog` class to call the new processing method
4. Update the documentation in `docs/user_guide/video_processing.md` and `docs/developer_guide/video_processing.md`

For example, to add a method for applying a filter to a video:

```python
@exception_handler
def apply_filter(self, input_path: str, output_path: str, filter_type: str) -> bool:
    """
    Apply a filter to a video.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the output video file
        filter_type (str): Type of filter to apply
        
    Returns:
        bool: True if the filter was applied successfully, False otherwise
    """
    # Implementation goes here
    pass
```

### Using External Libraries

The video processing module uses OpenCV for most of its functionality. If you need to use other libraries for specific tasks, you can import them in the `video_processor.py` module.

For example, to use FFmpeg for a specific task:

```python
import subprocess

def convert_video_with_ffmpeg(self, input_path: str, output_path: str, options: List[str]) -> bool:
    """
    Convert a video using FFmpeg.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the output video file
        options (List[str]): FFmpeg options
        
    Returns:
        bool: True if the conversion was successful, False otherwise
    """
    try:
        command = ['ffmpeg', '-i', input_path] + options + [output_path]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
```

## Best Practices

### Performance

Video processing can be computationally intensive. To ensure good performance:

- Use efficient algorithms and data structures
- Avoid unnecessary memory allocations
- Use caching where appropriate
- Consider using multiprocessing for CPU-intensive tasks
- Provide progress feedback for long-running operations

### Error Handling

Video processing can fail for various reasons. To ensure robust error handling:

- Use the `@exception_handler` decorator for all methods
- Check input parameters for validity
- Handle edge cases (e.g., empty videos, corrupted files)
- Provide meaningful error messages
- Log errors for debugging

### User Interface

The video processing dialog should provide a user-friendly interface:

- Use clear and concise labels
- Provide tooltips for complex options
- Show progress indicators for long-running operations
- Validate user input before processing
- Provide feedback on success or failure

## Future Improvements

Potential improvements to the video processing module:

- Add support for more video formats
- Add more processing methods (e.g., filters, effects)
- Improve performance with GPU acceleration
- Add batch processing capabilities
- Add preview functionality
- Add more export options (e.g., GIF, WebM)
- Add more metadata extraction (e.g., audio information)
- Add more error handling and recovery options
