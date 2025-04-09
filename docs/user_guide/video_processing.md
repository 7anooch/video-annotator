# Video Processing

This guide explains how to use the video processing capabilities of the Video Annotator.

## Overview

The Video Annotator includes several video processing tools that allow you to:

- Extract frames from a video
- Trim videos to specific frame ranges
- Get detailed information about video files

These tools can be accessed through the "Tools" menu in the main application.

## Video Processing Dialog

The Video Processing dialog provides a user-friendly interface for processing video files. To open the dialog, select "Tools" > "Video Processing" from the menu bar.

### Loading a Video

To load a video for processing:

1. Click the "Browse" button next to the "Video Path" field
2. Select a video file from your computer
3. Click "Load" to load the video information

The video information will be displayed in the "Video Information" section, including:

- Filename
- Resolution
- FPS (Frames Per Second)
- Total Frames
- Duration
- Codec
- File Size

### Extracting Frames

To extract frames from a video:

1. Select the "Frame Extraction" tab
2. Specify the output directory where the frames will be saved
3. Set the extraction options:
   - Start Frame: The first frame to extract
   - End Frame: The last frame to extract
   - Step: The number of frames to skip between extractions (e.g., a step of 2 will extract every other frame)
   - Format: The image format to use (JPG or PNG)
   - Quality: The image quality (1-100, higher is better)
4. Click "Extract Frames" to start the extraction process

The extracted frames will be saved in the specified output directory with filenames like `frame_000001.jpg`, `frame_000002.jpg`, etc.

### Trimming Videos

To trim a video to a specific frame range:

1. Select the "Video Trimming" tab
2. Specify the output file where the trimmed video will be saved
3. Set the trimming options:
   - Start Frame: The first frame to include in the trimmed video
   - End Frame: The last frame to include in the trimmed video
4. Click "Trim Video" to start the trimming process

The trimmed video will be saved to the specified output file.

## Supported Video Formats

The Video Annotator supports the following video formats:

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

## Tips for Video Processing

### Frame Extraction

- Extracting all frames from a long video can take a lot of time and disk space. Consider using a larger step value to extract fewer frames.
- PNG format provides better quality but larger file sizes compared to JPG.
- For JPG format, a quality setting of 90-95 provides a good balance between quality and file size.

### Video Trimming

- Trimming preserves the original video quality, but the output file may be larger than expected if the video uses certain codecs.
- The trimming process can take some time for large videos.

## Programmatic Video Processing

Advanced users can use the Video Processor API to process videos programmatically:

```python
from src.utils.video_processor import VideoProcessor

# Create a video processor
processor = VideoProcessor()

# Get video information
video_info = processor.get_video_info('path/to/video.mp4')
print(f"Video has {video_info['total_frames']} frames at {video_info['fps']} FPS")

# Extract frames
frames = processor.extract_frames(
    'path/to/video.mp4',
    'output/directory',
    start_frame=0,
    end_frame=100,
    step=5,
    format='jpg',
    quality=95
)
print(f"Extracted {len(frames)} frames")

# Trim video
success = processor.trim_video(
    'path/to/video.mp4',
    'path/to/output.mp4',
    start_frame=100,
    end_frame=200
)
if success:
    print("Video trimmed successfully")
else:
    print("Failed to trim video")
```

## Troubleshooting

### "Could not open video file"

This error can occur if:

- The video file does not exist
- The video file is corrupted
- The video codec is not supported

Try converting the video to a different format (e.g., MP4) using a video converter tool.

### "Failed to extract frames"

This error can occur if:

- The output directory does not exist and could not be created
- You don't have write permissions for the output directory
- The video file is corrupted

Check that you have write permissions for the output directory and that the video file is valid.

### "Failed to trim video"

This error can occur if:

- The output directory does not exist and could not be created
- You don't have write permissions for the output directory
- The video file is corrupted
- The specified frame range is invalid

Check that you have write permissions for the output directory, that the video file is valid, and that the frame range is valid.
