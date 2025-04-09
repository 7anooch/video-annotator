# Troubleshooting

This guide provides solutions to common issues you might encounter when using the Video Annotator application.

## Installation Issues

### Conda Environment Creation Fails

**Problem**: The conda environment creation fails with an error.

**Solution**:

1. Make sure you have the latest version of conda:
   ```bash
   conda update -n base conda
   ```

2. Try creating the environment with a specific Python version:
   ```bash
   conda create -n video-annotator python=3.12
   conda activate video-annotator
   pip install -r requirements.txt
   ```

3. If specific packages are causing issues, try installing them one by one:
   ```bash
   conda install -c conda-forge opencv
   conda install -c conda-forge pandas
   conda install -c conda-forge matplotlib
   ```

### Missing Dependencies

**Problem**: The application fails to start due to missing dependencies.

**Solution**:

1. Make sure you've activated the conda environment:
   ```bash
   conda activate video-annotator
   ```

2. Install the missing dependencies:
   ```bash
   pip install <missing_package>
   ```

3. If the issue persists, try reinstalling all dependencies:
   ```bash
   conda env update -f environment.yml --prune
   ```

### Tkinter Issues

**Problem**: The application fails to start with a Tkinter error.

**Solution**:

1. On Ubuntu/Debian:
   ```bash
   sudo apt-get install python3-tk
   ```

2. On macOS with Homebrew:
   ```bash
   brew install python-tk
   ```

3. On Windows, reinstall Python with the "tcl/tk and IDLE" option checked.

## Video Loading Issues

### Video File Not Found

**Problem**: The application cannot find the video file.

**Solution**:

1. Make sure the video file exists at the specified path.
2. Try using an absolute path instead of a relative path.
3. Check if the file has the correct permissions.

### Unsupported Video Format

**Problem**: The application cannot load the video file due to an unsupported format.

**Solution**:

1. Convert the video to a supported format (MP4, AVI) using a tool like FFmpeg:
   ```bash
   ffmpeg -i input.mov output.mp4
   ```

2. Make sure you have the necessary codecs installed:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install ubuntu-restricted-extras
   
   # On macOS with Homebrew
   brew install ffmpeg
   
   # On Windows
   # Install the K-Lite Codec Pack
   ```

### Video Playback is Slow

**Problem**: Video playback is slow or choppy.

**Solution**:

1. Reduce the video resolution:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:-1 output.mp4
   ```

2. Increase the cache size in the configuration:
   ```json
   "video": {
       "cache_size": 60
   }
   ```

3. Close other applications to free up system resources.

4. Try a different video player:
   ```bash
   # On Ubuntu/Debian
   vlc path/to/video.mp4
   
   # On macOS
   open -a QuickTime\ Player path/to/video.mp4
   
   # On Windows
   start path/to/video.mp4
   ```

## Annotation Issues

### Annotations Not Saving

**Problem**: Annotations are not being saved.

**Solution**:

1. Make sure the annotation file path is writable:
   ```bash
   # Check permissions
   ls -l path/to/annotations.csv
   
   # Change permissions if needed
   chmod 644 path/to/annotations.csv
   ```

2. Try saving to a different location:
   ```bash
   # Specify a different annotation file
   python main.py --mode annotator --csv path/to/different/annotations.csv
   ```

3. Check if the auto-save option is enabled in the configuration:
   ```json
   "annotations": {
       "auto_save": true
   }
   ```

### Annotations Not Loading

**Problem**: Annotations are not being loaded.

**Solution**:

1. Make sure the annotation file exists and is readable:
   ```bash
   # Check if the file exists
   ls -l path/to/annotations.csv
   
   # Check the file contents
   head path/to/annotations.csv
   ```

2. Make sure the annotation file has the correct format:
   ```
   frame,label
   0,0
   1,0
   2,1
   ```

3. Try loading a different annotation file:
   ```bash
   # Specify a different annotation file
   python main.py --mode annotator --csv path/to/different/annotations.csv
   ```

### Keyboard Shortcuts Not Working

**Problem**: Keyboard shortcuts for annotation are not working.

**Solution**:

1. Make sure the application window has focus.

2. Check if the keyboard shortcuts are correctly configured:
   ```bash
   # Open the configuration editor
   python main.py --mode config
   ```

3. Try using the buttons instead of keyboard shortcuts.

## UI Issues

### Window Size Too Large

**Problem**: The application window is too large for the screen.

**Solution**:

1. Adjust the window size in the configuration:
   ```json
   "ui": {
       "window_width": 800,
       "window_height": 600
   }
   ```

2. Use a lower resolution video:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:-1 output.mp4
   ```

### UI Elements Not Visible

**Problem**: Some UI elements are not visible or are cut off.

**Solution**:

1. Increase the window size in the configuration:
   ```json
   "ui": {
       "window_width": 1200,
       "window_height": 800
   }
   ```

2. Try using a different theme:
   ```python
   # In the code
   style = ttk.Style()
   style.theme_use('clam')  # or 'alt', 'default', 'classic'
   ```

3. Check your display settings and make sure the scaling is set to 100%.

## Performance Issues

### High Memory Usage

**Problem**: The application uses a lot of memory.

**Solution**:

1. Reduce the cache size in the configuration:
   ```json
   "video": {
       "cache_size": 10
   }
   ```

2. Use a lower resolution video:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:-1 output.mp4
   ```

3. Close other applications to free up system resources.

### Application Crashes

**Problem**: The application crashes unexpectedly.

**Solution**:

1. Check the log file for error messages:
   ```bash
   cat video_annotator.log
   ```

2. Run the application with a smaller video:
   ```bash
   ffmpeg -i input.mp4 -t 60 output.mp4  # Extract first 60 seconds
   python main.py --mode annotator --video output.mp4
   ```

3. Try running the application with a different configuration:
   ```bash
   # Create a minimal configuration
   echo '{"labels":[{"name":"Stop","key":"s","value":0,"color":"red"}],"ui":{"controls_right":false,"window_width":800,"window_height":600,"default_fps":30},"video":{"cache_size":10,"default_playback_speed":30},"annotations":{"auto_save":true,"auto_advance":true}}' > minimal_config.json
   
   # Run with the minimal configuration
   python main.py --mode annotator --config minimal_config.json
   ```

## Export Issues

### Export Fails

**Problem**: Exporting annotations fails.

**Solution**:

1. Make sure the output directory exists and is writable:
   ```bash
   # Create the directory if it doesn't exist
   mkdir -p path/to/output
   
   # Check permissions
   ls -l path/to/output
   
   # Change permissions if needed
   chmod 755 path/to/output
   ```

2. Try a different export format:
   ```bash
   # Use the export tool
   python main.py --mode export
   ```

3. Try exporting to a different location:
   ```bash
   # Specify a different output file
   # In the export tool, browse to a different location
   ```

## Getting Help

If you continue to experience issues, please:

1. Check the log file for error messages:
   ```bash
   cat video_annotator.log
   ```

2. Open an issue on the GitHub repository with:
   - A detailed description of the issue
   - Steps to reproduce the issue
   - The error message from the log file
   - Your system information (OS, Python version, etc.)

3. Contact the maintainers directly for urgent issues.
