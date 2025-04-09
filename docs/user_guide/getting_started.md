# Getting Started

This guide provides a quick introduction to using the Video Annotator application.

## Launch the Application

After [installing](installation.md) the Video Annotator, you can launch it using one of the following methods:

### Using the Run Scripts

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

This will display a menu with various options. Select "1. Run Annotator (Modular)" to launch the main application.

### Using the Command Line

```bash
# Run the annotator
python main.py --mode annotator
```

## Basic Workflow

### 1. Select a Video File

When you launch the application, a file dialog will appear. Select a video file (AVI or MP4) to annotate.

![Select Video File](../images/select_video.png)

### 2. Specify an Annotation File

After selecting a video file, you'll be prompted to enter a name for the annotation file. You can press Enter to use the default name (based on the video file name) or enter a custom name.

![Specify Annotation File](../images/specify_annotation.png)

### 3. Navigate the Video

Once the video is loaded, you can navigate through it using the following controls:

- **Left Arrow**: Go to the previous frame
- **Right Arrow**: Go to the next frame
- **Spacebar**: Play/Pause the video
- **Go to Frame**: Enter a specific frame number to jump to

![Video Navigation](../images/video_navigation.png)

### 4. Annotate Frames

You can annotate frames using the following methods:

#### Single Frame Annotation

1. Navigate to the frame you want to annotate
2. Click on one of the label buttons (e.g., "Stop", "Run", "Turn")
3. Alternatively, use the keyboard shortcuts (e.g., "s" for Stop, "r" for Run, "t" for Turn)

![Single Frame Annotation](../images/single_frame_annotation.png)

#### Range Annotation

1. Enter the start frame in the "Start Frame" field
2. Enter the end frame in the "End Frame" field
3. Click "Label Range"
4. Select a label from the dialog that appears
5. Click "OK" to annotate the range

![Range Annotation](../images/range_annotation.png)

### 5. Save Annotations

Annotations are automatically saved to the specified CSV file as you annotate frames.

### 6. Visualize Annotations

To visualize your annotations, you can use the visualization tool:

```bash
# Using the run scripts
# Select "8. Run Visualization Tool" from the menu

# Using the command line
python main.py --mode visualize
```

This will open the visualization tool, where you can select your annotation file and choose which visualizations to display.

![Visualization Tool](../images/visualization_tool.png)

### 7. Analyze Annotations

To analyze your annotations, you can use the analysis tool:

```bash
# Using the run scripts
# Select "5. Run Analysis Tool" from the menu

# Using the command line
python main.py --mode analyze
```

This will open the analysis tool, where you can select your annotation file and perform various analyses.

![Analysis Tool](../images/analysis_tool.png)

### 8. Export Annotations

To export your annotations to different formats, you can use the export tool:

```bash
# Using the run scripts
# Select "7. Run Export Tool" from the menu

# Using the command line
python main.py --mode export
```

This will open the export tool, where you can select your annotation file and choose the export format.

![Export Tool](../images/export_tool.png)

## Next Steps

Now that you're familiar with the basic workflow, you can explore the following guides for more detailed information:

- [Annotation Guide](annotation_guide.md): Detailed information about annotating videos
- [Visualization Guide](visualization_guide.md): How to visualize annotations
- [Analysis Guide](analysis_guide.md): How to analyze annotations
- [Export Guide](export_guide.md): How to export annotations
- [Configuration Guide](configuration_guide.md): How to configure the application
