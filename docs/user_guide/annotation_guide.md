# Annotation Guide

This guide provides detailed information about annotating videos using the Video Annotator application.

## Annotation Basics

### Understanding Labels

Labels are the categories you assign to each frame of the video. By default, the application comes with three labels:

- **Stop**: Typically used to indicate when the subject is not moving
- **Run**: Typically used to indicate when the subject is moving
- **Turn**: Typically used to indicate when the subject is changing direction

You can customize these labels in the configuration file. See the [Configuration Guide](configuration_guide.md) for more information.

### Annotation Methods

The Video Annotator provides two main methods for annotating frames:

1. **Single Frame Annotation**: Annotate one frame at a time
2. **Range Annotation**: Annotate a range of frames with the same label

## Single Frame Annotation

### Using Keyboard Shortcuts

The fastest way to annotate frames is using keyboard shortcuts:

- **s**: Annotate the current frame as "Stop"
- **r**: Annotate the current frame as "Run"
- **t**: Annotate the current frame as "Turn"

After annotating a frame, the application automatically advances to the next frame, allowing you to quickly annotate a sequence of frames.

### Using Buttons

You can also annotate frames by clicking on the label buttons in the interface:

1. Navigate to the frame you want to annotate
2. Click on the label button (e.g., "Stop", "Run", "Turn")
3. The application will automatically advance to the next frame

## Range Annotation

Range annotation is useful when you have a sequence of frames that should all have the same label.

### Steps for Range Annotation

1. Enter the start frame in the "Start Frame" field
2. Enter the end frame in the "End Frame" field
3. Click "Label Range"
4. Select a label from the dialog that appears
5. Click "OK" to annotate the range

![Range Annotation](../images/range_annotation.png)

### Progress Indicator

For long ranges, a progress indicator will be displayed to show the annotation progress.

## Annotation Strategies

### Frame-by-Frame Annotation

For the most accurate annotations, it's recommended to go through the video frame by frame and annotate each frame individually. This is especially important for behaviors that change rapidly.

### Range Annotation for Stable Behaviors

For stable behaviors that persist over many frames, range annotation can save time. For example, if the subject is running for 100 frames, you can use range annotation to label all 100 frames as "Run" at once.

### Combining Methods

A common strategy is to use range annotation for stable behaviors and single frame annotation for transitions between behaviors.

## Annotation Visualization

As you annotate frames, the annotations are displayed in the annotations listbox on the left side of the interface. Each frame is listed with its annotation, and the color of the text indicates the label:

- **Red**: Stop
- **Green**: Run
- **Blue**: Turn

You can click on an entry in the listbox to jump to that frame.

## Saving Annotations

Annotations are automatically saved to the specified CSV file as you annotate frames. The CSV file has the following format:

```
frame,label
0,0
1,0
2,1
3,1
...
```

Where:
- `frame`: The frame number (0-indexed)
- `label`: The label value (0 = Stop, 1 = Run, 2 = Turn, or your custom values)

## Continuing Annotation Sessions

You can continue an annotation session by selecting the same video file and annotation file. The application will load the existing annotations, allowing you to continue where you left off.

## Tips for Efficient Annotation

- **Use keyboard shortcuts** for faster annotation
- **Use range annotation** for stable behaviors
- **Take breaks** to avoid fatigue, which can lead to annotation errors
- **Review your annotations** using the visualization tool
- **Compare with other annotators** to ensure consistency

## Advanced Annotation Features

### Customizing Labels

You can customize the labels used for annotation by editing the configuration file or using the configuration editor. See the [Configuration Guide](configuration_guide.md) for more information.

### Annotating Multiple Videos

To annotate multiple videos, simply close the current video and open a new one. Each video will have its own annotation file.

### Collaborative Annotation

For collaborative annotation, multiple annotators can annotate the same video independently. The annotations can then be compared and merged using the ground truth generator. See the [Analysis Guide](analysis_guide.md) for more information.

## Next Steps

After annotating your videos, you can:

- [Visualize](visualization_guide.md) your annotations to identify patterns
- [Analyze](analysis_guide.md) your annotations to extract insights
- [Export](export_guide.md) your annotations for use in other applications
