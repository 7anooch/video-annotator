# Tools API

The Tools API provides standalone tools for visualization, analysis, and export.

## Visualization

The `visualization` module provides tools for visualizing annotations.

### Classes

#### VisualizationTool

The `VisualizationTool` class is responsible for visualizing annotations.

##### Constructor

```python
VisualizationTool(config_path='config.json')
```

- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

##### Methods

###### load_annotations

```python
load_annotations(csv_path)
```

Load annotations from a CSV file.

- `csv_path` (str): Path to the CSV file
- Returns: dict: Dictionary of frame numbers to labels

###### plot_ethogram

```python
plot_ethogram(annotations, title="Ethogram", show_legend=True, ax=None)
```

Plot an ethogram of the annotations.

- `annotations` (dict): Dictionary of frame numbers to labels
- `title` (str, optional): Title of the plot. Defaults to "Ethogram".
- `show_legend` (bool, optional): Whether to show the legend. Defaults to True.
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### plot_label_distribution

```python
plot_label_distribution(annotations, title="Label Distribution", ax=None)
```

Plot the distribution of labels in the annotations.

- `annotations` (dict): Dictionary of frame numbers to labels
- `title` (str, optional): Title of the plot. Defaults to "Label Distribution".
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### plot_label_timeline

```python
plot_label_timeline(annotations, title="Label Timeline", ax=None)
```

Plot a timeline of when each label occurs in the video.

- `annotations` (dict): Dictionary of frame numbers to labels
- `title` (str, optional): Title of the plot. Defaults to "Label Timeline".
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### plot_transition_matrix

```python
plot_transition_matrix(annotations, title="Transition Matrix", ax=None)
```

Plot a transition matrix showing how often one label transitions to another.

- `annotations` (dict): Dictionary of frame numbers to labels
- `title` (str, optional): Title of the plot. Defaults to "Transition Matrix".
- `ax` (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### visualize_annotations

```python
visualize_annotations(csv_path)
```

Visualize the annotations in a CSV file.

- `csv_path` (str): Path to the CSV file

#### VisualizationGUI

The `VisualizationGUI` class provides a graphical interface for visualizing annotations.

##### Constructor

```python
VisualizationGUI(master, config_path='config.json')
```

- `master` (tk.Tk): The main Tkinter window
- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

##### Methods

###### setup_ui

```python
setup_ui()
```

Set up the UI elements.

###### browse_file

```python
browse_file()
```

Browse for a CSV file.

###### visualize

```python
visualize()
```

Visualize the annotations.

## Analysis

The `analyze` module provides tools for analyzing annotations.

### Functions

#### load_annotations

```python
load_annotations(csv_path)
```

Load annotations from a CSV file.

- `csv_path` (str): Path to the CSV file
- Returns: dict: Dictionary of frame numbers to labels

#### get_csv_paths

```python
get_csv_paths()
```

Get paths to CSV files from command-line arguments or file dialog.

- Returns: list: List of CSV file paths

#### get_all_annotations

```python
get_all_annotations(csv_paths)
```

Get annotations from multiple CSV files.

- `csv_paths` (list): List of CSV file paths
- Returns: tuple: (dict of annotations, minimum length)

#### analyze_sequence

```python
analyze_sequence(annotations)
```

Analyze a sequence of annotations.

- `annotations` (dict): Dictionary of frame numbers to labels
- Returns: list: List of tuples (label, duration, frame_list)

#### analyze_annotations

```python
analyze_annotations(csv_paths)
```

Analyze annotations from multiple CSV files.

- `csv_paths` (list): List of CSV file paths

## Export

The `export_gui` module provides a graphical interface for exporting annotations.

### Classes

#### ExportGUI

The `ExportGUI` class provides a graphical interface for exporting annotations.

##### Constructor

```python
ExportGUI(master, annotation_path=None)
```

- `master` (tk.Tk): The main Tkinter window
- `annotation_path` (str, optional): Path to the annotation file. Defaults to None.

##### Methods

###### setup_ui

```python
setup_ui()
```

Set up the UI elements.

###### browse_file

```python
browse_file()
```

Browse for an annotation file.

###### browse_output

```python
browse_output()
```

Browse for an output file.

###### update_output_extension

```python
update_output_extension(event=None)
```

Update the output file extension based on the selected format.

- `event` (optional): The event that triggered the update. Defaults to None.

###### load_annotations

```python
load_annotations()
```

Load annotations from the file.

###### update_annotation_info

```python
update_annotation_info()
```

Update the annotation info labels.

###### export

```python
export()
```

Export the annotations to the selected format.

## Performance Profiler

The `performance_profiler` module provides tools for profiling application performance.

### Classes

#### PerformanceProfiler

The `PerformanceProfiler` class is responsible for profiling application performance.

##### Constructor

```python
PerformanceProfiler()
```

##### Methods

###### profile_video_loading

```python
profile_video_loading(video_path, num_frames=100)
```

Profile the performance of video loading.

- `video_path` (str): Path to the video file
- `num_frames` (int, optional): Number of frames to load. Defaults to 100.
- Returns: dict: Performance metrics

###### profile_annotation_loading

```python
profile_annotation_loading(annotation_path)
```

Profile the performance of annotation loading.

- `annotation_path` (str): Path to the annotation file
- Returns: dict: Performance metrics

###### profile_annotation_saving

```python
profile_annotation_saving(annotation_path, num_annotations=1000)
```

Profile the performance of annotation saving.

- `annotation_path` (str): Path to the annotation file
- `num_annotations` (int, optional): Number of annotations to save. Defaults to 1000.
- Returns: dict: Performance metrics

###### profile_function

```python
profile_function(func, *args, **kwargs)
```

Profile a specific function.

- `func` (function): The function to profile
- `*args`: Arguments to pass to the function
- `**kwargs`: Keyword arguments to pass to the function
- Returns: tuple: (result, profile_stats)

###### plot_frame_times

```python
plot_frame_times(frame_times, title="Frame Loading Times")
```

Plot the frame loading times.

- `frame_times` (list): List of frame loading times
- `title` (str, optional): Title of the plot. Defaults to "Frame Loading Times".
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### plot_frame_time_histogram

```python
plot_frame_time_histogram(frame_times, title="Frame Loading Time Distribution")
```

Plot a histogram of frame loading times.

- `frame_times` (list): List of frame loading times
- `title` (str, optional): Title of the plot. Defaults to "Frame Loading Time Distribution".
- Returns: matplotlib.figure.Figure: The figure containing the plot

###### generate_performance_report

```python
generate_performance_report(video_path, annotation_path, output_path=None)
```

Generate a comprehensive performance report.

- `video_path` (str): Path to the video file
- `annotation_path` (str): Path to the annotation file
- `output_path` (str, optional): Path to save the report. Defaults to None.
- Returns: dict: Performance metrics

#### PerformanceProfilerGUI

The `PerformanceProfilerGUI` class provides a graphical interface for profiling application performance.

##### Constructor

```python
PerformanceProfilerGUI(master)
```

- `master` (tk.Tk): The main Tkinter window

##### Methods

###### setup_ui

```python
setup_ui()
```

Set up the UI elements.

###### browse_video

```python
browse_video()
```

Browse for a video file.

###### browse_annotation

```python
browse_annotation()
```

Browse for an annotation file.

###### browse_output

```python
browse_output()
```

Browse for an output file.

###### profile

```python
profile()
```

Profile the application.

## Patch Gaps

The `patch_gaps` module provides tools for patching gaps in annotations.

### Functions

#### load_annotations

```python
load_annotations(csv_path)
```

Load annotations from a CSV file.

- `csv_path` (str): Path to the CSV file
- Returns: dict: Dictionary of frame numbers to labels

#### get_csv_paths

```python
get_csv_paths()
```

Get paths to CSV files from command-line arguments or file dialog.

- Returns: list: List of CSV file paths

#### get_all_annotations

```python
get_all_annotations(csv_paths)
```

Get annotations from multiple CSV files.

- `csv_paths` (list): List of CSV file paths
- Returns: tuple: (dict of annotations, minimum length)

#### connected_components_1D

```python
connected_components_1D(arr, dust_size=None)
```

Find connected components in a 1D array.

- `arr` (numpy.ndarray): The array to find connected components in
- `dust_size` (int, optional): Minimum size of components to keep. Defaults to None.
- Returns: dict: Dictionary of component indices to component arrays

#### analyze_sequence

```python
analyze_sequence(annotations)
```

Analyze a sequence of annotations.

- `annotations` (dict): Dictionary of frame numbers to labels
- Returns: list: List of tuples (label, duration, frame_list)

#### find_gaps

```python
find_gaps(updated_ground_truth_dict, max_gap_size=30)
```

Find gaps in annotations.

- `updated_ground_truth_dict` (dict): Dictionary of frame numbers to labels
- `max_gap_size` (int, optional): Maximum size of gaps to find. Defaults to 30.
- Returns: list: List of tuples (label, gap_size, gap_frames)

#### fill_missing_labels

```python
fill_missing_labels(ground_truth_path, ground_truth_ds_path, output_path)
```

Fill missing labels in ground truth annotations.

- `ground_truth_path` (str): Path to the ground truth CSV file
- `ground_truth_ds_path` (str): Path to the ground truth DS CSV file
- `output_path` (str): Path to the output CSV file
