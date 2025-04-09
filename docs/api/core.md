# Core API

The Core API provides the core functionality for video playback and annotation management.

## VideoPlayer

The `VideoPlayer` class is responsible for loading and playing videos.

### Constructor

```python
VideoPlayer(video_path, config_path='config.json')
```

- `video_path` (str): Path to the video file
- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

### Properties

- `video_path` (str): Path to the video file
- `total_frames` (int): Total number of frames in the video
- `frame_number` (int): Current frame number
- `fps` (float): Frames per second
- `playing` (bool): Whether the video is playing
- `current_frame` (numpy.ndarray): The current frame
- `current_frame_number` (int): The current frame number

### Methods

#### load_frame

```python
load_frame(frame_number)
```

Load a specific frame from the video.

- `frame_number` (int): The frame number to load
- Returns: numpy.ndarray: The loaded frame as an RGB image

#### next_frame

```python
next_frame()
```

Move to the next frame.

- Returns: numpy.ndarray: The next frame as an RGB image

#### prev_frame

```python
prev_frame()
```

Move to the previous frame.

- Returns: numpy.ndarray: The previous frame as an RGB image

#### go_to_frame

```python
go_to_frame(frame_number)
```

Go to a specific frame.

- `frame_number` (int): The frame number to go to
- Returns: numpy.ndarray: The specified frame as an RGB image

#### resize_frame

```python
resize_frame(frame, target_width=1200)
```

Resize a frame to a target width while maintaining aspect ratio.

- `frame` (numpy.ndarray): The frame to resize
- `target_width` (int, optional): The target width. Defaults to 1200.
- Returns: numpy.ndarray: The resized frame

## AnnotationManager

The `AnnotationManager` class is responsible for managing annotations.

### Constructor

```python
AnnotationManager(annotation_path, config_path='config.json')
```

- `annotation_path` (str): Path to the annotation CSV file
- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

### Properties

- `annotation_path` (str): Path to the annotation CSV file
- `annotations` (dict): Dictionary of frame numbers to labels
- `annotation_count` (int): Number of annotations

### Methods

#### load_annotations

```python
load_annotations()
```

Load annotations from the CSV file.

- Returns: dict: Dictionary of frame numbers to labels

#### save_annotations

```python
save_annotations()
```

Save annotations to the CSV file.

#### annotate_frame

```python
annotate_frame(frame, label)
```

Annotate a single frame.

- `frame` (int): Frame number to annotate
- `label` (int): Label to assign to the frame
- Returns: bool: True if successful, False otherwise

#### annotate_frame_range

```python
annotate_frame_range(start_frame, end_frame, label, callback=None)
```

Annotate a range of frames.

- `start_frame` (int): First frame in the range
- `end_frame` (int): Last frame in the range
- `label` (int): Label to assign to the frames
- `callback` (function, optional): Callback function to report progress. Defaults to None.
- Returns: bool: True if successful, False otherwise

#### get_annotation

```python
get_annotation(frame)
```

Get the annotation for a specific frame.

- `frame` (int): Frame number
- Returns: int or None: The label for the frame, or None if not annotated

#### clear_annotation

```python
clear_annotation(frame)
```

Clear the annotation for a specific frame.

- `frame` (int): Frame number
- Returns: bool: True if successful, False otherwise

#### clear_all_annotations

```python
clear_all_annotations()
```

Clear all annotations.

- Returns: bool: True if successful, False otherwise

## Config

The `Config` class is responsible for managing configuration settings.

### Constructor

```python
Config(config_path='config.json')
```

- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

### Properties

- `config_path` (str): Path to the configuration file
- `config` (dict): The loaded configuration

### Methods

#### load_config

```python
load_config()
```

Load the configuration from the file.

- Returns: dict: The loaded configuration

#### save_config

```python
save_config(config=None)
```

Save the configuration to the file.

- `config` (dict, optional): The configuration to save. If None, saves the current config. Defaults to None.

#### get_labels

```python
get_labels()
```

Get the labels from the configuration.

- Returns: list: The labels

#### get_ui_config

```python
get_ui_config()
```

Get the UI configuration.

- Returns: dict: The UI configuration

#### get_video_config

```python
get_video_config()
```

Get the video configuration.

- Returns: dict: The video configuration

#### get_annotations_config

```python
get_annotations_config()
```

Get the annotations configuration.

- Returns: dict: The annotations configuration

#### update_config

```python
update_config(new_config)
```

Update the configuration with new values.

- `new_config` (dict): The new configuration values

#### update_labels

```python
update_labels(labels)
```

Update the labels in the configuration.

- `labels` (list): The new labels

#### add_label

```python
add_label(name, key, value, color)
```

Add a new label to the configuration.

- `name` (str): The name of the label
- `key` (str): The keyboard shortcut for the label
- `value` (int): The value of the label
- `color` (str): The color of the label

#### remove_label

```python
remove_label(value)
```

Remove a label from the configuration.

- `value` (int): The value of the label to remove

#### get_label_by_value

```python
get_label_by_value(value)
```

Get a label by its value.

- `value` (int): The value of the label
- Returns: dict: The label, or None if not found

#### get_label_by_key

```python
get_label_by_key(key)
```

Get a label by its key.

- `key` (str): The key of the label
- Returns: dict: The label, or None if not found
