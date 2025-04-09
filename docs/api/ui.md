# UI API

The UI API provides user interface components for interacting with the application.

## UIController

The `UIController` class is responsible for managing the user interface.

### Constructor

```python
UIController(master, video_player, annotation_manager, config_path='config.json', controls_right=False)
```

- `master` (tk.Tk): The main Tkinter window
- `video_player` (VideoPlayer): The video player object
- `annotation_manager` (AnnotationManager): The annotation manager object
- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.
- `controls_right` (bool, optional): Whether to place controls on the right side. Defaults to False.

### Methods

#### setup_ui

```python
setup_ui()
```

Set up the UI elements.

#### setup_key_bindings

```python
setup_key_bindings()
```

Set up key bindings.

#### load_frame

```python
load_frame(frame_number)
```

Load a frame and display it.

- `frame_number` (int): The frame number to load

#### update_entry

```python
update_entry()
```

Update the frame entry with the current frame number.

#### on_progress_bar_click

```python
on_progress_bar_click(event)
```

Handle clicks on the progress bar.

- `event`: The click event

#### on_annotation_select

```python
on_annotation_select(event)
```

Handle selection in the annotations listbox.

- `event`: The selection event

#### toggle_play_pause

```python
toggle_play_pause()
```

Toggle between play and pause states.

#### play_frame_set

```python
play_frame_set()
```

Play a set of frames.

#### update_listbox_selection

```python
update_listbox_selection(frame_jump=25)
```

Update the selection in the annotations listbox.

- `frame_jump` (int, optional): Number of frames to jump ahead for scrolling. Defaults to 25.

#### process_frames

```python
process_frames()
```

Process frames from the queue.

#### display_frame

```python
display_frame(frame)
```

Display a frame.

- `frame`: The frame to display

#### prev_frame

```python
prev_frame()
```

Go to the previous frame.

#### next_frame

```python
next_frame()
```

Go to the next frame.

#### go_to_frame

```python
go_to_frame(frame_number=None)
```

Go to a specific frame.

- `frame_number` (int, optional): The frame number to go to. If None, uses the entry field. Defaults to None.

#### set_speed

```python
set_speed(speed)
```

Set the playback speed.

- `speed` (str): The speed in the format "X fps"

#### annotate_frame

```python
annotate_frame(label)
```

Annotate the current frame.

- `label` (int): The label to assign to the frame

#### label_range

```python
label_range()
```

Label a range of frames.

#### update_annotations_listbox

```python
update_annotations_listbox()
```

Update the annotations listbox.

## ConfigEditor

The `ConfigEditor` class provides a graphical interface for editing configuration settings.

### Constructor

```python
ConfigEditor(master, config_path='config.json')
```

- `master` (tk.Tk): The main Tkinter window
- `config_path` (str, optional): Path to the configuration file. Defaults to 'config.json'.

### Methods

#### setup_labels_tab

```python
setup_labels_tab()
```

Set up the labels tab.

#### setup_ui_tab

```python
setup_ui_tab()
```

Set up the UI tab.

#### setup_video_tab

```python
setup_video_tab()
```

Set up the video tab.

#### setup_annotations_tab

```python
setup_annotations_tab()
```

Set up the annotations tab.

#### populate_labels_tree

```python
populate_labels_tree()
```

Populate the labels treeview.

#### add_label

```python
add_label()
```

Add a new label.

#### edit_label

```python
edit_label()
```

Edit the selected label.

#### delete_label

```python
delete_label()
```

Delete the selected label.

#### choose_color

```python
choose_color(color_var)
```

Choose a color.

- `color_var` (tk.StringVar): The color variable to update

#### save_config

```python
save_config()
```

Save the configuration.
