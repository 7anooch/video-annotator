# Configuration Guide

This guide provides detailed information about configuring the Video Annotator application.

## Configuration Options

The Video Annotator can be configured in several ways:

1. **Configuration File**: Edit the `config.json` file directly
2. **Configuration Profiles**: Use different configuration profiles for different use cases
3. **Configuration Editor**: Use the graphical configuration editor
4. **Command-Line Arguments**: Specify configuration options on the command line

## Configuration File

The configuration file (`config.json`) is a JSON file that contains all the configuration options for the application. It is located in the root directory of the application.

## Configuration Profiles

The Video Annotator supports multiple configuration profiles, allowing you to create different configurations for different use cases. For example, you might have one profile for annotating running behavior and another for annotating feeding behavior.

Profiles are stored in the `profiles` directory in the root directory of the application. Each profile is stored in a separate JSON file with the name of the profile.

You can manage profiles using the Configuration Editor (see below).

### Structure

The configuration file has the following structure:

```json
{
    "profile": "default",
    "labels": [
        {
            "name": "Stop",
            "key": "s",
            "value": 0,
            "color": "red"
        },
        {
            "name": "Run",
            "key": "r",
            "value": 1,
            "color": "green"
        },
        {
            "name": "Turn",
            "key": "t",
            "value": 2,
            "color": "blue"
        }
    ],
    "ui": {
        "controls_right": false,
        "window_width": 1200,
        "window_height": 800,
        "default_fps": 30,
        "theme": "default",
        "font_size": 10,
        "show_frame_number": true,
        "show_timeline": true,
        "show_status_bar": true
    },
    "video": {
        "cache_size": 30,
        "default_playback_speed": 30,
        "frame_step": 1,
        "auto_play": false,
        "loop_playback": false,
        "show_grid": false,
        "grid_size": 50,
        "grid_color": "gray"
    },
    "annotations": {
        "auto_save": true,
        "auto_advance": true,
        "default_csv_name": "{video_name}_annotation.csv",
        "backup_interval": 300,
        "create_backups": true,
        "max_backups": 5
    },
    "keyboard_shortcuts": {
        "play_pause": "space",
        "next_frame": "Right",
        "prev_frame": "Left",
        "next_10_frames": "Shift+Right",
        "prev_10_frames": "Shift+Left",
        "save": "Control+s",
        "quit": "Control+q"
    }
}
```

### Labels

The `labels` section defines the labels that can be assigned to frames:

- `name`: The name of the label
- `key`: The keyboard shortcut for the label
- `value`: The numeric value of the label
- `color`: The color of the label (CSS color name or hex code)

### Profile

The `profile` field specifies the name of the current profile. This is used to identify which profile is currently active.

### UI

The `ui` section defines the user interface settings:

- `controls_right`: Whether to place controls on the right side
- `window_width`: The width of the application window
- `window_height`: The height of the application window
- `default_fps`: The default frames per second for playback
- `theme`: The UI theme (default: "default")
- `font_size`: The font size for UI elements (default: 10)
- `show_frame_number`: Whether to display the current frame number (default: true)
- `show_timeline`: Whether to display the timeline (default: true)
- `show_status_bar`: Whether to display the status bar (default: true)

### Video

The `video` section defines the video playback settings:

- `cache_size`: The number of frames to cache in memory
- `default_playback_speed`: The default playback speed in frames per second
- `frame_step`: The number of frames to skip when using next/previous frame buttons (default: 1)
- `auto_play`: Whether to automatically start playback when opening a video (default: false)
- `loop_playback`: Whether to loop playback when reaching the end of the video (default: false)
- `show_grid`: Whether to display a grid overlay on the video (default: false)
- `grid_size`: The size of the grid cells in pixels (default: 50)
- `grid_color`: The color of the grid (default: "gray")

### Annotations

The `annotations` section defines the annotation settings:

- `auto_save`: Whether to automatically save annotations
- `auto_advance`: Whether to automatically advance to the next frame after annotating
- `default_csv_name`: The default name for annotation CSV files (default: "{video_name}_annotation.csv")
- `backup_interval`: The interval in seconds between automatic backups (default: 300)
- `create_backups`: Whether to create backup files (default: true)
- `max_backups`: The maximum number of backup files to keep (default: 5)

### Keyboard Shortcuts

The `keyboard_shortcuts` section defines the keyboard shortcuts for various actions:

- `play_pause`: The key for play/pause (default: "space")
- `next_frame`: The key for next frame (default: "Right")
- `prev_frame`: The key for previous frame (default: "Left")
- `next_10_frames`: The key for next 10 frames (default: "Shift+Right")
- `prev_10_frames`: The key for previous 10 frames (default: "Shift+Left")
- `save`: The key for save (default: "Control+s")
- `quit`: The key for quit (default: "Control+q")

## Configuration Editor

The configuration editor provides a graphical interface for editing the configuration file.

### Launching the Configuration Editor

You can launch the configuration editor using one of the following methods:

#### Using the Run Scripts

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

Select "3. Run Configuration Editor" from the menu.

#### Using the Command Line

```bash
# Run the configuration editor
python main.py --mode config
```

### Using the Configuration Editor

The configuration editor has five tabs:

1. **Profiles**: Manage configuration profiles
2. **Labels**: Edit the labels
3. **UI**: Edit the user interface settings
4. **Video**: Edit the video playback settings
5. **Annotations**: Edit the annotation settings

#### Profiles Tab

The Profiles tab allows you to:

- Create new profiles
- Switch between profiles
- Delete profiles
- Import profiles from JSON files
- Export profiles to JSON files

![Profiles Tab](../images/config_profiles_tab.png)

To create a new profile:

1. Click the "Create" button
2. Enter a name for the profile
3. Select a base profile to use as a template
4. Click "OK"

To switch to a different profile:

1. Select the profile in the list
2. Click the "Switch" button

To delete a profile:

1. Select the profile in the list
2. Click the "Delete" button
3. Confirm the deletion

To import a profile:

1. Click the "Import" button
2. Select a JSON file to import
3. Enter a name for the profile
4. Click "OK"

To export a profile:

1. Select the profile in the list
2. Click the "Export" button
3. Choose a location to save the profile
4. Click "Save"

#### Labels Tab

The Labels tab allows you to:

- Add new labels
- Edit existing labels
- Delete labels

![Labels Tab](../images/config_labels_tab.png)

To add a new label:

1. Click the "Add" button
2. Enter the label name, key, value, and color
3. Click "OK"

To edit a label:

1. Select the label in the list
2. Click the "Edit" button
3. Modify the label properties
4. Click "OK"

To delete a label:

1. Select the label in the list
2. Click the "Delete" button
3. Confirm the deletion

#### UI Tab

The UI tab allows you to:

- Toggle controls on the right side
- Set the window width and height
- Set the default FPS

![UI Tab](../images/config_ui_tab.png)

#### Video Tab

The Video tab allows you to:

- Set the cache size
- Set the default playback speed

![Video Tab](../images/config_video_tab.png)

#### Annotations Tab

The Annotations tab allows you to:

- Toggle auto-save
- Toggle auto-advance

![Annotations Tab](../images/config_annotations_tab.png)

### Saving Configuration

To save your changes, click the "Save" button at the bottom of the configuration editor.

## Command-Line Arguments

You can specify some configuration options on the command line when launching the application:

```bash
# Run the annotator with controls on the right side
python main.py --mode annotator --side_controls

# Run the annotator with a specific configuration file
python main.py --mode annotator --config custom_config.json

# Run the annotator with a specific video file
python main.py --mode annotator --video path/to/video.mp4

# Run the annotator with a specific annotation file
python main.py --mode annotator --csv path/to/annotations.csv
```

## Programmatic Configuration

You can also configure the application programmatically using the Python API:

```python
from src.core.config import Config

# Load the default configuration
config = Config('config.json')

# Load a specific profile
config = Config('config.json', profile='my_profile')

# Get the labels
labels = config.get_labels()

# Add a new label
config.add_label("Jump", "j", 3, "purple")

# Remove a label
config.remove_label(2)  # Remove the label with value 2

# Update the UI configuration
ui_config = config.get_ui_config()
ui_config['controls_right'] = True
config.config['ui'] = ui_config

# Save the configuration
config.save_config()

# Create a new profile
config.create_profile('new_profile', 'default')

# Switch to a different profile
config.switch_profile('new_profile')

# Delete a profile
config.delete_profile('old_profile')

# Import a profile from a file
config.import_profile('path/to/profile.json', 'imported_profile')

# Export a profile to a file
config.export_profile('my_profile', 'path/to/export.json')
```

## Next Steps

After configuring the application, you can:

- [Annotate](annotation_guide.md) videos with your custom labels
- [Visualize](visualization_guide.md) your annotations
- [Analyze](analysis_guide.md) your annotations
- [Export](export_guide.md) your annotations
