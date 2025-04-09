# UI Improvements

This document describes the UI improvements implemented in the Video Annotator application.

## Overview

The Video Annotator UI has been enhanced with several new features to improve the user experience:

1. **Theme System**: A flexible theme system with multiple built-in themes
2. **Keyboard Shortcuts Manager**: Enhanced keyboard shortcut support
3. **Status Bar**: A status bar for displaying application information
4. **Timeline**: A timeline for navigating through video frames
5. **Help Dialog**: A help dialog for displaying keyboard shortcuts and other help information
6. **Menu Bar**: A menu bar for accessing application features

## Theme System

The theme system allows users to customize the appearance of the application. It supports multiple built-in themes and the ability to create custom themes.

### Built-in Themes

The application includes the following built-in themes:

- **Default**: The default light theme
- **Dark**: A dark theme for low-light environments
- **Light Blue**: A light blue theme
- **High Contrast**: A high contrast theme for accessibility

### Custom Themes

Users can create custom themes by creating JSON files in the `themes` directory. Each theme file should include the following properties:

```json
{
    "bg": "#f0f0f0",
    "fg": "#000000",
    "button_bg": "#e0e0e0",
    "button_fg": "#000000",
    "highlight_bg": "#4a6984",
    "highlight_fg": "#ffffff",
    "entry_bg": "#ffffff",
    "entry_fg": "#000000",
    "canvas_bg": "#000000",
    "label_bg": "#f0f0f0",
    "label_fg": "#000000",
    "menu_bg": "#f0f0f0",
    "menu_fg": "#000000",
    "status_bg": "#f0f0f0",
    "status_fg": "#000000",
    "timeline_bg": "#e0e0e0",
    "timeline_fg": "#000000",
    "timeline_marker": "#ff0000",
    "timeline_annotation": "#4a6984",
    "font_family": "Arial",
    "font_size": 10
}
```

### Theme Manager

The `ThemeManager` class in `src/ui/theme_manager.py` provides functionality for managing UI themes. It includes methods for:

- Loading themes from the `themes` directory
- Applying themes to the application
- Getting theme colors and fonts
- Saving custom themes
- Deleting custom themes

## Keyboard Shortcuts Manager

The keyboard shortcuts manager provides a centralized way to manage keyboard shortcuts in the application. It supports:

- Registering callback functions for keyboard actions
- Binding keyboard shortcuts to actions
- Getting and setting keyboard shortcuts
- Resetting shortcuts to their default values

### Default Shortcuts

The application includes the following default keyboard shortcuts:

- **Space**: Play/Pause
- **Right Arrow**: Next Frame
- **Left Arrow**: Previous Frame
- **Shift+Right Arrow**: Next 10 Frames
- **Shift+Left Arrow**: Previous 10 Frames
- **Ctrl+S**: Save
- **Ctrl+Q**: Quit
- **F11**: Toggle Fullscreen
- **F10**: Toggle Controls
- **F9**: Toggle Timeline
- **F8**: Toggle Status Bar
- **+**: Increase Speed
- **-**: Decrease Speed
- **0**: Reset Speed
- **Home**: Jump to Start
- **End**: Jump to End
- **Delete**: Delete Annotation
- **Ctrl+C**: Copy Annotation
- **Ctrl+V**: Paste Annotation
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **F1**: Help
- **F2**: Settings
- **F3**: Toggle Theme
- **Ctrl++**: Zoom In
- **Ctrl+-**: Zoom Out
- **Ctrl+0**: Zoom Reset

### Keyboard Manager

The `KeyboardManager` class in `src/ui/keyboard_manager.py` provides functionality for managing keyboard shortcuts. It includes methods for:

- Registering callback functions for keyboard actions
- Binding keyboard shortcuts to actions
- Getting and setting keyboard shortcuts
- Resetting shortcuts to their default values
- Getting human-readable descriptions of shortcuts and actions

## Status Bar

The status bar displays information about the current state of the application. It includes:

- Status messages
- Frame information
- Annotation information
- Playback speed
- Current time

### Status Bar Component

The `StatusBar` class in `src/ui/status_bar.py` provides a status bar component for displaying information to the user. It includes methods for:

- Setting the status message
- Setting frame information
- Setting annotation information
- Setting playback speed
- Updating the theme

## Timeline

The timeline provides a visual representation of the video frames and annotations. It allows users to:

- Navigate through the video by clicking on the timeline
- See the current frame position
- See annotations on the timeline
- See the current time

### Timeline Component

The `Timeline` class in `src/ui/timeline.py` provides a timeline component for navigating through video frames. It includes methods for:

- Updating the timeline with new frame information
- Updating annotations on the timeline
- Drawing the timeline
- Updating the theme

## Help Dialog

The help dialog provides information about keyboard shortcuts and other help topics. It includes:

- A list of keyboard shortcuts
- Information about the application
- Getting started information

### Help Dialog Component

The `HelpDialog` class in `src/ui/help_dialog.py` provides a help dialog for displaying keyboard shortcuts and other help information. It includes:

- A keyboard shortcuts tab
- An about tab
- A getting started tab

## Menu Bar

The menu bar provides access to application features. It includes:

- File menu (Open, Save, Exit)
- Edit menu (Copy, Paste, Delete, Settings)
- View menu (Theme, Timeline, Status Bar, Fullscreen)
- Help menu (Keyboard Shortcuts, About)

### Menu Bar Implementation

The menu bar is implemented in the `UIController` class in `src/ui/ui_controller.py`. It includes methods for:

- Setting up the menu bar
- Handling menu commands
- Registering keyboard shortcuts

## Integration with UI Controller

The UI improvements are integrated with the `UIController` class in `src/ui/ui_controller.py`. The controller:

- Initializes the theme manager, keyboard manager, status bar, and timeline
- Sets up the menu bar
- Registers keyboard shortcuts
- Handles UI events
- Updates the UI based on the current configuration

## Configuration Integration

The UI improvements are integrated with the configuration system. The configuration includes:

- Theme settings
- Timeline visibility settings
- Status bar visibility settings
- Keyboard shortcut settings

## Best Practices

When working with the UI components, follow these best practices:

1. **Use the Theme Manager**: Use the theme manager to get colors and fonts instead of hardcoding them
2. **Register Keyboard Shortcuts**: Register keyboard shortcuts with the keyboard manager instead of binding them directly
3. **Update the Status Bar**: Update the status bar with relevant information when actions are performed
4. **Update the Timeline**: Update the timeline when the video frame or annotations change
5. **Use the Menu Bar**: Add new features to the menu bar for easy access

## Future Improvements

Future UI improvements could include:

1. **Customizable Keyboard Shortcuts**: Allow users to customize keyboard shortcuts through the UI
2. **Theme Editor**: Add a theme editor for creating and editing themes
3. **Dockable Panels**: Add support for dockable panels
4. **Multiple Layouts**: Add support for multiple UI layouts
5. **Touch Support**: Add support for touch gestures
