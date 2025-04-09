# Technical Implementation Plan for Video Annotator

This document outlines specific technical changes needed to address the issues identified in the improvement plan.

## 1. Code Refactoring

### 1.1 Remove Global Variables

Current issue:
```python
# Global variables
annotations = {}
frame_counter = 0
```

Solution:
- Move `annotations` to be a class attribute of `VideoApp`
- Remove `frame_counter` if not needed or make it a class attribute

Implementation:
```python
class VideoApp:
    def __init__(self, master, video_path, annotation_path, controls_right=False):
        self.master = master
        self.annotations = {}
        self.frame_counter = 0
        # ... rest of initialization
```

Update all methods that use the global annotations:
```python
def load_annotations(self):
    if os.path.exists(self.annotation_path):
        df = pd.read_csv(self.annotation_path)
        self.annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
        self.update_annotations_listbox()
        print(f"Loaded annotations from {self.annotation_path}")
```

### 1.2 Modularize the Code

Split the large `VideoApp` class into smaller, focused classes:

1. **VideoPlayer**: Handles video loading, playback, and frame navigation
2. **AnnotationManager**: Manages annotations, saving, and loading
3. **UIController**: Manages UI elements and user interactions

Example structure:
```python
class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0
        # ... other video-related attributes

    def load_frame(self, frame_number):
        # Implementation of frame loading

    def next_frame(self):
        # Implementation of next frame

    def prev_frame(self):
        # Implementation of previous frame

class AnnotationManager:
    def __init__(self, annotation_path):
        self.annotation_path = annotation_path
        self.annotations = {}
        self.load_annotations()

    def load_annotations(self):
        # Implementation of annotation loading

    def save_annotations(self):
        # Implementation of annotation saving

    def annotate_frame(self, frame, label):
        # Implementation of frame annotation

class VideoApp:
    def __init__(self, master, video_path, annotation_path, controls_right=False):
        self.master = master
        self.video_player = VideoPlayer(video_path)
        self.annotation_manager = AnnotationManager(annotation_path)
        # ... UI setup
```

## 2. Improved Error Handling

### 2.1 Add Comprehensive Error Handling

Add try-except blocks for all file operations:

```python
def load_annotations(self):
    try:
        if os.path.exists(self.annotation_path):
            df = pd.read_csv(self.annotation_path)
            self.annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
            self.update_annotations_listbox()
            self.logger.info(f"Loaded annotations from {self.annotation_path}")
        else:
            self.logger.warning(f"No annotation file found at {self.annotation_path}")
    except Exception as e:
        self.logger.error(f"Error loading annotations: {str(e)}")
        self.show_error_message(f"Failed to load annotations: {str(e)}")
```

### 2.2 Implement a Logging System

Replace print statements with a proper logging system:

```python
import logging

def setup_logging():
    logger = logging.getLogger('video_annotator')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler('video_annotator.log')
    fh.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

class VideoApp:
    def __init__(self, master, video_path, annotation_path, controls_right=False):
        self.logger = setup_logging()
        # ... rest of initialization
```

### 2.3 User-Friendly Error Messages

Add methods to display error messages to the user:

```python
def show_error_message(self, message):
    tk.messagebox.showerror("Error", message)

def show_warning_message(self, message):
    tk.messagebox.showwarning("Warning", message)

def show_info_message(self, message):
    tk.messagebox.showinfo("Information", message)
```

## 3. Testing Framework

### 3.1 Unit Tests

Create unit tests for core functionality:

```python
# test_annotation_manager.py
import unittest
from unittest.mock import patch, mock_open
from annotation_manager import AnnotationManager

class TestAnnotationManager(unittest.TestCase):
    def setUp(self):
        self.annotation_manager = AnnotationManager('test.csv')
    
    def test_annotate_frame(self):
        self.annotation_manager.annotate_frame(1, 0)
        self.assertEqual(self.annotation_manager.annotations[1], 0)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.DataFrame.to_csv')
    def test_save_annotations(self, mock_to_csv, mock_file):
        self.annotation_manager.annotations = {1: 0, 2: 1}
        self.annotation_manager.save_annotations()
        mock_to_csv.assert_called_once()
```

### 3.2 Integration Tests

Create integration tests for the application workflow:

```python
# test_integration.py
import unittest
from unittest.mock import patch
import os
import tempfile
import pandas as pd
from video_app import VideoApp

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, 'test_video.mp4')
        self.annotation_path = os.path.join(self.temp_dir.name, 'test_annotation.csv')
        
        # Create a dummy video file
        with open(self.video_path, 'w') as f:
            f.write('dummy video content')
        
        # Create a dummy annotation file
        df = pd.DataFrame({'frame': [0, 1], 'label': [0, 1]})
        df.to_csv(self.annotation_path, index=False)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    @patch('tkinter.Tk')
    def test_load_and_save_annotations(self, mock_tk):
        app = VideoApp(mock_tk(), self.video_path, self.annotation_path)
        app.annotation_manager.annotate_frame(2, 2)
        app.annotation_manager.save_annotations()
        
        # Verify the annotation was saved
        df = pd.read_csv(self.annotation_path)
        self.assertEqual(df.loc[df['frame'] == 2, 'label'].iloc[0], 2)
```

## 4. Performance Optimization

### 4.1 Optimize Video Frame Loading

Implement frame caching to avoid reloading frames:

```python
class VideoPlayer:
    def __init__(self, video_path, cache_size=30):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0
        self.frame_cache = {}
        self.cache_size = cache_size
    
    def load_frame(self, frame_number):
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert frame to RGB
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add to cache
            if len(self.frame_cache) >= self.cache_size:
                # Remove oldest frame from cache
                oldest_frame = min(self.frame_cache.keys())
                del self.frame_cache[oldest_frame]
            
            self.frame_cache[frame_number] = frame
            return frame
        else:
            return None
```

### 4.2 Implement Progress Indicators

Add progress bars for long operations:

```python
def label_range(self):
    try:
        start_frame = int(self.start_frame_entry.get())
        end_frame = int(self.end_frame_entry.get())
        if start_frame < 0 or end_frame >= self.video_player.total_frames or start_frame > end_frame:
            raise ValueError("Invalid frame range")
        
        selected_label = self.selected_label.get()
        label_mapping = {
            "run": 1,
            "stop": 0,
            "turn": 2
        }
        
        label = label_mapping.get(selected_label.lower(), np.nan)
        
        # Create progress bar
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Labeling Progress")
        progress_label = tk.Label(progress_window, text="Labeling frames...")
        progress_label.pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        
        total_frames = end_frame - start_frame + 1
        progress_bar['maximum'] = total_frames
        
        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            self.annotation_manager.annotate_frame(frame, label, save=False)
            progress_bar['value'] = i + 1
            progress_window.update()
        
        # Save annotations once after labeling the entire range
        self.annotation_manager.save_annotations()
        self.update_annotations_listbox()
        
        progress_window.destroy()
        self.go_to_frame(end_frame)
        
    except ValueError as e:
        self.show_error_message(f"Error: {str(e)}")
```

## 5. Configuration and Customization

### 5.1 Make Labels Configurable

Create a configuration system for customizable labels:

```python
import json
import os

class Config:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.default_config = {
            'labels': [
                {'name': 'Stop', 'key': 's', 'value': 0, 'color': 'red'},
                {'name': 'Run', 'key': 'r', 'value': 1, 'color': 'green'},
                {'name': 'Turn', 'key': 't', 'value': 2, 'color': 'blue'}
            ],
            'default_fps': 30,
            'ui': {
                'controls_right': False,
                'window_width': 1200
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {str(e)}")
                return self.default_config
        else:
            self.save_config(self.default_config)
            return self.default_config
    
    def save_config(self, config):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {str(e)}")
    
    def get_labels(self):
        return self.config.get('labels', self.default_config['labels'])
    
    def get_default_fps(self):
        return self.config.get('default_fps', self.default_config['default_fps'])
    
    def get_ui_config(self):
        return self.config.get('ui', self.default_config['ui'])
```

Then use this configuration in the VideoApp:

```python
class VideoApp:
    def __init__(self, master, video_path, annotation_path, config_path='config.json'):
        self.config = Config(config_path)
        self.labels = self.config.get_labels()
        # ... rest of initialization
        
        # Create label buttons dynamically based on config
        for label in self.labels:
            button = tk.Button(self.controls_frame, text=label['name'], 
                              command=lambda val=label['value']: self.annotate_frame(val))
            # ... button placement
            
            # Bind keys dynamically
            self.master.bind(label['key'], lambda event, val=label['value']: self.annotate_frame(val))
```

## 6. Documentation

### 6.1 Comprehensive README

Update the README with detailed usage instructions, examples, and configuration options.

### 6.2 Code Documentation

Add docstrings to all classes and methods:

```python
class VideoPlayer:
    """
    Handles video loading, playback, and frame navigation.
    
    Attributes:
        video_path (str): Path to the video file
        cap (cv2.VideoCapture): OpenCV video capture object
        total_frames (int): Total number of frames in the video
        frame_number (int): Current frame number
        frame_cache (dict): Cache of loaded frames
        cache_size (int): Maximum number of frames to cache
    """
    
    def __init__(self, video_path, cache_size=30):
        """
        Initialize the VideoPlayer.
        
        Args:
            video_path (str): Path to the video file
            cache_size (int, optional): Maximum number of frames to cache. Defaults to 30.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0
        self.frame_cache = {}
        self.cache_size = cache_size
```

## Implementation Timeline

### Phase 1 (Week 1-2)
- Remove global variables
- Implement error handling
- Add logging system
- Update documentation

### Phase 2 (Week 3-4)
- Modularize the code
- Implement unit tests
- Add progress indicators
- Optimize frame loading

### Phase 3 (Week 5-6)
- Make labels configurable
- Add integration tests
- Implement additional features
- Final documentation updates
