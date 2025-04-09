import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestAnnotator(unittest.TestCase):
    @patch('tkinter.Tk')
    @patch('cv2.VideoCapture')
    def test_video_app_initialization(self, mock_video_capture, mock_tk):
        # Import here to avoid importing before mocks are in place
        from annotator import VideoApp
        
        # Setup mocks
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.get.return_value = 100  # Total frames
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4') as video_file, \
             tempfile.NamedTemporaryFile(suffix='.csv') as csv_file:
            
            # Create the VideoApp
            app = VideoApp(mock_tk(), video_file.name, csv_file.name)
            
            # Check that the app was initialized correctly
            self.assertEqual(app.video_path, video_file.name)
            self.assertEqual(app.annotation_path, csv_file.name)
            self.assertEqual(app.total_frames, 100)
            self.assertEqual(app.frame_number, 0)
            self.assertEqual(app.annotations, {})
    
    @patch('tkinter.Tk')
    @patch('cv2.VideoCapture')
    def test_annotate_frame(self, mock_video_capture, mock_tk):
        # Import here to avoid importing before mocks are in place
        from annotator import VideoApp
        
        # Setup mocks
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.get.return_value = 100  # Total frames
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4') as video_file, \
             tempfile.NamedTemporaryFile(suffix='.csv') as csv_file:
            
            # Create the VideoApp with a mock for save_annotations
            with patch('annotator.save_annotations') as mock_save:
                app = VideoApp(mock_tk(), video_file.name, csv_file.name)
                
                # Test annotating a frame
                app.annotate_frame(1, frame=10, save=True)
                
                # Check that the annotation was added
                self.assertEqual(app.annotations[10], 1)
                
                # Check that save_annotations was called
                mock_save.assert_called_once()
    
    @patch('tkinter.Tk')
    @patch('cv2.VideoCapture')
    def test_annotate_frame_range(self, mock_video_capture, mock_tk):
        # Import here to avoid importing before mocks are in place
        from annotator import VideoApp
        
        # Setup mocks
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.get.return_value = 100  # Total frames
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4') as video_file, \
             tempfile.NamedTemporaryFile(suffix='.csv') as csv_file:
            
            # Create the VideoApp with a mock for save_annotations
            with patch('annotator.save_annotations') as mock_save:
                app = VideoApp(mock_tk(), video_file.name, csv_file.name)
                
                # Test annotating a range of frames
                app.annotate_frame_range(2, start_frame=5, end_frame=15, save=True)
                
                # Check that the annotations were added
                for frame in range(5, 16):
                    self.assertEqual(app.annotations[frame], 2)
                
                # Check that save_annotations was called
                mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
