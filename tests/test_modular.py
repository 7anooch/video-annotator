import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestModularImplementation(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_video_player(self, mock_video_capture):
        # Import here to avoid importing before mocks are in place
        from video_player import VideoPlayer
        
        # Setup mocks
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.get.return_value = 100  # Total frames
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4') as video_file:
            # Create the VideoPlayer
            player = VideoPlayer(video_file.name)
            
            # Check that the player was initialized correctly
            self.assertEqual(player.video_path, video_file.name)
            self.assertEqual(player.total_frames, 100)
            self.assertEqual(player.frame_number, 0)
            
            # Test navigation methods
            player.next_frame()
            self.assertEqual(player.frame_number, 1)
            
            player.prev_frame()
            self.assertEqual(player.frame_number, 0)
            
            player.go_to_frame(50)
            self.assertEqual(player.frame_number, 50)
    
    def test_annotation_manager(self):
        # Import here
        from annotation_manager import AnnotationManager
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv') as csv_file:
            # Create the AnnotationManager
            manager = AnnotationManager(csv_file.name)
            
            # Check that the manager was initialized correctly
            self.assertEqual(manager.annotation_path, csv_file.name)
            self.assertEqual(manager.annotations, {})
            
            # Test annotation methods
            manager.annotate_frame(10, 1)
            self.assertEqual(manager.annotations[10], 1)
            
            manager.annotate_frame_range(20, 30, 2)
            for frame in range(20, 31):
                self.assertEqual(manager.annotations[frame], 2)
            
            # Test getting annotations
            self.assertEqual(manager.get_annotation(10), 1)
            self.assertEqual(manager.get_annotation(20), 2)
            self.assertIsNone(manager.get_annotation(5))
            
            # Test clearing annotations
            manager.clear_annotation(10)
            self.assertIsNone(manager.get_annotation(10))
            
            manager.clear_all_annotations()
            self.assertEqual(manager.annotations, {})
    
    @patch('tkinter.Tk')
    @patch('video_player.VideoPlayer')
    @patch('annotation_manager.AnnotationManager')
    def test_ui_controller(self, mock_annotation_manager, mock_video_player, mock_tk):
        # Import here
        from ui_controller import UIController
        
        # Setup mocks
        mock_video_player.total_frames = 100
        mock_video_player.frame_number = 0
        mock_video_player.fps = 30
        mock_video_player.load_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mock_annotation_manager.get_annotation.return_value = None
        
        # Create the UIController
        ui = UIController(mock_tk(), mock_video_player, mock_annotation_manager)
        
        # Test UI methods
        ui.next_frame()
        mock_video_player.next_frame.assert_called_once()
        
        ui.prev_frame()
        mock_video_player.prev_frame.assert_called_once()
        
        ui.annotate_frame(1)
        mock_annotation_manager.annotate_frame.assert_called_once_with(0, 1)
        mock_annotation_manager.save_annotations.assert_called_once()
    
    @patch('tkinter.Tk')
    @patch('tkinter.filedialog.askopenfilename')
    @patch('video_player.VideoPlayer')
    @patch('annotation_manager.AnnotationManager')
    @patch('ui_controller.UIController')
    def test_main(self, mock_ui_controller, mock_annotation_manager, mock_video_player, 
                 mock_askopenfilename, mock_tk):
        # Import here
        import annotator_modular
        
        # Setup mocks
        mock_askopenfilename.return_value = 'test.mp4'
        mock_video_player.return_value = MagicMock()
        mock_annotation_manager.return_value = MagicMock()
        mock_ui_controller.return_value = MagicMock()
        
        # Test main function
        result = annotator_modular.main()
        
        # Check that the components were created
        self.assertIsNotNone(result)
        root, player, manager, ui = result
        self.assertEqual(root, mock_tk.return_value)
        self.assertEqual(player, mock_video_player.return_value)
        self.assertEqual(manager, mock_annotation_manager.return_value)
        self.assertEqual(ui, mock_ui_controller.return_value)

if __name__ == '__main__':
    unittest.main()
