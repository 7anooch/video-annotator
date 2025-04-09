import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestAnnotationUtils(unittest.TestCase):
    """Test cases for the annotation utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.test_dir, 'test_video.mp4')
        
        # Create an empty file for the test video
        with open(self.test_video_path, 'w') as f:
            f.write('')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary files
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_save_annotations(self):
        """Test saving annotations to a CSV file."""
        from src.utils.annotation.funcs import save_annotations
        
        # Create test annotations
        annotations = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            5: 2  # Note: frame 4 is missing
        }
        
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_csv_path = temp_file.name
        
        try:
            # Save annotations
            save_annotations(annotations, output_csv_path)
            
            # Check that the file was created
            self.assertTrue(os.path.exists(output_csv_path))
            
            # Load the saved annotations
            df = pd.read_csv(output_csv_path)
            
            # Check the content
            self.assertEqual(len(df), 6)  # 0-5 inclusive
            self.assertEqual(df['frame'].tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(df['label'].tolist()[0], 0)
            self.assertEqual(df['label'].tolist()[1], 0)
            self.assertEqual(df['label'].tolist()[2], 1)
            self.assertEqual(df['label'].tolist()[3], 1)
            self.assertTrue(np.isnan(df['label'].tolist()[4]))  # Frame 4 should be NaN
            self.assertEqual(df['label'].tolist()[5], 2)
        finally:
            # Clean up
            if os.path.exists(output_csv_path):
                os.remove(output_csv_path)
    
    def test_get_csv_file_path(self):
        """Test getting the CSV file path."""
        from src.utils.annotation.funcs import get_csv_file_path
        
        # Test with default output name
        csv_path = get_csv_file_path(self.test_video_path)
        expected_path = os.path.join(self.test_dir, 'test_video_annotation.csv')
        self.assertEqual(csv_path, expected_path)
        
        # Test with custom output name
        csv_path = get_csv_file_path(self.test_video_path, 'custom_name')
        expected_path = os.path.join(self.test_dir, 'custom_name.csv')
        self.assertEqual(csv_path, expected_path)
        
        # Test with custom output name that already has .csv extension
        csv_path = get_csv_file_path(self.test_video_path, 'custom_name.csv')
        expected_path = os.path.join(self.test_dir, 'custom_name.csv')
        self.assertEqual(csv_path, expected_path)
    
    def test_get_common_substring(self):
        """Test getting the common substring from a list of strings."""
        from src.utils.annotation.funcs import get_common_substring
        
        # Test with strings that have a common substring
        strs = ['test_video1_annotation.csv', 'test_video2_annotation.csv', 'test_video3_annotation.csv']
        common = get_common_substring(strs)
        self.assertEqual(common, 'test_video')
        
        # Test with strings that have no common substring
        strs = ['abc.csv', 'def.csv', 'ghi.csv']
        common = get_common_substring(strs)
        self.assertEqual(common, '')
        
        # Test with empty list
        strs = []
        common = get_common_substring(strs)
        self.assertEqual(common, '')
        
        # Test with single string
        strs = ['test_video.csv']
        common = get_common_substring(strs)
        self.assertEqual(common, 'test_video.csv')
    
    def test_format_frames_and_ranges(self):
        """Test formatting frames and ranges."""
        from src.utils.annotation.funcs import format_frames_and_ranges
        
        # Test with consecutive frames
        frames = [1, 2, 3, 4, 5]
        formatted = format_frames_and_ranges(frames)
        self.assertEqual(formatted, '1-5')
        
        # Test with non-consecutive frames
        frames = [1, 2, 3, 5, 6, 8]
        formatted = format_frames_and_ranges(frames)
        self.assertEqual(formatted, '1-3, 5-6, 8')
        
        # Test with single frame
        frames = [1]
        formatted = format_frames_and_ranges(frames)
        self.assertEqual(formatted, '1')
        
        # Test with empty list
        frames = []
        formatted = format_frames_and_ranges(frames)
        self.assertEqual(formatted, '')
        
        # Test with unsorted frames
        frames = [5, 1, 3, 2, 4]
        formatted = format_frames_and_ranges(frames)
        self.assertEqual(formatted, '1-5')

if __name__ == '__main__':
    unittest.main()
