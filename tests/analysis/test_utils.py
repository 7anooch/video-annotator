#!/usr/bin/env python3
"""
Unit tests for the utils module.
"""

import os
import unittest
import tempfile
from src.analysis.utils import (
    interpolate_annotations,
    smooth_annotations,
    calculate_agreement,
    calculate_cohen_kappa,
    frames_to_time,
    time_to_frames
)

class TestUtils(unittest.TestCase):
    """Test cases for the utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample annotation dictionary with gaps
        self.annotations_with_gaps = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'walking'},
            # Gap
            5: {'label': 'running'},
            6: {'label': 'running'},
            # Gap
            10: {'label': 'jumping'},
            11: {'label': 'jumping'},
            # Gap
            15: {'label': 'walking'},
            16: {'label': 'walking'},
            17: {'label': 'walking'},
            18: {'label': 'walking'},
            19: {'label': 'walking'},
            20: {'label': 'standing'}
        }
        
        # Create another sample annotation dictionary
        self.annotations2 = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'walking'},
            5: {'label': 'running'},
            6: {'label': 'running'},
            10: {'label': 'jumping'},
            11: {'label': 'standing'},  # Different from annotations_with_gaps
            15: {'label': 'walking'},
            16: {'label': 'walking'},
            17: {'label': 'walking'},
            18: {'label': 'walking'},
            19: {'label': 'walking'},
            20: {'label': 'standing'}
        }
    
    def test_interpolate_annotations_nearest(self):
        """Test interpolate_annotations with nearest method."""
        interpolated = interpolate_annotations(self.annotations_with_gaps, method='nearest')
        
        # Check that the gaps are filled
        self.assertIn(4, interpolated)
        self.assertIn(7, interpolated)
        self.assertIn(8, interpolated)
        self.assertIn(9, interpolated)
        self.assertIn(12, interpolated)
        self.assertIn(13, interpolated)
        self.assertIn(14, interpolated)
        
        # Check that the interpolated values are correct
        self.assertEqual(interpolated[4]['label'], 'walking')  # Nearest to 3
        self.assertEqual(interpolated[7]['label'], 'running')  # Nearest to 6
        self.assertEqual(interpolated[8]['label'], 'jumping')  # Nearest to 10
        self.assertEqual(interpolated[9]['label'], 'jumping')  # Nearest to 10
        self.assertEqual(interpolated[12]['label'], 'jumping')  # Nearest to 11
        self.assertEqual(interpolated[13]['label'], 'walking')  # Nearest to 15
        self.assertEqual(interpolated[14]['label'], 'walking')  # Nearest to 15
    
    def test_interpolate_annotations_linear(self):
        """Test interpolate_annotations with linear method."""
        interpolated = interpolate_annotations(self.annotations_with_gaps, method='linear')
        
        # Check that the gaps are filled only when the labels on both sides are the same
        self.assertNotIn(4, interpolated)  # Different labels on both sides
        self.assertNotIn(7, interpolated)  # Different labels on both sides
        self.assertNotIn(8, interpolated)  # Different labels on both sides
        self.assertNotIn(9, interpolated)  # Different labels on both sides
        self.assertNotIn(12, interpolated)  # Different labels on both sides
        self.assertNotIn(13, interpolated)  # Different labels on both sides
        self.assertNotIn(14, interpolated)  # Different labels on both sides
    
    def test_interpolate_annotations_fill(self):
        """Test interpolate_annotations with fill method."""
        interpolated = interpolate_annotations(self.annotations_with_gaps, method='fill')
        
        # Check that the gaps are filled with the label from the start of the gap
        self.assertEqual(interpolated[4]['label'], 'walking')
        self.assertEqual(interpolated[7]['label'], 'running')
        self.assertEqual(interpolated[8]['label'], 'running')
        self.assertEqual(interpolated[9]['label'], 'running')
        self.assertEqual(interpolated[12]['label'], 'jumping')
        self.assertEqual(interpolated[13]['label'], 'jumping')
        self.assertEqual(interpolated[14]['label'], 'jumping')
    
    def test_smooth_annotations(self):
        """Test smooth_annotations."""
        # Create a sample annotation dictionary with noise
        annotations_with_noise = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'running'},  # Noise
            4: {'label': 'walking'},
            5: {'label': 'walking'},
            6: {'label': 'walking'},
            7: {'label': 'walking'},
            8: {'label': 'jumping'},  # Noise
            9: {'label': 'walking'},
            10: {'label': 'walking'}
        }
        
        # Smooth with window size 3
        smoothed = smooth_annotations(annotations_with_noise, window_size=3)
        
        # Check that the noise is smoothed out
        self.assertEqual(smoothed[3]['label'], 'walking')
        self.assertEqual(smoothed[8]['label'], 'walking')
    
    def test_calculate_agreement(self):
        """Test calculate_agreement."""
        agreement = calculate_agreement(self.annotations_with_gaps, self.annotations2)
        
        # Calculate expected agreement
        common_frames = set(self.annotations_with_gaps.keys()).intersection(set(self.annotations2.keys()))
        agreements = sum(1 for frame in common_frames
                        if self.annotations_with_gaps[frame].get('label') == self.annotations2[frame].get('label'))
        expected_agreement = agreements / len(common_frames)
        
        # Check that the agreement is correct
        self.assertAlmostEqual(agreement, expected_agreement)
    
    def test_calculate_cohen_kappa(self):
        """Test calculate_cohen_kappa."""
        try:
            from sklearn.metrics import cohen_kappa_score
            
            kappa = calculate_cohen_kappa(self.annotations_with_gaps, self.annotations2)
            
            # Check that the kappa is within a reasonable range
            self.assertGreaterEqual(kappa, -1.0)
            self.assertLessEqual(kappa, 1.0)
        except ImportError:
            self.skipTest("scikit-learn is not available")
    
    def test_frames_to_time(self):
        """Test frames_to_time."""
        # Test with various frame numbers and FPS values
        self.assertEqual(frames_to_time(0, 30.0), "00:00:00.000")
        self.assertEqual(frames_to_time(30, 30.0), "00:00:01.000")
        self.assertEqual(frames_to_time(90, 30.0), "00:00:03.000")
        self.assertEqual(frames_to_time(1800, 30.0), "00:01:00.000")
        self.assertEqual(frames_to_time(108000, 30.0), "01:00:00.000")
        
        # Test with non-integer FPS
        self.assertEqual(frames_to_time(25, 25.0), "00:00:01.000")
        self.assertEqual(frames_to_time(75, 25.0), "00:00:03.000")
    
    def test_time_to_frames(self):
        """Test time_to_frames."""
        # Test with various time strings and FPS values
        self.assertEqual(time_to_frames("00:00:00.000", 30.0), 0)
        self.assertEqual(time_to_frames("00:00:01.000", 30.0), 30)
        self.assertEqual(time_to_frames("00:00:03.000", 30.0), 90)
        self.assertEqual(time_to_frames("00:01:00.000", 30.0), 1800)
        self.assertEqual(time_to_frames("01:00:00.000", 30.0), 108000)
        
        # Test with non-integer FPS
        self.assertEqual(time_to_frames("00:00:01.000", 25.0), 25)
        self.assertEqual(time_to_frames("00:00:03.000", 25.0), 75)
        
        # Test with different time formats
        self.assertEqual(time_to_frames("00:00:01", 30.0), 30)
        self.assertEqual(time_to_frames("00:01", 30.0), 1800)
        self.assertEqual(time_to_frames("1", 30.0), 30)

if __name__ == '__main__':
    unittest.main()
