#!/usr/bin/env python3
"""
Unit tests for the data model module.
"""

import os
import unittest
import tempfile
import pandas as pd
from src.analysis.data_model import AnnotationData

class TestAnnotationData(unittest.TestCase):
    """Test cases for the AnnotationData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample annotation dictionary
        self.annotations = {
            1: {'label': 'walking'},
            5: {'label': 'running'},
            10: {'label': 'jumping'},
            15: {'label': 'walking'},
            20: {'label': 'standing'}
        }
        
        # Create a sample metadata dictionary
        self.metadata = {
            'source': 'test',
            'format': 'csv',
            'total_annotations': 5
        }
        
        # Create an AnnotationData object
        self.data = AnnotationData(self.annotations, self.metadata)
    
    def test_init(self):
        """Test initialization."""
        # Test with annotations and metadata
        data = AnnotationData(self.annotations, self.metadata)
        self.assertEqual(data.annotations, self.annotations)
        self.assertEqual(data.metadata, self.metadata)
        
        # Test with default values
        data = AnnotationData()
        self.assertEqual(data.annotations, {})
        self.assertEqual(data.metadata, {})
    
    def test_get_frames(self):
        """Test get_frames method."""
        frames = self.data.get_frames()
        self.assertEqual(frames, [1, 5, 10, 15, 20])
    
    def test_get_labels(self):
        """Test get_labels method."""
        labels = self.data.get_labels()
        self.assertEqual(labels, ['jumping', 'running', 'standing', 'walking'])
    
    def test_get_annotation(self):
        """Test get_annotation method."""
        # Test with existing frame
        annotation = self.data.get_annotation(10)
        self.assertEqual(annotation, {'label': 'jumping'})
        
        # Test with non-existing frame
        annotation = self.data.get_annotation(100)
        self.assertIsNone(annotation)
    
    def test_set_annotation(self):
        """Test set_annotation method."""
        # Test with new frame
        self.data.set_annotation(25, {'label': 'sitting'})
        self.assertEqual(self.data.get_annotation(25), {'label': 'sitting'})
        
        # Test with existing frame
        self.data.set_annotation(10, {'label': 'crawling'})
        self.assertEqual(self.data.get_annotation(10), {'label': 'crawling'})
    
    def test_delete_annotation(self):
        """Test delete_annotation method."""
        # Test with existing frame
        result = self.data.delete_annotation(10)
        self.assertTrue(result)
        self.assertIsNone(self.data.get_annotation(10))
        
        # Test with non-existing frame
        result = self.data.delete_annotation(100)
        self.assertFalse(result)
    
    def test_clear_annotations(self):
        """Test clear_annotations method."""
        self.data.clear_annotations()
        self.assertEqual(self.data.annotations, {})
    
    def test_get_annotation_count(self):
        """Test get_annotation_count method."""
        count = self.data.get_annotation_count()
        self.assertEqual(count, 5)
    
    def test_get_label_counts(self):
        """Test get_label_counts method."""
        counts = self.data.get_label_counts()
        self.assertEqual(counts, {'walking': 2, 'running': 1, 'jumping': 1, 'standing': 1})
    
    def test_filter_by_label(self):
        """Test filter_by_label method."""
        # Test with existing label
        filtered = self.data.filter_by_label('walking')
        self.assertEqual(filtered.get_annotation_count(), 2)
        self.assertEqual(filtered.get_frames(), [1, 15])
        
        # Test with non-existing label
        filtered = self.data.filter_by_label('swimming')
        self.assertEqual(filtered.get_annotation_count(), 0)
    
    def test_filter_by_frames(self):
        """Test filter_by_frames method."""
        # Test with valid range
        filtered = self.data.filter_by_frames(5, 15)
        self.assertEqual(filtered.get_annotation_count(), 3)
        self.assertEqual(filtered.get_frames(), [5, 10, 15])
        
        # Test with empty range
        filtered = self.data.filter_by_frames(100, 200)
        self.assertEqual(filtered.get_annotation_count(), 0)
    
    def test_merge(self):
        """Test merge method."""
        # Create another AnnotationData object
        other_annotations = {
            25: {'label': 'sitting'},
            30: {'label': 'lying'},
            10: {'label': 'crawling'}  # Overlapping frame
        }
        other_data = AnnotationData(other_annotations)
        
        # Test merge without overwrite
        merged = self.data.merge(other_data)
        self.assertEqual(merged.get_annotation_count(), 7)
        self.assertEqual(merged.get_annotation(10), {'label': 'jumping'})  # Original value preserved
        
        # Test merge with overwrite
        merged = self.data.merge(other_data, overwrite=True)
        self.assertEqual(merged.get_annotation_count(), 7)
        self.assertEqual(merged.get_annotation(10), {'label': 'crawling'})  # Value overwritten
    
    def test_to_dataframe(self):
        """Test to_dataframe method."""
        df = self.data.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn('frame', df.columns)
        self.assertIn('label', df.columns)
    
    def test_from_dataframe(self):
        """Test from_dataframe method."""
        # Create a sample DataFrame
        df = pd.DataFrame({
            'frame': [1, 5, 10],
            'label': ['walking', 'running', 'jumping']
        })
        
        # Test with valid DataFrame
        data = AnnotationData()
        result = data.from_dataframe(df)
        self.assertTrue(result)
        self.assertEqual(data.get_annotation_count(), 3)
        self.assertEqual(data.get_annotation(1), {'label': 'walking'})
        
        # Test with invalid DataFrame (missing 'frame' column)
        df = pd.DataFrame({
            'timestamp': [1, 5, 10],
            'label': ['walking', 'running', 'jumping']
        })
        data = AnnotationData()
        result = data.from_dataframe(df)
        self.assertFalse(result)
    
    def test_to_frame_label_lists(self):
        """Test to_frame_label_lists method."""
        frames, labels = self.data.to_frame_label_lists()
        self.assertEqual(frames, [1, 5, 10, 15, 20])
        self.assertEqual(labels, ['walking', 'running', 'jumping', 'walking', 'standing'])
    
    def test_from_frame_label_lists(self):
        """Test from_frame_label_lists method."""
        # Test with valid lists
        frames = [1, 5, 10]
        labels = ['walking', 'running', 'jumping']
        data = AnnotationData()
        result = data.from_frame_label_lists(frames, labels)
        self.assertTrue(result)
        self.assertEqual(data.get_annotation_count(), 3)
        self.assertEqual(data.get_annotation(1), {'label': 'walking'})
        
        # Test with lists of different lengths
        frames = [1, 5, 10]
        labels = ['walking', 'running']
        data = AnnotationData()
        result = data.from_frame_label_lists(frames, labels)
        self.assertFalse(result)
    
    def test_to_visualization_format(self):
        """Test to_visualization_format method."""
        viz_format = self.data.to_visualization_format()
        self.assertIn('frames', viz_format)
        self.assertIn('labels', viz_format)
        self.assertEqual(viz_format['frames'], [1, 5, 10, 15, 20])
        self.assertEqual(viz_format['labels'], ['walking', 'running', 'jumping', 'walking', 'standing'])
    
    def test_from_visualization_format(self):
        """Test from_visualization_format method."""
        # Test with valid format
        viz_format = {
            'frames': [1, 5, 10],
            'labels': ['walking', 'running', 'jumping']
        }
        data = AnnotationData()
        result = data.from_visualization_format(viz_format)
        self.assertTrue(result)
        self.assertEqual(data.get_annotation_count(), 3)
        self.assertEqual(data.get_annotation(1), {'label': 'walking'})
        
        # Test with invalid format (missing 'frames' key)
        viz_format = {
            'timestamps': [1, 5, 10],
            'labels': ['walking', 'running', 'jumping']
        }
        data = AnnotationData()
        result = data.from_visualization_format(viz_format)
        self.assertFalse(result)
    
    def test_to_analyze_format(self):
        """Test to_analyze_format method."""
        frames, labels = self.data.to_analyze_format()
        self.assertEqual(frames, [1, 5, 10, 15, 20])
        self.assertEqual(labels, ['walking', 'running', 'jumping', 'walking', 'standing'])
    
    def test_from_analyze_format(self):
        """Test from_analyze_format method."""
        # Test with valid format
        frames = [1, 5, 10]
        labels = ['walking', 'running', 'jumping']
        data = AnnotationData()
        result = data.from_analyze_format(frames, labels)
        self.assertTrue(result)
        self.assertEqual(data.get_annotation_count(), 3)
        self.assertEqual(data.get_annotation(1), {'label': 'walking'})
        
        # Test with lists of different lengths
        frames = [1, 5, 10]
        labels = ['walking', 'running']
        data = AnnotationData()
        result = data.from_analyze_format(frames, labels)
        self.assertFalse(result)
    
    def test_to_plot_format(self):
        """Test to_plot_format method."""
        frames, labels = self.data.to_plot_format()
        self.assertEqual(frames, [1, 5, 10, 15, 20])
        self.assertEqual(labels, ['walking', 'running', 'jumping', 'walking', 'standing'])
    
    def test_from_plot_format(self):
        """Test from_plot_format method."""
        # Test with valid format
        frames = [1, 5, 10]
        labels = ['walking', 'running', 'jumping']
        data = AnnotationData()
        result = data.from_plot_format(frames, labels)
        self.assertTrue(result)
        self.assertEqual(data.get_annotation_count(), 3)
        self.assertEqual(data.get_annotation(1), {'label': 'walking'})
        
        # Test with lists of different lengths
        frames = [1, 5, 10]
        labels = ['walking', 'running']
        data = AnnotationData()
        result = data.from_plot_format(frames, labels)
        self.assertFalse(result)
    
    def test_load_save_file(self):
        """Test load_from_file and save_to_file methods."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b'frame,label\n1,walking\n5,running\n10,jumping\n')
            temp_csv = f.name
        
        try:
            # Test load_from_file
            data = AnnotationData()
            result = data.load_from_file(temp_csv)
            self.assertTrue(result)
            self.assertEqual(data.get_annotation_count(), 3)
            self.assertEqual(data.get_annotation(1), {'label': 'walking'})
            
            # Test save_to_file
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                temp_csv_out = f.name
            
            result = data.save_to_file(temp_csv_out)
            self.assertTrue(result)
            
            # Verify the saved file
            data2 = AnnotationData()
            result = data2.load_from_file(temp_csv_out)
            self.assertTrue(result)
            self.assertEqual(data2.get_annotation_count(), 3)
            self.assertEqual(data2.get_annotation(1), {'label': 'walking'})
        finally:
            # Clean up temporary files
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            if os.path.exists(temp_csv_out):
                os.remove(temp_csv_out)

if __name__ == '__main__':
    unittest.main()
