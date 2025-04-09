#!/usr/bin/env python3
"""
Unit tests for the visualization module.
"""

import os
import unittest
import tempfile
import matplotlib.pyplot as plt
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager

class TestVisualizationManager(unittest.TestCase):
    """Test cases for the VisualizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample annotation dictionary
        self.annotations = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'walking'},
            4: {'label': 'walking'},
            5: {'label': 'running'},
            6: {'label': 'running'},
            7: {'label': 'running'},
            10: {'label': 'jumping'},
            11: {'label': 'jumping'},
            15: {'label': 'walking'},
            16: {'label': 'walking'},
            20: {'label': 'standing'},
            21: {'label': 'standing'},
            22: {'label': 'standing'}
        }
        
        # Create an AnnotationData object
        self.data = AnnotationData(self.annotations)
        
        # Create a VisualizationManager object
        self.viz = VisualizationManager()
    
    def test_timeline_plot(self):
        """Test timeline_plot method."""
        fig = self.viz.timeline_plot(self.data)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_label_distribution_plot(self):
        """Test label_distribution_plot method."""
        fig = self.viz.label_distribution_plot(self.data)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_duration_boxplot(self):
        """Test duration_boxplot method."""
        fig = self.viz.duration_boxplot(self.data)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_transition_heatmap(self):
        """Test transition_heatmap method."""
        fig = self.viz.transition_heatmap(self.data)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_time_series_plot(self):
        """Test time_series_plot method."""
        fig = self.viz.time_series_plot(self.data)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_comparison_plot(self):
        """Test comparison_plot method."""
        # Create another AnnotationData object
        annotations2 = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'walking'},
            4: {'label': 'running'},  # Different from annotations
            5: {'label': 'running'},
            6: {'label': 'running'},
            7: {'label': 'jumping'},  # Different from annotations
            10: {'label': 'jumping'},
            11: {'label': 'jumping'},
            15: {'label': 'walking'},
            16: {'label': 'walking'},
            20: {'label': 'standing'},
            21: {'label': 'standing'},
            22: {'label': 'standing'}
        }
        data2 = AnnotationData(annotations2)
        
        fig = self.viz.comparison_plot(self.data, data2)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_create_interactive_timeline(self):
        """Test create_interactive_timeline method."""
        try:
            import plotly.graph_objects as go
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_html = f.name
            
            try:
                # Create the interactive timeline
                output_path = self.viz.create_interactive_timeline(self.data, output_path=temp_html)
                
                # Check that the output path is the same as the input path
                self.assertEqual(output_path, temp_html)
                
                # Check that the file exists
                self.assertTrue(os.path.exists(temp_html))
                
                # Check that the file is not empty
                self.assertGreater(os.path.getsize(temp_html), 0)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_html):
                    os.remove(temp_html)
        except ImportError:
            self.skipTest("Plotly is not available")
    
    def test_create_3d_visualization(self):
        """Test create_3d_visualization method."""
        try:
            import plotly.graph_objects as go
            from sklearn.decomposition import PCA
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_html = f.name
            
            try:
                # Create the 3D visualization
                output_path = self.viz.create_3d_visualization(self.data, output_path=temp_html)
                
                # Check that the output path is the same as the input path
                self.assertEqual(output_path, temp_html)
                
                # Check that the file exists
                self.assertTrue(os.path.exists(temp_html))
                
                # Check that the file is not empty
                self.assertGreater(os.path.getsize(temp_html), 0)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_html):
                    os.remove(temp_html)
        except ImportError:
            self.skipTest("Plotly or scikit-learn is not available")
    
    def test_create_report(self):
        """Test create_report method."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the report
            output_files = self.viz.create_report(self.data, output_dir=temp_dir)
            
            # Check that the output files list is not empty
            self.assertGreater(len(output_files), 0)
            
            # Check that all the files exist
            for file_path in output_files:
                self.assertTrue(os.path.exists(file_path))
                
                # Check that the file is not empty
                self.assertGreater(os.path.getsize(file_path), 0)

if __name__ == '__main__':
    unittest.main()
