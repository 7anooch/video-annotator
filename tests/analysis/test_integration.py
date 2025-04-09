#!/usr/bin/env python3
"""
Integration tests for the enhanced analysis tools.

This module tests the integration between the enhanced analysis tools and the existing tools.
"""

import os
import unittest
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter
from src.analysis.adapters.visualization_adapter import VisualizationAdapter
from src.tools import analyze, plot, visualization

class TestIntegration(unittest.TestCase):
    """Test cases for the integration between enhanced and existing tools."""
    
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
        
        # Create the enhanced analysis objects
        self.stats = StatisticalAnalysis()
        self.viz = VisualizationManager()
        
        # Create the adapter objects
        self.analyze_adapter = AnalyzeAdapter()
        self.plot_adapter = PlotAdapter()
        self.visualization_adapter = VisualizationAdapter()
    
    def test_data_model_conversion(self):
        """Test conversion between enhanced data model and existing formats."""
        # Convert to the format used by the existing analysis tools
        frames, labels = self.data.to_analyze_format()
        
        # Check that the conversion is correct
        self.assertEqual(len(frames), len(self.annotations))
        self.assertEqual(len(labels), len(self.annotations))
        
        # Convert back to the enhanced data model
        data2 = AnnotationData()
        result = data2.from_analyze_format(frames, labels)
        
        # Check that the conversion is correct
        self.assertTrue(result)
        self.assertEqual(data2.get_annotation_count(), len(self.annotations))
        
        # Check that the annotations are the same
        for frame in self.annotations:
            self.assertIn(frame, data2.annotations)
            self.assertEqual(data2.annotations[frame]['label'], self.annotations[frame]['label'])
    
    def test_analyze_adapter(self):
        """Test the analyze adapter."""
        # Use the analyze adapter to call the existing analyze.py functions
        sequence_analysis = self.analyze_adapter.analyze_sequence(self.annotations)
        
        # Check that the sequence analysis contains the expected keys
        self.assertIn('sequences', sequence_analysis)
        self.assertIn('counts', sequence_analysis)
        
        # Use the existing analyze.py functions directly
        frames, labels = self.data.to_analyze_format()
        sequences, counts = analyze.analyze_sequences(labels)
        
        # Check that the results are the same
        self.assertEqual(len(sequence_analysis['sequences']), len(sequences))
        self.assertEqual(len(sequence_analysis['counts']), len(counts))
    
    def test_plot_adapter(self):
        """Test the plot adapter."""
        # Use the plot adapter to call the existing plot.py functions
        fig = self.plot_adapter.plot_ethogram(self.annotations)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
        
        # Use the existing plot.py functions directly
        frames, labels = self.data.to_plot_format()
        fig, ax = plt.subplots()
        plot.plot_ethogram(ax, frames, labels)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_visualization_adapter(self):
        """Test the visualization adapter."""
        # Skip this test if tkinter is not available
        try:
            import tkinter as tk
        except ImportError:
            self.skipTest("tkinter is not available")
        
        # Create a root window
        root = tk.Tk()
        
        try:
            # Use the visualization adapter to create a visualization GUI
            gui = self.visualization_adapter.create_visualization_gui(root, self.annotations)
            
            # Check that the GUI is created
            self.assertIsNotNone(gui)
        finally:
            # Destroy the root window
            root.destroy()
    
    def test_enhanced_analysis_with_existing_format(self):
        """Test using enhanced analysis tools with existing data format."""
        # Convert to the format used by the existing analysis tools
        frames, labels = self.data.to_analyze_format()
        
        # Create a new AnnotationData object from the existing format
        data2 = AnnotationData()
        data2.from_analyze_format(frames, labels)
        
        # Use the enhanced analysis tools with the new data
        basic_stats = self.stats.basic_statistics(data2)
        
        # Check that the basic statistics are correct
        self.assertEqual(basic_stats['total_annotations'], len(self.annotations))
        self.assertEqual(basic_stats['unique_labels'], len(set(annotation['label'] for annotation in self.annotations.values())))
    
    def test_existing_analysis_with_enhanced_format(self):
        """Test using existing analysis tools with enhanced data format."""
        # Convert to the format used by the existing analysis tools
        frames, labels = self.data.to_analyze_format()
        
        # Use the existing analysis tools with the enhanced data
        sequences, counts = analyze.analyze_sequences(labels)
        
        # Check that the sequences and counts are non-empty
        self.assertGreater(len(sequences), 0)
        self.assertGreater(len(counts), 0)
    
    def test_combined_visualization(self):
        """Test creating a visualization that combines enhanced and existing tools."""
        # Convert to the format used by the existing analysis tools
        frames, labels = self.data.to_analyze_format()
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Use the existing plot.py to create an ethogram in the first subplot
        plot.plot_ethogram(ax1, frames, labels, title="Ethogram (Existing Tool)")
        
        # Use the enhanced visualization to create a timeline in the second subplot
        self.viz.timeline_plot(self.data, ax=ax2, title="Timeline (Enhanced Tool)")
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
