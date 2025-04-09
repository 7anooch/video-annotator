#!/usr/bin/env python3
"""
Unit tests for the adapters module.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter

class TestAnalyzeAdapter(unittest.TestCase):
    """Test cases for the AnalyzeAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample annotation dictionary
        self.annotations1 = {
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
        
        # Create another sample annotation dictionary
        self.annotations2 = {
            1: {'label': 'walking'},
            2: {'label': 'walking'},
            3: {'label': 'walking'},
            4: {'label': 'running'},  # Different from annotations1
            5: {'label': 'running'},
            6: {'label': 'running'},
            7: {'label': 'jumping'},  # Different from annotations1
            10: {'label': 'jumping'},
            11: {'label': 'jumping'},
            15: {'label': 'walking'},
            16: {'label': 'walking'},
            20: {'label': 'standing'},
            21: {'label': 'standing'},
            22: {'label': 'standing'}
        }
        
        # Create an AnalyzeAdapter object
        self.analyze_adapter = AnalyzeAdapter()
    
    def test_analyze_sequence(self):
        """Test analyze_sequence method."""
        sequence_analysis = self.analyze_adapter.analyze_sequence(self.annotations1)
        
        # Check that the sequence_analysis dictionary contains the expected keys
        self.assertIn('sequences', sequence_analysis)
        self.assertIn('counts', sequence_analysis)
        
        # Check that the sequences and counts are non-empty
        self.assertGreater(len(sequence_analysis['sequences']), 0)
        self.assertGreater(len(sequence_analysis['counts']), 0)
    
    def test_calculate_precision_recall(self):
        """Test calculate_precision_recall method."""
        precision_recall = self.analyze_adapter.calculate_precision_recall(self.annotations1, self.annotations2)
        
        # Check that the precision_recall dictionary contains the expected keys
        self.assertIn('precision', precision_recall)
        self.assertIn('recall', precision_recall)
        self.assertIn('f1_score', precision_recall)
        self.assertIn('confusion_matrix', precision_recall)
        
        # Check that the values are within a reasonable range
        self.assertGreaterEqual(precision_recall['precision'], 0.0)
        self.assertLessEqual(precision_recall['precision'], 1.0)
        self.assertGreaterEqual(precision_recall['recall'], 0.0)
        self.assertLessEqual(precision_recall['recall'], 1.0)
        self.assertGreaterEqual(precision_recall['f1_score'], 0.0)
        self.assertLessEqual(precision_recall['f1_score'], 1.0)
    
    def test_calculate_confusion_matrix(self):
        """Test calculate_confusion_matrix method."""
        confusion_matrix = self.analyze_adapter.calculate_confusion_matrix(self.annotations1, self.annotations2)
        
        # Check that the confusion matrix is a numpy array
        self.assertIsInstance(confusion_matrix, np.ndarray)
        
        # Check that the confusion matrix is non-empty
        self.assertGreater(confusion_matrix.size, 0)
    
    def test_align_sequences(self):
        """Test align_sequences method."""
        sequence1 = ['walking', 'walking', 'running', 'jumping']
        sequence2 = ['walking', 'running', 'jumping', 'standing']
        
        alignment = self.analyze_adapter.align_sequences(sequence1, sequence2)
        
        # Check that the alignment dictionary contains the expected keys
        self.assertIn('alignment', alignment)
        self.assertIn('score', alignment)
        
        # Check that the alignment is a list of tuples
        self.assertIsInstance(alignment['alignment'], list)
        self.assertGreater(len(alignment['alignment']), 0)
        self.assertIsInstance(alignment['alignment'][0], tuple)

class TestPlotAdapter(unittest.TestCase):
    """Test cases for the PlotAdapter class."""
    
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
        
        # Create a PlotAdapter object
        self.plot_adapter = PlotAdapter()
    
    def test_plot_ethogram(self):
        """Test plot_ethogram method."""
        fig = self.plot_adapter.plot_ethogram(self.annotations)
        
        # Check that the figure is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to free up memory
        plt.close(fig)
    
    def test_get_color_map(self):
        """Test get_color_map method."""
        labels = ['walking', 'running', 'jumping', 'standing']
        
        color_map = self.plot_adapter.get_color_map(labels)
        
        # Check that the color map contains all the labels
        for label in labels:
            self.assertIn(label, color_map)
            
            # Check that the color is a string
            self.assertIsInstance(color_map[label], str)

if __name__ == '__main__':
    unittest.main()
