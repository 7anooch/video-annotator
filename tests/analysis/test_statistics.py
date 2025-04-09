#!/usr/bin/env python3
"""
Unit tests for the statistics module.
"""

import unittest
import numpy as np
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

class TestStatisticalAnalysis(unittest.TestCase):
    """Test cases for the StatisticalAnalysis class."""
    
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
        
        # Create AnnotationData objects
        self.data1 = AnnotationData(self.annotations1)
        self.data2 = AnnotationData(self.annotations2)
        
        # Create a StatisticalAnalysis object
        self.stats = StatisticalAnalysis()
    
    def test_basic_statistics(self):
        """Test basic_statistics method."""
        stats = self.stats.basic_statistics(self.data1)
        
        # Check that the stats dictionary contains the expected keys
        self.assertIn('total_annotations', stats)
        self.assertIn('unique_labels', stats)
        self.assertIn('label_counts', stats)
        self.assertIn('frame_min', stats)
        self.assertIn('frame_max', stats)
        self.assertIn('frame_range', stats)
        self.assertIn('total_gaps', stats)
        self.assertIn('total_gap_frames', stats)
        self.assertIn('avg_gap_size', stats)
        self.assertIn('max_gap_size', stats)
        
        # Check the values
        self.assertEqual(stats['total_annotations'], 14)
        self.assertEqual(stats['unique_labels'], 4)
        self.assertEqual(stats['label_counts'], {'walking': 6, 'running': 3, 'jumping': 2, 'standing': 3})
        self.assertEqual(stats['frame_min'], 1)
        self.assertEqual(stats['frame_max'], 22)
        self.assertEqual(stats['frame_range'], 21)
        self.assertEqual(stats['total_gaps'], 3)
        self.assertEqual(stats['total_gap_frames'], 6)
        self.assertAlmostEqual(stats['avg_gap_size'], 2.0)
        self.assertEqual(stats['max_gap_size'], 3)
    
    def test_label_transitions(self):
        """Test label_transitions method."""
        transitions = self.stats.label_transitions(self.data1)
        
        # Check the transitions
        self.assertEqual(transitions[('walking', 'running')], 1)
        self.assertEqual(transitions[('running', 'jumping')], 1)
        self.assertEqual(transitions[('jumping', 'walking')], 1)
        self.assertEqual(transitions[('walking', 'standing')], 1)
    
    def test_label_durations(self):
        """Test label_durations method."""
        durations = self.stats.label_durations(self.data1)
        
        # Check the durations
        self.assertEqual(durations['walking'], [4, 2])
        self.assertEqual(durations['running'], [3])
        self.assertEqual(durations['jumping'], [2])
        self.assertEqual(durations['standing'], [3])
    
    def test_duration_statistics(self):
        """Test duration_statistics method."""
        duration_stats = self.stats.duration_statistics(self.data1)
        
        # Check that the stats dictionary contains the expected keys
        for label in ['walking', 'running', 'jumping', 'standing']:
            self.assertIn(label, duration_stats)
            self.assertIn('count', duration_stats[label])
            self.assertIn('total_frames', duration_stats[label])
            self.assertIn('min', duration_stats[label])
            self.assertIn('max', duration_stats[label])
            self.assertIn('mean', duration_stats[label])
            self.assertIn('median', duration_stats[label])
            self.assertIn('std', duration_stats[label])
        
        # Check the values for 'walking'
        self.assertEqual(duration_stats['walking']['count'], 2)
        self.assertEqual(duration_stats['walking']['total_frames'], 6)
        self.assertEqual(duration_stats['walking']['min'], 2)
        self.assertEqual(duration_stats['walking']['max'], 4)
        self.assertAlmostEqual(duration_stats['walking']['mean'], 3.0)
        self.assertAlmostEqual(duration_stats['walking']['median'], 3.0)
        self.assertAlmostEqual(duration_stats['walking']['std'], 1.0)
    
    def test_compare_annotations(self):
        """Test compare_annotations method."""
        comparison = self.stats.compare_annotations(self.data1, self.data2)
        
        # Check that the comparison dictionary contains the expected keys
        self.assertIn('total_annotations_1', comparison)
        self.assertIn('total_annotations_2', comparison)
        self.assertIn('common_frames', comparison)
        self.assertIn('agreements', comparison)
        self.assertIn('disagreements', comparison)
        self.assertIn('agreement_rate', comparison)
        self.assertIn('unique_to_1', comparison)
        self.assertIn('unique_to_2', comparison)
        
        # Check the values
        self.assertEqual(comparison['total_annotations_1'], 14)
        self.assertEqual(comparison['total_annotations_2'], 14)
        self.assertEqual(comparison['common_frames'], 14)
        self.assertEqual(comparison['agreements'], 12)
        self.assertEqual(comparison['disagreements'], 2)
        self.assertAlmostEqual(comparison['agreement_rate'], 12/14)
        self.assertEqual(comparison['unique_to_1'], 0)
        self.assertEqual(comparison['unique_to_2'], 0)
    
    def test_correlation_analysis(self):
        """Test correlation_analysis method."""
        correlation = self.stats.correlation_analysis(self.data1, self.data2)
        
        # Check that the correlation dictionary contains the expected keys
        self.assertIn('pearson_correlation', correlation)
        self.assertIn('spearman_correlation', correlation)
        self.assertIn('cohen_kappa', correlation)
        
        # Check that the values are within a reasonable range
        self.assertGreaterEqual(correlation['pearson_correlation'], -1.0)
        self.assertLessEqual(correlation['pearson_correlation'], 1.0)
        self.assertGreaterEqual(correlation['spearman_correlation'], -1.0)
        self.assertLessEqual(correlation['spearman_correlation'], 1.0)
        self.assertGreaterEqual(correlation['cohen_kappa'], -1.0)
        self.assertLessEqual(correlation['cohen_kappa'], 1.0)
    
    def test_advanced_sequence_analysis(self):
        """Test advanced_sequence_analysis method."""
        sequence_analysis = self.stats.advanced_sequence_analysis(self.data1)
        
        # Check that the sequence_analysis dictionary contains the expected keys
        self.assertIn('basic_analysis', sequence_analysis)
        self.assertIn('transition_probabilities', sequence_analysis)
        self.assertIn('ngrams', sequence_analysis)
        self.assertIn('complexity', sequence_analysis)
        self.assertIn('recurring_patterns', sequence_analysis)
        
        # Check the transition probabilities
        transition_probs = sequence_analysis['transition_probabilities']
        self.assertGreaterEqual(transition_probs[('walking', 'running')], 0.0)
        self.assertLessEqual(transition_probs[('walking', 'running')], 1.0)
        
        # Check the n-grams
        ngrams = sequence_analysis['ngrams']
        self.assertIn(2, ngrams)  # 2-grams
        
        # Check the complexity measures
        complexity = sequence_analysis['complexity']
        self.assertIn('label_entropy', complexity)
        self.assertIn('bigram_entropy', complexity)
        self.assertIn('conditional_entropy', complexity)
        
        # Check the recurring patterns
        patterns = sequence_analysis['recurring_patterns']
        self.assertIsInstance(patterns, list)
    
    def test_time_series_analysis(self):
        """Test time_series_analysis method."""
        time_series = self.stats.time_series_analysis(self.data1)
        
        # Check that the time_series dictionary contains the expected keys
        self.assertIn('windows', time_series)
        self.assertIn('frequencies', time_series)
        self.assertIn('trends', time_series)
        self.assertIn('autocorrelations', time_series)
        
        # Check the windows
        windows = time_series['windows']
        self.assertIsInstance(windows, list)
        
        # Check the frequencies
        frequencies = time_series['frequencies']
        for label in ['walking', 'running', 'jumping', 'standing']:
            self.assertIn(label, frequencies)
            self.assertIsInstance(frequencies[label], list)
    
    def test_detect_anomalies(self):
        """Test detect_anomalies method."""
        # Test with Z-score method
        anomalies = self.stats.detect_anomalies(self.data1, method='zscore', threshold=3.0)
        self.assertIsInstance(anomalies, dict)
        
        # Test with IQR method
        anomalies = self.stats.detect_anomalies(self.data1, method='iqr', threshold=1.5)
        self.assertIsInstance(anomalies, dict)
        
        # Test with isolation_forest method (if scikit-learn is available)
        try:
            from sklearn.ensemble import IsolationForest
            anomalies = self.stats.detect_anomalies(self.data1, method='isolation_forest')
            self.assertIsInstance(anomalies, dict)
        except ImportError:
            pass  # Skip this test if scikit-learn is not available

if __name__ == '__main__':
    unittest.main()
