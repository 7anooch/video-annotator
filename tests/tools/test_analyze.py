#!/usr/bin/env python3
"""
Unit tests for the analyze.py module.
"""

import os
import unittest
import tempfile
import pandas as pd
try:
    from src.tools.analysis.analyze import load_annotations, analyze_sequence
except ImportError:
    from src.tools.analyze import load_annotations, analyze_sequence

class TestAnalyze(unittest.TestCase):
    """Test cases for the analyze.py module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create test CSV files
        self.create_test_files()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        for filename in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, filename))

        # Remove test directory
        os.rmdir(self.test_dir)

    def create_test_files(self):
        """Create test CSV files."""
        # Create a CSV file with 1 column
        df1 = pd.DataFrame({0: [1, 2, 3, 4, 5]})
        self.csv1_path = os.path.join(self.test_dir, 'test1.csv')
        df1.to_csv(self.csv1_path, header=False, index=False)

        # Create a CSV file with 2 columns
        df2 = pd.DataFrame({0: [1, 2, 3, 4, 5], 1: ['a', 'b', 'c', 'd', 'e']})
        self.csv2_path = os.path.join(self.test_dir, 'test2.csv')
        df2.to_csv(self.csv2_path, header=False, index=False)

        # Create a CSV file with 3 columns
        df3 = pd.DataFrame({0: [1, 2, 3, 4, 5], 1: ['a', 'b', 'c', 'd', 'e'], 2: ['x', 'y', 'z', 'w', 'v']})
        self.csv3_path = os.path.join(self.test_dir, 'test3.csv')
        df3.to_csv(self.csv3_path, header=False, index=False)

        # Create a CSV file with headers
        df4 = pd.DataFrame({'frame': [1, 2, 3, 4, 5], 'label': ['a', 'b', 'c', 'd', 'e']})
        self.csv4_path = os.path.join(self.test_dir, 'test4.csv')
        df4.to_csv(self.csv4_path, header=True, index=False)

    def test_load_annotations_nonexistent_file(self):
        """Test loading annotations from a nonexistent file."""
        annotations = load_annotations('nonexistent.csv', verbose=False)
        self.assertEqual(annotations, {})

    def test_load_annotations_1_column(self):
        """Test loading annotations from a CSV file with 1 column."""
        annotations = load_annotations(self.csv1_path, verbose=False)
        self.assertEqual(len(annotations), 5)
        self.assertEqual(annotations[1], 1)
        self.assertEqual(annotations[5], 5)

    def test_load_annotations_2_columns(self):
        """Test loading annotations from a CSV file with 2 columns."""
        annotations = load_annotations(self.csv2_path, verbose=False, col=1)
        self.assertEqual(len(annotations), 5)
        self.assertEqual(annotations[1], 'a')
        self.assertEqual(annotations[5], 'e')

    def test_load_annotations_3_columns(self):
        """Test loading annotations from a CSV file with 3 columns."""
        annotations = load_annotations(self.csv3_path, verbose=False, col=2)
        self.assertEqual(len(annotations), 5)
        self.assertEqual(annotations[1], 'x')
        self.assertEqual(annotations[5], 'v')

    def test_load_annotations_column_out_of_bounds(self):
        """Test loading annotations with a column index that is out of bounds."""
        annotations = load_annotations(self.csv2_path, verbose=False, col=5)
        self.assertEqual(len(annotations), 5)
        self.assertEqual(annotations[1], 'a')
        self.assertEqual(annotations[5], 'e')

    def test_load_annotations_with_header(self):
        """Test loading annotations from a CSV file with headers."""
        annotations = load_annotations(self.csv4_path, verbose=False, col=1)
        self.assertEqual(len(annotations), 5)
        self.assertEqual(annotations[1], 'a')
        self.assertEqual(annotations[5], 'e')

    def test_analyze_sequence_empty(self):
        """Test analyzing an empty sequence."""
        sequence = analyze_sequence({})
        self.assertEqual(sequence, [])

    def test_analyze_sequence_single_label(self):
        """Test analyzing a sequence with a single label."""
        annotations = {1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'a'}
        sequence = analyze_sequence(annotations)
        self.assertEqual(len(sequence), 1)
        self.assertEqual(sequence[0][0], 'a')  # Label
        self.assertEqual(sequence[0][1], 5)    # Duration

    def test_analyze_sequence_multiple_labels(self):
        """Test analyzing a sequence with multiple labels."""
        annotations = {1: 'a', 2: 'a', 3: 'b', 4: 'b', 5: 'c'}
        sequence = analyze_sequence(annotations)
        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence[0][0], 'a')  # First label
        self.assertEqual(sequence[0][1], 2)    # First duration
        self.assertEqual(sequence[1][0], 'b')  # Second label
        self.assertEqual(sequence[1][1], 2)    # Second duration
        self.assertEqual(sequence[2][0], 'c')  # Third label
        self.assertEqual(sequence[2][1], 1)    # Third duration

    def test_analyze_sequence_with_non_numeric_keys(self):
        """Test analyzing a sequence with non-numeric keys."""
        annotations = {1: 'a', 2: 'a', 3: 'b', 4: 'b', 5: 'c', 'frame': 'label'}
        sequence = analyze_sequence(annotations)
        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence[0][0], 'a')  # First label
        self.assertEqual(sequence[0][1], 2)    # First duration
        self.assertEqual(sequence[1][0], 'b')  # Second label
        self.assertEqual(sequence[1][1], 2)    # Second duration
        self.assertEqual(sequence[2][0], 'c')  # Third label
        self.assertEqual(sequence[2][1], 1)    # Third duration

    def test_analyze_sequence_with_tuple_values(self):
        """Test analyzing a sequence with tuple values."""
        annotations = {1: ('1.0', 265), 2: ('1.0', 104), 3: ('2.0', 12), 4: ('2.0', 10), 5: ('1.0', 162)}
        sequence = analyze_sequence(annotations)
        self.assertEqual(len(sequence), 3)
        self.assertEqual(sequence[0][0], 1)  # First label (converted to integer)
        self.assertEqual(sequence[0][1], 2)  # First duration
        self.assertEqual(sequence[1][0], 2)  # Second label (converted to integer)
        self.assertEqual(sequence[1][1], 2)  # Second duration
        self.assertEqual(sequence[2][0], 1)  # Third label (converted to integer)
        self.assertEqual(sequence[2][1], 1)  # Third duration

if __name__ == '__main__':
    unittest.main()
