import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

class TestEnhancedVisualizer(unittest.TestCase):
    """Test cases for the EnhancedVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid importing before mocks are in place
        try:
            from src.tools.analysis.visualization_enhanced import EnhancedVisualizer
        except ImportError:
            from src.tools.visualization_enhanced import EnhancedVisualizer
        self.visualizer = EnhancedVisualizer()

        # Create a temporary CSV file with test data
        self.test_data = pd.DataFrame({
            'frame': list(range(100)),
            'label': [0] * 30 + [1] * 40 + [2] * 30
        })

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test_annotation.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)

        # Disable matplotlib show to avoid displaying plots during tests
        plt.show = lambda: None

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary files
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_load_annotations(self):
        """Test loading annotations from a CSV file."""
        # Test loading valid annotations
        annotations = self.visualizer.load_annotations(self.test_csv_path)
        self.assertEqual(len(annotations), 100)
        self.assertEqual(annotations[0], 0)
        self.assertEqual(annotations[30], 1)
        self.assertEqual(annotations[70], 2)

        # Test loading from non-existent file
        annotations = self.visualizer.load_annotations('non_existent_file.csv')
        self.assertEqual(annotations, {})

    @patch('matplotlib.pyplot.figure')
    def test_create_heatmap(self, mock_figure):
        """Test creating a heatmap visualization."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        # Test creating a heatmap with valid data
        fig = self.visualizer.create_heatmap([self.test_csv_path], bin_size=10)
        self.assertIsNotNone(fig)

        # Test creating a heatmap with non-existent file
        fig = self.visualizer.create_heatmap(['non_existent_file.csv'], bin_size=10)
        self.assertIsNone(fig)

    @patch('matplotlib.pyplot.figure')
    def test_create_comparison_visualization(self, mock_figure):
        """Test creating a comparison visualization."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        # Test creating a comparison visualization with valid data
        fig = self.visualizer.create_comparison_visualization([self.test_csv_path])
        self.assertIsNotNone(fig)

        # Test creating a comparison visualization with non-existent file
        fig = self.visualizer.create_comparison_visualization(['non_existent_file.csv'])
        self.assertIsNone(fig)

    @patch('matplotlib.pyplot.figure')
    def test_create_timeline_visualization(self, mock_figure):
        """Test creating a timeline visualization."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        # Test creating a timeline visualization with valid data
        fig = self.visualizer.create_timeline_visualization(self.test_csv_path)
        self.assertIsNotNone(fig)

        # Test creating a timeline visualization with non-existent file
        fig = self.visualizer.create_timeline_visualization('non_existent_file.csv')
        self.assertIsNone(fig)

    @patch('matplotlib.pyplot.figure')
    def test_create_annotation_statistics(self, mock_figure):
        """Test creating annotation statistics visualization."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.subplots.return_value = (mock_fig, mock_axes)

        # Test creating annotation statistics with valid data
        fig = self.visualizer.create_annotation_statistics([self.test_csv_path])
        self.assertIsNotNone(fig)

        # Test creating annotation statistics with non-existent file
        fig = self.visualizer.create_annotation_statistics(['non_existent_file.csv'])
        self.assertIsNone(fig)

class TestVisualizationEnhancedMain(unittest.TestCase):
    """Test cases for the main function in visualization_enhanced.py."""

    @patch('tkinter.Tk')
    @patch('tkinter.filedialog.askopenfilenames')
    @patch('src.tools.analysis.visualization_enhanced.EnhancedVisualizer', new=MagicMock())
    @patch('src.tools.visualization_enhanced.EnhancedVisualizer', new=MagicMock())
    def test_main_function(self, mock_visualizer_old, mock_visualizer, mock_askopenfilenames, mock_tk):
        """Test the main function."""
        # Import here to avoid importing before mocks are in place
        try:
            from src.tools.analysis.visualization_enhanced import main
        except ImportError:
            from src.tools.visualization_enhanced import main

        # Setup mocks
        mock_askopenfilenames.return_value = ['test1.csv', 'test2.csv']
        mock_visualizer_instance = MagicMock()
        mock_visualizer.return_value = mock_visualizer_instance

        # Mock the visualization methods
        mock_visualizer_instance.create_heatmap.return_value = MagicMock()
        mock_visualizer_instance.create_comparison_visualization.return_value = MagicMock()
        mock_visualizer_instance.create_timeline_visualization.return_value = MagicMock()
        mock_visualizer_instance.create_annotation_statistics.return_value = MagicMock()

        # Test main function
        with patch('sys.argv', ['visualization_enhanced.py']):
            main()

        # Check that the visualizer was created
        mock_visualizer.assert_called_once()

        # Check that the visualization methods were called
        mock_visualizer_instance.create_heatmap.assert_called_once()
        mock_visualizer_instance.create_comparison_visualization.assert_called_once()
        mock_visualizer_instance.create_timeline_visualization.assert_called()
        mock_visualizer_instance.create_annotation_statistics.assert_called_once()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.tools.visualization_enhanced.EnhancedVisualizer')
    def test_main_with_command_line_args(self, mock_visualizer, mock_parse_args):
        """Test the main function with command line arguments."""
        # Import here to avoid importing before mocks are in place
        from src.tools.visualization_enhanced import main

        # Setup mocks
        mock_args = MagicMock()
        mock_args.csv = ['test1.csv', 'test2.csv']
        mock_args.bin_size = 50
        mock_args.mode = 'heatmap'
        mock_parse_args.return_value = mock_args

        mock_visualizer_instance = MagicMock()
        mock_visualizer.return_value = mock_visualizer_instance

        # Mock the visualization methods
        mock_visualizer_instance.create_heatmap.return_value = MagicMock()

        # Test main function
        with patch('sys.argv', ['visualization_enhanced.py', '--csv', 'test1.csv', 'test2.csv', '--bin_size', '50', '--mode', 'heatmap']):
            main()

        # Check that the visualizer was created
        mock_visualizer.assert_called_once()

        # Check that only the heatmap method was called
        mock_visualizer_instance.create_heatmap.assert_called_once()
        mock_visualizer_instance.create_comparison_visualization.assert_not_called()
        mock_visualizer_instance.create_timeline_visualization.assert_not_called()
        mock_visualizer_instance.create_annotation_statistics.assert_not_called()

if __name__ == '__main__':
    unittest.main()
