#!/usr/bin/env python3
"""
Adapter for the visualization.py module.

This module provides an adapter class that interfaces with the existing visualization.py module.
"""

import os
import tkinter as tk
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel
try:
    from src.tools.analysis import visualization
except ImportError:
    from src.tools import visualization

# Set up logger
logger = setup_logger('visualization_adapter')

class VisualizationAdapter:
    """
    Adapter for the visualization.py module.

    This class provides an interface to the functionality in the existing visualization.py module.
    It allows the enhanced analysis tools to use the existing visualization functions.

    Attributes:
        logger: The logger instance
    """

    def __init__(self):
        """Initialize the VisualizationAdapter."""
        self.logger = setup_logger('visualization_adapter')

    @exception_handler
    def create_visualization_gui(self, master: tk.Tk, annotations: Dict[int, Dict[str, Any]]) -> visualization.VisualizationGUI:
        """
        Create a visualization GUI using the existing visualization.py module.

        Args:
            master (tk.Tk): The Tkinter master window
            annotations (Dict[int, Dict[str, Any]]): Annotations to visualize

        Returns:
            visualization.VisualizationGUI: The visualization GUI instance
        """
        try:
            # Convert annotations to the format expected by visualization.py
            frames = []
            labels = []

            for frame, annotation in sorted(annotations.items()):
                if 'label' in annotation:
                    frames.append(frame)
                    labels.append(annotation['label'])

            # Create a visualization GUI
            gui = visualization.VisualizationGUI(master)

            # Set the annotations
            gui.set_annotations(frames, labels)

            self.logger.info(f"Created visualization GUI with {len(frames)} annotations")
            return gui
        except Exception as e:
            self.logger.error(f"Error creating visualization GUI: {str(e)}")
            return None

    @exception_handler
    def visualize_annotations(self, annotations: Dict[int, Dict[str, Any]]) -> None:
        """
        Visualize annotations using the existing visualization.py module.

        Args:
            annotations (Dict[int, Dict[str, Any]]): Annotations to visualize
        """
        try:
            # Convert annotations to the format expected by visualization.py
            frames = []
            labels = []

            for frame, annotation in sorted(annotations.items()):
                if 'label' in annotation:
                    frames.append(frame)
                    labels.append(annotation['label'])

            # Call the existing visualization.py function
            visualization.visualize_annotations(frames, labels)

            self.logger.info(f"Visualized {len(frames)} annotations")
        except Exception as e:
            self.logger.error(f"Error visualizing annotations: {str(e)}")

    @exception_handler
    def compare_annotations(self, annotations1: Dict[int, Dict[str, Any]],
                          annotations2: Dict[int, Dict[str, Any]],
                          labels1: str = 'Set 1',
                          labels2: str = 'Set 2') -> None:
        """
        Compare two sets of annotations using the existing visualization.py module.

        Args:
            annotations1 (Dict[int, Dict[str, Any]]): First set of annotations
            annotations2 (Dict[int, Dict[str, Any]]): Second set of annotations
            labels1 (str, optional): Label for the first set. Defaults to 'Set 1'.
            labels2 (str, optional): Label for the second set. Defaults to 'Set 2'.
        """
        try:
            # Convert annotations to the format expected by visualization.py
            frames1 = []
            labels1_list = []

            for frame, annotation in sorted(annotations1.items()):
                if 'label' in annotation:
                    frames1.append(frame)
                    labels1_list.append(annotation['label'])

            frames2 = []
            labels2_list = []

            for frame, annotation in sorted(annotations2.items()):
                if 'label' in annotation:
                    frames2.append(frame)
                    labels2_list.append(annotation['label'])

            # Call the existing visualization.py function
            visualization.compare_annotations(frames1, labels1_list, frames2, labels2_list, labels1, labels2)

            self.logger.info(f"Compared {len(frames1)} and {len(frames2)} annotations")
        except Exception as e:
            self.logger.error(f"Error comparing annotations: {str(e)}")
