#!/usr/bin/env python3
"""
Adapter for the plot.py module.

This module provides an adapter class that interfaces with the existing plot.py module.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel
try:
    from src.tools.analysis import visualization as plot
except ImportError:
    from src.tools import plot

# Set up logger
logger = setup_logger('plot_adapter')

class PlotAdapter:
    """
    Adapter for the plot.py module.

    This class provides an interface to the functionality in the existing plot.py module.
    It allows the enhanced analysis tools to use the existing plotting functions.

    Attributes:
        logger: The logger instance
    """

    def __init__(self):
        """Initialize the PlotAdapter."""
        self.logger = setup_logger('plot_adapter')

    @exception_handler
    def plot_ethogram(self, annotations: Dict[int, Dict[str, Any]],
                     figsize: Tuple[int, int] = (12, 6),
                     title: str = 'Ethogram',
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot an ethogram using the existing plot.py module.

        Args:
            annotations (Dict[int, Dict[str, Any]]): Annotations to plot
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to 'Ethogram'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Convert annotations to the format expected by plot.py
            frames = []
            labels = []

            for frame, annotation in sorted(annotations.items()):
                if 'label' in annotation:
                    frames.append(frame)
                    labels.append(annotation['label'])

            # Create a figure
            fig, ax = plt.subplots(figsize=figsize)

            # Call the existing plot.py function
            plot.plot_ethogram(ax, frames, labels, title=title)

            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved ethogram plot to {save_path}")

            return fig
        except Exception as e:
            self.logger.error(f"Error plotting ethogram: {str(e)}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error plotting ethogram: {str(e)}", ha='center', va='center')
            return fig

    @exception_handler
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             labels: List[str],
                             figsize: Tuple[int, int] = (10, 8),
                             title: str = 'Confusion Matrix',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a confusion matrix using the existing plot.py module.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix to plot
            labels (List[str]): Labels for the confusion matrix
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
            title (str, optional): Plot title. Defaults to 'Confusion Matrix'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=figsize)

            # Call the existing plot.py function
            plot.plot_confusion_matrix(ax, confusion_matrix, labels, title=title)

            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved confusion matrix plot to {save_path}")

            return fig
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error plotting confusion matrix: {str(e)}", ha='center', va='center')
            return fig

    @exception_handler
    def plot_precision_recall(self, precision: float, recall: float, f1: float,
                            figsize: Tuple[int, int] = (8, 6),
                            title: str = 'Precision and Recall',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision and recall metrics using the existing plot.py module.

        Args:
            precision (float): Precision value
            recall (float): Recall value
            f1 (float): F1 score
            figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).
            title (str, optional): Plot title. Defaults to 'Precision and Recall'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=figsize)

            # Call the existing plot.py function
            plot.plot_precision_recall(ax, precision, recall, f1, title=title)

            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved precision-recall plot to {save_path}")

            return fig
        except Exception as e:
            self.logger.error(f"Error plotting precision and recall: {str(e)}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error plotting precision and recall: {str(e)}", ha='center', va='center')
            return fig

    @exception_handler
    def get_color_map(self, labels: List[str]) -> Dict[str, str]:
        """
        Get a color map for labels using the existing plot.py module.

        Args:
            labels (List[str]): Labels to map to colors

        Returns:
            Dict[str, str]: Dictionary mapping labels to colors
        """
        try:
            # Call the existing plot.py function
            color_map = plot.get_color_map(labels)

            self.logger.info(f"Created color map for {len(labels)} labels")
            return color_map
        except Exception as e:
            self.logger.error(f"Error creating color map: {str(e)}")
            return {}
