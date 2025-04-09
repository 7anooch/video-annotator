#!/usr/bin/env python3
"""
Adapter for the analyze.py module.

This module provides an adapter class that interfaces with the existing analyze.py module.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel
try:
    from src.tools.analysis import analyze
except ImportError:
    from src.tools import analyze

# Set up logger
logger = setup_logger('analyze_adapter')

class AnalyzeAdapter:
    """
    Adapter for the analyze.py module.

    This class provides an interface to the functionality in the existing analyze.py module.
    It allows the enhanced analysis tools to use the existing analysis functions.

    Attributes:
        logger: The logger instance
    """

    def __init__(self):
        """Initialize the AnalyzeAdapter."""
        self.logger = setup_logger('analyze_adapter')

    @exception_handler
    def calculate_precision_recall(self, ground_truth: Dict[int, Dict[str, Any]],
                                 predictions: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate precision and recall using the existing analyze.py module.

        Args:
            ground_truth (Dict[int, Dict[str, Any]]): Ground truth annotations
            predictions (Dict[int, Dict[str, Any]]): Predicted annotations

        Returns:
            Dict[str, Any]: Precision and recall metrics
        """
        try:
            # Convert annotations to the format expected by analyze.py
            gt_frames = []
            gt_labels = []
            pred_frames = []
            pred_labels = []

            for frame, annotation in ground_truth.items():
                if 'label' in annotation:
                    gt_frames.append(frame)
                    gt_labels.append(annotation['label'])

            for frame, annotation in predictions.items():
                if 'label' in annotation:
                    pred_frames.append(frame)
                    pred_labels.append(annotation['label'])

            # Call the existing analyze.py function
            precision, recall, f1, confusion_matrix = analyze.calculate_precision_recall(
                gt_frames, gt_labels, pred_frames, pred_labels
            )

            # Format the results
            results = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix
            }

            self.logger.info(f"Calculated precision and recall: {precision:.4f}, {recall:.4f}")
            return results
        except Exception as e:
            self.logger.error(f"Error calculating precision and recall: {str(e)}")
            return {}

    @exception_handler
    def analyze_sequence(self, annotations: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sequence patterns using the existing analyze.py module.

        Args:
            annotations (Dict[int, Dict[str, Any]]): Annotations to analyze

        Returns:
            Dict[str, Any]: Sequence analysis results
        """
        try:
            # Convert annotations to the format expected by analyze.py
            frames = []
            labels = []

            for frame, annotation in sorted(annotations.items()):
                if 'label' in annotation:
                    frames.append(frame)
                    labels.append(annotation['label'])

            # Call the existing analyze.py function
            sequences, counts = analyze.analyze_sequences(labels)

            # Format the results
            results = {
                'sequences': sequences,
                'counts': counts
            }

            self.logger.info(f"Analyzed {len(sequences)} sequences")
            return results
        except Exception as e:
            self.logger.error(f"Error analyzing sequences: {str(e)}")
            return {}

    @exception_handler
    def calculate_confusion_matrix(self, ground_truth: Dict[int, Dict[str, Any]],
                                 predictions: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """
        Calculate confusion matrix using the existing analyze.py module.

        Args:
            ground_truth (Dict[int, Dict[str, Any]]): Ground truth annotations
            predictions (Dict[int, Dict[str, Any]]): Predicted annotations

        Returns:
            np.ndarray: Confusion matrix
        """
        try:
            # Convert annotations to the format expected by analyze.py
            gt_frames = []
            gt_labels = []
            pred_frames = []
            pred_labels = []

            for frame, annotation in ground_truth.items():
                if 'label' in annotation:
                    gt_frames.append(frame)
                    gt_labels.append(annotation['label'])

            for frame, annotation in predictions.items():
                if 'label' in annotation:
                    pred_frames.append(frame)
                    pred_labels.append(annotation['label'])

            # Call the existing analyze.py function
            _, _, _, confusion_matrix = analyze.calculate_precision_recall(
                gt_frames, gt_labels, pred_frames, pred_labels
            )

            self.logger.info(f"Calculated confusion matrix of shape {confusion_matrix.shape}")
            return confusion_matrix
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {str(e)}")
            return np.array([])

    @exception_handler
    def align_sequences(self, sequence1: List[str], sequence2: List[str]) -> Dict[str, Any]:
        """
        Align two sequences using the existing analyze.py module.

        Args:
            sequence1 (List[str]): First sequence
            sequence2 (List[str]): Second sequence

        Returns:
            Dict[str, Any]: Alignment results
        """
        try:
            # Call the existing analyze.py function
            alignment, score = analyze.align_sequences(sequence1, sequence2)

            # Format the results
            results = {
                'alignment': alignment,
                'score': score
            }

            self.logger.info(f"Aligned sequences with score {score}")
            return results
        except Exception as e:
            self.logger.error(f"Error aligning sequences: {str(e)}")
            return {}
