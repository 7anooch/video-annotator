#!/usr/bin/env python3
"""
Core analysis functionality for Video Annotator.

This module provides the core functionality for analyzing annotation data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('analysis_core')

class AnalysisCore:
    """
    Core functionality for all analysis tools.
    
    This class provides the foundation for all analysis tools, including
    data loading, basic analysis, and result management.
    
    Attributes:
        logger: The logger instance
    """
    
    def __init__(self):
        """Initialize the AnalysisCore."""
        self.logger = setup_logger('analysis_core')
        self.data = {}
        self.results = {}
    
    @exception_handler
    def load_annotations(self, path: str) -> Dict[int, Any]:
        """
        Load annotations from a file.
        
        Args:
            path (str): Path to the annotation file
            
        Returns:
            Dict[int, Any]: Dictionary of annotations with frame numbers as keys
        """
        if not os.path.exists(path):
            self.logger.error(f"Annotation file not found: {path}")
            return {}
        
        try:
            # Determine file type based on extension
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            
            if ext == '.csv':
                return self._load_from_csv(path)
            elif ext == '.json':
                return self._load_from_json(path)
            elif ext == '.xlsx' or ext == '.xls':
                return self._load_from_excel(path)
            else:
                self.logger.error(f"Unsupported file format: {ext}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            return {}
    
    @exception_handler
    def _load_from_csv(self, path: str) -> Dict[int, Any]:
        """
        Load annotations from a CSV file.
        
        Args:
            path (str): Path to the CSV file
            
        Returns:
            Dict[int, Any]: Dictionary of annotations with frame numbers as keys
        """
        try:
            df = pd.read_csv(path)
            
            # Check if the required columns exist
            if 'frame' not in df.columns:
                self.logger.error(f"CSV file does not contain a 'frame' column: {path}")
                return {}
            
            # Convert to dictionary
            annotations = {}
            for _, row in df.iterrows():
                frame = int(row['frame'])
                annotation = row.to_dict()
                del annotation['frame']  # Remove frame from the annotation
                annotations[frame] = annotation
            
            self.logger.info(f"Loaded {len(annotations)} annotations from {path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            return {}
    
    @exception_handler
    def _load_from_json(self, path: str) -> Dict[int, Any]:
        """
        Load annotations from a JSON file.
        
        Args:
            path (str): Path to the JSON file
            
        Returns:
            Dict[int, Any]: Dictionary of annotations with frame numbers as keys
        """
        try:
            import json
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Check if the data is a list of objects with 'frame' property
            if isinstance(data, list) and all('frame' in item for item in data):
                annotations = {}
                for item in data:
                    frame = int(item['frame'])
                    annotation = item.copy()
                    del annotation['frame']  # Remove frame from the annotation
                    annotations[frame] = annotation
                
                self.logger.info(f"Loaded {len(annotations)} annotations from {path}")
                return annotations
            else:
                self.logger.error(f"JSON file does not contain a list of objects with 'frame' property: {path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {str(e)}")
            return {}
    
    @exception_handler
    def _load_from_excel(self, path: str) -> Dict[int, Any]:
        """
        Load annotations from an Excel file.
        
        Args:
            path (str): Path to the Excel file
            
        Returns:
            Dict[int, Any]: Dictionary of annotations with frame numbers as keys
        """
        try:
            df = pd.read_excel(path)
            
            # Check if the required columns exist
            if 'frame' not in df.columns:
                self.logger.error(f"Excel file does not contain a 'frame' column: {path}")
                return {}
            
            # Convert to dictionary
            annotations = {}
            for _, row in df.iterrows():
                frame = int(row['frame'])
                annotation = row.to_dict()
                del annotation['frame']  # Remove frame from the annotation
                annotations[frame] = annotation
            
            self.logger.info(f"Loaded {len(annotations)} annotations from {path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            return {}
    
    @exception_handler
    def load_multiple_annotations(self, paths: List[str]) -> Dict[str, Dict[int, Any]]:
        """
        Load multiple annotation files.
        
        Args:
            paths (List[str]): List of paths to annotation files
            
        Returns:
            Dict[str, Dict[int, Any]]: Dictionary of annotations with file paths as keys
        """
        annotations = {}
        
        for path in paths:
            annotation = self.load_annotations(path)
            if annotation:
                annotations[path] = annotation
        
        self.logger.info(f"Loaded annotations from {len(annotations)} files")
        return annotations
    
    @exception_handler
    def get_annotation_statistics(self, annotations: Dict[int, Any]) -> Dict[str, Any]:
        """
        Get basic statistics for annotations.
        
        Args:
            annotations (Dict[int, Any]): Dictionary of annotations with frame numbers as keys
            
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        if not annotations:
            self.logger.warning("No annotations to analyze")
            return {}
        
        # Get all unique labels
        labels = set()
        for annotation in annotations.values():
            if 'label' in annotation:
                labels.add(annotation['label'])
        
        # Count occurrences of each label
        label_counts = {}
        for label in labels:
            count = sum(1 for annotation in annotations.values() if annotation.get('label') == label)
            label_counts[label] = count
        
        # Calculate frame statistics
        frame_numbers = list(annotations.keys())
        frame_min = min(frame_numbers)
        frame_max = max(frame_numbers)
        frame_range = frame_max - frame_min
        
        # Calculate gaps
        frame_numbers.sort()
        gaps = []
        for i in range(1, len(frame_numbers)):
            gap = frame_numbers[i] - frame_numbers[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        
        # Calculate statistics
        stats = {
            'total_annotations': len(annotations),
            'unique_labels': len(labels),
            'label_counts': label_counts,
            'frame_min': frame_min,
            'frame_max': frame_max,
            'frame_range': frame_range,
            'total_gaps': len(gaps),
            'total_gap_frames': sum(gaps) if gaps else 0,
            'avg_gap_size': np.mean(gaps) if gaps else 0,
            'max_gap_size': max(gaps) if gaps else 0
        }
        
        self.logger.info(f"Calculated statistics for {len(annotations)} annotations")
        return stats
    
    @exception_handler
    def get_label_transitions(self, annotations: Dict[int, Any]) -> Dict[Tuple[str, str], int]:
        """
        Get transitions between labels.
        
        Args:
            annotations (Dict[int, Any]): Dictionary of annotations with frame numbers as keys
            
        Returns:
            Dict[Tuple[str, str], int]: Dictionary of transitions with (from_label, to_label) as keys
        """
        if not annotations:
            self.logger.warning("No annotations to analyze")
            return {}
        
        # Sort annotations by frame number
        sorted_frames = sorted(annotations.keys())
        
        # Count transitions
        transitions = {}
        prev_label = None
        
        for frame in sorted_frames:
            annotation = annotations[frame]
            if 'label' in annotation:
                current_label = annotation['label']
                
                if prev_label is not None:
                    transition = (prev_label, current_label)
                    transitions[transition] = transitions.get(transition, 0) + 1
                
                prev_label = current_label
        
        self.logger.info(f"Calculated {len(transitions)} label transitions")
        return transitions
    
    @exception_handler
    def get_label_durations(self, annotations: Dict[int, Any]) -> Dict[str, List[int]]:
        """
        Get durations of continuous label segments.
        
        Args:
            annotations (Dict[int, Any]): Dictionary of annotations with frame numbers as keys
            
        Returns:
            Dict[str, List[int]]: Dictionary of durations with labels as keys
        """
        if not annotations:
            self.logger.warning("No annotations to analyze")
            return {}
        
        # Sort annotations by frame number
        sorted_frames = sorted(annotations.keys())
        
        # Calculate durations
        durations = {}
        current_label = None
        current_start = None
        
        for i, frame in enumerate(sorted_frames):
            annotation = annotations[frame]
            if 'label' in annotation:
                label = annotation['label']
                
                # If this is a new label or the first annotation
                if label != current_label or current_start is None:
                    # If we were tracking a label, calculate its duration
                    if current_label is not None and current_start is not None:
                        duration = frame - current_start
                        if current_label not in durations:
                            durations[current_label] = []
                        durations[current_label].append(duration)
                    
                    # Start tracking the new label
                    current_label = label
                    current_start = frame
        
        # Handle the last segment
        if current_label is not None and current_start is not None:
            duration = sorted_frames[-1] - current_start + 1
            if current_label not in durations:
                durations[current_label] = []
            durations[current_label].append(duration)
        
        self.logger.info(f"Calculated durations for {len(durations)} labels")
        return durations
    
    @exception_handler
    def compare_annotations(self, annotations1: Dict[int, Any], annotations2: Dict[int, Any]) -> Dict[str, Any]:
        """
        Compare two sets of annotations.
        
        Args:
            annotations1 (Dict[int, Any]): First set of annotations
            annotations2 (Dict[int, Any]): Second set of annotations
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        if not annotations1 or not annotations2:
            self.logger.warning("Cannot compare empty annotation sets")
            return {}
        
        # Get common frames
        frames1 = set(annotations1.keys())
        frames2 = set(annotations2.keys())
        common_frames = frames1.intersection(frames2)
        
        # Calculate agreement
        agreements = 0
        disagreements = 0
        
        for frame in common_frames:
            label1 = annotations1[frame].get('label')
            label2 = annotations2[frame].get('label')
            
            if label1 == label2:
                agreements += 1
            else:
                disagreements += 1
        
        # Calculate statistics
        total_common = len(common_frames)
        agreement_rate = agreements / total_common if total_common > 0 else 0
        
        # Calculate unique frames
        unique_to_1 = frames1 - frames2
        unique_to_2 = frames2 - frames1
        
        comparison = {
            'total_annotations_1': len(annotations1),
            'total_annotations_2': len(annotations2),
            'common_frames': total_common,
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreement_rate,
            'unique_to_1': len(unique_to_1),
            'unique_to_2': len(unique_to_2)
        }
        
        self.logger.info(f"Compared annotations with {total_common} common frames")
        return comparison
