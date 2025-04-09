#!/usr/bin/env python3
"""
Data model for annotation analysis that is compatible with existing tools.

This module provides a common data model for working with annotations that is
compatible with the existing annotation formats used in the Video Annotator.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('data_model')

class AnnotationData:
    """
    Data model for annotations that is compatible with existing tools.

    This class provides a common data model for working with annotations,
    including loading, saving, and manipulating annotation data. It is designed
    to be compatible with the existing annotation formats used in the Video Annotator.

    Attributes:
        logger: The logger instance
        annotations: Dictionary of annotations with frame numbers as keys
        metadata: Dictionary of metadata about the annotations
    """

    def __init__(self, annotations: Optional[Dict[int, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the AnnotationData.

        Args:
            annotations (Dict[int, Any], optional): Dictionary of annotations. Defaults to None.
            metadata (Dict[str, Any], optional): Dictionary of metadata. Defaults to None.
        """
        self.logger = setup_logger('annotation_data')
        self.annotations = annotations or {}
        self.metadata = metadata or {}

    @exception_handler
    def load_from_file(self, path: str, column: int = 1) -> bool:
        """
        Load annotations from a file.

        Args:
            path (str): Path to the annotation file
            column (int, optional): Column index for the label in CSV files. Defaults to 1.

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(path):
            self.logger.error(f"Annotation file not found: {path}")
            return False

        try:
            # Determine file type based on extension
            _, ext = os.path.splitext(path)
            ext = ext.lower()

            if ext == '.csv':
                return self._load_from_csv(path, column)
            elif ext == '.json':
                return self._load_from_json(path)
            elif ext == '.xlsx' or ext == '.xls':
                return self._load_from_excel(path)
            else:
                self.logger.error(f"Unsupported file format: {ext}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            return False

    @exception_handler
    def _load_from_csv(self, path: str, column: int = 1) -> bool:
        """
        Load annotations from a CSV file.

        Args:
            path (str): Path to the CSV file
            column (int, optional): Column index for the label. Defaults to 1.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to load as a standard CSV with headers
            try:
                df = pd.read_csv(path)

                # Check if the required columns exist
                if 'frame' in df.columns:
                    # Standard format with 'frame' and 'label' columns
                    annotations = {}
                    for _, row in df.iterrows():
                        frame = int(row['frame'])
                        annotation = row.to_dict()
                        del annotation['frame']  # Remove frame from the annotation
                        annotations[frame] = annotation

                    self.annotations = annotations
                    self.metadata = {
                        'source': path,
                        'format': 'csv',
                        'total_annotations': len(annotations)
                    }

                    self.logger.info(f"Loaded {len(annotations)} annotations from {path} (standard format)")
                    return True
            except Exception as e:
                self.logger.warning(f"Could not load as standard CSV: {str(e)}. Trying simple format...")

            # Try to load as a simple CSV without headers (compatible with existing tools)
            try:
                # Read the CSV file as plain text
                with open(path, 'r') as f:
                    lines = f.readlines()

                annotations = {}
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) > column:
                        try:
                            frame = int(parts[0])
                            label = parts[column]
                            annotations[frame] = {'label': label}
                        except ValueError:
                            # Skip header or invalid lines
                            continue

                self.annotations = annotations
                self.metadata = {
                    'source': path,
                    'format': 'csv_simple',
                    'total_annotations': len(annotations)
                }

                self.logger.info(f"Loaded {len(annotations)} annotations from {path} (simple format)")
                return True
            except Exception as e:
                self.logger.error(f"Error loading simple CSV: {str(e)}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            return False

    @exception_handler
    def _load_from_json(self, path: str) -> bool:
        """
        Load annotations from a JSON file.

        Args:
            path (str): Path to the JSON file

        Returns:
            bool: True if successful, False otherwise
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

                self.annotations = annotations
                self.metadata = {
                    'source': path,
                    'format': 'json',
                    'total_annotations': len(annotations)
                }

                self.logger.info(f"Loaded {len(annotations)} annotations from {path}")
                return True
            else:
                self.logger.error(f"JSON file does not contain a list of objects with 'frame' property: {path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {str(e)}")
            return False

    @exception_handler
    def _load_from_excel(self, path: str) -> bool:
        """
        Load annotations from an Excel file.

        Args:
            path (str): Path to the Excel file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_excel(path)

            # Check if the required columns exist
            if 'frame' not in df.columns:
                self.logger.error(f"Excel file does not contain a 'frame' column: {path}")
                return False

            # Convert to dictionary
            annotations = {}
            for _, row in df.iterrows():
                frame = int(row['frame'])
                annotation = row.to_dict()
                del annotation['frame']  # Remove frame from the annotation
                annotations[frame] = annotation

            self.annotations = annotations
            self.metadata = {
                'source': path,
                'format': 'excel',
                'total_annotations': len(annotations)
            }

            self.logger.info(f"Loaded {len(annotations)} annotations from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            return False

    @exception_handler
    def save_to_file(self, path: str) -> bool:
        """
        Save annotations to a file.

        Args:
            path (str): Path to save the annotations

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine file type based on extension
            _, ext = os.path.splitext(path)
            ext = ext.lower()

            if ext == '.csv':
                return self._save_to_csv(path)
            elif ext == '.json':
                return self._save_to_json(path)
            elif ext == '.xlsx':
                return self._save_to_excel(path)
            else:
                self.logger.error(f"Unsupported file format: {ext}")
                return False
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            return False

    @exception_handler
    def _save_to_csv(self, path: str) -> bool:
        """
        Save annotations to a CSV file.

        Args:
            path (str): Path to save the CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert annotations to a list of dictionaries
            rows = []
            for frame, annotation in self.annotations.items():
                row = {'frame': frame}
                row.update(annotation)
                rows.append(row)

            # Convert to DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)

            self.logger.info(f"Saved {len(rows)} annotations to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {str(e)}")
            return False

    @exception_handler
    def _save_to_json(self, path: str) -> bool:
        """
        Save annotations to a JSON file.

        Args:
            path (str): Path to save the JSON file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import json

            # Convert annotations to a list of dictionaries
            items = []
            for frame, annotation in self.annotations.items():
                item = {'frame': frame}
                item.update(annotation)
                items.append(item)

            # Save to JSON
            with open(path, 'w') as f:
                json.dump(items, f, indent=4)

            self.logger.info(f"Saved {len(items)} annotations to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to JSON: {str(e)}")
            return False

    @exception_handler
    def _save_to_excel(self, path: str) -> bool:
        """
        Save annotations to an Excel file.

        Args:
            path (str): Path to save the Excel file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert annotations to a list of dictionaries
            rows = []
            for frame, annotation in self.annotations.items():
                row = {'frame': frame}
                row.update(annotation)
                rows.append(row)

            # Convert to DataFrame and save
            df = pd.DataFrame(rows)
            df.to_excel(path, index=False)

            self.logger.info(f"Saved {len(rows)} annotations to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to Excel: {str(e)}")
            return False

    @exception_handler
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert annotations to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representation of the annotations
        """
        # Convert annotations to a list of dictionaries
        rows = []
        for frame, annotation in self.annotations.items():
            row = {'frame': frame}
            row.update(annotation)
            rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        return df

    @exception_handler
    def from_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Load annotations from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing annotations

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the required columns exist
            if 'frame' not in df.columns:
                self.logger.error("DataFrame does not contain a 'frame' column")
                return False

            # Convert to dictionary
            annotations = {}
            for _, row in df.iterrows():
                frame = int(row['frame'])
                annotation = row.to_dict()
                del annotation['frame']  # Remove frame from the annotation
                annotations[frame] = annotation

            self.annotations = annotations
            self.metadata = {
                'source': 'dataframe',
                'total_annotations': len(annotations)
            }

            self.logger.info(f"Loaded {len(annotations)} annotations from DataFrame")
            return True
        except Exception as e:
            self.logger.error(f"Error loading from DataFrame: {str(e)}")
            return False

    @exception_handler
    def get_frames(self) -> List[int]:
        """
        Get a list of all frame numbers.

        Returns:
            List[int]: List of frame numbers
        """
        return sorted(self.annotations.keys())

    @exception_handler
    def get_labels(self) -> List[str]:
        """
        Get a list of all unique labels.

        Returns:
            List[str]: List of unique labels
        """
        labels = set()
        for annotation in self.annotations.values():
            if 'label' in annotation:
                labels.add(annotation['label'])
        return sorted(labels)

    @exception_handler
    def to_frame_label_lists(self) -> Tuple[List[int], List[str]]:
        """
        Convert annotations to separate lists of frames and labels.
        This format is compatible with the existing analysis tools.

        Returns:
            Tuple[List[int], List[str]]: Tuple of (frames, labels)
        """
        frames = []
        labels = []

        for frame, annotation in sorted(self.annotations.items()):
            if 'label' in annotation:
                frames.append(frame)
                labels.append(annotation['label'])

        return frames, labels

    @exception_handler
    def from_frame_label_lists(self, frames: List[int], labels: List[str]) -> bool:
        """
        Load annotations from separate lists of frames and labels.
        This format is used by the existing analysis tools.

        Args:
            frames (List[int]): List of frame numbers
            labels (List[str]): List of labels

        Returns:
            bool: True if successful, False otherwise
        """
        if len(frames) != len(labels):
            self.logger.error(f"Frame and label lists have different lengths: {len(frames)} vs {len(labels)}")
            return False

        annotations = {}
        for frame, label in zip(frames, labels):
            annotations[frame] = {'label': label}

        self.annotations = annotations
        self.metadata = {
            'source': 'frame_label_lists',
            'total_annotations': len(annotations)
        }

        self.logger.info(f"Loaded {len(annotations)} annotations from frame and label lists")
        return True

    @exception_handler
    def get_annotation(self, frame: int) -> Optional[Dict[str, Any]]:
        """
        Get the annotation for a specific frame.

        Args:
            frame (int): Frame number

        Returns:
            Optional[Dict[str, Any]]: Annotation for the frame, or None if not found
        """
        return self.annotations.get(frame)

    @exception_handler
    def set_annotation(self, frame: int, annotation: Dict[str, Any]) -> None:
        """
        Set the annotation for a specific frame.

        Args:
            frame (int): Frame number
            annotation (Dict[str, Any]): Annotation data
        """
        self.annotations[frame] = annotation

    @exception_handler
    def delete_annotation(self, frame: int) -> bool:
        """
        Delete the annotation for a specific frame.

        Args:
            frame (int): Frame number

        Returns:
            bool: True if the annotation was deleted, False if it didn't exist
        """
        if frame in self.annotations:
            del self.annotations[frame]
            return True
        return False

    @exception_handler
    def clear_annotations(self) -> None:
        """Clear all annotations."""
        self.annotations = {}

    @exception_handler
    def get_annotation_count(self) -> int:
        """
        Get the total number of annotations.

        Returns:
            int: Total number of annotations
        """
        return len(self.annotations)

    @exception_handler
    def get_label_counts(self) -> Dict[str, int]:
        """
        Get counts of each label.

        Returns:
            Dict[str, int]: Dictionary of label counts
        """
        counts = {}
        for annotation in self.annotations.values():
            if 'label' in annotation:
                label = annotation['label']
                counts[label] = counts.get(label, 0) + 1
        return counts

    @exception_handler
    def to_visualization_format(self) -> Dict[str, Any]:
        """
        Convert annotations to the format used by the existing visualization tools.

        Returns:
            Dict[str, Any]: Dictionary with 'frames' and 'labels' keys
        """
        frames, labels = self.to_frame_label_lists()
        return {
            'frames': frames,
            'labels': labels
        }

    @exception_handler
    def from_visualization_format(self, data: Dict[str, Any]) -> bool:
        """
        Load annotations from the format used by the existing visualization tools.

        Args:
            data (Dict[str, Any]): Dictionary with 'frames' and 'labels' keys

        Returns:
            bool: True if successful, False otherwise
        """
        if 'frames' not in data or 'labels' not in data:
            self.logger.error("Visualization format data must contain 'frames' and 'labels' keys")
            return False

        return self.from_frame_label_lists(data['frames'], data['labels'])

    @exception_handler
    def filter_by_label(self, label: str) -> 'AnnotationData':
        """
        Filter annotations by label.

        Args:
            label (str): Label to filter by

        Returns:
            AnnotationData: New AnnotationData object with filtered annotations
        """
        filtered = {}
        for frame, annotation in self.annotations.items():
            if annotation.get('label') == label:
                filtered[frame] = annotation.copy()

        return AnnotationData(filtered, {
            'source': 'filter',
            'filter_type': 'label',
            'filter_value': label,
            'total_annotations': len(filtered)
        })

    @exception_handler
    def filter_by_frames(self, start_frame: int, end_frame: int) -> 'AnnotationData':
        """
        Filter annotations by frame range.

        Args:
            start_frame (int): Start frame (inclusive)
            end_frame (int): End frame (inclusive)

        Returns:
            AnnotationData: New AnnotationData object with filtered annotations
        """
        filtered = {}
        for frame, annotation in self.annotations.items():
            if start_frame <= frame <= end_frame:
                filtered[frame] = annotation.copy()

        return AnnotationData(filtered, {
            'source': 'filter',
            'filter_type': 'frames',
            'filter_start': start_frame,
            'filter_end': end_frame,
            'total_annotations': len(filtered)
        })

    @exception_handler
    def merge(self, other: 'AnnotationData', overwrite: bool = False) -> 'AnnotationData':
        """
        Merge with another AnnotationData object.

        Args:
            other (AnnotationData): Other AnnotationData object
            overwrite (bool, optional): Whether to overwrite existing annotations. Defaults to False.

        Returns:
            AnnotationData: New AnnotationData object with merged annotations
        """
        merged = self.annotations.copy()

        for frame, annotation in other.annotations.items():
            if frame not in merged or overwrite:
                merged[frame] = annotation.copy()

        return AnnotationData(merged, {
            'source': 'merge',
            'total_annotations': len(merged)
        })

    @exception_handler
    def to_analyze_format(self) -> Tuple[List[int], List[str]]:
        """
        Convert annotations to the format used by the existing analyze.py module.

        Returns:
            Tuple[List[int], List[str]]: Tuple of (frames, labels)
        """
        return self.to_frame_label_lists()

    @exception_handler
    def from_analyze_format(self, frames: List[int], labels: List[str]) -> bool:
        """
        Load annotations from the format used by the existing analyze.py module.

        Args:
            frames (List[int]): List of frame numbers
            labels (List[str]): List of labels

        Returns:
            bool: True if successful, False otherwise
        """
        return self.from_frame_label_lists(frames, labels)

    @exception_handler
    def to_plot_format(self) -> Tuple[List[int], List[str]]:
        """
        Convert annotations to the format used by the existing plot.py module.

        Returns:
            Tuple[List[int], List[str]]: Tuple of (frames, labels)
        """
        return self.to_frame_label_lists()

    @exception_handler
    def from_plot_format(self, frames: List[int], labels: List[str]) -> bool:
        """
        Load annotations from the format used by the existing plot.py module.

        Args:
            frames (List[int]): List of frame numbers
            labels (List[str]): List of labels

        Returns:
            bool: True if successful, False otherwise
        """
        return self.from_frame_label_lists(frames, labels)
