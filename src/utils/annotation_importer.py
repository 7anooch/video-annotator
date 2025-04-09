#!/usr/bin/env python3
"""
Annotation import utilities for the Video Annotator.

This module provides tools for importing annotations from different formats.
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('annotation_importer')

class AnnotationImporter:
    """
    Provides annotation import capabilities.
    
    Attributes:
        supported_formats (List[str]): List of supported import formats
    """
    
    def __init__(self):
        """Initialize the AnnotationImporter."""
        self.logger = setup_logger('annotation_importer')
        
        # List of supported import formats
        self.supported_formats = ['csv', 'json', 'excel', 'matlab', 'numpy']
    
    @exception_handler
    def import_annotations(self, input_path: str, format: str = None) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from a file.
        
        Args:
            input_path (str): Path to the file to import
            format (str, optional): Import format. Defaults to None (inferred from input_path).
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            return {}
        
        # Determine import format
        if format is None:
            format = os.path.splitext(input_path)[1].lower().lstrip('.')
            if not format:
                format = 'csv'  # Default to CSV
        
        # Check if format is supported
        if format not in self.supported_formats:
            self.logger.error(f"Unsupported import format: {format}")
            return {}
        
        # Import annotations based on format
        if format == 'csv':
            return self._import_from_csv(input_path)
        elif format == 'json':
            return self._import_from_json(input_path)
        elif format == 'excel':
            return self._import_from_excel(input_path)
        elif format == 'matlab':
            return self._import_from_matlab(input_path)
        elif format == 'numpy':
            return self._import_from_numpy(input_path)
        else:
            self.logger.error(f"Unsupported import format: {format}")
            return {}
    
    @exception_handler
    def _import_from_csv(self, input_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from a CSV file.
        
        Args:
            input_path (str): Path to the CSV file
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        try:
            # Read CSV file
            with open(input_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Convert rows to annotations
            annotations = {}
            for row in rows:
                # Get frame number
                frame_number = int(row.pop('frame'))
                
                # Convert string values to appropriate types
                annotation = {}
                for key, value in row.items():
                    # Try to convert to int or float
                    try:
                        if value.isdigit():
                            annotation[key] = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            annotation[key] = float(value)
                        else:
                            annotation[key] = value
                    except (ValueError, AttributeError):
                        annotation[key] = value
                
                # Add annotation
                annotations[frame_number] = annotation
            
            self.logger.info(f"Imported {len(annotations)} annotations from {input_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error importing from CSV: {str(e)}")
            return {}
    
    @exception_handler
    def _import_from_json(self, input_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from a JSON file.
        
        Args:
            input_path (str): Path to the JSON file
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        try:
            # Read JSON file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert data to annotations
            annotations = {}
            for item in data:
                # Get frame number
                frame_number = int(item.pop('frame'))
                
                # Add annotation
                annotations[frame_number] = item
            
            self.logger.info(f"Imported {len(annotations)} annotations from {input_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error importing from JSON: {str(e)}")
            return {}
    
    @exception_handler
    def _import_from_excel(self, input_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from an Excel file.
        
        Args:
            input_path (str): Path to the Excel file
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        try:
            # Read Excel file
            df = pd.read_excel(input_path)
            
            # Convert DataFrame to annotations
            annotations = {}
            for _, row in df.iterrows():
                # Get frame number
                frame_number = int(row['frame'])
                
                # Create annotation
                annotation = {}
                for column in df.columns:
                    if column != 'frame':
                        annotation[column] = row[column]
                
                # Add annotation
                annotations[frame_number] = annotation
            
            self.logger.info(f"Imported {len(annotations)} annotations from {input_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error importing from Excel: {str(e)}")
            return {}
    
    @exception_handler
    def _import_from_matlab(self, input_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from a MATLAB file.
        
        Args:
            input_path (str): Path to the MATLAB file
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        try:
            # Import scipy.io here to avoid dependency issues
            import scipy.io as sio
            
            # Read MATLAB file
            data = sio.loadmat(input_path)
            
            # Convert data to annotations
            annotations = {}
            
            # Get frame numbers
            frame_numbers = data.get('frame', [])
            if len(frame_numbers) == 0:
                self.logger.error("No frame numbers found in MATLAB file")
                return {}
            
            # Get number of frames
            num_frames = len(frame_numbers)
            
            # Create annotations
            for i in range(num_frames):
                frame_number = int(frame_numbers[i])
                annotation = {}
                
                # Add other fields
                for key, value in data.items():
                    if key != 'frame' and not key.startswith('__'):
                        if len(value) > i:
                            annotation[key] = value[i]
                
                # Add annotation
                annotations[frame_number] = annotation
            
            self.logger.info(f"Imported {len(annotations)} annotations from {input_path}")
            return annotations
        except ImportError:
            self.logger.error("scipy.io is required for MATLAB import")
            return {}
        except Exception as e:
            self.logger.error(f"Error importing from MATLAB: {str(e)}")
            return {}
    
    @exception_handler
    def _import_from_numpy(self, input_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Import annotations from a NumPy file.
        
        Args:
            input_path (str): Path to the NumPy file
            
        Returns:
            Dict[int, Dict[str, Any]]: The imported annotations
        """
        try:
            # Read NumPy file
            data = np.load(input_path)
            
            # Convert data to annotations
            annotations = {}
            
            # Get frame numbers
            frame_numbers = data.get('frame', [])
            if len(frame_numbers) == 0:
                self.logger.error("No frame numbers found in NumPy file")
                return {}
            
            # Get number of frames
            num_frames = len(frame_numbers)
            
            # Create annotations
            for i in range(num_frames):
                frame_number = int(frame_numbers[i])
                annotation = {}
                
                # Add other fields
                for key in data.files:
                    if key != 'frame':
                        if len(data[key]) > i:
                            value = data[key][i]
                            if isinstance(value, np.ndarray) and value.size == 1:
                                value = value.item()
                            annotation[key] = value
                
                # Add annotation
                annotations[frame_number] = annotation
            
            self.logger.info(f"Imported {len(annotations)} annotations from {input_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error importing from NumPy: {str(e)}")
            return {}
    
    @exception_handler
    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported import formats.
        
        Returns:
            List[str]: List of supported import formats
        """
        return self.supported_formats.copy()
    
    @exception_handler
    def is_supported_format(self, format: str) -> bool:
        """
        Check if a format is supported.
        
        Args:
            format (str): The format to check
            
        Returns:
            bool: True if the format is supported, False otherwise
        """
        return format.lower() in self.supported_formats
