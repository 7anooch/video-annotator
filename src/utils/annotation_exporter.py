#!/usr/bin/env python3
"""
Annotation export utilities for the Video Annotator.

This module provides tools for exporting annotations to different formats.
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
logger = setup_logger('annotation_exporter')

class AnnotationExporter:
    """
    Provides annotation export capabilities.
    
    Attributes:
        supported_formats (List[str]): List of supported export formats
    """
    
    def __init__(self):
        """Initialize the AnnotationExporter."""
        self.logger = setup_logger('annotation_exporter')
        
        # List of supported export formats
        self.supported_formats = ['csv', 'json', 'excel', 'matlab', 'numpy']
    
    @exception_handler
    def export_annotations(self, annotations: Dict[int, Dict[str, Any]], 
                          output_path: str, format: str = None) -> bool:
        """
        Export annotations to a file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            format (str, optional): Export format. Defaults to None (inferred from output_path).
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        # Check if annotations is empty
        if not annotations:
            self.logger.warning("No annotations to export")
            return False
        
        # Determine export format
        if format is None:
            format = os.path.splitext(output_path)[1].lower().lstrip('.')
            if not format:
                format = 'csv'  # Default to CSV
        
        # Check if format is supported
        if format not in self.supported_formats:
            self.logger.error(f"Unsupported export format: {format}")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Export annotations based on format
        if format == 'csv':
            return self._export_to_csv(annotations, output_path)
        elif format == 'json':
            return self._export_to_json(annotations, output_path)
        elif format == 'excel':
            return self._export_to_excel(annotations, output_path)
        elif format == 'matlab':
            return self._export_to_matlab(annotations, output_path)
        elif format == 'numpy':
            return self._export_to_numpy(annotations, output_path)
        else:
            self.logger.error(f"Unsupported export format: {format}")
            return False
    
    @exception_handler
    def _export_to_csv(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
        """
        Export annotations to a CSV file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Convert annotations to a list of dictionaries
            rows = []
            for frame_number, annotation in annotations.items():
                row = {'frame': frame_number}
                row.update(annotation)
                rows.append(row)
            
            # Sort rows by frame number
            rows.sort(key=lambda x: x['frame'])
            
            # Write to CSV
            with open(output_path, 'w', newline='') as f:
                if not rows:
                    self.logger.warning("No annotations to export")
                    return False
                
                # Get fieldnames from the first row
                fieldnames = list(rows[0].keys())
                
                # Create CSV writer
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header and rows
                writer.writeheader()
                writer.writerows(rows)
            
            self.logger.info(f"Exported {len(rows)} annotations to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    @exception_handler
    def _export_to_json(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
        """
        Export annotations to a JSON file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Convert annotations to a list of dictionaries
            data = []
            for frame_number, annotation in annotations.items():
                item = {'frame': frame_number}
                item.update(annotation)
                data.append(item)
            
            # Sort data by frame number
            data.sort(key=lambda x: x['frame'])
            
            # Write to JSON
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            self.logger.info(f"Exported {len(data)} annotations to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
            return False
    
    @exception_handler
    def _export_to_excel(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
        """
        Export annotations to an Excel file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Convert annotations to a list of dictionaries
            rows = []
            for frame_number, annotation in annotations.items():
                row = {'frame': frame_number}
                row.update(annotation)
                rows.append(row)
            
            # Sort rows by frame number
            rows.sort(key=lambda x: x['frame'])
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            
            # Write to Excel
            df.to_excel(output_path, index=False)
            
            self.logger.info(f"Exported {len(rows)} annotations to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}")
            return False
    
    @exception_handler
    def _export_to_matlab(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
        """
        Export annotations to a MATLAB file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Import scipy.io here to avoid dependency issues
            import scipy.io as sio
            
            # Convert annotations to a dictionary of arrays
            data = {}
            
            # Get all keys from annotations
            all_keys = set()
            for annotation in annotations.values():
                all_keys.update(annotation.keys())
            
            # Initialize arrays for each key
            for key in all_keys:
                data[key] = []
            
            # Add frame numbers
            data['frame'] = []
            
            # Fill arrays
            for frame_number, annotation in sorted(annotations.items()):
                data['frame'].append(frame_number)
                
                for key in all_keys:
                    if key in annotation:
                        data[key].append(annotation[key])
                    else:
                        data[key].append(None)
            
            # Convert lists to numpy arrays
            for key in data:
                data[key] = np.array(data[key])
            
            # Write to MATLAB file
            sio.savemat(output_path, data)
            
            self.logger.info(f"Exported {len(annotations)} annotations to {output_path}")
            return True
        except ImportError:
            self.logger.error("scipy.io is required for MATLAB export")
            return False
        except Exception as e:
            self.logger.error(f"Error exporting to MATLAB: {str(e)}")
            return False
    
    @exception_handler
    def _export_to_numpy(self, annotations: Dict[int, Dict[str, Any]], output_path: str) -> bool:
        """
        Export annotations to a NumPy file.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            output_path (str): Path to save the exported annotations
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Convert annotations to a dictionary of arrays
            data = {}
            
            # Get all keys from annotations
            all_keys = set()
            for annotation in annotations.values():
                all_keys.update(annotation.keys())
            
            # Initialize arrays for each key
            for key in all_keys:
                data[key] = []
            
            # Add frame numbers
            data['frame'] = []
            
            # Fill arrays
            for frame_number, annotation in sorted(annotations.items()):
                data['frame'].append(frame_number)
                
                for key in all_keys:
                    if key in annotation:
                        data[key].append(annotation[key])
                    else:
                        data[key].append(None)
            
            # Convert lists to numpy arrays
            for key in data:
                data[key] = np.array(data[key])
            
            # Write to NumPy file
            np.savez(output_path, **data)
            
            self.logger.info(f"Exported {len(annotations)} annotations to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to NumPy: {str(e)}")
            return False
    
    @exception_handler
    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported export formats.
        
        Returns:
            List[str]: List of supported export formats
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
