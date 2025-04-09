#!/usr/bin/env python3
"""
Export module for Video Annotator.

This module provides functions for exporting annotations to different formats.
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

class AnnotationExporter:
    """
    Exporter for annotations to different formats.

    Attributes:
        logger: The logger instance
    """

    def __init__(self):
        """Initialize the AnnotationExporter."""
        self.logger = setup_logger('annotation_exporter')

    @exception_handler
    def export_csv(self, annotations, output_path):
        """
        Export annotations to CSV format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert annotations to DataFrame
            max_frame = max(annotations.keys(), default=0)
            all_frames = list(range(int(max_frame) + 1))
            labels = [annotations.get(frame, np.nan) for frame in all_frames]
            df = pd.DataFrame({'frame': all_frames, 'label': labels})

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export to CSV
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported annotations to CSV: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting annotations to CSV: {str(e)}")
            return False

    @exception_handler
    def export_json(self, annotations, output_path):
        """
        Export annotations to JSON format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert annotations to serializable format
            serializable_annotations = {str(k): int(v) if not np.isnan(v) else None
                                       for k, v in annotations.items()}

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export to JSON
            with open(output_path, 'w') as f:
                json.dump(serializable_annotations, f, indent=4)
            self.logger.info(f"Exported annotations to JSON: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting annotations to JSON: {str(e)}")
            return False

    @exception_handler
    def export_txt(self, annotations, output_path):
        """
        Export annotations to TXT format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export to TXT
            with open(output_path, 'w') as f:
                f.write("frame,label\n")
                for frame in sorted(annotations.keys()):
                    label = annotations[frame]
                    if not np.isnan(label):
                        f.write(f"{frame},{int(label)}\n")
            self.logger.info(f"Exported annotations to TXT: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting annotations to TXT: {str(e)}")
            return False

    @exception_handler
    def export_matlab(self, annotations, output_path):
        """
        Export annotations to MATLAB format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if scipy is available
            try:
                from scipy.io import savemat
            except ImportError:
                self.logger.error("scipy is not installed. Cannot export to MATLAB format.")
                return False

            # Convert annotations to arrays
            frames = np.array(list(annotations.keys()))
            labels = np.array([annotations[frame] for frame in frames])

            # Create a dictionary for MATLAB
            matlab_dict = {
                'frames': frames,
                'labels': labels
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export to MATLAB
            savemat(output_path, matlab_dict)
            self.logger.info(f"Exported annotations to MATLAB: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting annotations to MATLAB: {str(e)}")
            return False

    @exception_handler
    def export_excel(self, annotations, output_path):
        """
        Export annotations to Excel format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert annotations to DataFrame
            max_frame = max(annotations.keys(), default=0)
            all_frames = list(range(int(max_frame) + 1))
            labels = [annotations.get(frame, np.nan) for frame in all_frames]
            df = pd.DataFrame({'frame': all_frames, 'label': labels})

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Export to Excel
            df.to_excel(output_path, index=False)
            self.logger.info(f"Exported annotations to Excel: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting annotations to Excel: {str(e)}")
            return False

    @exception_handler
    def export(self, annotations, output_path, format_type):
        """
        Export annotations to the specified format.

        Args:
            annotations (dict): Dictionary of frame numbers to labels
            output_path (str): Path to the output file
            format_type (str): Format type (csv, json, txt, matlab, excel)

        Returns:
            bool: True if successful, False otherwise
        """
        format_type = format_type.lower()

        if format_type == 'csv':
            return self.export_csv(annotations, output_path)
        elif format_type == 'json':
            return self.export_json(annotations, output_path)
        elif format_type == 'txt':
            return self.export_txt(annotations, output_path)
        elif format_type == 'matlab':
            return self.export_matlab(annotations, output_path)
        elif format_type == 'excel':
            return self.export_excel(annotations, output_path)
        else:
            self.logger.error(f"Unsupported format type: {format_type}")
            return False
