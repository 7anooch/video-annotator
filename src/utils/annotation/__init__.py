"""
Annotation utilities for the Video Annotator application.

This package provides utility functions for working with annotations.
"""

from src.utils.annotation.funcs import (
    save_annotations,
    get_csv_file_path,
    get_common_substring,
    format_frames_and_ranges
)

__all__ = [
    'save_annotations',
    'get_csv_file_path',
    'get_common_substring',
    'format_frames_and_ranges'
]
