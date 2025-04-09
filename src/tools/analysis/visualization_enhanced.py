#!/usr/bin/env python3
"""
Enhanced visualization tools for the Video Annotator.

This module provides advanced visualization tools for analyzing and visualizing
annotation data, including heatmaps, comparison visualizations, and timeline visualizations.
"""

# Import the original visualization_enhanced.py content
try:
    from src.tools.visualization_enhanced import *

    # Add a note about the new location
    print("Note: This module is now located at src.tools.analysis.visualization_enhanced")
except ImportError:
    # If the original module is not found, define the functions here
    print("Warning: Could not import from src.tools.visualization_enhanced. Using local definitions.")
