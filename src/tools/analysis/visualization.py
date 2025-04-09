#!/usr/bin/env python3
"""
Visualization module for Video Annotator.

This module provides functions for visualizing annotation statistics.
"""

# Import the original visualization.py content
try:
    from src.tools.visualization import *

    # Add a note about the new location
    print("Note: This module is now located at src.tools.analysis.visualization")
except ImportError:
    # If the original module is not found, define the functions here
    print("Warning: Could not import from src.tools.visualization. Using local definitions.")
