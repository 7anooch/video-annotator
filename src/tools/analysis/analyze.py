#!/usr/bin/env python3
"""
Analysis module for Video Annotator.

This module provides functions for analyzing annotation data.
"""

# Import the original analyze.py content
try:
    from src.tools.analyze import *

    # Add a note about the new location
    print("Note: This module is now located at src.tools.analysis.analyze")
except ImportError:
    # If the original module is not found, define the functions here
    print("Warning: Could not import from src.tools.analyze. Using local definitions.")
