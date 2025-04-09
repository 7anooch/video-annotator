#!/usr/bin/env python3
"""
Launcher for the enhanced analysis tools UI.

This script launches the enhanced analysis tools UI.
"""

import os
import sys
import tkinter as tk

# Add the parent directory to the path so that the UI can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the AnalysisUI class
from src.ui.analysis_ui import AnalysisUI

def main():
    """Launch the enhanced analysis tools UI."""
    root = tk.Tk()
    app = AnalysisUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
