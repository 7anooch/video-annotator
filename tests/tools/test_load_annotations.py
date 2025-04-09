#!/usr/bin/env python3
"""
Test script for the load_annotations function.
"""

import os
import pandas as pd

def test_load_annotations(csv_path, col=2):
    """Test version of load_annotations that handles any CSV structure."""
    print(f"Testing load_annotations with {csv_path}, col={col}")
    
    annotations = {}
    if not os.path.exists(csv_path):
        print(f"File does not exist: {csv_path}")
        return annotations
    
    try:
        # First, read the entire CSV to see what we're working with
        print(f"Reading entire CSV file: {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        print(f"CSV shape: {df.shape}")
        
        # Determine which columns to use
        num_columns = df.shape[1]
        print(f"CSV has {num_columns} columns")
        
        if num_columns < 2:
            print(f"CSV has only {num_columns} column(s), need at least 2")
            return annotations
        
        # Use first column for frame, and adjust label column if needed
        frame_col = 0
        label_col = min(col, num_columns - 1)
        print(f"Using column {frame_col} for frames and column {label_col} for labels")
        
        # Create annotations dictionary
        annotations = {int(row[frame_col]): row[label_col] for _, row in df.iterrows()}
        print(f"Created {len(annotations)} annotations")
        
        return annotations
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("Enter CSV file path: ")
    
    col = 2
    if len(sys.argv) > 2:
        try:
            col = int(sys.argv[2])
        except ValueError:
            print(f"Invalid column number: {sys.argv[2]}, using default: {col}")
    
    annotations = test_load_annotations(csv_path, col)
    print(f"Loaded {len(annotations)} annotations")
    
    if annotations:
        print("First 5 annotations:")
        for i, (frame, label) in enumerate(sorted(annotations.items())[:5]):
            print(f"  {frame}: {label}")
