#!/usr/bin/env python3
"""
Basic analysis example using the enhanced analysis tools.

This script demonstrates how to use the enhanced analysis tools for basic analysis tasks.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis

def main():
    """Run the basic analysis example."""
    print("Basic Analysis Example")
    print("=====================")
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python basic_analysis_example.py <annotation_file>")
        return
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loading annotations from: {file_path}")
    
    # Load annotations
    data = AnnotationData()
    result = data.load_from_file(file_path)
    
    if not result:
        print(f"Error: Failed to load file: {file_path}")
        return
    
    print(f"Loaded {data.get_annotation_count()} annotations")
    print(f"Labels: {', '.join(data.get_labels())}")
    print(f"Frame range: {min(data.get_frames())} - {max(data.get_frames())}")
    
    # Create a statistical analysis object
    stats = StatisticalAnalysis()
    
    # Calculate basic statistics
    print("\nBasic Statistics:")
    basic_stats = stats.basic_statistics(data)
    print(f"Total annotations: {basic_stats['total_annotations']}")
    print(f"Unique labels: {basic_stats['unique_labels']}")
    print(f"Frame range: {basic_stats['frame_min']} - {basic_stats['frame_max']}")
    print(f"Total gaps: {basic_stats['total_gaps']}")
    print(f"Total gap frames: {basic_stats['total_gap_frames']}")
    print(f"Average gap size: {basic_stats['avg_gap_size']:.2f}")
    print(f"Maximum gap size: {basic_stats['max_gap_size']}")
    
    print("\nLabel counts:")
    for label, count in basic_stats['label_counts'].items():
        print(f"  {label}: {count}")
    
    # Calculate label transitions
    print("\nLabel Transitions:")
    transitions = stats.label_transitions(data)
    for (from_label, to_label), count in sorted(transitions.items()):
        print(f"  {from_label} -> {to_label}: {count}")
    
    # Calculate label durations
    print("\nLabel Durations:")
    durations = stats.label_durations(data)
    for label, duration_list in durations.items():
        print(f"  {label}: {duration_list}")
    
    # Calculate duration statistics
    print("\nDuration Statistics:")
    duration_stats = stats.duration_statistics(data)
    for label, stats_dict in duration_stats.items():
        print(f"  {label}:")
        print(f"    Count: {stats_dict['count']}")
        print(f"    Total frames: {stats_dict['total_frames']}")
        print(f"    Min: {stats_dict['min']}")
        print(f"    Max: {stats_dict['max']}")
        print(f"    Mean: {stats_dict['mean']:.2f}")
        print(f"    Median: {stats_dict['median']:.2f}")
        print(f"    Std: {stats_dict['std']:.2f}")
    
    print("\nBasic analysis completed successfully!")

if __name__ == "__main__":
    main()
