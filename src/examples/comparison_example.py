#!/usr/bin/env python3
"""
Comparison example using the enhanced analysis tools.

This script demonstrates how to use the enhanced analysis tools to compare two annotation sets.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager
from src.analysis.utils import calculate_agreement, calculate_cohen_kappa

def main():
    """Run the comparison example."""
    print("Comparison Example")
    print("=================")
    
    # Check if two file paths were provided
    if len(sys.argv) < 3:
        print("Usage: python comparison_example.py <annotation_file_1> <annotation_file_2>")
        return
    
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    
    # Check if the files exist
    if not os.path.exists(file_path_1):
        print(f"Error: File not found: {file_path_1}")
        return
    
    if not os.path.exists(file_path_2):
        print(f"Error: File not found: {file_path_2}")
        return
    
    print(f"Loading annotations from: {file_path_1}")
    
    # Load the first annotation set
    data1 = AnnotationData()
    result = data1.load_from_file(file_path_1)
    
    if not result:
        print(f"Error: Failed to load file: {file_path_1}")
        return
    
    print(f"Loaded {data1.get_annotation_count()} annotations from {file_path_1}")
    print(f"Labels: {', '.join(data1.get_labels())}")
    print(f"Frame range: {min(data1.get_frames())} - {max(data1.get_frames())}")
    
    print(f"\nLoading annotations from: {file_path_2}")
    
    # Load the second annotation set
    data2 = AnnotationData()
    result = data2.load_from_file(file_path_2)
    
    if not result:
        print(f"Error: Failed to load file: {file_path_2}")
        return
    
    print(f"Loaded {data2.get_annotation_count()} annotations from {file_path_2}")
    print(f"Labels: {', '.join(data2.get_labels())}")
    print(f"Frame range: {min(data2.get_frames())} - {max(data2.get_frames())}")
    
    # Create the analysis objects
    stats = StatisticalAnalysis()
    viz = VisualizationManager()
    
    # Compare the annotation sets
    print("\nComparing annotation sets...")
    comparison = stats.compare_annotations(data1, data2)
    
    print(f"Total annotations in set 1: {comparison['total_annotations_1']}")
    print(f"Total annotations in set 2: {comparison['total_annotations_2']}")
    print(f"Common frames: {comparison['common_frames']}")
    print(f"Agreements: {comparison['agreements']}")
    print(f"Disagreements: {comparison['disagreements']}")
    print(f"Agreement rate: {comparison['agreement_rate']:.2f}")
    print(f"Unique to set 1: {comparison['unique_to_1']}")
    print(f"Unique to set 2: {comparison['unique_to_2']}")
    
    # Calculate agreement
    print("\nCalculating agreement...")
    agreement = calculate_agreement(data1.annotations, data2.annotations)
    print(f"Agreement: {agreement:.2f}")
    
    # Calculate Cohen's kappa
    try:
        print("\nCalculating Cohen's kappa...")
        kappa = calculate_cohen_kappa(data1.annotations, data2.annotations)
        print(f"Cohen's kappa: {kappa:.2f}")
    except Exception as e:
        print(f"Error calculating Cohen's kappa: {str(e)}")
    
    # Calculate correlation
    print("\nCalculating correlation...")
    correlation = stats.correlation_analysis(data1, data2)
    print(f"Pearson correlation: {correlation['pearson_correlation']:.2f}")
    print(f"Spearman correlation: {correlation['spearman_correlation']:.2f}")
    print(f"Cohen's kappa: {correlation['cohen_kappa']:.2f}")
    
    # Create a comparison plot
    print("\nCreating comparison plot...")
    fig = viz.comparison_plot(data1, data2)
    plt.savefig("comparison.png")
    print(f"Comparison plot saved to: comparison.png")
    plt.close(fig)
    
    # Create a confusion matrix
    print("\nCreating confusion matrix...")
    fig = viz.confusion_matrix_plot(data1, data2)
    plt.savefig("confusion_matrix.png")
    print(f"Confusion matrix saved to: confusion_matrix.png")
    plt.close(fig)
    
    print("\nComparison example completed successfully!")

if __name__ == "__main__":
    main()
