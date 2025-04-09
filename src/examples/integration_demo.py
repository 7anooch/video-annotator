#!/usr/bin/env python3
"""
Proof-of-concept integration of enhanced analysis tools with existing tools.

This script demonstrates how the enhanced analysis tools can be integrated with
the existing analysis tools in the Video Annotator application.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager
from src.analysis.adapters.analyze_adapter import AnalyzeAdapter
from src.analysis.adapters.plot_adapter import PlotAdapter
from src.analysis.adapters.visualization_adapter import VisualizationAdapter

# Import the existing analysis tools
try:
    from src.tools.analysis import analyze, visualization
    from src.tools import plot
except ImportError:
    from src.tools import analyze, plot, visualization

def main():
    """Run the integration demo."""
    print("Enhanced Analysis Tools Integration Demo")
    print("=======================================")

    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python integration_demo.py <annotation_file>")
        return

    file_path = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading annotations from: {file_path}")

    # Load annotations using the enhanced data model
    data = AnnotationData()
    result = data.load_from_file(file_path)

    if not result:
        print(f"Error: Failed to load file: {file_path}")
        return

    print(f"Loaded {data.get_annotation_count()} annotations")
    print(f"Labels: {', '.join(data.get_labels())}")
    print(f"Frame range: {min(data.get_frames())} - {max(data.get_frames())}")

    # Create the enhanced analysis objects
    stats = StatisticalAnalysis()
    viz = VisualizationManager()

    # Create the adapter objects
    analyze_adapter = AnalyzeAdapter()
    plot_adapter = PlotAdapter()
    visualization_adapter = VisualizationAdapter()

    print("\nDemonstration 1: Using Enhanced Tools with Existing Data Format")
    print("-------------------------------------------------------------")

    # Convert to the format used by the existing analysis tools
    frames, labels = data.to_analyze_format()

    print(f"Converted to existing format: {len(frames)} frames, {len(labels)} labels")

    # Use the existing analysis tools
    print("\nUsing existing analyze.py:")
    sequences, counts = analyze.analyze_sequences(labels)
    print(f"Sequences: {sequences[:3]}...")
    print(f"Counts: {counts[:3]}...")

    # Use the enhanced analysis tools with the existing format
    print("\nUsing enhanced StatisticalAnalysis:")
    basic_stats = stats.basic_statistics(data)
    print(f"Total annotations: {basic_stats['total_annotations']}")
    print(f"Unique labels: {basic_stats['unique_labels']}")
    print(f"Label counts: {basic_stats['label_counts']}")

    print("\nDemonstration 2: Using Adapters to Bridge Enhanced and Existing Tools")
    print("--------------------------------------------------------------------")

    # Use the analyze adapter to call the existing analyze.py functions
    print("\nUsing AnalyzeAdapter:")
    sequence_analysis = analyze_adapter.analyze_sequence(data.annotations)
    print(f"Sequences: {sequence_analysis['sequences'][:3]}...")
    print(f"Counts: {sequence_analysis['counts'][:3]}...")

    # Use the plot adapter to call the existing plot.py functions
    print("\nUsing PlotAdapter:")
    fig = plot_adapter.plot_ethogram(data.annotations)
    plt.savefig("ethogram.png")
    print(f"Ethogram saved to: ethogram.png")
    plt.close(fig)

    print("\nDemonstration 3: Combining Enhanced and Existing Tools")
    print("-----------------------------------------------------")

    # Use the enhanced analysis tools to detect anomalies
    print("\nUsing enhanced anomaly detection:")
    anomalies = stats.detect_anomalies(data, method="zscore", threshold=3.0)
    print(f"Detected {len(anomalies)} anomalies")

    # Convert the anomalies to a format that can be used by the existing visualization tools
    anomaly_frames = list(anomalies.keys())
    print(f"Anomaly frames: {anomaly_frames}")

    # Create a visualization that combines enhanced and existing tools
    print("\nCreating a combined visualization:")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Use the existing plot.py to create an ethogram in the first subplot
    plot.plot_ethogram(ax1, frames, labels, title="Ethogram (Existing Tool)")

    # Use the enhanced visualization to create a timeline in the second subplot
    viz.timeline_plot(data, ax=ax2, title="Timeline with Anomalies (Enhanced Tool)")

    # Highlight the anomalies in the second subplot
    for frame in anomaly_frames:
        ax2.axvline(x=frame, color='red', linestyle='--', alpha=0.5)

    # Save the combined visualization
    plt.tight_layout()
    plt.savefig("combined_visualization.png")
    print(f"Combined visualization saved to: combined_visualization.png")
    plt.close(fig)

    print("\nDemonstration 4: Creating Interactive Visualizations")
    print("---------------------------------------------------")

    # Create an interactive timeline
    print("\nCreating an interactive timeline:")
    interactive_path = viz.create_interactive_timeline(data, output_path="interactive_timeline.html")
    print(f"Interactive timeline saved to: {interactive_path}")

    # Create a 3D visualization
    print("\nCreating a 3D visualization:")
    visualization_3d_path = viz.create_3d_visualization(data, output_path="3d_visualization.html")
    print(f"3D visualization saved to: {visualization_3d_path}")

    print("\nDemonstration 5: Generating a Comprehensive Report")
    print("--------------------------------------------------")

    # Create a report directory
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)

    # Generate a comprehensive report
    print("\nGenerating a comprehensive report:")
    output_files = viz.create_report(data, output_dir=report_dir)
    print(f"Report generated in: {report_dir}")
    print("Generated files:")
    for file_path in output_files:
        print(f"  {os.path.basename(file_path)}")

    print("\nIntegration demo completed successfully!")

if __name__ == "__main__":
    main()
