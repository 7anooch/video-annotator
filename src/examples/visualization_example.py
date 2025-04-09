#!/usr/bin/env python3
"""
Visualization example using the enhanced analysis tools.

This script demonstrates how to use the enhanced analysis tools for visualization tasks.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path so that the script can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the enhanced analysis tools
from src.analysis.data_model import AnnotationData
from src.analysis.visualization import VisualizationManager

def main():
    """Run the visualization example."""
    print("Visualization Example")
    print("====================")
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python visualization_example.py <annotation_file>")
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
    
    # Create a visualization manager
    viz = VisualizationManager()
    
    # Create a timeline plot
    print("\nCreating timeline plot...")
    fig = viz.timeline_plot(data)
    plt.savefig("timeline.png")
    print(f"Timeline plot saved to: timeline.png")
    plt.close(fig)
    
    # Create a label distribution plot
    print("\nCreating label distribution plot...")
    fig = viz.label_distribution_plot(data)
    plt.savefig("label_distribution.png")
    print(f"Label distribution plot saved to: label_distribution.png")
    plt.close(fig)
    
    # Create a duration boxplot
    print("\nCreating duration boxplot...")
    fig = viz.duration_boxplot(data)
    plt.savefig("duration_boxplot.png")
    print(f"Duration boxplot saved to: duration_boxplot.png")
    plt.close(fig)
    
    # Create a transition heatmap
    print("\nCreating transition heatmap...")
    fig = viz.transition_heatmap(data)
    plt.savefig("transition_heatmap.png")
    print(f"Transition heatmap saved to: transition_heatmap.png")
    plt.close(fig)
    
    # Create a time series plot
    print("\nCreating time series plot...")
    fig = viz.time_series_plot(data)
    plt.savefig("time_series.png")
    print(f"Time series plot saved to: time_series.png")
    plt.close(fig)
    
    # Create an interactive timeline
    try:
        print("\nCreating interactive timeline...")
        interactive_path = viz.create_interactive_timeline(data, output_path="interactive_timeline.html")
        print(f"Interactive timeline saved to: {interactive_path}")
        
        # Ask if the user wants to open the interactive timeline
        response = input("Do you want to open the interactive timeline in your default browser? (y/n): ")
        if response.lower() == 'y':
            import webbrowser
            webbrowser.open(interactive_path)
    except Exception as e:
        print(f"Error creating interactive timeline: {str(e)}")
    
    # Create a 3D visualization
    try:
        print("\nCreating 3D visualization...")
        visualization_3d_path = viz.create_3d_visualization(data, output_path="3d_visualization.html")
        print(f"3D visualization saved to: {visualization_3d_path}")
        
        # Ask if the user wants to open the 3D visualization
        response = input("Do you want to open the 3D visualization in your default browser? (y/n): ")
        if response.lower() == 'y':
            import webbrowser
            webbrowser.open(visualization_3d_path)
    except Exception as e:
        print(f"Error creating 3D visualization: {str(e)}")
    
    # Generate a comprehensive report
    print("\nGenerating comprehensive report...")
    
    # Create a report directory
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate the report
    output_files = viz.create_report(data, output_dir=report_dir)
    
    print(f"Report generated in: {report_dir}")
    print("Generated files:")
    for file_path in output_files:
        print(f"  {os.path.basename(file_path)}")
    
    # Ask if the user wants to open the HTML report
    html_report = next((f for f in output_files if f.endswith('_report.html')), None)
    if html_report:
        response = input("Do you want to open the HTML report in your default browser? (y/n): ")
        if response.lower() == 'y':
            import webbrowser
            webbrowser.open(html_report)
    
    print("\nVisualization example completed successfully!")

if __name__ == "__main__":
    main()
