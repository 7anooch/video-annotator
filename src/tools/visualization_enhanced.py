#!/usr/bin/env python3
"""
Enhanced visualization tools for the Video Annotator.

This module provides advanced visualization tools for analyzing and visualizing
annotation data, including heatmaps, comparison visualizations, and timeline visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tkinter import filedialog
import tkinter as tk
import argparse
from matplotlib.patches import Patch
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.utils.annotation.funcs import get_common_substring, format_frames_and_ranges

# Set up logger
logger = setup_logger('visualization_enhanced')

class EnhancedVisualizer:
    """
    Enhanced visualization tools for annotation data.
    
    This class provides methods for creating advanced visualizations of annotation data,
    including heatmaps, comparison visualizations, and timeline visualizations.
    """
    
    def __init__(self):
        """Initialize the EnhancedVisualizer."""
        self.logger = setup_logger('enhanced_visualizer')
        self.label_colors = {
            0: 'red',    # Stop
            1: 'green',  # Run
            2: 'blue'    # Turn
        }
        self.label_names = {
            0: 'Stop',
            1: 'Run',
            2: 'Turn'
        }
    
    @exception_handler
    def load_annotations(self, csv_path):
        """
        Load annotations from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            dict: Dictionary of frame numbers to labels
        """
        self.logger.info(f"Loading annotations from {csv_path}")
        if not os.path.exists(csv_path):
            self.logger.error(f"File not found: {csv_path}")
            return {}
        
        try:
            df = pd.read_csv(csv_path)
            annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
            self.logger.info(f"Loaded {len(annotations)} annotations from {csv_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            return {}
    
    @exception_handler
    def create_heatmap(self, csv_paths, bin_size=100, title=None):
        """
        Create a heatmap visualization of annotation density.
        
        Args:
            csv_paths (list): List of paths to CSV files
            bin_size (int, optional): Size of bins for the heatmap. Defaults to 100.
            title (str, optional): Title for the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.logger.info(f"Creating heatmap for {len(csv_paths)} files with bin_size={bin_size}")
        
        # Load annotations from all files
        all_annotations = {}
        for csv_path in csv_paths:
            file_name = os.path.basename(csv_path)
            all_annotations[file_name] = self.load_annotations(csv_path)
        
        if not all_annotations:
            self.logger.warning("No annotations loaded")
            return None
        
        # Find the maximum frame number across all files
        max_frame = max([max(annotations.keys()) if annotations else 0 
                         for annotations in all_annotations.values()])
        
        # Create bins
        num_bins = int(max_frame / bin_size) + 1
        bins = np.zeros((len(all_annotations), num_bins, 3))  # 3 labels: Stop, Run, Turn
        
        # Fill bins with annotation counts
        for i, (file_name, annotations) in enumerate(all_annotations.items()):
            for frame, label in annotations.items():
                if 0 <= label <= 2:  # Only count valid labels
                    bin_index = int(frame / bin_size)
                    if bin_index < num_bins:
                        bins[i, bin_index, int(label)] += 1
        
        # Normalize bins
        for i in range(len(all_annotations)):
            for j in range(num_bins):
                total = np.sum(bins[i, j, :])
                if total > 0:
                    bins[i, j, :] = bins[i, j, :] / total
        
        # Create figure
        fig, axes = plt.subplots(len(all_annotations), 3, figsize=(15, 3 * len(all_annotations)),
                                 sharex=True, sharey=True)
        
        # If only one file, make axes 2D
        if len(all_annotations) == 1:
            axes = np.array([axes])
        
        # Plot heatmaps
        for i, (file_name, _) in enumerate(all_annotations.items()):
            for j, label_name in enumerate(['Stop', 'Run', 'Turn']):
                im = axes[i, j].imshow(bins[i, :, j].reshape(-1, 1),
                                      aspect='auto',
                                      cmap='viridis',
                                      vmin=0, vmax=1)
                axes[i, j].set_title(f"{file_name} - {label_name}")
                axes[i, j].set_ylabel("Frame Bin")
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i, j])
        
        # Set common x-axis label
        for ax in axes[-1, :]:
            ax.set_xlabel("Density")
        
        # Set overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)
        
        return fig
    
    @exception_handler
    def create_comparison_visualization(self, csv_paths, title=None):
        """
        Create a visualization comparing annotations from multiple files.
        
        Args:
            csv_paths (list): List of paths to CSV files
            title (str, optional): Title for the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.logger.info(f"Creating comparison visualization for {len(csv_paths)} files")
        
        # Load annotations from all files
        all_annotations = {}
        for csv_path in csv_paths:
            file_name = os.path.basename(csv_path)
            all_annotations[file_name] = self.load_annotations(csv_path)
        
        if not all_annotations:
            self.logger.warning("No annotations loaded")
            return None
        
        # Find the maximum frame number across all files
        max_frame = max([max(annotations.keys()) if annotations else 0 
                         for annotations in all_annotations.values()])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5 * len(all_annotations)))
        
        # Plot annotations for each file
        y_offset = 0
        file_positions = {}
        
        for file_name, annotations in all_annotations.items():
            file_positions[file_name] = y_offset
            
            # Plot segments for each label
            for label in range(3):
                label_frames = [frame for frame, l in annotations.items() if l == label]
                if label_frames:
                    # Group consecutive frames
                    segments = []
                    current_segment = [label_frames[0]]
                    
                    for frame in label_frames[1:]:
                        if frame == current_segment[-1] + 1:
                            current_segment.append(frame)
                        else:
                            segments.append(current_segment)
                            current_segment = [frame]
                    
                    segments.append(current_segment)
                    
                    # Plot segments
                    for segment in segments:
                        ax.plot([segment[0], segment[-1]], [y_offset, y_offset],
                                color=self.label_colors[label], linewidth=5)
            
            y_offset += 1
        
        # Set y-ticks to file names
        ax.set_yticks(list(file_positions.values()))
        ax.set_yticklabels(list(file_positions.keys()))
        
        # Set x-axis limits and label
        ax.set_xlim(0, max_frame)
        ax.set_xlabel("Frame Number")
        
        # Add legend
        legend_elements = [Patch(facecolor=self.label_colors[i], label=self.label_names[i])
                          for i in range(3)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        return fig
    
    @exception_handler
    def create_timeline_visualization(self, csv_path, title=None):
        """
        Create a timeline visualization of annotations.
        
        Args:
            csv_path (str): Path to the CSV file
            title (str, optional): Title for the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.logger.info(f"Creating timeline visualization for {csv_path}")
        
        # Load annotations
        annotations = self.load_annotations(csv_path)
        
        if not annotations:
            self.logger.warning("No annotations loaded")
            return None
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(list(annotations.items()), columns=['frame', 'label'])
        df = df.sort_values('frame')
        
        # Find transitions between labels
        df['next_label'] = df['label'].shift(-1)
        df['transition'] = df['label'] != df['next_label']
        transitions = df[df['transition']].copy()
        
        # Add the last frame
        last_row = pd.DataFrame({'frame': [df['frame'].max()], 
                                'label': [df['label'].iloc[-1]],
                                'next_label': [np.nan],
                                'transition': [True]})
        transitions = pd.concat([transitions, last_row], ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot segments
        prev_frame = 0
        for _, row in transitions.iterrows():
            frame = row['frame']
            label = row['label']
            
            # Plot segment
            ax.axvspan(prev_frame, frame, alpha=0.5, color=self.label_colors[label])
            
            # Add label text
            mid_frame = (prev_frame + frame) / 2
            ax.text(mid_frame, 0.5, self.label_names[label],
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.get_xaxis_transform())
            
            prev_frame = frame
        
        # Set x-axis limits and label
        ax.set_xlim(0, df['frame'].max())
        ax.set_xlabel("Frame Number")
        
        # Remove y-axis
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Add legend
        legend_elements = [Patch(facecolor=self.label_colors[i], label=self.label_names[i])
                          for i in range(3)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title if provided
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Timeline Visualization - {os.path.basename(csv_path)}")
        
        plt.tight_layout()
        
        return fig
    
    @exception_handler
    def create_annotation_statistics(self, csv_paths, title=None):
        """
        Create a visualization of annotation statistics.
        
        Args:
            csv_paths (list): List of paths to CSV files
            title (str, optional): Title for the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.logger.info(f"Creating annotation statistics for {len(csv_paths)} files")
        
        # Load annotations from all files
        all_annotations = {}
        for csv_path in csv_paths:
            file_name = os.path.basename(csv_path)
            all_annotations[file_name] = self.load_annotations(csv_path)
        
        if not all_annotations:
            self.logger.warning("No annotations loaded")
            return None
        
        # Calculate statistics for each file
        stats = []
        for file_name, annotations in all_annotations.items():
            # Count labels
            label_counts = {0: 0, 1: 0, 2: 0}
            for label in annotations.values():
                if 0 <= label <= 2:
                    label_counts[label] += 1
            
            # Calculate percentages
            total = sum(label_counts.values())
            if total > 0:
                label_percentages = {label: count / total * 100 
                                    for label, count in label_counts.items()}
            else:
                label_percentages = {0: 0, 1: 0, 2: 0}
            
            stats.append({
                'file_name': file_name,
                'total_frames': len(annotations),
                'stop_count': label_counts[0],
                'run_count': label_counts[1],
                'turn_count': label_counts[2],
                'stop_percent': label_percentages[0],
                'run_percent': label_percentages[1],
                'turn_percent': label_percentages[2]
            })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot label counts
        counts_df = stats_df.melt(id_vars=['file_name'], 
                                 value_vars=['stop_count', 'run_count', 'turn_count'],
                                 var_name='label', value_name='count')
        counts_df['label'] = counts_df['label'].str.replace('_count', '')
        
        sns.barplot(x='file_name', y='count', hue='label', data=counts_df, ax=axes[0])
        axes[0].set_title('Label Counts by File')
        axes[0].set_xlabel('File')
        axes[0].set_ylabel('Count')
        
        # Plot label percentages
        percentages_df = stats_df.melt(id_vars=['file_name'], 
                                      value_vars=['stop_percent', 'run_percent', 'turn_percent'],
                                      var_name='label', value_name='percentage')
        percentages_df['label'] = percentages_df['label'].str.replace('_percent', '')
        
        sns.barplot(x='file_name', y='percentage', hue='label', data=percentages_df, ax=axes[1])
        axes[1].set_title('Label Percentages by File')
        axes[1].set_xlabel('File')
        axes[1].set_ylabel('Percentage')
        
        # Set overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)
        
        plt.tight_layout()
        
        return fig

def get_csv_paths():
    """
    Get paths to CSV files from command-line arguments or file dialog.
    
    Returns:
        list: List of CSV file paths
    """
    parser = argparse.ArgumentParser(description="Enhanced Visualization Tool")
    parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    parser.add_argument('--bin_size', type=int, default=100, help="Bin size for heatmap")
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'heatmap', 'comparison', 'timeline', 'statistics'],
                       help="Visualization mode")
    args = parser.parse_args()
    
    if args.csv:
        return args.csv, args.bin_size, args.mode
    else:
        root = tk.Tk()
        root.withdraw()
        csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        return csv_paths, args.bin_size, args.mode

def main():
    """Main function to run the visualization tool."""
    csv_paths, bin_size, mode = get_csv_paths()
    
    if not csv_paths:
        logger.warning("No CSV files selected")
        return
    
    visualizer = EnhancedVisualizer()
    
    # Create common title from file names
    file_names = [os.path.basename(path) for path in csv_paths]
    common_substring = get_common_substring(file_names)
    title = f"Visualization for {common_substring}" if common_substring else "Annotation Visualization"
    
    # Create visualizations based on mode
    if mode == 'all' or mode == 'heatmap':
        heatmap_fig = visualizer.create_heatmap(csv_paths, bin_size=bin_size, 
                                               title=f"Annotation Density Heatmap - {title}")
        if heatmap_fig:
            plt.figure(heatmap_fig.number)
            plt.show()
    
    if mode == 'all' or mode == 'comparison':
        comparison_fig = visualizer.create_comparison_visualization(csv_paths, 
                                                                   title=f"Annotation Comparison - {title}")
        if comparison_fig:
            plt.figure(comparison_fig.number)
            plt.show()
    
    if mode == 'all' or mode == 'timeline':
        for csv_path in csv_paths:
            timeline_fig = visualizer.create_timeline_visualization(csv_path)
            if timeline_fig:
                plt.figure(timeline_fig.number)
                plt.show()
    
    if mode == 'all' or mode == 'statistics':
        stats_fig = visualizer.create_annotation_statistics(csv_paths, 
                                                          title=f"Annotation Statistics - {title}")
        if stats_fig:
            plt.figure(stats_fig.number)
            plt.show()

if __name__ == "__main__":
    main()
