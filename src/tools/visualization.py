#!/usr/bin/env python3
"""
Visualization module for Video Annotator.

This module provides functions for visualizing annotation statistics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.core.config import Config

class VisualizationTool:
    """
    Tool for visualizing annotation statistics.
    
    Attributes:
        logger: The logger instance
        config: The configuration instance
    """
    
    def __init__(self, config_path='config.json'):
        """
        Initialize the VisualizationTool.
        
        Args:
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
        """
        self.logger = setup_logger('visualization_tool')
        self.config = Config(config_path)
        self.labels = self.config.get_labels()
        
        # Create a mapping of label values to colors
        self.color_mapping = {}
        for label in self.labels:
            self.color_mapping[label['value']] = label['color']
    
    @exception_handler
    def load_annotations(self, csv_path):
        """
        Load annotations from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            dict: Dictionary of frame numbers to labels
        """
        try:
            if not os.path.exists(csv_path):
                self.logger.error(f"No annotation file found at {csv_path}")
                return {}
            
            df = pd.read_csv(csv_path)
            annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
            self.logger.info(f"Loaded {len(annotations)} annotations from {csv_path}")
            return annotations
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            return {}
    
    @exception_handler
    def plot_ethogram(self, annotations, title="Ethogram", show_legend=True, ax=None):
        """
        Plot an ethogram of the annotations.
        
        Args:
            annotations (dict): Dictionary of frame numbers to labels
            title (str, optional): Title of the plot. Defaults to "Ethogram".
            show_legend (bool, optional): Whether to show the legend. Defaults to True.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not annotations:
            self.logger.warning("No annotations to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
        
        frames = sorted(annotations.keys())
        behaviors = [annotations[frame] for frame in frames]
        colors = [self.color_mapping.get(int(behavior), "#FFFF00") 
                 if not np.isnan(behavior) else "#FFFF00" for behavior in behaviors]
        
        # Fill the region for each frame
        for i, frame in enumerate(frames[:-1]):  # Exclude the last frame to avoid out-of-bounds
            ax.fill_between(
                [frame, frames[i + 1]],  # x range for the current frame
                0,  # ymin
                1,  # ymax
                color=colors[i],  # Color for the current behavior
                step="pre"  # Step style to create sharp transitions
            )
        
        ax.set_yticks([])
        ax.set_xlabel("Frame Number")
        ax.set_title(title)
        ax.set_xlim(frames[0], frames[-1])
        
        if show_legend:
            legend_patches = []
            for label in self.labels:
                patch = mpatches.Patch(color=label['color'], label=label['name'])
                legend_patches.append(patch)
            ax.legend(handles=legend_patches, loc='upper right')
        
        return fig
    
    @exception_handler
    def plot_label_distribution(self, annotations, title="Label Distribution", ax=None):
        """
        Plot the distribution of labels in the annotations.
        
        Args:
            annotations (dict): Dictionary of frame numbers to labels
            title (str, optional): Title of the plot. Defaults to "Label Distribution".
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not annotations:
            self.logger.warning("No annotations to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # Count the occurrences of each label
        label_counts = {}
        for label in annotations.values():
            if not np.isnan(label):
                label_int = int(label)
                label_counts[label_int] = label_counts.get(label_int, 0) + 1
        
        # Get the label names and colors
        label_names = []
        label_colors = []
        for label_value in sorted(label_counts.keys()):
            label_info = next((l for l in self.labels if l['value'] == label_value), None)
            if label_info:
                label_names.append(label_info['name'])
                label_colors.append(label_info['color'])
            else:
                label_names.append(f"Label {label_value}")
                label_colors.append("#FFFF00")
        
        # Plot the distribution
        bars = ax.bar(label_names, [label_counts[int(label_value)] for label_value in sorted(label_counts.keys())], 
                     color=label_colors)
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f"{height}", ha='center', va='bottom')
        
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title(title)
        
        return fig
    
    @exception_handler
    def plot_label_timeline(self, annotations, title="Label Timeline", ax=None):
        """
        Plot a timeline of when each label occurs in the video.
        
        Args:
            annotations (dict): Dictionary of frame numbers to labels
            title (str, optional): Title of the plot. Defaults to "Label Timeline".
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not annotations:
            self.logger.warning("No annotations to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
        
        # Group frames by label
        label_frames = {}
        for frame, label in annotations.items():
            if not np.isnan(label):
                label_int = int(label)
                if label_int not in label_frames:
                    label_frames[label_int] = []
                label_frames[label_int].append(frame)
        
        # Get the label names and colors
        label_names = []
        label_colors = []
        for label_value in sorted(label_frames.keys()):
            label_info = next((l for l in self.labels if l['value'] == label_value), None)
            if label_info:
                label_names.append(label_info['name'])
                label_colors.append(label_info['color'])
            else:
                label_names.append(f"Label {label_value}")
                label_colors.append("#FFFF00")
        
        # Plot the timeline
        for i, (label_value, frames) in enumerate(sorted(label_frames.items())):
            label_info = next((l for l in self.labels if l['value'] == label_value), None)
            if label_info:
                label_name = label_info['name']
                label_color = label_info['color']
            else:
                label_name = f"Label {label_value}"
                label_color = "#FFFF00"
            
            ax.scatter(frames, [i] * len(frames), color=label_color, label=label_name, s=10)
        
        ax.set_yticks(range(len(label_names)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Frame Number")
        ax.set_title(title)
        
        # Add a grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        return fig
    
    @exception_handler
    def plot_transition_matrix(self, annotations, title="Transition Matrix", ax=None):
        """
        Plot a transition matrix showing how often one label transitions to another.
        
        Args:
            annotations (dict): Dictionary of frame numbers to labels
            title (str, optional): Title of the plot. Defaults to "Transition Matrix".
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
            
        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if not annotations:
            self.logger.warning("No annotations to plot")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Get the sorted frames and labels
        frames = sorted(annotations.keys())
        labels = [annotations[frame] for frame in frames]
        
        # Count transitions
        transitions = {}
        for i in range(len(frames) - 1):
            if np.isnan(labels[i]) or np.isnan(labels[i + 1]):
                continue
            
            from_label = int(labels[i])
            to_label = int(labels[i + 1])
            
            if from_label not in transitions:
                transitions[from_label] = {}
            
            if to_label not in transitions[from_label]:
                transitions[from_label][to_label] = 0
            
            transitions[from_label][to_label] += 1
        
        # Get all unique labels
        unique_labels = sorted(set([int(label) for label in labels if not np.isnan(label)]))
        
        # Create the transition matrix
        matrix = np.zeros((len(unique_labels), len(unique_labels)))
        for i, from_label in enumerate(unique_labels):
            if from_label in transitions:
                for j, to_label in enumerate(unique_labels):
                    if to_label in transitions[from_label]:
                        matrix[i, j] = transitions[from_label][to_label]
        
        # Normalize the matrix
        row_sums = matrix.sum(axis=1)
        matrix_norm = np.zeros_like(matrix)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                matrix_norm[i, :] = matrix[i, :] / row_sums[i]
        
        # Plot the matrix
        im = ax.imshow(matrix_norm, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Transition Probability", rotation=-90, va="bottom")
        
        # Set ticks and labels
        label_names = []
        for label_value in unique_labels:
            label_info = next((l for l in self.labels if l['value'] == label_value), None)
            if label_info:
                label_names.append(label_info['name'])
            else:
                label_names.append(f"Label {label_value}")
        
        ax.set_xticks(np.arange(len(unique_labels)))
        ax.set_yticks(np.arange(len(unique_labels)))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if matrix_norm[i, j] > 0:
                    ax.text(j, i, f"{matrix_norm[i, j]:.2f}", ha="center", va="center", 
                           color="white" if matrix_norm[i, j] > 0.5 else "black")
        
        ax.set_title(title)
        ax.set_xlabel("To Label")
        ax.set_ylabel("From Label")
        
        fig.tight_layout()
        
        return fig
    
    @exception_handler
    def visualize_annotations(self, csv_path):
        """
        Visualize the annotations in a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
        """
        annotations = self.load_annotations(csv_path)
        if not annotations:
            show_error_message(f"No annotations found in {csv_path}")
            return
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot the ethogram
        self.plot_ethogram(annotations, title="Ethogram", ax=axs[0, 0])
        
        # Plot the label distribution
        self.plot_label_distribution(annotations, title="Label Distribution", ax=axs[0, 1])
        
        # Plot the label timeline
        self.plot_label_timeline(annotations, title="Label Timeline", ax=axs[1, 0])
        
        # Plot the transition matrix
        self.plot_transition_matrix(annotations, title="Transition Matrix", ax=axs[1, 1])
        
        # Adjust the layout
        fig.tight_layout()
        
        # Show the figure
        plt.show()

class VisualizationGUI:
    """
    GUI for visualizing annotation statistics.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        visualization_tool (VisualizationTool): The visualization tool
    """
    
    def __init__(self, master, config_path='config.json'):
        """
        Initialize the VisualizationGUI.
        
        Args:
            master (tk.Tk): The main Tkinter window
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
        """
        self.logger = setup_logger('visualization_gui')
        self.master = master
        self.master.title("Annotation Visualization")
        self.master.geometry("500x300")
        
        # Create the visualization tool
        self.visualization_tool = VisualizationTool(config_path)
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI elements."""
        # Create a frame for the file selection
        self.file_frame = ttk.LabelFrame(self.master, text="File Selection")
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        # Create a label for the CSV file
        self.file_label = ttk.Label(self.file_frame, text="CSV File:")
        self.file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Create an entry for the CSV file
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_var, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Create a button for browsing the CSV file
        self.browse_button = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Create a frame for the visualization options
        self.options_frame = ttk.LabelFrame(self.master, text="Visualization Options")
        self.options_frame.pack(fill="x", padx=10, pady=10)
        
        # Create checkboxes for the different visualizations
        self.ethogram_var = tk.BooleanVar(value=True)
        self.ethogram_check = ttk.Checkbutton(self.options_frame, text="Ethogram", 
                                             variable=self.ethogram_var)
        self.ethogram_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.distribution_var = tk.BooleanVar(value=True)
        self.distribution_check = ttk.Checkbutton(self.options_frame, text="Label Distribution", 
                                                variable=self.distribution_var)
        self.distribution_check.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.timeline_var = tk.BooleanVar(value=True)
        self.timeline_check = ttk.Checkbutton(self.options_frame, text="Label Timeline", 
                                             variable=self.timeline_var)
        self.timeline_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.transition_var = tk.BooleanVar(value=True)
        self.transition_check = ttk.Checkbutton(self.options_frame, text="Transition Matrix", 
                                              variable=self.transition_var)
        self.transition_check.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Create a frame for the buttons
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(fill="x", padx=10, pady=10)
        
        # Create the visualize button
        self.visualize_button = ttk.Button(self.button_frame, text="Visualize", 
                                          command=self.visualize)
        self.visualize_button.pack(side="right", padx=5)
        
        # Create the cancel button
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", 
                                       command=self.master.destroy)
        self.cancel_button.pack(side="right", padx=5)
    
    @exception_handler
    def browse_file(self):
        """Browse for a CSV file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_var.set(file_path)
    
    @exception_handler
    def visualize(self):
        """Visualize the annotations."""
        csv_path = self.file_var.get()
        if not csv_path:
            show_error_message("Please select a CSV file")
            return
        
        if not os.path.exists(csv_path):
            show_error_message(f"File not found: {csv_path}")
            return
        
        # Load the annotations
        annotations = self.visualization_tool.load_annotations(csv_path)
        if not annotations:
            show_error_message(f"No annotations found in {csv_path}")
            return
        
        # Determine how many plots to create
        num_plots = sum([self.ethogram_var.get(), self.distribution_var.get(), 
                         self.timeline_var.get(), self.transition_var.get()])
        
        if num_plots == 0:
            show_error_message("Please select at least one visualization option")
            return
        
        # Create a figure with subplots
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            axs = [ax]
        elif num_plots == 2:
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            axs = axs.flatten()
        elif num_plots == 3:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            axs = axs.flatten()
        
        # Plot the selected visualizations
        plot_index = 0
        
        if self.ethogram_var.get():
            self.visualization_tool.plot_ethogram(annotations, title="Ethogram", ax=axs[plot_index])
            plot_index += 1
        
        if self.distribution_var.get():
            self.visualization_tool.plot_label_distribution(annotations, title="Label Distribution", 
                                                          ax=axs[plot_index])
            plot_index += 1
        
        if self.timeline_var.get():
            self.visualization_tool.plot_label_timeline(annotations, title="Label Timeline", 
                                                      ax=axs[plot_index])
            plot_index += 1
        
        if self.transition_var.get():
            self.visualization_tool.plot_transition_matrix(annotations, title="Transition Matrix", 
                                                         ax=axs[plot_index])
            plot_index += 1
        
        # Hide any unused subplots
        for i in range(plot_index, len(axs)):
            axs[i].axis('off')
        
        # Adjust the layout
        fig.tight_layout()
        
        # Show the figure
        plt.show()

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = VisualizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
