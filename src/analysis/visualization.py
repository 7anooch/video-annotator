#!/usr/bin/env python3
"""
Enhanced visualization functions for annotation analysis that complement existing tools.

This module provides enhanced visualization functions for annotation data that
complement and extend the functionality in the existing visualization tools.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel
from src.analysis.data_model import AnnotationData

# Set up logger
logger = setup_logger('visualization')

class VisualizationManager:
    """
    Enhanced visualization manager for annotation data that complements existing tools.

    This class provides enhanced visualization functions for annotation data that
    complement and extend the functionality in the existing visualization tools.

    Attributes:
        logger: The logger instance
        plot_adapter: Adapter for the plot.py module
        visualization_adapter: Adapter for the visualization.py module
    """

    def __init__(self):
        """Initialize the VisualizationManager."""
        self.logger = setup_logger('visualization_manager')

        # Set default colors
        self.default_colors = list(mcolors.TABLEAU_COLORS.values())

        # Import the adapters
        from src.analysis.adapters.plot_adapter import PlotAdapter
        from src.analysis.adapters.visualization_adapter import VisualizationAdapter
        self.plot_adapter = PlotAdapter()
        self.visualization_adapter = VisualizationAdapter()

    @exception_handler
    def timeline_plot(self, data: AnnotationData, figsize: Tuple[int, int] = (12, 6),
                     title: str = 'Annotation Timeline', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a timeline plot of annotations.

        Args:
            data (AnnotationData): Annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to 'Annotation Timeline'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No annotations to visualize", ha='center', va='center')
            return fig

        # Get frames and labels
        frames = data.get_frames()
        labels = data.get_labels()

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(labels)}

        # Create arrays for plotting
        x = []
        y = []
        colors = []

        # Assign colors to labels
        label_colors = {}
        for i, label in enumerate(labels):
            label_colors[label] = self.default_colors[i % len(self.default_colors)]

        # Create data for plotting
        for frame in frames:
            annotation = data.get_annotation(frame)
            if annotation and 'label' in annotation:
                label = annotation['label']
                x.append(frame)
                y.append(label_to_index[label])
                colors.append(label_colors[label])

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.scatter(x, y, c=colors, s=50, alpha=0.7)

        # Set the y-axis ticks and labels
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Label')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved timeline plot to {save_path}")

        return fig

    @exception_handler
    def label_distribution_plot(self, data: AnnotationData, figsize: Tuple[int, int] = (10, 6),
                               title: str = 'Label Distribution', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar plot of label distribution.

        Args:
            data (AnnotationData): Annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (str, optional): Plot title. Defaults to 'Label Distribution'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No annotations to visualize", ha='center', va='center')
            return fig

        # Get label counts
        label_counts = data.get_label_counts()

        # Sort labels by count
        labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
        counts = [label_counts[label] for label in labels]

        # Assign colors to labels
        colors = [self.default_colors[i % len(self.default_colors)] for i in range(len(labels))]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        bars = ax.bar(labels, counts, color=colors, alpha=0.7)

        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')

        # Rotate x-axis labels if there are many labels
        if len(labels) > 5:
            plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved label distribution plot to {save_path}")

        return fig

    @exception_handler
    def duration_boxplot(self, data: AnnotationData, figsize: Tuple[int, int] = (12, 6),
                        title: str = 'Label Duration Distribution', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a box plot of label durations.

        Args:
            data (AnnotationData): Annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to 'Label Duration Distribution'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No annotations to visualize", ha='center', va='center')
            return fig

        # Calculate durations
        from src.analysis.statistics import StatisticalAnalysis
        stats = StatisticalAnalysis()
        durations = stats.label_durations(data)

        # Check if there are any durations
        if not durations:
            self.logger.warning("No label durations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No label durations to visualize", ha='center', va='center')
            return fig

        # Sort labels by median duration
        labels = sorted(durations.keys(), key=lambda x: np.median(durations[x]) if durations[x] else 0)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.boxplot([durations[label] for label in labels], labels=labels, vert=False, patch_artist=True)

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('Duration (frames)')
        ax.set_ylabel('Label')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved duration boxplot to {save_path}")

        return fig

    @exception_handler
    def transition_heatmap(self, data: AnnotationData, figsize: Tuple[int, int] = (10, 8),
                          title: str = 'Label Transitions', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap of label transitions.

        Args:
            data (AnnotationData): Annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
            title (str, optional): Plot title. Defaults to 'Label Transitions'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No annotations to visualize", ha='center', va='center')
            return fig

        # Calculate transitions
        from src.analysis.statistics import StatisticalAnalysis
        stats = StatisticalAnalysis()
        transitions = stats.label_transitions(data)

        # Check if there are any transitions
        if not transitions:
            self.logger.warning("No label transitions to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No label transitions to visualize", ha='center', va='center')
            return fig

        # Get all unique labels
        labels = data.get_labels()

        # Create a transition matrix
        matrix = np.zeros((len(labels), len(labels)))

        # Fill the matrix
        for (from_label, to_label), count in transitions.items():
            if from_label in labels and to_label in labels:
                from_idx = labels.index(from_label)
                to_idx = labels.index(to_label)
                matrix[from_idx, to_idx] = count

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the heatmap
        im = ax.imshow(matrix, cmap='YlOrRd')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

        # Set the ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Rotate x-axis labels if there are many labels
        if len(labels) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i, j] > 0:
                    ax.text(j, i, int(matrix[i, j]), ha='center', va='center',
                           color='white' if matrix[i, j] > np.max(matrix) / 2 else 'black')

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('To Label')
        ax.set_ylabel('From Label')

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved transition heatmap to {save_path}")

        return fig

    @exception_handler
    def time_series_plot(self, data: AnnotationData, figsize: Tuple[int, int] = (12, 6),
                        title: str = 'Label Frequency Over Time', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a time series plot of label frequencies.

        Args:
            data (AnnotationData): Annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to 'Label Frequency Over Time'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No annotations to visualize", ha='center', va='center')
            return fig

        # Calculate time series
        from src.analysis.statistics import StatisticalAnalysis
        stats = StatisticalAnalysis()
        time_series = stats.time_series_analysis(data)

        # Check if there are any time series data
        if not time_series or 'windows' not in time_series or 'frequencies' not in time_series:
            self.logger.warning("No time series data to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No time series data to visualize", ha='center', va='center')
            return fig

        # Get windows and frequencies
        windows = time_series['windows']
        frequencies = time_series['frequencies']

        # Check if there are any windows
        if not windows:
            self.logger.warning("No time windows to visualize")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No time windows to visualize", ha='center', va='center')
            return fig

        # Create x-axis values (midpoints of windows)
        x = [(start + end) / 2 for start, end in windows]

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data for each label
        for i, (label, freq) in enumerate(frequencies.items()):
            if freq:  # Check if there are any frequencies for this label
                color = self.default_colors[i % len(self.default_colors)]
                ax.plot(x, freq, marker='o', linestyle='-', label=label, color=color)

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frequency')

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved time series plot to {save_path}")

        return fig

    @exception_handler
    def comparison_plot(self, data1: AnnotationData, data2: AnnotationData, figsize: Tuple[int, int] = (12, 6),
                       title: str = 'Annotation Comparison', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison plot of two annotation sets.

        Args:
            data1 (AnnotationData): First annotation data
            data2 (AnnotationData): Second annotation data
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to 'Annotation Comparison'.
            save_path (Optional[str], optional): Path to save the plot. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure
        """
        if data1.get_annotation_count() == 0 or data2.get_annotation_count() == 0:
            self.logger.warning("Cannot compare empty annotation sets")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Cannot compare empty annotation sets", ha='center', va='center')
            return fig

        # Get frames
        frames1 = data1.get_frames()
        frames2 = data2.get_frames()

        # Get all unique labels
        labels1 = data1.get_labels()
        labels2 = data2.get_labels()
        all_labels = sorted(set(labels1 + labels2))

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(all_labels)}

        # Create arrays for plotting
        x1 = []
        y1 = []
        x2 = []
        y2 = []

        # Create data for plotting
        for frame in frames1:
            annotation = data1.get_annotation(frame)
            if annotation and 'label' in annotation:
                label = annotation['label']
                x1.append(frame)
                y1.append(label_to_index[label])

        for frame in frames2:
            annotation = data2.get_annotation(frame)
            if annotation and 'label' in annotation:
                label = annotation['label']
                x2.append(frame)
                y2.append(label_to_index[label])

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.scatter(x1, y1, marker='o', label='Set 1', alpha=0.7)
        ax.scatter(x2, y2, marker='x', label='Set 2', alpha=0.7)

        # Set the y-axis ticks and labels
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels(all_labels)

        # Set the title and labels
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Label')

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {save_path}")

        return fig

    @exception_handler
    def create_interactive_timeline(self, data: AnnotationData, output_path: str = 'interactive_timeline.html') -> str:
        """
        Create an interactive timeline visualization using Plotly.
        This extends the existing visualization tools with interactive capabilities.

        Args:
            data (AnnotationData): Annotation data
            output_path (str, optional): Path to save the HTML file. Defaults to 'interactive_timeline.html'.

        Returns:
            str: Path to the generated HTML file
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            return ""

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            self.logger.error("Plotly is required for interactive visualizations")
            return ""

        # Get frames and labels
        frames = data.get_frames()
        labels = []
        for frame in frames:
            annotation = data.get_annotation(frame)
            if annotation and 'label' in annotation:
                labels.append(annotation['label'])
            else:
                labels.append("Unknown")

        # Get unique labels
        unique_labels = sorted(set(labels))

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        # Create a color map
        colors = {}
        for i, label in enumerate(unique_labels):
            color_index = i % len(self.default_colors)
            colors[label] = self.default_colors[color_index]

        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Annotation Timeline', 'Label Distribution'),
                           row_heights=[0.7, 0.3])

        # Add timeline trace
        y = [label_to_index[label] for label in labels]
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Timeline'
            ),
            row=1, col=1
        )

        # Add markers for each label
        for label in unique_labels:
            label_frames = [frame for frame, l in zip(frames, labels) if l == label]
            label_y = [label_to_index[label]] * len(label_frames)

            fig.add_trace(
                go.Scatter(
                    x=label_frames,
                    y=label_y,
                    mode='markers',
                    marker=dict(color=colors[label], size=8),
                    name=label
                ),
                row=1, col=1
            )

        # Add label distribution bar chart
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        fig.add_trace(
            go.Bar(
                x=list(label_counts.keys()),
                y=list(label_counts.values()),
                marker_color=[colors[label] for label in label_counts.keys()],
                name='Label Distribution'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title='Interactive Annotation Visualization',
            height=800,
            width=1200,
            hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        # Update y-axis for timeline
        fig.update_yaxes(
            title_text='Label',
            ticktext=unique_labels,
            tickvals=list(range(len(unique_labels))),
            row=1, col=1
        )

        # Update x-axis
        fig.update_xaxes(title_text='Frame', row=2, col=1)

        # Add hover information
        fig.update_traces(
            hovertemplate='Frame: %{x}<br>Label: %{text}',
            text=labels,
            row=1, col=1
        )

        # Save the figure
        fig.write_html(output_path)

        self.logger.info(f"Created interactive timeline visualization at {output_path}")
        return output_path

    @exception_handler
    def create_3d_visualization(self, data: AnnotationData, output_path: str = '3d_visualization.html') -> str:
        """
        Create a 3D visualization of annotation patterns using Plotly.
        This extends the existing visualization tools with 3D capabilities.

        Args:
            data (AnnotationData): Annotation data
            output_path (str, optional): Path to save the HTML file. Defaults to '3d_visualization.html'.

        Returns:
            str: Path to the generated HTML file
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            return ""

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from sklearn.decomposition import PCA
        except ImportError:
            self.logger.error("Plotly and scikit-learn are required for 3D visualizations")
            return ""

        # Get frames and labels
        frames = data.get_frames()
        labels = []
        for frame in frames:
            annotation = data.get_annotation(frame)
            if annotation and 'label' in annotation:
                labels.append(annotation['label'])
            else:
                labels.append("Unknown")

        # Get unique labels
        unique_labels = sorted(set(labels))

        # Create a color map
        colors = {}
        for i, label in enumerate(unique_labels):
            color_index = i % len(self.default_colors)
            colors[label] = self.default_colors[color_index]

        # Create features for 3D visualization
        # We'll use the frame number, label index, and duration as features

        # Calculate durations
        durations = []
        current_label = None
        current_start = None

        for i, (frame, label) in enumerate(zip(frames, labels)):
            # If this is a new label or the first annotation
            if label != current_label or current_start is None:
                # If we were tracking a label, calculate its duration
                if current_label is not None and current_start is not None:
                    duration = frame - current_start
                    durations.extend([duration] * (i - len(durations)))

                # Start tracking the new label
                current_label = label
                current_start = frame

        # Handle the last segment
        if current_label is not None and current_start is not None and frames:
            duration = frames[-1] - current_start + 1
            durations.extend([duration] * (len(frames) - len(durations)))

        # If we still don't have enough durations, fill with zeros
        if len(durations) < len(frames):
            durations.extend([0] * (len(frames) - len(durations)))

        # Create feature matrix
        label_indices = [unique_labels.index(label) for label in labels]
        features = np.column_stack((frames, label_indices, durations))

        # Apply PCA if we have enough data points
        if len(features) > 3:
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features)
        else:
            # If we don't have enough data points, just use the original features
            features_3d = features[:, :3]

        # Create 3D scatter plot
        fig = go.Figure()

        for label in unique_labels:
            # Get indices for this label
            indices = [i for i, l in enumerate(labels) if l == label]

            # Add scatter plot for this label
            fig.add_trace(
                go.Scatter3d(
                    x=features_3d[indices, 0],
                    y=features_3d[indices, 1],
                    z=features_3d[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[label],
                        opacity=0.8
                    ),
                    name=label,
                    text=[f"Frame: {frames[i]}<br>Label: {label}<br>Duration: {durations[i]}" for i in indices],
                    hoverinfo='text'
                )
            )

        # Update layout
        fig.update_layout(
            title='3D Visualization of Annotation Patterns',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3'
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Save the figure
        fig.write_html(output_path)

        self.logger.info(f"Created 3D visualization at {output_path}")
        return output_path

    @exception_handler
    def create_report(self, data: AnnotationData, output_dir: str, prefix: str = 'report') -> List[str]:
        """
        Create a comprehensive report with multiple visualizations.
        This extends the existing reporting capabilities with more visualizations and interactive features.

        Args:
            data (AnnotationData): Annotation data
            output_dir (str): Directory to save the report
            prefix (str, optional): Prefix for output files. Defaults to 'report'.

        Returns:
            List[str]: List of paths to the generated files
        """
        if data.get_annotation_count() == 0:
            self.logger.warning("No annotations to visualize")
            return []

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate plots
        output_files = []

        # Timeline plot
        timeline_path = os.path.join(output_dir, f"{prefix}_timeline.png")
        self.timeline_plot(data, save_path=timeline_path)
        output_files.append(timeline_path)

        # Label distribution plot
        distribution_path = os.path.join(output_dir, f"{prefix}_distribution.png")
        self.label_distribution_plot(data, save_path=distribution_path)
        output_files.append(distribution_path)

        # Duration boxplot
        duration_path = os.path.join(output_dir, f"{prefix}_duration.png")
        self.duration_boxplot(data, save_path=duration_path)
        output_files.append(duration_path)

        # Transition heatmap
        transition_path = os.path.join(output_dir, f"{prefix}_transition.png")
        self.transition_heatmap(data, save_path=transition_path)
        output_files.append(transition_path)

        # Time series plot
        time_series_path = os.path.join(output_dir, f"{prefix}_time_series.png")
        self.time_series_plot(data, save_path=time_series_path)
        output_files.append(time_series_path)

        # Interactive timeline visualization
        try:
            interactive_timeline_path = os.path.join(output_dir, f"{prefix}_interactive_timeline.html")
            self.create_interactive_timeline(data, output_path=interactive_timeline_path)
            output_files.append(interactive_timeline_path)
        except Exception as e:
            self.logger.warning(f"Could not create interactive timeline: {str(e)}")

        # 3D visualization
        try:
            visualization_3d_path = os.path.join(output_dir, f"{prefix}_3d_visualization.html")
            self.create_3d_visualization(data, output_path=visualization_3d_path)
            output_files.append(visualization_3d_path)
        except Exception as e:
            self.logger.warning(f"Could not create 3D visualization: {str(e)}")

        # Generate statistics
        from src.analysis.statistics import StatisticalAnalysis
        stats = StatisticalAnalysis()

        # Basic statistics
        basic_stats = stats.basic_statistics(data)

        # Duration statistics
        duration_stats = stats.duration_statistics(data)

        # Generate HTML report
        html_path = os.path.join(output_dir, f"{prefix}_report.html")

        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Annotation Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .stats {{ margin-bottom: 20px; }}
                    .stats table {{ border-collapse: collapse; width: 100%; }}
                    .stats th, .stats td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .stats th {{ background-color: #f2f2f2; }}
                    .plot {{ margin-bottom: 30px; }}
                    .plot img {{ max-width: 100%; height: auto; }}
                    .interactive {{ margin-bottom: 30px; }}
                    .interactive iframe {{ width: 100%; height: 600px; border: none; }}
                    .button {{ display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; margin-bottom: 10px; }}
                    .button:hover {{ background-color: #45a049; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Enhanced Annotation Analysis Report</h1>

                    <div class="interactive-links">
                        <h2>Interactive Visualizations</h2>
                        <a href="{prefix}_interactive_timeline.html" class="button" target="_blank">Interactive Timeline</a>
                        <a href="{prefix}_3d_visualization.html" class="button" target="_blank">3D Visualization</a>
                    </div>

                    <div class="stats">
                        <h2>Basic Statistics</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Total Annotations</td><td>{basic_stats.get('total_annotations', 0)}</td></tr>
                            <tr><td>Unique Labels</td><td>{basic_stats.get('unique_labels', 0)}</td></tr>
                            <tr><td>Frame Range</td><td>{basic_stats.get('frame_min', 0)} - {basic_stats.get('frame_max', 0)}</td></tr>
                            <tr><td>Total Gaps</td><td>{basic_stats.get('total_gaps', 0)}</td></tr>
                            <tr><td>Total Gap Frames</td><td>{basic_stats.get('total_gap_frames', 0)}</td></tr>
                            <tr><td>Average Gap Size</td><td>{basic_stats.get('avg_gap_size', 0):.2f}</td></tr>
                            <tr><td>Maximum Gap Size</td><td>{basic_stats.get('max_gap_size', 0)}</td></tr>
                        </table>
                    </div>

                    <div class="stats">
                        <h2>Label Counts</h2>
                        <table>
                            <tr><th>Label</th><th>Count</th></tr>
            """)

            # Add label counts
            label_counts = basic_stats.get('label_counts', {})
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"<tr><td>{label}</td><td>{count}</td></tr>\n")

            f.write("""
                        </table>
                    </div>

                    <div class="stats">
                        <h2>Duration Statistics</h2>
                        <table>
                            <tr><th>Label</th><th>Count</th><th>Total Frames</th><th>Min</th><th>Max</th><th>Mean</th><th>Median</th><th>Std</th></tr>
            """)

            # Add duration statistics
            for label, stats_dict in duration_stats.items():
                f.write(f"""
                <tr>
                    <td>{label}</td>
                    <td>{stats_dict.get('count', 0)}</td>
                    <td>{stats_dict.get('total_frames', 0)}</td>
                    <td>{stats_dict.get('min', 0)}</td>
                    <td>{stats_dict.get('max', 0)}</td>
                    <td>{stats_dict.get('mean', 0):.2f}</td>
                    <td>{stats_dict.get('median', 0):.2f}</td>
                    <td>{stats_dict.get('std', 0):.2f}</td>
                </tr>
                """)

            f.write("""
                        </table>
                    </div>

                    <div class="plot">
                        <h2>Timeline Plot</h2>
                        <img src="{prefix}_timeline.png" alt="Timeline Plot">
                    </div>

                    <div class="plot">
                        <h2>Label Distribution</h2>
                        <img src="{prefix}_distribution.png" alt="Label Distribution">
                    </div>

                    <div class="plot">
                        <h2>Duration Boxplot</h2>
                        <img src="{prefix}_duration.png" alt="Duration Boxplot">
                    </div>

                    <div class="plot">
                        <h2>Transition Heatmap</h2>
                        <img src="{prefix}_transition.png" alt="Transition Heatmap">
                    </div>

                    <div class="plot">
                        <h2>Time Series Plot</h2>
                        <img src="{prefix}_time_series.png" alt="Time Series Plot">
                    </div>
                </div>
            </body>
            </html>
            """.format(prefix=prefix))

        output_files.append(html_path)

        self.logger.info(f"Created report with {len(output_files)} files in {output_dir}")
        return output_files
