#!/usr/bin/env python3
"""
Performance profiling tool for Video Annotator.

This module provides tools for profiling the performance of the Video Annotator application.
"""

import os
import time
import cProfile
import pstats
import io
import psutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.core.video_player import VideoPlayer
from src.core.annotation_manager import AnnotationManager
from src.core.config import Config

class PerformanceProfiler:
    """
    Tool for profiling the performance of the Video Annotator application.

    Attributes:
        logger: The logger instance
        process: The current process for memory tracking
    """

    def __init__(self):
        """Initialize the PerformanceProfiler."""
        self.logger = setup_logger('performance_profiler')
        self.process = psutil.Process(os.getpid())

    @exception_handler
    def profile_video_loading(self, video_path, num_frames=100):
        """
        Profile the performance of video loading.

        Args:
            video_path (str): Path to the video file
            num_frames (int, optional): Number of frames to load. Defaults to 100.

        Returns:
            dict: Performance metrics
        """
        self.logger.info(f"Profiling video loading for {video_path}")

        # Create a video player
        video_player = VideoPlayer(video_path)

        # Measure the time to load frames
        frame_times = []
        for i in range(min(num_frames, video_player.total_frames)):
            start_time = time.time()
            frame = video_player.load_frame(i)
            end_time = time.time()
            frame_times.append(end_time - start_time)

        # Calculate statistics
        metrics = {
            'avg_frame_time': np.mean(frame_times),
            'min_frame_time': np.min(frame_times),
            'max_frame_time': np.max(frame_times),
            'std_frame_time': np.std(frame_times),
            'total_time': sum(frame_times),
            'fps': 1.0 / np.mean(frame_times) if np.mean(frame_times) > 0 else 0,
            'frame_times': frame_times
        }

        self.logger.info(f"Video loading metrics: {metrics}")
        return metrics

    @exception_handler
    def profile_annotation_loading(self, annotation_path):
        """
        Profile the performance of annotation loading.

        Args:
            annotation_path (str): Path to the annotation file

        Returns:
            dict: Performance metrics
        """
        self.logger.info(f"Profiling annotation loading for {annotation_path}")

        # Measure the time to load annotations
        start_time = time.time()
        annotation_manager = AnnotationManager(annotation_path)
        end_time = time.time()

        # Calculate statistics
        metrics = {
            'load_time': end_time - start_time,
            'num_annotations': len(annotation_manager.annotations)
        }

        self.logger.info(f"Annotation loading metrics: {metrics}")
        return metrics

    @exception_handler
    def profile_annotation_saving(self, annotation_path, num_annotations=1000):
        """
        Profile the performance of annotation saving.

        Args:
            annotation_path (str): Path to the annotation file
            num_annotations (int, optional): Number of annotations to save. Defaults to 1000.

        Returns:
            dict: Performance metrics
        """
        self.logger.info(f"Profiling annotation saving for {annotation_path}")

        # Create a temporary annotation manager
        annotation_manager = AnnotationManager(annotation_path)

        # Create dummy annotations
        for i in range(num_annotations):
            annotation_manager.annotations[i] = i % 3

        # Measure the time to save annotations
        start_time = time.time()
        annotation_manager.save_annotations()
        end_time = time.time()

        # Calculate statistics
        metrics = {
            'save_time': end_time - start_time,
            'num_annotations': num_annotations
        }

        self.logger.info(f"Annotation saving metrics: {metrics}")
        return metrics

    @exception_handler
    def profile_function(self, func, *args, **kwargs):
        """
        Profile a specific function.

        Args:
            func (function): The function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            tuple: (result, profile_stats)
        """
        self.logger.info(f"Profiling function {func.__name__}")

        # Create a profiler
        profiler = cProfile.Profile()

        # Start profiling
        profiler.enable()

        # Call the function
        result = func(*args, **kwargs)

        # Stop profiling
        profiler.disable()

        # Get the profile stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()

        self.logger.info(f"Profile stats for {func.__name__}:\n{s.getvalue()}")
        return result, s.getvalue()

    @exception_handler
    def plot_frame_times(self, frame_times, title="Frame Loading Times"):
        """
        Plot the frame loading times.

        Args:
            frame_times (list): List of frame loading times
            title (str, optional): Title of the plot. Defaults to "Frame Loading Times".

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the frame times
        ax.plot(frame_times)

        # Add a horizontal line for the average
        ax.axhline(y=np.mean(frame_times), color='r', linestyle='--', label=f'Average: {np.mean(frame_times):.4f}s')

        # Add labels and title
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Loading Time (s)")
        ax.set_title(title)
        ax.legend()

        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig

    @exception_handler
    def plot_frame_time_histogram(self, frame_times, title="Frame Loading Time Distribution"):
        """
        Plot a histogram of frame loading times.

        Args:
            frame_times (list): List of frame loading times
            title (str, optional): Title of the plot. Defaults to "Frame Loading Time Distribution".

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the histogram
        ax.hist(frame_times, bins=20, alpha=0.7)

        # Add a vertical line for the average
        ax.axvline(x=np.mean(frame_times), color='r', linestyle='--', label=f'Average: {np.mean(frame_times):.4f}s')

        # Add labels and title
        ax.set_xlabel("Loading Time (s)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()

        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig

    @exception_handler
    def track_memory_usage(self, interval=1.0, duration=10.0):
        """
        Track memory usage over time.

        Args:
            interval (float, optional): Interval between measurements in seconds. Defaults to 1.0.
            duration (float, optional): Duration of tracking in seconds. Defaults to 10.0.

        Returns:
            dict: Memory usage statistics
        """
        self.logger.info(f"Tracking memory usage for {duration} seconds with interval {interval} seconds")

        # Initialize tracking data
        timestamps = []
        memory_usage = []

        # Track memory usage
        start_time = time.time()
        while time.time() - start_time < duration:
            # Get memory info
            mem_info = self.process.memory_info()

            # Record timestamp and memory usage
            timestamps.append(time.time() - start_time)
            memory_usage.append(mem_info.rss)  # Resident Set Size in bytes

            # Sleep for the specified interval
            time.sleep(interval)

        # Calculate statistics
        memory_mb = [m / (1024 * 1024) for m in memory_usage]  # Convert to MB
        metrics = {
            'timestamps': timestamps,
            'memory_usage': memory_usage,
            'memory_mb': memory_mb,
            'min_memory_mb': min(memory_mb),
            'max_memory_mb': max(memory_mb),
            'avg_memory_mb': np.mean(memory_mb),
            'start_memory_mb': memory_mb[0],
            'end_memory_mb': memory_mb[-1],
            'diff_memory_mb': memory_mb[-1] - memory_mb[0]
        }

        self.logger.info(f"Memory usage metrics: {metrics}")
        return metrics

    @exception_handler
    def plot_memory_usage(self, memory_metrics, title="Memory Usage Over Time"):
        """
        Plot memory usage over time.

        Args:
            memory_metrics (dict): Memory usage metrics from track_memory_usage
            title (str, optional): Title of the plot. Defaults to "Memory Usage Over Time".

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot memory usage
        ax.plot(memory_metrics['timestamps'], memory_metrics['memory_mb'])

        # Add a horizontal line for the average
        ax.axhline(y=memory_metrics['avg_memory_mb'], color='r', linestyle='--',
                  label=f'Average: {memory_metrics["avg_memory_mb"]:.2f} MB')

        # Add labels and title
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title(title)
        ax.legend()

        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig

    @exception_handler
    def generate_performance_report(self, video_path, annotation_path, output_path=None):
        """
        Generate a comprehensive performance report.

        Args:
            video_path (str): Path to the video file
            annotation_path (str): Path to the annotation file
            output_path (str, optional): Path to save the report. Defaults to None.

        Returns:
            dict: Performance metrics
        """
        self.logger.info(f"Generating performance report for {video_path} and {annotation_path}")

        # Profile video loading
        video_metrics = self.profile_video_loading(video_path)

        # Profile annotation loading
        annotation_loading_metrics = self.profile_annotation_loading(annotation_path)

        # Profile annotation saving
        annotation_saving_metrics = self.profile_annotation_saving(annotation_path)

        # Track memory usage
        memory_metrics = self.track_memory_usage(interval=0.5, duration=5.0)

        # Combine metrics
        metrics = {
            'video': video_metrics,
            'annotation_loading': annotation_loading_metrics,
            'annotation_saving': annotation_saving_metrics,
            'memory': memory_metrics
        }

        # Generate plots
        fig1 = self.plot_frame_times(video_metrics['frame_times'])
        fig2 = self.plot_frame_time_histogram(video_metrics['frame_times'])
        fig3 = self.plot_memory_usage(memory_metrics)

        # Save the report if output_path is provided
        if output_path:
            # Save the metrics as CSV
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Average Frame Time (s)',
                    'Minimum Frame Time (s)',
                    'Maximum Frame Time (s)',
                    'Standard Deviation of Frame Time (s)',
                    'Total Frame Loading Time (s)',
                    'Frames Per Second',
                    'Annotation Loading Time (s)',
                    'Number of Annotations',
                    'Annotation Saving Time (s)',
                    'Minimum Memory Usage (MB)',
                    'Maximum Memory Usage (MB)',
                    'Average Memory Usage (MB)',
                    'Memory Usage at Start (MB)',
                    'Memory Usage at End (MB)',
                    'Memory Usage Difference (MB)'
                ],
                'Value': [
                    video_metrics['avg_frame_time'],
                    video_metrics['min_frame_time'],
                    video_metrics['max_frame_time'],
                    video_metrics['std_frame_time'],
                    video_metrics['total_time'],
                    video_metrics['fps'],
                    annotation_loading_metrics['load_time'],
                    annotation_loading_metrics['num_annotations'],
                    annotation_saving_metrics['save_time'],
                    memory_metrics['min_memory_mb'],
                    memory_metrics['max_memory_mb'],
                    memory_metrics['avg_memory_mb'],
                    memory_metrics['start_memory_mb'],
                    memory_metrics['end_memory_mb'],
                    memory_metrics['diff_memory_mb']
                ]
            })

            metrics_df.to_csv(output_path, index=False)

            # Save the plots
            fig1.savefig(os.path.splitext(output_path)[0] + '_frame_times.png')
            fig2.savefig(os.path.splitext(output_path)[0] + '_frame_time_histogram.png')
            fig3.savefig(os.path.splitext(output_path)[0] + '_memory_usage.png')

            self.logger.info(f"Performance report saved to {output_path}")

        # Show the plots
        plt.show()

        return metrics

class PerformanceProfilerGUI:
    """
    GUI for the performance profiler.

    Attributes:
        master (tk.Tk): The main Tkinter window
        profiler (PerformanceProfiler): The performance profiler
    """

    def __init__(self, master):
        """
        Initialize the PerformanceProfilerGUI.

        Args:
            master (tk.Tk): The main Tkinter window
        """
        self.logger = setup_logger('performance_profiler_gui')
        self.master = master
        self.master.title("Performance Profiler")
        self.master.geometry("600x400")

        # Create the profiler
        self.profiler = PerformanceProfiler()

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI elements."""
        # Create a frame for the video file
        self.video_frame = ttk.LabelFrame(self.master, text="Video File")
        self.video_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the video file
        self.video_label = ttk.Label(self.video_frame, text="Video File:")
        self.video_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the video file
        self.video_var = tk.StringVar()
        self.video_entry = ttk.Entry(self.video_frame, textvariable=self.video_var, width=40)
        self.video_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create a button for browsing the video file
        self.video_button = ttk.Button(self.video_frame, text="Browse", command=self.browse_video)
        self.video_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Create a frame for the annotation file
        self.annotation_frame = ttk.LabelFrame(self.master, text="Annotation File")
        self.annotation_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the annotation file
        self.annotation_label = ttk.Label(self.annotation_frame, text="Annotation File:")
        self.annotation_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the annotation file
        self.annotation_var = tk.StringVar()
        self.annotation_entry = ttk.Entry(self.annotation_frame, textvariable=self.annotation_var, width=40)
        self.annotation_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create a button for browsing the annotation file
        self.annotation_button = ttk.Button(self.annotation_frame, text="Browse", command=self.browse_annotation)
        self.annotation_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Create a frame for the profiling options
        self.options_frame = ttk.LabelFrame(self.master, text="Profiling Options")
        self.options_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the number of frames
        self.frames_label = ttk.Label(self.options_frame, text="Number of Frames:")
        self.frames_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the number of frames
        self.frames_var = tk.StringVar(value="100")
        self.frames_entry = ttk.Entry(self.options_frame, textvariable=self.frames_var, width=10)
        self.frames_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create a label for the output file
        self.output_label = ttk.Label(self.options_frame, text="Output File:")
        self.output_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the output file
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.options_frame, textvariable=self.output_var, width=40)
        self.output_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        # Create a button for browsing the output file
        self.output_button = ttk.Button(self.options_frame, text="Browse", command=self.browse_output)
        self.output_button.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Create a frame for memory profiling
        self.memory_frame = ttk.LabelFrame(self.master, text="Memory Profiling")
        self.memory_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the duration
        self.duration_label = ttk.Label(self.memory_frame, text="Duration (s):")
        self.duration_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the duration
        self.duration_var = tk.StringVar(value="10.0")
        self.duration_entry = ttk.Entry(self.memory_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=0, column=1, padx=5, pady=5)

        # Create a label for the interval
        self.interval_label = ttk.Label(self.memory_frame, text="Interval (s):")
        self.interval_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Create an entry for the interval
        self.interval_var = tk.StringVar(value="0.5")
        self.interval_entry = ttk.Entry(self.memory_frame, textvariable=self.interval_var, width=10)
        self.interval_entry.grid(row=0, column=3, padx=5, pady=5)

        # Create a button for memory profiling
        self.memory_button = ttk.Button(self.memory_frame, text="Profile Memory", command=self.profile_memory)
        self.memory_button.grid(row=0, column=4, padx=5, pady=5)

        # Create a frame for the buttons
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(fill="x", padx=10, pady=10)

        # Create the profile button
        self.profile_button = ttk.Button(self.button_frame, text="Profile", command=self.profile)
        self.profile_button.pack(side="right", padx=5)

        # Create the cancel button
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", command=self.master.destroy)
        self.cancel_button.pack(side="right", padx=5)

    @exception_handler
    def browse_video(self):
        """Browse for a video file."""
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if file_path:
            self.video_var.set(file_path)

            # Set a default output file
            if not self.output_var.get():
                output_path = os.path.splitext(file_path)[0] + "_performance.csv"
                self.output_var.set(output_path)

    @exception_handler
    def browse_annotation(self):
        """Browse for an annotation file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.annotation_var.set(file_path)

    @exception_handler
    def browse_output(self):
        """Browse for an output file."""
        file_path = filedialog.asksaveasfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            self.output_var.set(file_path)

    @exception_handler
    def profile_memory(self):
        """Profile memory usage."""
        try:
            duration = float(self.duration_var.get())
            interval = float(self.interval_var.get())
        except ValueError:
            show_error_message("Duration and interval must be numbers")
            return

        # Track memory usage
        memory_metrics = self.profiler.track_memory_usage(interval=interval, duration=duration)

        # Plot memory usage
        fig = self.profiler.plot_memory_usage(memory_metrics)
        plt.show()

        # Show memory metrics in a message box
        message = f"Memory Usage Metrics:\n"
        message += f"Minimum: {memory_metrics['min_memory_mb']:.2f} MB\n"
        message += f"Maximum: {memory_metrics['max_memory_mb']:.2f} MB\n"
        message += f"Average: {memory_metrics['avg_memory_mb']:.2f} MB\n"
        message += f"Start: {memory_metrics['start_memory_mb']:.2f} MB\n"
        message += f"End: {memory_metrics['end_memory_mb']:.2f} MB\n"
        message += f"Difference: {memory_metrics['diff_memory_mb']:.2f} MB"

        messagebox.showinfo("Memory Usage Metrics", message)

    @exception_handler
    def profile(self):
        """Profile the application."""
        video_path = self.video_var.get()
        annotation_path = self.annotation_var.get()
        output_path = self.output_var.get()

        if not video_path:
            show_error_message("Please select a video file")
            return

        if not annotation_path:
            show_error_message("Please select an annotation file")
            return

        try:
            num_frames = int(self.frames_var.get())
        except ValueError:
            show_error_message("Number of frames must be an integer")
            return

        # Show a progress dialog
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Profiling")
        progress_window.geometry("300x100")

        progress_label = ttk.Label(progress_window, text="Profiling in progress...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(fill="x", padx=10, pady=10)
        progress_bar.start()

        # Update the UI
        self.master.update()

        try:
            # Generate the performance report
            self.profiler.generate_performance_report(video_path, annotation_path, output_path)

            # Close the progress dialog
            progress_window.destroy()

            # Show a success message
            messagebox.showinfo("Success", "Performance profiling completed successfully")
        except Exception as e:
            # Close the progress dialog
            progress_window.destroy()

            # Show an error message
            show_error_message(f"Error during profiling: {str(e)}")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = PerformanceProfilerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
