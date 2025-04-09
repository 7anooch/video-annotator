#!/usr/bin/env python3
"""
Simple UI for the enhanced analysis tools.

This module provides a simple Tkinter-based UI for the enhanced analysis tools.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from src.analysis.data_model import AnnotationData
from src.analysis.statistics import StatisticalAnalysis
from src.analysis.visualization import VisualizationManager

class AnalysisUI:
    """
    Simple UI for the enhanced analysis tools.
    
    This class provides a simple Tkinter-based UI for the enhanced analysis tools.
    """
    
    def __init__(self, root):
        """
        Initialize the AnalysisUI.
        
        Args:
            root (tk.Tk): The root Tkinter window
        """
        self.root = root
        self.root.title("Enhanced Analysis Tools")
        self.root.geometry("800x600")
        
        # Create the data model, statistics, and visualization objects
        self.data = None
        self.stats = StatisticalAnalysis()
        self.viz = VisualizationManager()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the file frame
        self.file_frame = ttk.LabelFrame(self.main_frame, text="File", padding=10)
        self.file_frame.pack(fill=tk.X, pady=5)
        
        # Create the file path entry
        self.file_path_var = tk.StringVar()
        ttk.Label(self.file_frame, text="File Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(self.file_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
        ttk.Button(self.file_frame, text="Load", command=self.load_file).grid(row=0, column=3, sticky=tk.E, padx=5, pady=5)
        
        # Create the analysis frame
        self.analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis", padding=10)
        self.analysis_frame.pack(fill=tk.X, pady=5)
        
        # Create the analysis buttons
        ttk.Button(self.analysis_frame, text="Basic Statistics", command=self.show_basic_statistics).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.analysis_frame, text="Label Transitions", command=self.show_label_transitions).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.analysis_frame, text="Duration Statistics", command=self.show_duration_statistics).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.analysis_frame, text="Advanced Sequence Analysis", command=self.show_advanced_sequence_analysis).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.analysis_frame, text="Time Series Analysis", command=self.show_time_series_analysis).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.analysis_frame, text="Detect Anomalies", command=self.show_anomalies).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Create the visualization frame
        self.visualization_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding=10)
        self.visualization_frame.pack(fill=tk.X, pady=5)
        
        # Create the visualization buttons
        ttk.Button(self.visualization_frame, text="Timeline Plot", command=self.show_timeline_plot).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.visualization_frame, text="Label Distribution Plot", command=self.show_label_distribution_plot).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.visualization_frame, text="Duration Boxplot", command=self.show_duration_boxplot).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.visualization_frame, text="Transition Heatmap", command=self.show_transition_heatmap).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.visualization_frame, text="Time Series Plot", command=self.show_time_series_plot).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(self.visualization_frame, text="Interactive Timeline", command=self.show_interactive_timeline).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Create the report frame
        self.report_frame = ttk.LabelFrame(self.main_frame, text="Report", padding=10)
        self.report_frame.pack(fill=tk.X, pady=5)
        
        # Create the report buttons
        ttk.Button(self.report_frame, text="Generate Report", command=self.generate_report).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Create the output frame
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output", padding=10)
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create the output text
        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, width=80, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Create a scrollbar for the output text
        scrollbar = ttk.Scrollbar(self.output_text, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
    
    def browse_file(self):
        """Browse for a file."""
        file_path = filedialog.askopenfilename(
            title="Select Annotation File",
            filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("Excel Files", "*.xlsx;*.xls"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def load_file(self):
        """Load the selected file."""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        
        try:
            self.data = AnnotationData()
            result = self.data.load_from_file(file_path)
            
            if result:
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Loaded {self.data.get_annotation_count()} annotations from {file_path}\n")
                self.output_text.insert(tk.END, f"Labels: {', '.join(self.data.get_labels())}\n")
                self.output_text.insert(tk.END, f"Frame range: {min(self.data.get_frames())} - {max(self.data.get_frames())}\n")
            else:
                messagebox.showerror("Error", f"Failed to load file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def show_basic_statistics(self):
        """Show basic statistics."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            stats = self.stats.basic_statistics(self.data)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Basic Statistics:\n\n")
            self.output_text.insert(tk.END, f"Total annotations: {stats['total_annotations']}\n")
            self.output_text.insert(tk.END, f"Unique labels: {stats['unique_labels']}\n")
            self.output_text.insert(tk.END, f"Frame range: {stats['frame_min']} - {stats['frame_max']}\n")
            self.output_text.insert(tk.END, f"Total gaps: {stats['total_gaps']}\n")
            self.output_text.insert(tk.END, f"Total gap frames: {stats['total_gap_frames']}\n")
            self.output_text.insert(tk.END, f"Average gap size: {stats['avg_gap_size']:.2f}\n")
            self.output_text.insert(tk.END, f"Maximum gap size: {stats['max_gap_size']}\n\n")
            
            self.output_text.insert(tk.END, "Label counts:\n")
            for label, count in stats['label_counts'].items():
                self.output_text.insert(tk.END, f"  {label}: {count}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating basic statistics: {str(e)}")
    
    def show_label_transitions(self):
        """Show label transitions."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            transitions = self.stats.label_transitions(self.data)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Label Transitions:\n\n")
            
            for (from_label, to_label), count in sorted(transitions.items()):
                self.output_text.insert(tk.END, f"  {from_label} -> {to_label}: {count}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating label transitions: {str(e)}")
    
    def show_duration_statistics(self):
        """Show duration statistics."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            duration_stats = self.stats.duration_statistics(self.data)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Duration Statistics:\n\n")
            
            for label, stats in duration_stats.items():
                self.output_text.insert(tk.END, f"{label}:\n")
                self.output_text.insert(tk.END, f"  Count: {stats['count']}\n")
                self.output_text.insert(tk.END, f"  Total frames: {stats['total_frames']}\n")
                self.output_text.insert(tk.END, f"  Min: {stats['min']}\n")
                self.output_text.insert(tk.END, f"  Max: {stats['max']}\n")
                self.output_text.insert(tk.END, f"  Mean: {stats['mean']:.2f}\n")
                self.output_text.insert(tk.END, f"  Median: {stats['median']:.2f}\n")
                self.output_text.insert(tk.END, f"  Std: {stats['std']:.2f}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating duration statistics: {str(e)}")
    
    def show_advanced_sequence_analysis(self):
        """Show advanced sequence analysis."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            sequence_analysis = self.stats.advanced_sequence_analysis(self.data)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Advanced Sequence Analysis:\n\n")
            
            # Show transition probabilities
            self.output_text.insert(tk.END, "Transition Probabilities:\n")
            for (from_label, to_label), prob in sorted(sequence_analysis['transition_probabilities'].items()):
                self.output_text.insert(tk.END, f"  {from_label} -> {to_label}: {prob:.2f}\n")
            
            # Show complexity measures
            self.output_text.insert(tk.END, "\nComplexity Measures:\n")
            complexity = sequence_analysis['complexity']
            if complexity:
                self.output_text.insert(tk.END, f"  Label entropy: {complexity.get('label_entropy', 0):.2f}\n")
                self.output_text.insert(tk.END, f"  Bigram entropy: {complexity.get('bigram_entropy', 0):.2f}\n")
                self.output_text.insert(tk.END, f"  Conditional entropy: {complexity.get('conditional_entropy', 0):.2f}\n")
            
            # Show recurring patterns
            self.output_text.insert(tk.END, "\nRecurring Patterns:\n")
            for pattern in sequence_analysis['recurring_patterns']:
                self.output_text.insert(tk.END, f"  Pattern: {pattern['pattern']}\n")
                self.output_text.insert(tk.END, f"    Length: {pattern['length']}\n")
                self.output_text.insert(tk.END, f"    Occurrences: {pattern['occurrences']}\n")
                self.output_text.insert(tk.END, f"    Positions: {pattern['positions']}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error performing advanced sequence analysis: {str(e)}")
    
    def show_time_series_analysis(self):
        """Show time series analysis."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            time_series = self.stats.time_series_analysis(self.data)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Time Series Analysis:\n\n")
            
            # Show windows
            self.output_text.insert(tk.END, "Windows:\n")
            for i, (start, end) in enumerate(time_series['windows']):
                self.output_text.insert(tk.END, f"  Window {i+1}: {start} - {end}\n")
            
            # Show trends
            self.output_text.insert(tk.END, "\nTrends:\n")
            for label, trend in time_series['trends'].items():
                self.output_text.insert(tk.END, f"  {label}:\n")
                self.output_text.insert(tk.END, f"    Slope: {trend['slope']:.2f}\n")
                self.output_text.insert(tk.END, f"    Intercept: {trend['intercept']:.2f}\n")
                self.output_text.insert(tk.END, f"    Increasing: {trend['increasing']}\n")
                self.output_text.insert(tk.END, f"    Decreasing: {trend['decreasing']}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error performing time series analysis: {str(e)}")
    
    def show_anomalies(self):
        """Show anomalies."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            # Create a dialog to select the anomaly detection method
            method_dialog = tk.Toplevel(self.root)
            method_dialog.title("Select Anomaly Detection Method")
            method_dialog.geometry("300x150")
            method_dialog.transient(self.root)
            method_dialog.grab_set()
            
            method_var = tk.StringVar(value="zscore")
            threshold_var = tk.DoubleVar(value=3.0)
            
            ttk.Label(method_dialog, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Radiobutton(method_dialog, text="Z-score", variable=method_var, value="zscore").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
            ttk.Radiobutton(method_dialog, text="IQR", variable=method_var, value="iqr").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
            ttk.Radiobutton(method_dialog, text="Isolation Forest", variable=method_var, value="isolation_forest").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
            
            ttk.Label(method_dialog, text="Threshold:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
            ttk.Entry(method_dialog, textvariable=threshold_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
            
            def on_ok():
                method = method_var.get()
                threshold = threshold_var.get()
                method_dialog.destroy()
                
                try:
                    anomalies = self.stats.detect_anomalies(self.data, method=method, threshold=threshold)
                    
                    self.output_text.delete(1.0, tk.END)
                    self.output_text.insert(tk.END, f"Anomalies ({method} method, threshold={threshold}):\n\n")
                    
                    if not anomalies:
                        self.output_text.insert(tk.END, "No anomalies detected.\n")
                    else:
                        for frame, anomaly in sorted(anomalies.items()):
                            self.output_text.insert(tk.END, f"Frame {frame}:\n")
                            self.output_text.insert(tk.END, f"  Label: {anomaly['label']}\n")
                            self.output_text.insert(tk.END, f"  Duration: {anomaly['duration']}\n")
                            
                            if method == "zscore":
                                self.output_text.insert(tk.END, f"  Z-score: {anomaly['zscore']:.2f}\n")
                                self.output_text.insert(tk.END, f"  Mean duration: {anomaly['mean_duration']:.2f}\n")
                                self.output_text.insert(tk.END, f"  Std duration: {anomaly['std_duration']:.2f}\n")
                            elif method == "iqr":
                                self.output_text.insert(tk.END, f"  Q1: {anomaly['q1']:.2f}\n")
                                self.output_text.insert(tk.END, f"  Q3: {anomaly['q3']:.2f}\n")
                                self.output_text.insert(tk.END, f"  IQR: {anomaly['iqr']:.2f}\n")
                                self.output_text.insert(tk.END, f"  Lower bound: {anomaly['lower_bound']:.2f}\n")
                                self.output_text.insert(tk.END, f"  Upper bound: {anomaly['upper_bound']:.2f}\n")
                            elif method == "isolation_forest":
                                self.output_text.insert(tk.END, f"  Anomaly score: {anomaly['anomaly_score']:.2f}\n")
                            
                            self.output_text.insert(tk.END, "\n")
                except Exception as e:
                    messagebox.showerror("Error", f"Error detecting anomalies: {str(e)}")
            
            ttk.Button(method_dialog, text="OK", command=on_ok).grid(row=4, column=0, columnspan=2, pady=10)
            
            method_dialog.wait_window()
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting anomalies: {str(e)}")
    
    def show_timeline_plot(self):
        """Show timeline plot."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            fig = self.viz.timeline_plot(self.data)
            fig.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error creating timeline plot: {str(e)}")
    
    def show_label_distribution_plot(self):
        """Show label distribution plot."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            fig = self.viz.label_distribution_plot(self.data)
            fig.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error creating label distribution plot: {str(e)}")
    
    def show_duration_boxplot(self):
        """Show duration boxplot."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            fig = self.viz.duration_boxplot(self.data)
            fig.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error creating duration boxplot: {str(e)}")
    
    def show_transition_heatmap(self):
        """Show transition heatmap."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            fig = self.viz.transition_heatmap(self.data)
            fig.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error creating transition heatmap: {str(e)}")
    
    def show_time_series_plot(self):
        """Show time series plot."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            fig = self.viz.time_series_plot(self.data)
            fig.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error creating time series plot: {str(e)}")
    
    def show_interactive_timeline(self):
        """Show interactive timeline."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            # Create a temporary file for the interactive timeline
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                temp_html = f.name
            
            # Create the interactive timeline
            output_path = self.viz.create_interactive_timeline(self.data, output_path=temp_html)
            
            # Open the interactive timeline in the default browser
            import webbrowser
            webbrowser.open(output_path)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Interactive timeline saved to: {output_path}\n")
            self.output_text.insert(tk.END, "The interactive timeline has been opened in your default browser.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating interactive timeline: {str(e)}")
    
    def generate_report(self):
        """Generate a comprehensive report."""
        if not self.data:
            messagebox.showerror("Error", "Please load a file first.")
            return
        
        try:
            # Ask for the output directory
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if not output_dir:
                return
            
            # Generate the report
            output_files = self.viz.create_report(self.data, output_dir=output_dir)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Report generated in: {output_dir}\n\n")
            self.output_text.insert(tk.END, "Generated files:\n")
            for file_path in output_files:
                self.output_text.insert(tk.END, f"  {os.path.basename(file_path)}\n")
            
            # Ask if the user wants to open the HTML report
            html_report = next((f for f in output_files if f.endswith('_report.html')), None)
            if html_report:
                if messagebox.askyesno("Open Report", "Do you want to open the HTML report in your default browser?"):
                    import webbrowser
                    webbrowser.open(html_report)
        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {str(e)}")

def main():
    """Run the AnalysisUI."""
    root = tk.Tk()
    app = AnalysisUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
