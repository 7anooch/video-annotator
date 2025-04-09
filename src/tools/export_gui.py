#!/usr/bin/env python3
"""
Export GUI for Video Annotator.

This module provides a GUI for exporting annotations to different formats.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from src.utils.export import AnnotationExporter
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message

class ExportGUI:
    """
    GUI for exporting annotations to different formats.

    Attributes:
        master (tk.Tk): The main Tkinter window
        annotation_path (str): Path to the annotation file
    """

    def __init__(self, master, annotation_path=None):
        """
        Initialize the ExportGUI.

        Args:
            master (tk.Tk): The main Tkinter window
            annotation_path (str, optional): Path to the annotation file. Defaults to None.
        """
        self.logger = setup_logger('export_gui')
        self.master = master
        self.master.title("Export Annotations")
        self.master.geometry("500x400")

        self.annotation_path = annotation_path
        self.annotations = {}

        # Create the exporter
        self.exporter = AnnotationExporter()

        # Set up the UI
        self.setup_ui()

        # Load annotations if path is provided
        if self.annotation_path:
            self.load_annotations()

    def setup_ui(self):
        """Set up the UI elements."""
        # Create a frame for the annotation file
        self.file_frame = ttk.LabelFrame(self.master, text="Annotation File")
        self.file_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the annotation file
        self.file_label = ttk.Label(self.file_frame, text="File:")
        self.file_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the annotation file
        self.file_var = tk.StringVar(value=self.annotation_path or "")
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_var, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create a button for browsing the annotation file
        self.browse_button = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Create a button for loading the annotation file
        self.load_button = ttk.Button(self.file_frame, text="Load", command=self.load_annotations)
        self.load_button.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Create a frame for the export options
        self.export_frame = ttk.LabelFrame(self.master, text="Export Options")
        self.export_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a label for the export format
        self.format_label = ttk.Label(self.export_frame, text="Format:")
        self.format_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create a combobox for the export format
        self.format_var = tk.StringVar(value="CSV")
        self.format_combo = ttk.Combobox(self.export_frame, textvariable=self.format_var,
                                        values=["CSV", "JSON", "TXT", "MATLAB", "Excel"],
                                        state="readonly", width=10)
        self.format_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create a label for the output file
        self.output_label = ttk.Label(self.export_frame, text="Output File:")
        self.output_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Create an entry for the output file
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.export_frame, textvariable=self.output_var, width=40)
        self.output_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        # Create a button for browsing the output file
        self.output_button = ttk.Button(self.export_frame, text="Browse", command=self.browse_output)
        self.output_button.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Create a frame for the annotation info
        self.info_frame = ttk.LabelFrame(self.export_frame, text="Annotation Info")
        self.info_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        # Create a label for the number of annotations
        self.count_label = ttk.Label(self.info_frame, text="Number of Annotations: 0")
        self.count_label.pack(padx=5, pady=5, anchor="w")

        # Create a label for the frame range
        self.range_label = ttk.Label(self.info_frame, text="Frame Range: N/A")
        self.range_label.pack(padx=5, pady=5, anchor="w")

        # Create a label for the label distribution
        self.distribution_label = ttk.Label(self.info_frame, text="Label Distribution: N/A")
        self.distribution_label.pack(padx=5, pady=5, anchor="w")

        # Configure the export frame to expand
        self.export_frame.columnconfigure(1, weight=1)
        self.export_frame.rowconfigure(2, weight=1)

        # Create a frame for the buttons
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(fill="x", padx=10, pady=10)

        # Create the export button
        self.export_button = ttk.Button(self.button_frame, text="Export", command=self.export)
        self.export_button.pack(side="right", padx=5)

        # Create the cancel button
        self.cancel_button = ttk.Button(self.button_frame, text="Cancel", command=self.master.destroy)
        self.cancel_button.pack(side="right", padx=5)

        # Bind the format combobox to update the output file extension
        self.format_combo.bind("<<ComboboxSelected>>", self.update_output_extension)

    @exception_handler
    def browse_file(self):
        """Browse for an annotation file."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_var.set(file_path)
            self.annotation_path = file_path
            self.load_annotations()

    @exception_handler
    def browse_output(self):
        """Browse for an output file."""
        format_type = self.format_var.get().lower()
        file_types = {
            "csv": [("CSV Files", "*.csv")],
            "json": [("JSON Files", "*.json")],
            "txt": [("Text Files", "*.txt")],
            "matlab": [("MATLAB Files", "*.mat")],
            "excel": [("Excel Files", "*.xlsx")]
        }

        file_path = filedialog.asksaveasfilename(filetypes=file_types.get(format_type, [("All Files", "*.*")]))
        if file_path:
            # Add extension if not present
            if not file_path.lower().endswith(f".{format_type}"):
                if format_type == "excel":
                    file_path += ".xlsx"
                else:
                    file_path += f".{format_type}"
            self.output_var.set(file_path)

    @exception_handler
    def update_output_extension(self, event=None):
        """Update the output file extension based on the selected format."""
        format_type = self.format_var.get().lower()
        output_path = self.output_var.get()

        if output_path:
            # Remove the old extension
            base_path = os.path.splitext(output_path)[0]

            # Add the new extension
            if format_type == "excel":
                self.output_var.set(f"{base_path}.xlsx")
            else:
                self.output_var.set(f"{base_path}.{format_type}")

    @exception_handler
    def load_annotations(self):
        """Load annotations from the file."""
        if not self.annotation_path and self.file_var.get():
            self.annotation_path = self.file_var.get()

        if not self.annotation_path or not os.path.exists(self.annotation_path):
            show_error_message("Please select a valid annotation file")
            return

        try:
            # Load the annotations
            df = pd.read_csv(self.annotation_path)
            self.annotations = {row['frame']: row['label'] for _, row in df.iterrows()}

            # Update the annotation info
            self.update_annotation_info()

            # Set the default output file
            base_path = os.path.splitext(self.annotation_path)[0]
            format_type = self.format_var.get().lower()
            if format_type == "excel":
                self.output_var.set(f"{base_path}_export.xlsx")
            else:
                self.output_var.set(f"{base_path}_export.{format_type}")

            self.logger.info(f"Loaded {len(self.annotations)} annotations from {self.annotation_path}")
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            show_error_message(f"Error loading annotations: {str(e)}")

    @exception_handler
    def update_annotation_info(self):
        """Update the annotation info labels."""
        if not self.annotations:
            self.count_label.config(text="Number of Annotations: 0")
            self.range_label.config(text="Frame Range: N/A")
            self.distribution_label.config(text="Label Distribution: N/A")
            return

        # Update the count label
        self.count_label.config(text=f"Number of Annotations: {len(self.annotations)}")

        # Update the range label
        min_frame = min(self.annotations.keys())
        max_frame = max(self.annotations.keys())
        self.range_label.config(text=f"Frame Range: {min_frame} - {max_frame}")

        # Update the distribution label
        label_counts = {}
        for label in self.annotations.values():
            if not pd.isna(label):
                label_int = int(label)
                label_counts[label_int] = label_counts.get(label_int, 0) + 1

        distribution_text = "Label Distribution: "
        for label, count in sorted(label_counts.items()):
            distribution_text += f"Label {label}: {count}, "

        # Remove the trailing comma and space
        distribution_text = distribution_text.rstrip(", ")

        self.distribution_label.config(text=distribution_text)

    @exception_handler
    def export(self):
        """Export the annotations to the selected format."""
        if not self.annotations:
            show_error_message("No annotations to export")
            return

        output_path = self.output_var.get()
        if not output_path:
            show_error_message("Please specify an output file")
            return

        format_type = self.format_var.get().lower()

        # Export the annotations
        success = self.exporter.export(self.annotations, output_path, format_type)

        if success:
            messagebox.showinfo("Success", f"Annotations exported to {output_path}")
        else:
            show_error_message(f"Error exporting annotations to {format_type} format")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = ExportGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
