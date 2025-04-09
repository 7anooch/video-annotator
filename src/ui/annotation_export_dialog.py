#!/usr/bin/env python3
"""
Annotation export dialog for the Video Annotator.

This module provides a dialog for exporting annotations to different formats.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.utils.annotation_exporter import AnnotationExporter

class AnnotationExportDialog(tk.Toplevel):
    """
    Dialog for exporting annotations to different formats.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        annotations (Dict[int, Dict[str, Any]]): The annotations to export
        theme_manager: The theme manager
    """
    
    def __init__(self, master: tk.Tk, annotations: Dict[int, Dict[str, Any]], theme_manager=None):
        """
        Initialize the AnnotationExportDialog.
        
        Args:
            master (tk.Tk): The main Tkinter window
            annotations (Dict[int, Dict[str, Any]]): The annotations to export
            theme_manager: The theme manager. Defaults to None.
        """
        self.logger = setup_logger('annotation_export_dialog')
        
        # Initialize the toplevel window
        super().__init__(master)
        self.title("Export Annotations")
        self.geometry("500x300")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()
        
        # Store references
        self.master = master
        self.annotations = annotations
        self.theme_manager = theme_manager
        self.exporter = AnnotationExporter()
        
        # Apply theme if available
        if theme_manager:
            bg_color = theme_manager.get_theme_color('bg')
            fg_color = theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
        
        # Initialize variables
        self.output_path = tk.StringVar(value="")
        self.format = tk.StringVar(value="csv")
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI elements."""
        # Create the main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the output file frame
        output_frame = ttk.LabelFrame(main_frame, text="Output File")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the output file entry
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=40)
        output_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the browse button
        browse_button = ttk.Button(output_frame, text="Browse", command=self.browse_output_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Create the format frame
        format_frame = ttk.LabelFrame(main_frame, text="Export Format")
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the format combobox
        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        format_combo = ttk.Combobox(format_frame, textvariable=self.format, 
                                   values=self.exporter.get_supported_formats(), width=15)
        format_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create the format description
        self.format_description = ttk.Label(format_frame, text="", wraplength=400)
        self.format_description.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        # Update format description when format changes
        format_combo.bind("<<ComboboxSelected>>", self.update_format_description)
        
        # Set initial format description
        self.update_format_description()
        
        # Create the annotation info frame
        info_frame = ttk.LabelFrame(main_frame, text="Annotation Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the annotation info label
        annotation_count = len(self.annotations)
        ttk.Label(info_frame, text=f"Number of annotations: {annotation_count}").pack(padx=5, pady=5, anchor=tk.W)
        
        # Create the buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the export button
        export_button = ttk.Button(buttons_frame, text="Export", command=self.export_annotations)
        export_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    @exception_handler
    def browse_output_file(self):
        """Browse for an output file."""
        # Get the current format
        format = self.format.get()
        
        # Create a file filter
        file_filter = [(f"{format.upper()} Files", f"*.{format}")]
        
        # Open a file dialog
        file_path = filedialog.asksaveasfilename(filetypes=file_filter, defaultextension=f".{format}")
        
        if file_path:
            self.output_path.set(file_path)
    
    @exception_handler
    def update_format_description(self, event=None):
        """Update the format description."""
        # Get the current format
        format = self.format.get()
        
        # Set the description based on the format
        if format == 'csv':
            description = "CSV (Comma-Separated Values) format. Simple text format that can be opened in Excel or other spreadsheet software."
        elif format == 'json':
            description = "JSON (JavaScript Object Notation) format. Structured text format that can be parsed by many programming languages."
        elif format == 'excel':
            description = "Excel format. Can be opened directly in Microsoft Excel or other compatible spreadsheet software."
        elif format == 'matlab':
            description = "MATLAB format. Can be loaded directly into MATLAB for analysis."
        elif format == 'numpy':
            description = "NumPy format. Can be loaded directly into Python using NumPy for analysis."
        else:
            description = ""
        
        # Update the description label
        self.format_description.config(text=description)
    
    @exception_handler
    def export_annotations(self):
        """Export the annotations."""
        # Get the output path
        output_path = self.output_path.get()
        
        if not output_path:
            show_error_message("Please specify an output file")
            return
        
        # Get the format
        format = self.format.get()
        
        # Export the annotations
        success = self.exporter.export_annotations(self.annotations, output_path, format)
        
        if not success:
            show_error_message("Failed to export annotations")
            return
        
        # Show a success message
        messagebox.showinfo("Success", f"Annotations exported to {output_path}")
        
        # Close the dialog
        self.destroy()
    
    def update_theme(self):
        """Update the dialog theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('bg')
            fg_color = self.theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
