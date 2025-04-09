#!/usr/bin/env python3
"""
Annotation import dialog for the Video Annotator.

This module provides a dialog for importing annotations from different formats.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Callable
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.utils.annotation_importer import AnnotationImporter

class AnnotationImportDialog(tk.Toplevel):
    """
    Dialog for importing annotations from different formats.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        callback (Callable[[Dict[int, Dict[str, Any]]], None]): Callback function to handle imported annotations
        theme_manager: The theme manager
    """
    
    def __init__(self, master: tk.Tk, callback: Callable[[Dict[int, Dict[str, Any]]], None], theme_manager=None):
        """
        Initialize the AnnotationImportDialog.
        
        Args:
            master (tk.Tk): The main Tkinter window
            callback (Callable[[Dict[int, Dict[str, Any]]], None]): Callback function to handle imported annotations
            theme_manager: The theme manager. Defaults to None.
        """
        self.logger = setup_logger('annotation_import_dialog')
        
        # Initialize the toplevel window
        super().__init__(master)
        self.title("Import Annotations")
        self.geometry("500x300")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()
        
        # Store references
        self.master = master
        self.callback = callback
        self.theme_manager = theme_manager
        self.importer = AnnotationImporter()
        
        # Apply theme if available
        if theme_manager:
            bg_color = theme_manager.get_theme_color('bg')
            fg_color = theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
        
        # Initialize variables
        self.input_path = tk.StringVar(value="")
        self.format = tk.StringVar(value="csv")
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI elements."""
        # Create the main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the input file frame
        input_frame = ttk.LabelFrame(main_frame, text="Input File")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the input file entry
        ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        input_entry = ttk.Entry(input_frame, textvariable=self.input_path, width=40)
        input_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the browse button
        browse_button = ttk.Button(input_frame, text="Browse", command=self.browse_input_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Create the format frame
        format_frame = ttk.LabelFrame(main_frame, text="Import Format")
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the format combobox
        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        format_combo = ttk.Combobox(format_frame, textvariable=self.format, 
                                   values=self.importer.get_supported_formats(), width=15)
        format_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create the format description
        self.format_description = ttk.Label(format_frame, text="", wraplength=400)
        self.format_description.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        # Update format description when format changes
        format_combo.bind("<<ComboboxSelected>>", self.update_format_description)
        
        # Set initial format description
        self.update_format_description()
        
        # Create the buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the import button
        import_button = ttk.Button(buttons_frame, text="Import", command=self.import_annotations)
        import_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    @exception_handler
    def browse_input_file(self):
        """Browse for an input file."""
        # Get the current format
        format = self.format.get()
        
        # Create a file filter
        file_filter = [(f"{format.upper()} Files", f"*.{format}")]
        
        # Open a file dialog
        file_path = filedialog.askopenfilename(filetypes=file_filter)
        
        if file_path:
            self.input_path.set(file_path)
    
    @exception_handler
    def update_format_description(self, event=None):
        """Update the format description."""
        # Get the current format
        format = self.format.get()
        
        # Set the description based on the format
        if format == 'csv':
            description = "CSV (Comma-Separated Values) format. Simple text format that can be exported from Excel or other spreadsheet software."
        elif format == 'json':
            description = "JSON (JavaScript Object Notation) format. Structured text format that can be generated by many programming languages."
        elif format == 'excel':
            description = "Excel format. Can be exported from Microsoft Excel or other compatible spreadsheet software."
        elif format == 'matlab':
            description = "MATLAB format. Can be exported from MATLAB for use in the Video Annotator."
        elif format == 'numpy':
            description = "NumPy format. Can be exported from Python using NumPy for use in the Video Annotator."
        else:
            description = ""
        
        # Update the description label
        self.format_description.config(text=description)
    
    @exception_handler
    def import_annotations(self):
        """Import the annotations."""
        # Get the input path
        input_path = self.input_path.get()
        
        if not input_path:
            show_error_message("Please specify an input file")
            return
        
        # Get the format
        format = self.format.get()
        
        # Import the annotations
        annotations = self.importer.import_annotations(input_path, format)
        
        if not annotations:
            show_error_message("Failed to import annotations or no annotations found")
            return
        
        # Call the callback function with the imported annotations
        self.callback(annotations)
        
        # Show a success message
        messagebox.showinfo("Success", f"Imported {len(annotations)} annotations from {input_path}")
        
        # Close the dialog
        self.destroy()
    
    def update_theme(self):
        """Update the dialog theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('bg')
            fg_color = self.theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
