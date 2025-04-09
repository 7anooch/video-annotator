#!/usr/bin/env python3
"""
Video processing dialog for the Video Annotator.

This module provides a dialog for processing video files.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any, List
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message
from src.utils.video_processor import VideoProcessor

class VideoProcessingDialog(tk.Toplevel):
    """
    Dialog for processing video files.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        video_processor (VideoProcessor): The video processor
        theme_manager: The theme manager
    """
    
    def __init__(self, master: tk.Tk, video_path: str = None, theme_manager=None):
        """
        Initialize the VideoProcessingDialog.
        
        Args:
            master (tk.Tk): The main Tkinter window
            video_path (str, optional): Path to the video file. Defaults to None.
            theme_manager: The theme manager. Defaults to None.
        """
        self.logger = setup_logger('video_processing_dialog')
        
        # Initialize the toplevel window
        super().__init__(master)
        self.title("Video Processing")
        self.geometry("800x600")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()
        
        # Store references
        self.master = master
        self.theme_manager = theme_manager
        self.video_processor = VideoProcessor()
        
        # Apply theme if available
        if theme_manager:
            bg_color = theme_manager.get_theme_color('bg')
            fg_color = theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
        
        # Initialize variables
        self.video_path = tk.StringVar(value=video_path if video_path else "")
        self.output_path = tk.StringVar(value="")
        self.start_frame = tk.IntVar(value=0)
        self.end_frame = tk.IntVar(value=0)
        self.step = tk.IntVar(value=1)
        self.format = tk.StringVar(value="jpg")
        self.quality = tk.IntVar(value=95)
        self.video_info = {}
        
        # Set up the UI
        self.setup_ui()
        
        # Load video info if a video path is provided
        if video_path:
            self.load_video_info()
    
    def setup_ui(self):
        """Set up the UI elements."""
        # Create the main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Video Selection")
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the video path entry
        ttk.Label(video_frame, text="Video Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        video_entry = ttk.Entry(video_frame, textvariable=self.video_path, width=50)
        video_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the browse button
        browse_button = ttk.Button(video_frame, text="Browse", command=self.browse_video)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Create the load button
        load_button = ttk.Button(video_frame, text="Load", command=self.load_video_info)
        load_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Create the video info frame
        info_frame = ttk.LabelFrame(main_frame, text="Video Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the video info text
        self.info_text = tk.Text(info_frame, height=10, width=80, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        self.info_text.config(state=tk.DISABLED)
        
        # Create the notebook for different processing options
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the frame extraction tab
        self.extraction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.extraction_frame, text="Frame Extraction")
        self.setup_extraction_tab()
        
        # Create the video trimming tab
        self.trimming_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trimming_frame, text="Video Trimming")
        self.setup_trimming_tab()
        
        # Create the buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the close button
        close_button = ttk.Button(buttons_frame, text="Close", command=self.destroy)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def setup_extraction_tab(self):
        """Set up the frame extraction tab."""
        # Create the output directory frame
        output_frame = ttk.LabelFrame(self.extraction_frame, text="Output Directory")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the output directory entry
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=50)
        output_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the browse button
        browse_button = ttk.Button(output_frame, text="Browse", command=self.browse_output_directory)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Create the extraction options frame
        options_frame = ttk.LabelFrame(self.extraction_frame, text="Extraction Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the start frame entry
        ttk.Label(options_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        start_entry = ttk.Entry(options_frame, textvariable=self.start_frame, width=10)
        start_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create the end frame entry
        ttk.Label(options_frame, text="End Frame:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        end_entry = ttk.Entry(options_frame, textvariable=self.end_frame, width=10)
        end_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Create the step entry
        ttk.Label(options_frame, text="Step:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        step_entry = ttk.Entry(options_frame, textvariable=self.step, width=10)
        step_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create the format combobox
        ttk.Label(options_frame, text="Format:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        format_combo = ttk.Combobox(options_frame, textvariable=self.format, values=["jpg", "png"], width=8)
        format_combo.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Create the quality scale
        ttk.Label(options_frame, text="Quality:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        quality_scale = ttk.Scale(options_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                variable=self.quality, length=200)
        quality_scale.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the extract button
        extract_button = ttk.Button(options_frame, text="Extract Frames", command=self.extract_frames)
        extract_button.grid(row=3, column=0, columnspan=4, padx=5, pady=5)
    
    def setup_trimming_tab(self):
        """Set up the video trimming tab."""
        # Create the output file frame
        output_frame = ttk.LabelFrame(self.trimming_frame, text="Output File")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the output file entry
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, width=50)
        output_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create the browse button
        browse_button = ttk.Button(output_frame, text="Browse", command=self.browse_output_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Create the trimming options frame
        options_frame = ttk.LabelFrame(self.trimming_frame, text="Trimming Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create the start frame entry
        ttk.Label(options_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        start_entry = ttk.Entry(options_frame, textvariable=self.start_frame, width=10)
        start_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Create the end frame entry
        ttk.Label(options_frame, text="End Frame:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        end_entry = ttk.Entry(options_frame, textvariable=self.end_frame, width=10)
        end_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Create the trim button
        trim_button = ttk.Button(options_frame, text="Trim Video", command=self.trim_video)
        trim_button.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
    
    @exception_handler
    def browse_video(self):
        """Browse for a video file."""
        # Get the supported formats
        formats = self.video_processor.get_supported_formats()
        
        # Create a file filter
        file_filter = [("Video Files", " ".join(["*" + fmt for fmt in formats]))]
        
        # Open a file dialog
        file_path = filedialog.askopenfilename(filetypes=file_filter)
        
        if file_path:
            self.video_path.set(file_path)
            self.load_video_info()
    
    @exception_handler
    def browse_output_directory(self):
        """Browse for an output directory."""
        # Open a directory dialog
        directory = filedialog.askdirectory()
        
        if directory:
            self.output_path.set(directory)
    
    @exception_handler
    def browse_output_file(self):
        """Browse for an output file."""
        # Get the supported formats
        formats = self.video_processor.get_supported_formats()
        
        # Create a file filter
        file_filter = [("Video Files", " ".join(["*" + fmt for fmt in formats]))]
        
        # Open a file dialog
        file_path = filedialog.asksaveasfilename(filetypes=file_filter, defaultextension=".mp4")
        
        if file_path:
            self.output_path.set(file_path)
    
    @exception_handler
    def load_video_info(self):
        """Load information about the video."""
        # Get the video path
        video_path = self.video_path.get()
        
        if not video_path:
            show_error_message("Please select a video file")
            return
        
        # Get video info
        self.video_info = self.video_processor.get_video_info(video_path)
        
        if not self.video_info:
            show_error_message("Could not load video information")
            return
        
        # Update the info text
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        # Add video info to the text widget
        self.info_text.insert(tk.END, f"Filename: {self.video_info['filename']}\n")
        self.info_text.insert(tk.END, f"Resolution: {self.video_info['width']}x{self.video_info['height']}\n")
        self.info_text.insert(tk.END, f"FPS: {self.video_info['fps']:.2f}\n")
        self.info_text.insert(tk.END, f"Total Frames: {self.video_info['total_frames']}\n")
        self.info_text.insert(tk.END, f"Duration: {self.video_info['duration']:.2f} seconds\n")
        self.info_text.insert(tk.END, f"Codec: {self.video_info['codec']}\n")
        self.info_text.insert(tk.END, f"File Size: {self.video_info['file_size_mb']:.2f} MB\n")
        
        self.info_text.config(state=tk.DISABLED)
        
        # Update the end frame
        self.end_frame.set(self.video_info['total_frames'] - 1)
    
    @exception_handler
    def extract_frames(self):
        """Extract frames from the video."""
        # Get the video path
        video_path = self.video_path.get()
        
        if not video_path:
            show_error_message("Please select a video file")
            return
        
        # Get the output directory
        output_dir = self.output_path.get()
        
        if not output_dir:
            show_error_message("Please select an output directory")
            return
        
        # Get the extraction options
        start_frame = self.start_frame.get()
        end_frame = self.end_frame.get()
        step = self.step.get()
        format = self.format.get()
        quality = self.quality.get()
        
        # Validate the options
        if start_frame < 0:
            show_error_message("Start frame must be non-negative")
            return
        
        if end_frame < start_frame:
            show_error_message("End frame must be greater than or equal to start frame")
            return
        
        if step < 1:
            show_error_message("Step must be at least 1")
            return
        
        # Extract the frames
        frame_paths = self.video_processor.extract_frames(
            video_path, output_dir, start_frame, end_frame, step, format, quality
        )
        
        if not frame_paths:
            show_error_message("Failed to extract frames")
            return
        
        # Show a success message
        messagebox.showinfo("Success", f"Extracted {len(frame_paths)} frames to {output_dir}")
    
    @exception_handler
    def trim_video(self):
        """Trim the video."""
        # Get the video path
        video_path = self.video_path.get()
        
        if not video_path:
            show_error_message("Please select a video file")
            return
        
        # Get the output file
        output_path = self.output_path.get()
        
        if not output_path:
            show_error_message("Please select an output file")
            return
        
        # Get the trimming options
        start_frame = self.start_frame.get()
        end_frame = self.end_frame.get()
        
        # Validate the options
        if start_frame < 0:
            show_error_message("Start frame must be non-negative")
            return
        
        if end_frame < start_frame:
            show_error_message("End frame must be greater than or equal to start frame")
            return
        
        # Trim the video
        success = self.video_processor.trim_video(video_path, output_path, start_frame, end_frame)
        
        if not success:
            show_error_message("Failed to trim video")
            return
        
        # Show a success message
        messagebox.showinfo("Success", f"Trimmed video saved to {output_path}")
    
    def update_theme(self):
        """Update the dialog theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('bg')
            fg_color = self.theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
