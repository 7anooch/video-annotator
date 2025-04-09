#!/usr/bin/env python3
"""
Status bar component for the Video Annotator.

This module provides a status bar component for displaying information to the user.
"""

import tkinter as tk
from tkinter import ttk
import time
from typing import Optional, Dict, Any
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

# Set up logger
logger = setup_logger('status_bar')

class StatusBar(ttk.Frame):
    """
    Status bar component for displaying information to the user.
    
    Attributes:
        master (tk.Widget): The parent widget
        theme_manager (ThemeManager, optional): The theme manager
    """
    
    def __init__(self, master: tk.Widget, theme_manager=None):
        """
        Initialize the StatusBar.
        
        Args:
            master (tk.Widget): The parent widget
            theme_manager (ThemeManager, optional): The theme manager. Defaults to None.
        """
        self.logger = setup_logger('status_bar')
        
        # Get colors from theme manager if available
        bg_color = '#f0f0f0'
        fg_color = '#000000'
        if theme_manager:
            bg_color = theme_manager.get_theme_color('status_bg')
            fg_color = theme_manager.get_theme_color('status_fg')
        
        # Initialize the frame
        super().__init__(master, style='Status.TFrame')
        
        # Create a style for the status bar
        style = ttk.Style()
        style.configure('Status.TFrame', background=bg_color)
        style.configure('Status.TLabel', background=bg_color, foreground=fg_color)
        
        # Create the status message label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self, textvariable=self.status_var, style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Create the frame info label
        self.frame_info_var = tk.StringVar(value="Frame: 0/0")
        self.frame_info_label = ttk.Label(self, textvariable=self.frame_info_var, style='Status.TLabel')
        self.frame_info_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create the annotation info label
        self.annotation_info_var = tk.StringVar(value="Annotation: None")
        self.annotation_info_label = ttk.Label(self, textvariable=self.annotation_info_var, style='Status.TLabel')
        self.annotation_info_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create the playback speed label
        self.speed_var = tk.StringVar(value="Speed: 1.0x")
        self.speed_label = ttk.Label(self, textvariable=self.speed_var, style='Status.TLabel')
        self.speed_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create the time label
        self.time_var = tk.StringVar(value=time.strftime("%H:%M:%S"))
        self.time_label = ttk.Label(self, textvariable=self.time_var, style='Status.TLabel')
        self.time_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Start the clock update
        self._update_clock()
        
        # Store the theme manager
        self.theme_manager = theme_manager
    
    def _update_clock(self):
        """Update the clock display."""
        self.time_var.set(time.strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)
    
    def set_status(self, message: str) -> None:
        """
        Set the status message.
        
        Args:
            message (str): The status message
        """
        self.status_var.set(message)
        self.logger.info(f"Status: {message}")
    
    def set_frame_info(self, current_frame: int, total_frames: int) -> None:
        """
        Set the frame information.
        
        Args:
            current_frame (int): The current frame number
            total_frames (int): The total number of frames
        """
        self.frame_info_var.set(f"Frame: {current_frame}/{total_frames}")
    
    def set_annotation_info(self, annotation: Optional[Dict[str, Any]]) -> None:
        """
        Set the annotation information.
        
        Args:
            annotation (Optional[Dict[str, Any]]): The current annotation
        """
        if annotation:
            label_name = annotation.get('name', 'Unknown')
            self.annotation_info_var.set(f"Annotation: {label_name}")
        else:
            self.annotation_info_var.set("Annotation: None")
    
    def set_playback_speed(self, speed: float) -> None:
        """
        Set the playback speed.
        
        Args:
            speed (float): The playback speed
        """
        self.speed_var.set(f"Speed: {speed:.1f}x")
    
    def update_theme(self) -> None:
        """Update the status bar theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('status_bg')
            fg_color = self.theme_manager.get_theme_color('status_fg')
            
            style = ttk.Style()
            style.configure('Status.TFrame', background=bg_color)
            style.configure('Status.TLabel', background=bg_color, foreground=fg_color)
