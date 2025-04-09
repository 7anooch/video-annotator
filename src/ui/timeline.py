#!/usr/bin/env python3
"""
Timeline component for the Video Annotator.

This module provides a timeline component for navigating through video frames.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

# Set up logger
logger = setup_logger('timeline')

class Timeline(ttk.Frame):
    """
    Timeline component for navigating through video frames.
    
    Attributes:
        master (tk.Widget): The parent widget
        video_player: The video player object
        annotation_manager: The annotation manager object
        theme_manager (ThemeManager, optional): The theme manager
    """
    
    def __init__(self, master: tk.Widget, video_player, annotation_manager, theme_manager=None):
        """
        Initialize the Timeline.
        
        Args:
            master (tk.Widget): The parent widget
            video_player: The video player object
            annotation_manager: The annotation manager object
            theme_manager (ThemeManager, optional): The theme manager. Defaults to None.
        """
        self.logger = setup_logger('timeline')
        
        # Get colors from theme manager if available
        bg_color = '#e0e0e0'
        fg_color = '#000000'
        marker_color = '#ff0000'
        annotation_color = '#4a6984'
        if theme_manager:
            bg_color = theme_manager.get_theme_color('timeline_bg')
            fg_color = theme_manager.get_theme_color('timeline_fg')
            marker_color = theme_manager.get_theme_color('timeline_marker')
            annotation_color = theme_manager.get_theme_color('timeline_annotation')
        
        # Initialize the frame
        super().__init__(master, style='Timeline.TFrame')
        
        # Create a style for the timeline
        style = ttk.Style()
        style.configure('Timeline.TFrame', background=bg_color)
        
        # Store references to the video player and annotation manager
        self.video_player = video_player
        self.annotation_manager = annotation_manager
        
        # Store the theme manager
        self.theme_manager = theme_manager
        
        # Create the timeline canvas
        self.canvas = tk.Canvas(self, bg=bg_color, height=50, highlightthickness=0)
        self.canvas.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create the frame slider
        self.slider_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL, 
                               variable=self.slider_var, command=self._on_slider_change)
        self.slider.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Create the frame counter
        self.frame_counter = ttk.Label(self, text="Frame: 0/0", style='Timeline.TLabel')
        self.frame_counter.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create the time counter
        self.time_counter = ttk.Label(self, text="Time: 00:00:00", style='Timeline.TLabel')
        self.time_counter.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Initialize variables
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.annotations = {}
        self.label_colors = {}
        self.marker_color = marker_color
        self.annotation_color = annotation_color
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
    
    def _on_slider_change(self, value):
        """
        Handle slider change events.
        
        Args:
            value: The new slider value
        """
        # Convert the slider value to a frame number
        frame = int(float(value))
        
        # Update the video player
        if frame != self.current_frame:
            self.video_player.set_frame(frame)
    
    def _on_canvas_click(self, event):
        """
        Handle canvas click events.
        
        Args:
            event: The click event
        """
        # Convert the click position to a frame number
        if self.total_frames > 0:
            frame = int(event.x / self.canvas.winfo_width() * self.total_frames)
            
            # Update the video player
            self.video_player.set_frame(frame)
    
    def _on_canvas_resize(self, event):
        """
        Handle canvas resize events.
        
        Args:
            event: The resize event
        """
        # Redraw the timeline
        self.draw_timeline()
    
    @exception_handler
    def update(self, current_frame: int, total_frames: int, fps: float) -> None:
        """
        Update the timeline.
        
        Args:
            current_frame (int): The current frame number
            total_frames (int): The total number of frames
            fps (float): The frames per second
        """
        # Update variables
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.fps = fps
        
        # Update the slider
        self.slider.configure(to=total_frames - 1)
        self.slider_var.set(current_frame)
        
        # Update the frame counter
        self.frame_counter.configure(text=f"Frame: {current_frame}/{total_frames}")
        
        # Update the time counter
        hours, remainder = divmod(current_frame / fps, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.time_counter.configure(text=f"Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Redraw the timeline
        self.draw_timeline()
    
    @exception_handler
    def update_annotations(self, annotations: Dict[int, Dict[str, Any]], label_colors: Dict[int, str]) -> None:
        """
        Update the annotations.
        
        Args:
            annotations (Dict[int, Dict[str, Any]]): The annotations
            label_colors (Dict[int, str]): The label colors
        """
        # Update variables
        self.annotations = annotations
        self.label_colors = label_colors
        
        # Redraw the timeline
        self.draw_timeline()
    
    @exception_handler
    def draw_timeline(self) -> None:
        """Draw the timeline."""
        # Clear the canvas
        self.canvas.delete("all")
        
        # Get the canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # If the canvas is not yet visible, return
        if width <= 1 or height <= 1:
            return
        
        # Draw the timeline background
        self.canvas.create_rectangle(0, 0, width, height, fill=self.theme_manager.get_theme_color('timeline_bg'))
        
        # Draw the annotations
        if self.total_frames > 0:
            for frame, annotation in self.annotations.items():
                # Calculate the x position
                x = frame / self.total_frames * width
                
                # Get the label color
                label_value = annotation.get('value', 0)
                color = self.label_colors.get(label_value, self.annotation_color)
                
                # Draw the annotation marker
                self.canvas.create_line(x, 0, x, height, fill=color, width=1)
            
            # Draw the current frame marker
            x = self.current_frame / self.total_frames * width
            self.canvas.create_line(x, 0, x, height, fill=self.marker_color, width=2)
    
    def update_theme(self) -> None:
        """Update the timeline theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('timeline_bg')
            fg_color = self.theme_manager.get_theme_color('timeline_fg')
            self.marker_color = self.theme_manager.get_theme_color('timeline_marker')
            self.annotation_color = self.theme_manager.get_theme_color('timeline_annotation')
            
            style = ttk.Style()
            style.configure('Timeline.TFrame', background=bg_color)
            style.configure('Timeline.TLabel', background=bg_color, foreground=fg_color)
            
            self.canvas.configure(bg=bg_color)
            
            # Redraw the timeline
            self.draw_timeline()
