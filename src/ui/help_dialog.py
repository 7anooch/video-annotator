#!/usr/bin/env python3
"""
Help dialog for the Video Annotator.

This module provides a help dialog for displaying keyboard shortcuts and other help information.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

# Set up logger
logger = setup_logger('help_dialog')

class HelpDialog(tk.Toplevel):
    """
    Help dialog for displaying keyboard shortcuts and other help information.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        keyboard_manager: The keyboard manager
        theme_manager (ThemeManager, optional): The theme manager
    """
    
    def __init__(self, master: tk.Tk, keyboard_manager, theme_manager=None):
        """
        Initialize the HelpDialog.
        
        Args:
            master (tk.Tk): The main Tkinter window
            keyboard_manager: The keyboard manager
            theme_manager (ThemeManager, optional): The theme manager. Defaults to None.
        """
        self.logger = setup_logger('help_dialog')
        
        # Initialize the toplevel window
        super().__init__(master)
        self.title("Help")
        self.geometry("600x500")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()
        
        # Store references
        self.keyboard_manager = keyboard_manager
        self.theme_manager = theme_manager
        
        # Apply theme if available
        if theme_manager:
            bg_color = theme_manager.get_theme_color('bg')
            fg_color = theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
        
        # Create the notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the shortcuts tab
        self.shortcuts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.shortcuts_frame, text="Keyboard Shortcuts")
        
        # Create the about tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")
        
        # Create the getting started tab
        self.getting_started_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.getting_started_frame, text="Getting Started")
        
        # Set up the tabs
        self._setup_shortcuts_tab()
        self._setup_about_tab()
        self._setup_getting_started_tab()
        
        # Create the close button
        self.close_button = ttk.Button(self, text="Close", command=self.destroy)
        self.close_button.pack(side=tk.BOTTOM, padx=10, pady=10)
    
    def _setup_shortcuts_tab(self):
        """Set up the shortcuts tab."""
        # Create a frame for the shortcuts
        shortcuts_frame = ttk.Frame(self.shortcuts_frame)
        shortcuts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a treeview for the shortcuts
        columns = ("Action", "Shortcut")
        self.shortcuts_tree = ttk.Treeview(shortcuts_frame, columns=columns, show="headings")
        
        # Set column headings
        self.shortcuts_tree.heading("Action", text="Action")
        self.shortcuts_tree.heading("Shortcut", text="Shortcut")
        
        # Set column widths
        self.shortcuts_tree.column("Action", width=200)
        self.shortcuts_tree.column("Shortcut", width=200)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(shortcuts_frame, orient=tk.VERTICAL, command=self.shortcuts_tree.yview)
        self.shortcuts_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.shortcuts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate the treeview
        self._populate_shortcuts_tree()
    
    def _populate_shortcuts_tree(self):
        """Populate the shortcuts treeview."""
        # Clear the treeview
        for item in self.shortcuts_tree.get_children():
            self.shortcuts_tree.delete(item)
        
        # Get all shortcuts
        shortcuts = self.keyboard_manager.get_all_shortcuts()
        
        # Add shortcuts to the treeview
        for action, key in shortcuts.items():
            # Get the action description
            action_desc = self.keyboard_manager.get_action_description(action)
            
            # Get the shortcut description
            shortcut_desc = self.keyboard_manager.get_shortcut_description(action)
            
            # Add to the treeview
            self.shortcuts_tree.insert("", tk.END, values=(action_desc, shortcut_desc))
    
    def _setup_about_tab(self):
        """Set up the about tab."""
        # Create a frame for the about information
        about_frame = ttk.Frame(self.about_frame)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a label for the title
        title_label = ttk.Label(about_frame, text="Video Annotator", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create a label for the version
        version_label = ttk.Label(about_frame, text="Version 1.0.0")
        version_label.pack(pady=5)
        
        # Create a label for the description
        description_text = (
            "Video Annotator is a tool for annotating video frames with behavioral labels. "
            "It is primarily designed for neuroscience research, but can be used for any "
            "application that requires frame-by-frame video annotation."
        )
        description_label = ttk.Label(about_frame, text=description_text, wraplength=500, justify=tk.CENTER)
        description_label.pack(pady=10)
        
        # Create a label for the copyright
        copyright_label = ttk.Label(about_frame, text="© 2024 Video Annotator Team")
        copyright_label.pack(pady=5)
        
        # Create a label for the license
        license_text = (
            "This software is licensed under the MIT License. "
            "See the LICENSE file for details."
        )
        license_label = ttk.Label(about_frame, text=license_text, wraplength=500, justify=tk.CENTER)
        license_label.pack(pady=10)
    
    def _setup_getting_started_tab(self):
        """Set up the getting started tab."""
        # Create a frame for the getting started information
        getting_started_frame = ttk.Frame(self.getting_started_frame)
        getting_started_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a scrollable text widget
        text = tk.Text(getting_started_frame, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(getting_started_frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        # Pack the text widget and scrollbar
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add getting started information
        text.insert(tk.END, "Getting Started with Video Annotator\n\n", "heading")
        text.insert(tk.END, "Video Annotator is a tool for annotating video frames with behavioral labels. "
                           "This guide will help you get started with the application.\n\n")
        
        text.insert(tk.END, "1. Loading a Video\n\n", "subheading")
        text.insert(tk.END, "To load a video, click the 'Open Video' button or use the File menu. "
                           "You can also drag and drop a video file onto the application window.\n\n")
        
        text.insert(tk.END, "2. Navigating the Video\n\n", "subheading")
        text.insert(tk.END, "Use the playback controls to navigate through the video:\n"
                           "- Play/Pause: Space\n"
                           "- Next Frame: Right Arrow\n"
                           "- Previous Frame: Left Arrow\n"
                           "- Next 10 Frames: Shift+Right Arrow\n"
                           "- Previous 10 Frames: Shift+Left Arrow\n"
                           "- Jump to Start: Home\n"
                           "- Jump to End: End\n\n")
        
        text.insert(tk.END, "3. Annotating Frames\n\n", "subheading")
        text.insert(tk.END, "To annotate a frame, press the key corresponding to the label you want to assign. "
                           "The default labels are:\n"
                           "- Stop: S\n"
                           "- Run: R\n"
                           "- Turn: T\n\n"
                           "You can customize these labels in the Settings.\n\n")
        
        text.insert(tk.END, "4. Saving Annotations\n\n", "subheading")
        text.insert(tk.END, "Annotations are automatically saved to a CSV file in the same directory as the video. "
                           "You can also manually save annotations using the File menu or by pressing Ctrl+S.\n\n")
        
        text.insert(tk.END, "5. Visualizing Annotations\n\n", "subheading")
        text.insert(tk.END, "You can visualize your annotations using the Visualization menu. "
                           "This allows you to see patterns in your annotations and identify areas that need attention.\n\n")
        
        text.insert(tk.END, "6. Customizing the Application\n\n", "subheading")
        text.insert(tk.END, "You can customize the application using the Settings menu. "
                           "This allows you to change the labels, keyboard shortcuts, UI theme, and other settings.\n\n")
        
        text.insert(tk.END, "For more information, please refer to the documentation or contact the developers.")
        
        # Configure tags
        text.tag_configure("heading", font=("Arial", 14, "bold"))
        text.tag_configure("subheading", font=("Arial", 12, "bold"))
        
        # Make the text widget read-only
        text.configure(state=tk.DISABLED)
    
    def update_theme(self):
        """Update the dialog theme."""
        if self.theme_manager:
            bg_color = self.theme_manager.get_theme_color('bg')
            fg_color = self.theme_manager.get_theme_color('fg')
            self.configure(background=bg_color)
