#!/usr/bin/env python3
"""
Theme manager for the Video Annotator.

This module provides functionality for managing UI themes.
"""

import tkinter as tk
from tkinter import ttk
import os
import json
from typing import Dict, Any, Optional
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

# Set up logger
logger = setup_logger('theme_manager')

class ThemeManager:
    """
    Manages UI themes for the application.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        current_theme (str): The name of the current theme
        themes (Dict[str, Dict[str, Any]]): Dictionary of available themes
    """
    
    def __init__(self, master: tk.Tk, theme_name: str = 'default'):
        """
        Initialize the ThemeManager.
        
        Args:
            master (tk.Tk): The main Tkinter window
            theme_name (str, optional): The name of the theme to use. Defaults to 'default'.
        """
        self.logger = setup_logger('theme_manager')
        self.master = master
        self.current_theme = theme_name
        
        # Define built-in themes
        self.themes = {
            'default': {
                'bg': '#f0f0f0',
                'fg': '#000000',
                'button_bg': '#e0e0e0',
                'button_fg': '#000000',
                'highlight_bg': '#4a6984',
                'highlight_fg': '#ffffff',
                'entry_bg': '#ffffff',
                'entry_fg': '#000000',
                'canvas_bg': '#000000',
                'label_bg': '#f0f0f0',
                'label_fg': '#000000',
                'menu_bg': '#f0f0f0',
                'menu_fg': '#000000',
                'status_bg': '#f0f0f0',
                'status_fg': '#000000',
                'timeline_bg': '#e0e0e0',
                'timeline_fg': '#000000',
                'timeline_marker': '#ff0000',
                'timeline_annotation': '#4a6984',
                'font_family': 'Arial',
                'font_size': 10
            },
            'dark': {
                'bg': '#2d2d2d',
                'fg': '#ffffff',
                'button_bg': '#3d3d3d',
                'button_fg': '#ffffff',
                'highlight_bg': '#4a6984',
                'highlight_fg': '#ffffff',
                'entry_bg': '#3d3d3d',
                'entry_fg': '#ffffff',
                'canvas_bg': '#000000',
                'label_bg': '#2d2d2d',
                'label_fg': '#ffffff',
                'menu_bg': '#2d2d2d',
                'menu_fg': '#ffffff',
                'status_bg': '#2d2d2d',
                'status_fg': '#ffffff',
                'timeline_bg': '#3d3d3d',
                'timeline_fg': '#ffffff',
                'timeline_marker': '#ff0000',
                'timeline_annotation': '#4a6984',
                'font_family': 'Arial',
                'font_size': 10
            },
            'light_blue': {
                'bg': '#e6f3ff',
                'fg': '#000000',
                'button_bg': '#c1e0ff',
                'button_fg': '#000000',
                'highlight_bg': '#4a6984',
                'highlight_fg': '#ffffff',
                'entry_bg': '#ffffff',
                'entry_fg': '#000000',
                'canvas_bg': '#000000',
                'label_bg': '#e6f3ff',
                'label_fg': '#000000',
                'menu_bg': '#e6f3ff',
                'menu_fg': '#000000',
                'status_bg': '#e6f3ff',
                'status_fg': '#000000',
                'timeline_bg': '#c1e0ff',
                'timeline_fg': '#000000',
                'timeline_marker': '#ff0000',
                'timeline_annotation': '#4a6984',
                'font_family': 'Arial',
                'font_size': 10
            },
            'high_contrast': {
                'bg': '#000000',
                'fg': '#ffffff',
                'button_bg': '#000000',
                'button_fg': '#ffffff',
                'highlight_bg': '#ffff00',
                'highlight_fg': '#000000',
                'entry_bg': '#000000',
                'entry_fg': '#ffffff',
                'canvas_bg': '#000000',
                'label_bg': '#000000',
                'label_fg': '#ffffff',
                'menu_bg': '#000000',
                'menu_fg': '#ffffff',
                'status_bg': '#000000',
                'status_fg': '#ffffff',
                'timeline_bg': '#000000',
                'timeline_fg': '#ffffff',
                'timeline_marker': '#ffff00',
                'timeline_annotation': '#ffff00',
                'font_family': 'Arial',
                'font_size': 12
            }
        }
        
        # Load custom themes if available
        self.load_custom_themes()
        
        # Apply the specified theme
        self.apply_theme(theme_name)
    
    @exception_handler
    def load_custom_themes(self) -> None:
        """Load custom themes from the themes directory."""
        themes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'themes')
        
        if not os.path.exists(themes_dir):
            os.makedirs(themes_dir, exist_ok=True)
            self.logger.info(f"Created themes directory at {themes_dir}")
            return
        
        for filename in os.listdir(themes_dir):
            if filename.endswith('.json'):
                theme_path = os.path.join(themes_dir, filename)
                theme_name = os.path.splitext(filename)[0]
                
                try:
                    with open(theme_path, 'r') as f:
                        theme_data = json.load(f)
                    
                    # Validate theme data
                    if self._validate_theme(theme_data):
                        self.themes[theme_name] = theme_data
                        self.logger.info(f"Loaded custom theme: {theme_name}")
                    else:
                        self.logger.warning(f"Invalid theme data in {theme_path}")
                except Exception as e:
                    self.logger.error(f"Error loading theme from {theme_path}: {str(e)}")
    
    def _validate_theme(self, theme_data: Dict[str, Any]) -> bool:
        """
        Validate theme data.
        
        Args:
            theme_data (Dict[str, Any]): The theme data to validate
            
        Returns:
            bool: True if the theme data is valid, False otherwise
        """
        required_keys = [
            'bg', 'fg', 'button_bg', 'button_fg', 'highlight_bg', 'highlight_fg',
            'entry_bg', 'entry_fg', 'canvas_bg', 'label_bg', 'label_fg',
            'menu_bg', 'menu_fg', 'status_bg', 'status_fg',
            'timeline_bg', 'timeline_fg', 'timeline_marker', 'timeline_annotation',
            'font_family', 'font_size'
        ]
        
        return all(key in theme_data for key in required_keys)
    
    @exception_handler
    def apply_theme(self, theme_name: str) -> None:
        """
        Apply a theme to the application.
        
        Args:
            theme_name (str): The name of the theme to apply
        """
        if theme_name not in self.themes:
            self.logger.warning(f"Theme '{theme_name}' not found. Using default theme.")
            theme_name = 'default'
        
        theme = self.themes[theme_name]
        self.current_theme = theme_name
        
        # Configure ttk styles
        style = ttk.Style(self.master)
        
        # Configure TButton
        style.configure('TButton', 
                        background=theme['button_bg'],
                        foreground=theme['button_fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TLabel
        style.configure('TLabel', 
                        background=theme['label_bg'],
                        foreground=theme['label_fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TEntry
        style.configure('TEntry', 
                        fieldbackground=theme['entry_bg'],
                        foreground=theme['entry_fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TFrame
        style.configure('TFrame', background=theme['bg'])
        
        # Configure TLabelframe
        style.configure('TLabelframe', background=theme['bg'])
        style.configure('TLabelframe.Label', 
                        background=theme['bg'],
                        foreground=theme['fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TNotebook
        style.configure('TNotebook', background=theme['bg'])
        style.configure('TNotebook.Tab', 
                        background=theme['button_bg'],
                        foreground=theme['button_fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TProgressbar
        style.configure('TProgressbar', 
                        background=theme['highlight_bg'],
                        troughcolor=theme['bg'])
        
        # Configure TScale
        style.configure('TScale', 
                        background=theme['bg'],
                        troughcolor=theme['timeline_bg'])
        
        # Configure TScrollbar
        style.configure('TScrollbar', 
                        background=theme['button_bg'],
                        troughcolor=theme['bg'])
        
        # Configure TCombobox
        style.configure('TCombobox', 
                        fieldbackground=theme['entry_bg'],
                        foreground=theme['entry_fg'],
                        background=theme['button_bg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TCheckbutton
        style.configure('TCheckbutton', 
                        background=theme['bg'],
                        foreground=theme['fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure TRadiobutton
        style.configure('TRadiobutton', 
                        background=theme['bg'],
                        foreground=theme['fg'],
                        font=(theme['font_family'], theme['font_size']))
        
        # Configure the root window
        self.master.configure(background=theme['bg'])
        
        # Log the theme change
        self.logger.info(f"Applied theme: {theme_name}")
    
    def get_theme_color(self, color_key: str) -> str:
        """
        Get a color from the current theme.
        
        Args:
            color_key (str): The key of the color to get
            
        Returns:
            str: The color value
        """
        theme = self.themes[self.current_theme]
        return theme.get(color_key, '#000000')
    
    def get_theme_font(self, size_modifier: int = 0) -> tuple:
        """
        Get the font from the current theme.
        
        Args:
            size_modifier (int, optional): Modifier to add to the font size. Defaults to 0.
            
        Returns:
            tuple: The font family and size
        """
        theme = self.themes[self.current_theme]
        return (theme['font_family'], theme['font_size'] + size_modifier)
    
    def get_available_themes(self) -> list:
        """
        Get a list of available themes.
        
        Returns:
            list: List of theme names
        """
        return list(self.themes.keys())
    
    @exception_handler
    def save_custom_theme(self, theme_name: str, theme_data: Dict[str, Any]) -> bool:
        """
        Save a custom theme.
        
        Args:
            theme_name (str): The name of the theme
            theme_data (Dict[str, Any]): The theme data
            
        Returns:
            bool: True if the theme was saved successfully, False otherwise
        """
        # Validate theme data
        if not self._validate_theme(theme_data):
            self.logger.warning(f"Invalid theme data for {theme_name}")
            return False
        
        # Create themes directory if it doesn't exist
        themes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'themes')
        if not os.path.exists(themes_dir):
            os.makedirs(themes_dir, exist_ok=True)
        
        # Save the theme
        theme_path = os.path.join(themes_dir, f"{theme_name}.json")
        try:
            with open(theme_path, 'w') as f:
                json.dump(theme_data, f, indent=4)
            
            # Add the theme to the available themes
            self.themes[theme_name] = theme_data
            
            self.logger.info(f"Saved custom theme: {theme_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving theme to {theme_path}: {str(e)}")
            return False
    
    @exception_handler
    def delete_custom_theme(self, theme_name: str) -> bool:
        """
        Delete a custom theme.
        
        Args:
            theme_name (str): The name of the theme to delete
            
        Returns:
            bool: True if the theme was deleted successfully, False otherwise
        """
        # Don't allow deleting built-in themes
        if theme_name in ['default', 'dark', 'light_blue', 'high_contrast']:
            self.logger.warning(f"Cannot delete built-in theme: {theme_name}")
            return False
        
        # Check if the theme exists
        if theme_name not in self.themes:
            self.logger.warning(f"Theme '{theme_name}' not found")
            return False
        
        # Delete the theme file
        themes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'themes')
        theme_path = os.path.join(themes_dir, f"{theme_name}.json")
        
        if os.path.exists(theme_path):
            try:
                os.remove(theme_path)
                
                # Remove the theme from the available themes
                del self.themes[theme_name]
                
                # If the current theme was deleted, switch to the default theme
                if self.current_theme == theme_name:
                    self.apply_theme('default')
                
                self.logger.info(f"Deleted custom theme: {theme_name}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting theme {theme_name}: {str(e)}")
                return False
        else:
            self.logger.warning(f"Theme file not found: {theme_path}")
            return False
