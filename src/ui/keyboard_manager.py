#!/usr/bin/env python3
"""
Keyboard shortcuts manager for the Video Annotator.

This module provides functionality for managing keyboard shortcuts.
"""

import tkinter as tk
from typing import Dict, Any, Callable, Optional
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

# Set up logger
logger = setup_logger('keyboard_manager')

class KeyboardManager:
    """
    Manages keyboard shortcuts for the application.
    
    Attributes:
        master (tk.Tk): The main Tkinter window
        shortcuts (Dict[str, Dict[str, Any]]): Dictionary of keyboard shortcuts
    """
    
    def __init__(self, master: tk.Tk, shortcuts: Optional[Dict[str, str]] = None):
        """
        Initialize the KeyboardManager.
        
        Args:
            master (tk.Tk): The main Tkinter window
            shortcuts (Dict[str, str], optional): Dictionary of keyboard shortcuts. Defaults to None.
        """
        self.logger = setup_logger('keyboard_manager')
        self.master = master
        
        # Default shortcuts
        self.default_shortcuts = {
            'play_pause': 'space',
            'next_frame': 'Right',
            'prev_frame': 'Left',
            'next_10_frames': 'Shift+Right',
            'prev_10_frames': 'Shift+Left',
            'save': 'Control+s',
            'quit': 'Control+q',
            'toggle_fullscreen': 'F11',
            'toggle_controls': 'F10',
            'toggle_timeline': 'F9',
            'toggle_status_bar': 'F8',
            'increase_speed': 'plus',
            'decrease_speed': 'minus',
            'reset_speed': '0',
            'jump_to_start': 'Home',
            'jump_to_end': 'End',
            'delete_annotation': 'Delete',
            'copy_annotation': 'Control+c',
            'paste_annotation': 'Control+v',
            'undo': 'Control+z',
            'redo': 'Control+y',
            'help': 'F1',
            'settings': 'F2',
            'toggle_theme': 'F3',
            'zoom_in': 'Control+plus',
            'zoom_out': 'Control+minus',
            'zoom_reset': 'Control+0'
        }
        
        # Use provided shortcuts or defaults
        self.shortcuts = shortcuts or self.default_shortcuts
        
        # Dictionary to store callback functions
        self.callbacks = {}
        
        # Dictionary to store key bindings
        self.bindings = {}
    
    def register_callback(self, action: str, callback: Callable) -> None:
        """
        Register a callback function for a keyboard action.
        
        Args:
            action (str): The action name
            callback (Callable): The callback function
        """
        self.callbacks[action] = callback
        
        # If the action has a shortcut, bind it
        if action in self.shortcuts:
            self.bind_shortcut(action, self.shortcuts[action])
    
    def bind_shortcut(self, action: str, key: str) -> None:
        """
        Bind a keyboard shortcut to an action.
        
        Args:
            action (str): The action name
            key (str): The keyboard shortcut
        """
        # Convert the key to Tkinter format
        tk_key = self._convert_to_tk_key(key)
        
        # If the action already has a binding, unbind it
        if action in self.bindings:
            self.master.unbind(self.bindings[action])
        
        # Bind the new shortcut
        if action in self.callbacks:
            binding_id = self.master.bind(tk_key, lambda event: self.callbacks[action]())
            self.bindings[action] = binding_id
            self.logger.info(f"Bound {key} to {action}")
        else:
            self.logger.warning(f"No callback registered for action: {action}")
    
    def _convert_to_tk_key(self, key: str) -> str:
        """
        Convert a key string to Tkinter format.
        
        Args:
            key (str): The key string (e.g., 'Control+s')
            
        Returns:
            str: The key in Tkinter format (e.g., '<Control-s>')
        """
        # Special case for space
        if key.lower() == 'space':
            return '<space>'
        
        # Split the key into modifiers and key
        parts = key.split('+')
        
        # Convert modifiers to Tkinter format
        for i, part in enumerate(parts):
            if part.lower() in ['control', 'ctrl']:
                parts[i] = 'Control'
            elif part.lower() in ['alt']:
                parts[i] = 'Alt'
            elif part.lower() in ['shift']:
                parts[i] = 'Shift'
        
        # Join the parts with '-'
        tk_key = '-'.join(parts)
        
        # Add angle brackets
        return f'<{tk_key}>'
    
    def get_shortcut(self, action: str) -> str:
        """
        Get the keyboard shortcut for an action.
        
        Args:
            action (str): The action name
            
        Returns:
            str: The keyboard shortcut
        """
        return self.shortcuts.get(action, '')
    
    def set_shortcut(self, action: str, key: str) -> None:
        """
        Set the keyboard shortcut for an action.
        
        Args:
            action (str): The action name
            key (str): The keyboard shortcut
        """
        self.shortcuts[action] = key
        
        # If the action has a callback, update the binding
        if action in self.callbacks:
            self.bind_shortcut(action, key)
    
    def reset_shortcuts(self) -> None:
        """Reset all shortcuts to their default values."""
        # Unbind all current shortcuts
        for action in self.bindings:
            self.master.unbind(self.bindings[action])
        
        # Reset shortcuts to defaults
        self.shortcuts = self.default_shortcuts.copy()
        
        # Rebind all shortcuts
        for action in self.callbacks:
            if action in self.shortcuts:
                self.bind_shortcut(action, self.shortcuts[action])
        
        self.logger.info("Reset all keyboard shortcuts to defaults")
    
    def get_all_shortcuts(self) -> Dict[str, str]:
        """
        Get all keyboard shortcuts.
        
        Returns:
            Dict[str, str]: Dictionary of all keyboard shortcuts
        """
        return self.shortcuts.copy()
    
    def get_shortcut_description(self, action: str) -> str:
        """
        Get a human-readable description of a keyboard shortcut.
        
        Args:
            action (str): The action name
            
        Returns:
            str: The shortcut description
        """
        shortcut = self.get_shortcut(action)
        
        if not shortcut:
            return "No shortcut assigned"
        
        # Replace + with space+space for better readability
        shortcut = shortcut.replace('+', ' + ')
        
        # Capitalize first letter of each part
        parts = shortcut.split(' + ')
        parts = [part.capitalize() for part in parts]
        
        return ' + '.join(parts)
    
    def get_action_description(self, action: str) -> str:
        """
        Get a human-readable description of an action.
        
        Args:
            action (str): The action name
            
        Returns:
            str: The action description
        """
        descriptions = {
            'play_pause': 'Play/Pause',
            'next_frame': 'Next Frame',
            'prev_frame': 'Previous Frame',
            'next_10_frames': 'Next 10 Frames',
            'prev_10_frames': 'Previous 10 Frames',
            'save': 'Save',
            'quit': 'Quit',
            'toggle_fullscreen': 'Toggle Fullscreen',
            'toggle_controls': 'Toggle Controls',
            'toggle_timeline': 'Toggle Timeline',
            'toggle_status_bar': 'Toggle Status Bar',
            'increase_speed': 'Increase Speed',
            'decrease_speed': 'Decrease Speed',
            'reset_speed': 'Reset Speed',
            'jump_to_start': 'Jump to Start',
            'jump_to_end': 'Jump to End',
            'delete_annotation': 'Delete Annotation',
            'copy_annotation': 'Copy Annotation',
            'paste_annotation': 'Paste Annotation',
            'undo': 'Undo',
            'redo': 'Redo',
            'help': 'Help',
            'settings': 'Settings',
            'toggle_theme': 'Toggle Theme',
            'zoom_in': 'Zoom In',
            'zoom_out': 'Zoom Out',
            'zoom_reset': 'Reset Zoom'
        }
        
        return descriptions.get(action, action.replace('_', ' ').title())
